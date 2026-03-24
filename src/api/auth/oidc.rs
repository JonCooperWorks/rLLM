// ===========================================================================
// OIDC auth provider — JWT validation against an OpenID Connect issuer.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements AuthProvider for OIDC (OpenID Connect) JWT validation.
//   Fetches the issuer's discovery document and JWKS at startup, then
//   validates Bearer tokens on every request against the cached keys.
//
// What OIDC is and why we use it:
//   OIDC is a standard identity layer built on top of OAuth 2.0.  Any
//   compliant identity provider (Google, Okta, Auth0, Keycloak, etc.)
//   publishes a discovery document at /.well-known/openid-configuration
//   that tells us where to find the signing keys (JWKS endpoint).  We
//   fetch those keys once, cache them, and use them to verify JWTs
//   without ever talking to the identity provider on the request path.
//
//   This means rLLM doesn't need to know anything about the org's auth
//   system — it just needs the issuer URL and the expected audience.
//   The gateway (or whatever mints the token) handles the complexity
//   of user sessions, OAuth flows, token exchange, etc.
//
// Three hooks:
//   init()         — GET /.well-known/openid-configuration to find the
//                    JWKS URI, then GET the JWKS and cache it.  Fails
//                    fast if the issuer is unreachable or misconfigured.
//
//   authenticate() — extract Bearer token from Authorization header,
//                    decode the JWT header to get the kid (key ID),
//                    look up the key in the cache, verify signature +
//                    exp + iss + aud claims.  All local — no network I/O
//                    on the request path unless a key rotation triggers
//                    an eager refresh.
//
//   background()   — sleep 60 minutes, re-fetch JWKS, compare key IDs.
//                    Only acquires the write lock if keys actually
//                    changed — request-path readers are never blocked
//                    by a routine refresh where nothing rotated.
//
// Locking strategy:
//   The JWKS is behind a tokio::sync::RwLock.  The request path takes a
//   read lock (non-exclusive, many concurrent readers).  The background
//   task and eager refresh take a write lock only when keys change.
//   This means the common case (keys haven't rotated) has zero
//   contention between the background task and request handlers.
//
// Eager refresh:
//   When a JWT's kid isn't in the cache, the request-path code tries one
//   eager JWKS refresh before rejecting.  This handles the window between
//   key rotation and the next background refresh.  Rate-limited to once
//   per 30 seconds to prevent an attacker from causing refresh storms by
//   sending JWTs with random kids.
//
// Config format (auth.json):
//   {
//     "provider": "oidc",
//     "issuer": "https://accounts.google.com",
//     "audience": "my-rllm-instance"
//   }
//
//   - issuer: the OIDC issuer URL.  Must match the `iss` claim in tokens
//     and the `issuer` field in the discovery document.
//   - audience: the expected `aud` claim.  Typically the client ID or a
//     service identifier that the gateway uses when minting tokens.
//
// What we validate on each request:
//   1. Authorization header present and starts with "Bearer "
//   2. JWT header is valid and contains a kid
//   3. kid exists in the cached JWKS
//   4. Signature is valid (RS256, ES256, etc. — whatever the JWK specifies)
//   5. exp claim is in the future (token not expired)
//   6. iss claim matches the configured issuer
//   7. aud claim matches the configured audience
//
// What we do NOT validate:
//   - Authorization (all authenticated users have equal access)
//   - Token revocation (no introspection endpoint, no CRL checks)
//   - Scopes or roles (not relevant — rLLM has no permission model)
//
// Related files:
//   api/auth/mod.rs         — AuthProvider trait, AuthDecision, AuthUser
//   api/mod.rs              — wires provider into ServerState
//   docs/authentication.md  — full design rationale
// ===========================================================================

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::{self, HeaderMap, StatusCode};
use jsonwebtoken::jwk::JwkSet;
use tokio::sync::RwLock;

use super::{AuthDecision, AuthProvider, AuthUser};

// ---------------------------------------------------------------------------
// Config parsed from auth.json.
//
// Only the fields the OIDC provider needs.  The framework passes the full
// JSON value — other fields (like "provider") are ignored by serde.
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct OidcConfig {
    /// OIDC issuer URL (e.g. "https://accounts.google.com").
    issuer: String,
    /// Expected audience claim in JWTs.
    audience: String,
}

// ---------------------------------------------------------------------------
// Discovery document — the subset of /.well-known/openid-configuration
// that we actually need.  OIDC discovery documents have many more fields
// (authorization_endpoint, token_endpoint, supported scopes, etc.) but
// we only care about the issuer (for validation) and jwks_uri (to fetch keys).
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct DiscoveryDocument {
    /// Must match the configured issuer — if it doesn't, someone is doing
    /// something wrong (typo, MITM, wrong endpoint).
    issuer: String,
    /// URL of the JWKS endpoint where we fetch the signing keys.
    jwks_uri: String,
}

// ---------------------------------------------------------------------------
// OidcProvider — the auth provider instance.
//
// All fields are immutable after init() except `jwks` (updated by background
// refresh) and `last_eager_refresh` (updated by request-path eager refresh).
// Both are behind RwLock for safe concurrent access.
// ---------------------------------------------------------------------------

pub(crate) struct OidcProvider {
    /// Cached JWKS (JSON Web Key Set).  Read on every request, written only
    /// when keys rotate (background refresh or eager refresh).
    jwks: Arc<RwLock<JwkSet>>,
    /// Expected issuer (`iss` claim).  Set once in init(), never changes.
    issuer: String,
    /// Expected audience (`aud` claim).  Set once in init(), never changes.
    audience: String,
    /// JWKS endpoint URL, discovered from the OIDC discovery document.
    jwks_uri: String,
    /// HTTP client for JWKS fetches.  Reuses the rustls TLS stack already
    /// in the dependency tree (via reqwest's rustls-tls feature).
    http: reqwest::Client,
    /// Timestamp of the last eager refresh.  Rate-limits request-triggered
    /// JWKS refreshes to one per 30 seconds, preventing an attacker from
    /// causing refresh storms by sending JWTs with fabricated kid values.
    last_eager_refresh: Arc<RwLock<Instant>>,
}

impl AuthProvider for OidcProvider {
    /// Hook 1: Init — discover JWKS endpoint and fetch signing keys.
    ///
    /// Startup flow:
    ///   1. Parse issuer + audience from auth.json
    ///   2. GET {issuer}/.well-known/openid-configuration
    ///   3. Validate that discovery.issuer matches config.issuer
    ///   4. GET {discovery.jwks_uri} → parse into JwkSet
    ///   5. Cache the JwkSet in an Arc<RwLock> for request-path use
    ///
    /// If any step fails, init() returns an error and the server refuses
    /// to start.  This is intentional — running with auth "enabled" but
    /// broken keys is worse than not starting at all.
    async fn init(config: &serde_json::Value) -> anyhow::Result<Self> {
        let oidc_config: OidcConfig = serde_json::from_value(config.clone())
            .map_err(|e| anyhow::anyhow!("invalid OIDC config: {e}"))?;

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;

        // Step 2: fetch OIDC discovery document.
        let discovery_url = format!(
            "{}/.well-known/openid-configuration",
            oidc_config.issuer.trim_end_matches('/')
        );
        let discovery: DiscoveryDocument = http
            .get(&discovery_url)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("failed to fetch OIDC discovery from {discovery_url}: {e}"))?
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("invalid OIDC discovery document: {e}"))?;

        // Step 3: validate issuer.  OIDC spec requires exact match.
        // We trim trailing slashes for robustness (some issuers include
        // them, some don't).
        if discovery.issuer.trim_end_matches('/') != oidc_config.issuer.trim_end_matches('/') {
            anyhow::bail!(
                "OIDC issuer mismatch: config has '{}', discovery has '{}'",
                oidc_config.issuer,
                discovery.issuer
            );
        }

        // Step 4: fetch JWKS.
        let jwks: JwkSet = http
            .get(&discovery.jwks_uri)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("failed to fetch JWKS from {}: {e}", discovery.jwks_uri))?
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("invalid JWKS: {e}"))?;

        eprintln!(
            "  oidc      : {} ({} keys cached)",
            oidc_config.issuer,
            jwks.keys.len()
        );

        Ok(Self {
            jwks: Arc::new(RwLock::new(jwks)),
            issuer: oidc_config.issuer,
            audience: oidc_config.audience,
            jwks_uri: discovery.jwks_uri,
            http,
            // Initialise to 60 seconds ago so the first eager refresh (if
            // needed) isn't rate-limited.
            last_eager_refresh: Arc::new(RwLock::new(
                Instant::now() - Duration::from_secs(60),
            )),
        })
    }

    /// Hook 2: Request — validate Bearer JWT against cached JWKS.
    ///
    /// Request-path flow:
    ///   1. Extract "Authorization: Bearer <token>" from headers
    ///   2. Decode the JWT header (unverified) to get the kid (key ID)
    ///   3. Look up the kid in the cached JWKS (read lock)
    ///   4. If kid not found, try one eager JWKS refresh (rate-limited)
    ///   5. Build a DecodingKey from the matched JWK
    ///   6. Verify signature + exp + iss + aud via jsonwebtoken::decode()
    ///   7. Return Allow(AuthUser) or Deny(status, reason)
    ///
    /// The common case (key found in cache, token valid) involves one read
    /// lock and one signature verification — microseconds, no network I/O.
    async fn authenticate(&self, headers: &HeaderMap) -> AuthDecision {
        // Step 1: extract Bearer token.
        let token = match extract_bearer_token(headers) {
            Ok(t) => t,
            Err(reason) => {
                return AuthDecision::Deny {
                    status: StatusCode::UNAUTHORIZED,
                    reason,
                }
            }
        };

        // Step 2: decode JWT header (unverified) to get kid.
        // We need the kid to look up the right key in the JWKS.
        // decode_header() only parses the base64 header — it doesn't
        // verify the signature, so this is cheap.
        let jwt_header = match jsonwebtoken::decode_header(token) {
            Ok(h) => h,
            Err(e) => {
                return AuthDecision::Deny {
                    status: StatusCode::UNAUTHORIZED,
                    reason: format!("invalid JWT header: {e}"),
                }
            }
        };

        let kid = match &jwt_header.kid {
            Some(kid) => kid.clone(),
            None => {
                return AuthDecision::Deny {
                    status: StatusCode::UNAUTHORIZED,
                    reason: "JWT missing kid header".into(),
                }
            }
        };

        // Step 3-4: look up kid, with eager refresh fallback.
        let jwk = self.find_key(&kid).await;
        let jwk = match jwk {
            Some(jwk) => jwk,
            None => {
                // Key not found — the issuer may have rotated keys since our
                // last refresh.  Try one eager fetch, rate-limited to prevent
                // an attacker from triggering refresh storms with random kids.
                if self.try_eager_refresh().await {
                    match self.find_key(&kid).await {
                        Some(jwk) => jwk,
                        None => {
                            return AuthDecision::Deny {
                                status: StatusCode::UNAUTHORIZED,
                                reason: "unknown signing key".into(),
                            }
                        }
                    }
                } else {
                    return AuthDecision::Deny {
                        status: StatusCode::UNAUTHORIZED,
                        reason: "unknown signing key".into(),
                    };
                }
            }
        };

        // Step 5: build decoding key from the JWK.
        let decoding_key = match jsonwebtoken::DecodingKey::from_jwk(&jwk) {
            Ok(key) => key,
            Err(e) => {
                return AuthDecision::Deny {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    reason: format!("failed to build decoding key: {e}"),
                }
            }
        };

        // Step 6: verify signature + claims.
        // Validation checks: algorithm matches JWK, exp is in the future,
        // iss matches our configured issuer, aud matches our audience.
        let mut validation = jsonwebtoken::Validation::new(jwt_header.alg);
        validation.set_issuer(&[&self.issuer]);
        validation.set_audience(&[&self.audience]);

        match jsonwebtoken::decode::<Claims>(token, &decoding_key, &validation) {
            Ok(token_data) => AuthDecision::Allow(AuthUser {
                sub: token_data.claims.sub,
                name: token_data.claims.name,
            }),
            Err(e) => {
                // Map specific error kinds to clear messages.
                let reason = match e.kind() {
                    jsonwebtoken::errors::ErrorKind::ExpiredSignature => {
                        "token expired".to_string()
                    }
                    jsonwebtoken::errors::ErrorKind::InvalidAudience => {
                        "invalid audience".to_string()
                    }
                    jsonwebtoken::errors::ErrorKind::InvalidIssuer => {
                        "invalid issuer".to_string()
                    }
                    _ => format!("token validation failed: {e}"),
                };
                AuthDecision::Deny {
                    status: StatusCode::UNAUTHORIZED,
                    reason,
                }
            }
        }
    }

    /// Hook 3: Background — periodic JWKS refresh for key rotation.
    ///
    /// OIDC issuers rotate signing keys periodically (Google rotates roughly
    /// every 6 hours).  This task fetches fresh JWKS every 60 minutes and
    /// updates the cache only if the key set changed.  The write lock is
    /// never acquired if keys haven't rotated — routine refreshes don't
    /// block request-path readers.
    async fn background(self: Arc<Self>) {
        let interval = Duration::from_secs(60 * 60);
        loop {
            tokio::time::sleep(interval).await;
            if let Err(e) = self.refresh_jwks().await {
                eprintln!("  warning   : JWKS refresh failed: {e}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JWT claims — the subset of standard OIDC claims we extract.
//
// OIDC tokens contain many more claims (email, email_verified, picture,
// locale, etc.) but we only need `sub` (who) and optionally `name` (for
// nicer log output).  Unknown fields are silently ignored by serde.
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct Claims {
    /// Subject identifier — the unique, stable user ID from the issuer.
    sub: String,
    /// Display name — optional, used for log output.
    #[serde(default)]
    name: Option<String>,
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Extract the Bearer token from the Authorization header.
///
/// Returns the token string (without the "Bearer " prefix) or an error
/// message suitable for returning to the client.
fn extract_bearer_token(headers: &HeaderMap) -> Result<&str, String> {
    let header = headers
        .get(http::header::AUTHORIZATION)
        .ok_or_else(|| "missing authorization header".to_string())?;

    let value = header
        .to_str()
        .map_err(|_| "authorization header is not valid UTF-8".to_string())?;

    value
        .strip_prefix("Bearer ")
        .ok_or_else(|| "authorization header must start with 'Bearer '".to_string())
}

impl OidcProvider {
    /// Look up a JWK by kid in the cached JWKS.
    ///
    /// Takes a read lock — many concurrent requests can look up keys
    /// simultaneously without blocking each other.
    async fn find_key(&self, kid: &str) -> Option<jsonwebtoken::jwk::Jwk> {
        let jwks = self.jwks.read().await;
        jwks.find(kid).cloned()
    }

    /// Attempt an eager JWKS refresh, rate-limited to once per 30 seconds.
    ///
    /// Called when a JWT's kid isn't in the cache — the issuer may have
    /// rotated keys between background refreshes.  The rate limit prevents
    /// an attacker from triggering refresh storms by sending JWTs with
    /// fabricated kid values.  Returns true if a refresh was performed.
    async fn try_eager_refresh(&self) -> bool {
        let now = Instant::now();
        {
            let last = self.last_eager_refresh.read().await;
            if now.duration_since(*last) < Duration::from_secs(30) {
                return false;
            }
        }
        // Update timestamp before refreshing to prevent concurrent refreshes.
        {
            let mut last = self.last_eager_refresh.write().await;
            *last = now;
        }
        self.refresh_jwks().await.is_ok()
    }

    /// Fetch fresh JWKS from the issuer and update the cache if keys changed.
    ///
    /// Locking strategy: fetch the JWKS (no lock held), then take a read lock
    /// to compare key IDs.  Only if the key set actually changed do we take
    /// the write lock to swap in the new JWKS.  This means the common case
    /// (keys haven't rotated) never blocks request-path readers with a write.
    async fn refresh_jwks(&self) -> anyhow::Result<()> {
        let new_jwks: JwkSet = self
            .http
            .get(&self.jwks_uri)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("JWKS fetch failed: {e}"))?
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("invalid JWKS response: {e}"))?;

        // Compare key IDs to detect rotation.  Read lock only — cheap.
        let changed = {
            let cached = self.jwks.read().await;
            let cached_kids: Vec<_> = cached
                .keys
                .iter()
                .filter_map(|k| k.common.key_id.as_deref())
                .collect();
            let new_kids: Vec<_> = new_jwks
                .keys
                .iter()
                .filter_map(|k| k.common.key_id.as_deref())
                .collect();
            cached_kids != new_kids
        };

        if changed {
            // Keys rotated — take the write lock and swap.
            let count = new_jwks.keys.len();
            *self.jwks.write().await = new_jwks;
            eprintln!("  oidc      : JWKS rotated ({count} keys)");
        }

        Ok(())
    }

    /// Construct an OidcProvider directly for testing (bypasses HTTP discovery).
    #[cfg(test)]
    pub(crate) fn new_for_test(
        jwks: JwkSet,
        issuer: String,
        audience: String,
    ) -> Self {
        Self {
            jwks: Arc::new(RwLock::new(jwks)),
            issuer,
            audience,
            jwks_uri: String::new(),
            http: reqwest::Client::new(),
            last_eager_refresh: Arc::new(RwLock::new(
                Instant::now() - Duration::from_secs(60),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests.
//
// All OIDC tests use a pre-generated 2048-bit RSA keypair (NOT secret —
// test-only).  The private key signs test JWTs; the public components
// (n, e) are used to build a JWK for the provider's cache.  This avoids
// runtime key generation and external dependencies.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{EncodingKey, Header};

    // Pre-generated 2048-bit RSA test key.  NOT secret — used only in tests.
    // Generated via: openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048
    const TEST_RSA_PRIVATE_PEM: &str = "\
-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDBvELblXFFas4O
NAqE9nKMOCswE8SPAWcVWE/+owmOxNYogBbArwjSWTGiS9jqPZy8DOnz0bUddu+l
pfDtRLtCNO2abDHGooXWNljev80eOu5X1SpgZ3n5Ai31hSTEKGNW6QSwGjS6ASJt
7EoyxnTwGDo0tezEKBrn0J63ckJXknjtZEnjD6uvbVJFH4Cksm3qHl+A4ROOkgQD
oZdP1h3iEvOC9JPJHqADORG8kFe5N5m0eWmjdeehaTV60itEUlCJ/vi0s2Pi7HJ2
xQVUXED53/sGFD0zjfdhzlLyxPKafVOLyudXGNJwZPuB1pXI7SmPLUDOAH0vqlL2
pV/j4manAgMBAAECggEADwTBqBUjNU7sz2QQZrNWOPGHw4/cvntM2vZAKUzJSmSO
94/Kc8B97hSVCPBZTftdwaQ8pLaelDyqokxFa5dW7vB8dOqrRytFNdB7vjTfOVc3
L97qDJQo+/lDx9M9AOninZOt1dsKOFsrKJtXOy1wVkaCiKCLzL8RKuRc0XrNks04
3a/w/nQfHhWyR+hFUithfkjg3jPhgKH5vkl6/PORHVjKp7Hmvim5UQ0Dw5B4i2ya
RdFsblqKXt9wZw96BUP4fcd16YJa1psnqJuiwXpJRwcUkCO8PJeveAGqC6grU6FT
9VcGmVVfz+JezkLYuJQkF5yPpCWMxQwVEByaN9o1bQKBgQD0g62wTJoFODgd0Bmy
OmcY3tfPyNiyQhKUuNFAvhLvPItNTeH5R82+Okih1+cdidj/0VX6rJNcdoyrq/mD
sDhxq/BrpyVMbYuM2D2pnqOysDoQCWP952jEyI54TapLc5KfFP3oW25PFZFddYom
I8Gci2KCITbqxo0BjBfdAqsrYwKBgQDK1fU/zN+jrxK4eGvbcZcouAPDcFebJPgL
4yCxwFGguybAaP7UP1Vkjhf59FHy0I5CSlmQSFb7D1vbVkWqo9kVOWn/Nc1cxSte
6roU1RX2s9l8eTWOkU7KXB8F8BjQiAIUsra6LzCdiz2ziD9JGgDc6ytaPV5F90s3
dMxuHzKU7QKBgCufQmFtiRzdRsWq1qrBWJtLRl0/i8lhmEcIIezW/DHKL1//QQ1k
DgSeCU82YXkXmqspAZnTKAca70XBTKZ9zdQZxK6wByt3b6oU+gtEzheW8QTjZ/9o
RXy1+xTjZjpHyCSxbgsCJM/fHSv7SEY9otD23QAyRMXl3kokYC2ByF/ZAoGAKM93
ssiDzqkw+RCxkst+AGFV0ILP/ZUomyutrlXllpNRLrFxZD8B7WNxi4cO3e38UXYo
IxGK/qSOdMkc50JkMRMGMqUelqXNHiHYIszkyGhTP+obTn4J/kkerNEsDPjwgj2a
6kcIXwpe9bpaEVk8BzcB1/w09ZrV9Wh4oUeBo9kCgYEA6d4bxUTVl+kzSBOe7q8N
Q0kFvB6jG1HSERY+YV55T7r2snP6I/u0t5CLZku4KOAPcKBCySSJ5l4qalUgnALj
bCTi+XuneFCq5dXqB1FAe7fXzGPQlftfE9rPuMfGii5IbENbzbN+G0m/L2C+J+Mo
jEbiqV8OJp7dP2SZJvhLidY=
-----END PRIVATE KEY-----";

    // Base64url-encoded RSA modulus (n) and exponent (e) for the test key above.
    // These are the public-key components needed to build a JWK for verification.
    // Extracted via: openssl rsa -modulus -noout / openssl rsa -text -noout
    const TEST_RSA_N: &str = "wbxC25VxRWrODjQKhPZyjDgrMBPEjwFnFVhP_qMJjsTWKIAWwK8I0lkxokvY6j2cvAzp89G1HXbvpaXw7US7QjTtmmwxxqKF1jZY3r_NHjruV9UqYGd5-QIt9YUkxChjVukEsBo0ugEibexKMsZ08Bg6NLXsxCga59Cet3JCV5J47WRJ4w-rr21SRR-ApLJt6h5fgOETjpIEA6GXT9Yd4hLzgvSTyR6gAzkRvJBXuTeZtHlpo3XnoWk1etIrRFJQif74tLNj4uxydsUFVFxA-d_7BhQ9M433Yc5S8sTymn1Ti8rnVxjScGT7gdaVyO0pjy1AzgB9L6pS9qVf4-Jmpw";
    const TEST_RSA_E: &str = "AQAB";

    /// Build test RSA keys: (encoding_key for signing, jwk_set for verifying, kid).
    fn test_rsa_keys() -> (EncodingKey, JwkSet, String) {
        use jsonwebtoken::jwk::{
            CommonParameters, Jwk, KeyAlgorithm, RSAKeyParameters, RSAKeyType,
        };

        let kid = "test-key-1".to_string();
        let encoding_key = EncodingKey::from_rsa_pem(TEST_RSA_PRIVATE_PEM.as_bytes()).unwrap();

        let jwk = Jwk {
            common: CommonParameters {
                public_key_use: Some(jsonwebtoken::jwk::PublicKeyUse::Signature),
                key_operations: None,
                key_algorithm: Some(KeyAlgorithm::RS256),
                key_id: Some(kid.clone()),
                x509_url: None,
                x509_chain: None,
                x509_sha1_fingerprint: None,
                x509_sha256_fingerprint: None,
            },
            algorithm: jsonwebtoken::jwk::AlgorithmParameters::RSA(RSAKeyParameters {
                key_type: RSAKeyType::RSA,
                n: TEST_RSA_N.to_string(),
                e: TEST_RSA_E.to_string(),
            }),
        };

        let jwks = JwkSet { keys: vec![jwk] };
        (encoding_key, jwks, kid)
    }

    /// Create a signed test JWT with the given claims.
    fn make_test_token(
        encoding_key: &EncodingKey,
        kid: &str,
        sub: &str,
        iss: &str,
        aud: &str,
        exp_offset_secs: i64,
        name: Option<&str>,
    ) -> String {
        let now = jsonwebtoken::get_current_timestamp();
        let exp = (now as i64 + exp_offset_secs) as u64;

        let mut header = Header::new(jsonwebtoken::Algorithm::RS256);
        header.kid = Some(kid.to_string());

        #[derive(serde::Serialize)]
        struct TestClaims {
            sub: String,
            iss: String,
            aud: String,
            exp: u64,
            iat: u64,
            #[serde(skip_serializing_if = "Option::is_none")]
            name: Option<String>,
        }

        let claims = TestClaims {
            sub: sub.to_string(),
            iss: iss.to_string(),
            aud: aud.to_string(),
            exp,
            iat: now,
            name: name.map(|s| s.to_string()),
        };

        jsonwebtoken::encode(&header, &claims, encoding_key).unwrap()
    }

    /// Build a HeaderMap with an Authorization: Bearer <token> header.
    fn headers_with_bearer(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::AUTHORIZATION,
            format!("Bearer {token}").parse().unwrap(),
        );
        headers
    }

    // --- Test cases ---

    #[test]
    fn test_oidc_rejects_missing_bearer() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (_, jwks, _) = test_rsa_keys();
        let provider = OidcProvider::new_for_test(
            jwks,
            "https://issuer.example.com".into(),
            "test-audience".into(),
        );
        let decision = rt.block_on(provider.authenticate(&HeaderMap::new()));
        match decision {
            AuthDecision::Deny { status, reason } => {
                assert_eq!(status, StatusCode::UNAUTHORIZED);
                assert!(reason.contains("missing"), "reason: {reason}");
            }
            AuthDecision::Allow(_) => panic!("expected Deny"),
        }
    }

    #[test]
    fn test_oidc_rejects_malformed_bearer() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (_, jwks, _) = test_rsa_keys();
        let provider = OidcProvider::new_for_test(
            jwks,
            "https://issuer.example.com".into(),
            "test-audience".into(),
        );
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::AUTHORIZATION,
            "NotBearer xyz".parse().unwrap(),
        );
        let decision = rt.block_on(provider.authenticate(&headers));
        match decision {
            AuthDecision::Deny { status, .. } => {
                assert_eq!(status, StatusCode::UNAUTHORIZED);
            }
            AuthDecision::Allow(_) => panic!("expected Deny"),
        }
    }

    #[test]
    fn test_oidc_rejects_expired_token() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (encoding_key, jwks, kid) = test_rsa_keys();
        let provider = OidcProvider::new_for_test(
            jwks,
            "https://issuer.example.com".into(),
            "test-audience".into(),
        );
        let token = make_test_token(
            &encoding_key,
            &kid,
            "user-1",
            "https://issuer.example.com",
            "test-audience",
            -3600, // expired 1 hour ago
            None,
        );
        let headers = headers_with_bearer(&token);
        let decision = rt.block_on(provider.authenticate(&headers));
        match decision {
            AuthDecision::Deny { status, reason } => {
                assert_eq!(status, StatusCode::UNAUTHORIZED);
                assert!(reason.contains("expired"), "reason: {reason}");
            }
            AuthDecision::Allow(_) => panic!("expected Deny"),
        }
    }

    #[test]
    fn test_oidc_rejects_wrong_audience() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (encoding_key, jwks, kid) = test_rsa_keys();
        let provider = OidcProvider::new_for_test(
            jwks,
            "https://issuer.example.com".into(),
            "test-audience".into(),
        );
        let token = make_test_token(
            &encoding_key,
            &kid,
            "user-1",
            "https://issuer.example.com",
            "wrong-audience",
            3600,
            None,
        );
        let headers = headers_with_bearer(&token);
        let decision = rt.block_on(provider.authenticate(&headers));
        match decision {
            AuthDecision::Deny { status, reason } => {
                assert_eq!(status, StatusCode::UNAUTHORIZED);
                assert!(reason.contains("audience"), "reason: {reason}");
            }
            AuthDecision::Allow(_) => panic!("expected Deny"),
        }
    }

    #[test]
    fn test_oidc_rejects_wrong_issuer() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (encoding_key, jwks, kid) = test_rsa_keys();
        let provider = OidcProvider::new_for_test(
            jwks,
            "https://issuer.example.com".into(),
            "test-audience".into(),
        );
        let token = make_test_token(
            &encoding_key,
            &kid,
            "user-1",
            "https://wrong-issuer.example.com",
            "test-audience",
            3600,
            None,
        );
        let headers = headers_with_bearer(&token);
        let decision = rt.block_on(provider.authenticate(&headers));
        match decision {
            AuthDecision::Deny { status, reason } => {
                assert_eq!(status, StatusCode::UNAUTHORIZED);
                assert!(reason.contains("issuer"), "reason: {reason}");
            }
            AuthDecision::Allow(_) => panic!("expected Deny"),
        }
    }

    #[test]
    fn test_oidc_rejects_unknown_kid() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (encoding_key, jwks, _) = test_rsa_keys();
        let provider = OidcProvider::new_for_test(
            jwks,
            "https://issuer.example.com".into(),
            "test-audience".into(),
        );
        let token = make_test_token(
            &encoding_key,
            "unknown-kid", // not in JWKS
            "user-1",
            "https://issuer.example.com",
            "test-audience",
            3600,
            None,
        );
        let headers = headers_with_bearer(&token);
        let decision = rt.block_on(provider.authenticate(&headers));
        match decision {
            AuthDecision::Deny { status, reason } => {
                assert_eq!(status, StatusCode::UNAUTHORIZED);
                assert!(reason.contains("signing key"), "reason: {reason}");
            }
            AuthDecision::Allow(_) => panic!("expected Deny"),
        }
    }

    #[test]
    fn test_oidc_accepts_valid_token() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (encoding_key, jwks, kid) = test_rsa_keys();
        let provider = OidcProvider::new_for_test(
            jwks,
            "https://issuer.example.com".into(),
            "test-audience".into(),
        );
        let token = make_test_token(
            &encoding_key,
            &kid,
            "user-42",
            "https://issuer.example.com",
            "test-audience",
            3600,
            Some("Alice"),
        );
        let headers = headers_with_bearer(&token);
        let decision = rt.block_on(provider.authenticate(&headers));
        match decision {
            AuthDecision::Allow(user) => {
                assert_eq!(user.sub, "user-42");
                assert_eq!(user.name.as_deref(), Some("Alice"));
            }
            AuthDecision::Deny { reason, .. } => panic!("expected Allow, got Deny: {reason}"),
        }
    }

    #[test]
    fn test_oidc_accepts_token_without_name() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (encoding_key, jwks, kid) = test_rsa_keys();
        let provider = OidcProvider::new_for_test(
            jwks,
            "https://issuer.example.com".into(),
            "test-audience".into(),
        );
        let token = make_test_token(
            &encoding_key,
            &kid,
            "user-99",
            "https://issuer.example.com",
            "test-audience",
            3600,
            None,
        );
        let headers = headers_with_bearer(&token);
        let decision = rt.block_on(provider.authenticate(&headers));
        match decision {
            AuthDecision::Allow(user) => {
                assert_eq!(user.sub, "user-99");
                assert!(user.name.is_none());
            }
            AuthDecision::Deny { reason, .. } => panic!("expected Allow, got Deny: {reason}"),
        }
    }
}
