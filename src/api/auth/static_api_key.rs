// ===========================================================================
// Static API key auth provider — argon2id hash comparison for personal deploys.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements AuthProvider for static API key authentication.  The server
//   stores an argon2id hash of the API key in auth.json; clients send the
//   plaintext key as a Bearer token.  On each request, argon2id verify
//   compares the token against the hash (constant-time, timing-safe).
//
// Why argon2id instead of plaintext comparison:
//   1. The key never appears in plaintext on disk — only the hash is stored.
//   2. argon2id verify is inherently constant-time, eliminating timing
//      side-channel attacks that affect naive string comparison.
//   3. If the config file is leaked, the attacker gets a hash, not the key.
//
// When to use this provider:
//   Personal inference servers on a LAN or behind a simple reverse proxy
//   where OIDC is overkill.  You just want a shared secret to keep
//   unauthorized callers out.
//
// Security gaps vs OIDC (documented intentionally):
//   - Single shared key, no per-user identity (sub is always "apikey")
//   - No token expiry (key is valid forever until rotated)
//   - No rate limiting on failed attempts (delegated to gateway/firewall)
//   - argon2id verify is ~30-50ms per request (prevents brute-force but
//     adds latency — acceptable for personal use, not for high-QPS)
//
// Hot reload:
//   The background task watches the config file's mtime every 30 seconds.
//   If the file changes, it re-reads the config and swaps the hash behind
//   a RwLock — same pattern as OIDC's JWKS cache.  This allows key rotation
//   without restarting the server.
//
// Config format:
//   {
//     "provider": "static_api_key",
//     "key_hash": "$argon2id$v=19$m=19456,t=2,p=1$..."
//   }
//
// Generating a hash:
//   echo -n "my-secret-key" | argon2 $(openssl rand -hex 16) -id -e
//
// Related files:
//   auth/mod.rs     — AuthProvider trait, AuthProviderKind enum dispatch
//   api/mod.rs      — factory that routes "static_api_key" to this provider
//   docs/authentication.md — user-facing documentation
// ===========================================================================

use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use axum::http::{HeaderMap, StatusCode};
use tokio::sync::RwLock;

use argon2::password_hash::PasswordHash;
use argon2::{Argon2, PasswordVerifier};

use super::{AuthDecision, AuthProvider, AuthUser};

// ---------------------------------------------------------------------------
// Provider struct.
// ---------------------------------------------------------------------------

pub(crate) struct StaticApiKeyProvider {
    /// argon2id hash of the API key (PHC string format).
    /// Behind RwLock for hot reload — read lock on every request, write lock
    /// only when the config file changes.
    key_hash: Arc<RwLock<String>>,
    /// Path to the auth config file — re-read by the background task for
    /// hot reload of the key hash.
    config_path: PathBuf,
}

// ---------------------------------------------------------------------------
// Config.
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct StaticApiKeyConfig {
    key_hash: String,
}

// ---------------------------------------------------------------------------
// AuthProvider implementation.
// ---------------------------------------------------------------------------

impl AuthProvider for StaticApiKeyProvider {
    /// Parse the config and validate the hash is a valid argon2id PHC string.
    async fn init(config: &serde_json::Value) -> anyhow::Result<Self> {
        let cfg: StaticApiKeyConfig = serde_json::from_value(config.clone())?;

        // Validate the hash is parseable as a PHC string.
        PasswordHash::new(&cfg.key_hash)
            .map_err(|e| anyhow::anyhow!("invalid argon2id hash in auth config: {e}"))?;

        // Extract config_path from the config JSON (injected by the factory).
        let config_path = config
            .get("_config_path")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .unwrap_or_default();

        Ok(Self {
            key_hash: Arc::new(RwLock::new(cfg.key_hash)),
            config_path,
        })
    }

    /// Verify the Bearer token against the stored argon2id hash.
    async fn authenticate(&self, headers: &HeaderMap) -> AuthDecision {
        // Extract "Authorization: Bearer <token>".
        let token = match extract_bearer_token(headers) {
            Some(t) => t,
            None => {
                return AuthDecision::Deny {
                    status: StatusCode::UNAUTHORIZED,
                    reason: "missing or malformed Authorization header".into(),
                }
            }
        };

        if token.is_empty() {
            return AuthDecision::Deny {
                status: StatusCode::UNAUTHORIZED,
                reason: "empty bearer token".into(),
            };
        }

        let hash_str = self.key_hash.read().await;

        // Parse the PHC hash string.  This should always succeed because we
        // validated it in init() and hot reload, but handle the error anyway.
        let hash = match PasswordHash::new(&hash_str) {
            Ok(h) => h,
            Err(_) => {
                return AuthDecision::Deny {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    reason: "server auth configuration error".into(),
                }
            }
        };

        // argon2id verify is constant-time and takes ~30-50ms (intentional).
        match Argon2::default().verify_password(token.as_bytes(), &hash) {
            Ok(()) => AuthDecision::Allow(AuthUser {
                sub: "apikey".into(),
                name: None,
            }),
            Err(_) => AuthDecision::Deny {
                status: StatusCode::UNAUTHORIZED,
                reason: "invalid API key".into(),
            },
        }
    }

    /// Watch the config file for changes and hot-reload the key hash.
    async fn background(self: Arc<Self>) {
        if self.config_path.as_os_str().is_empty() {
            return;
        }

        let mut last_mtime = file_mtime(&self.config_path);

        loop {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;

            let current_mtime = file_mtime(&self.config_path);
            if current_mtime == last_mtime {
                continue;
            }
            last_mtime = current_mtime;

            // Config file changed — try to reload.
            match reload_hash(&self.config_path) {
                Ok(new_hash) => {
                    let mut hash = self.key_hash.write().await;
                    *hash = new_hash;
                    tracing::info!("static API key hash reloaded");
                }
                Err(e) => {
                    // Keep the old hash — don't break a running server.
                    tracing::warn!(error = %e, "failed to reload static API key config");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Extract the Bearer token from the Authorization header.
fn extract_bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
}

/// Get the mtime of a file, or None if the file doesn't exist.
fn file_mtime(path: &std::path::Path) -> Option<SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

/// Re-read the config file and extract + validate the new key_hash.
fn reload_hash(path: &std::path::Path) -> anyhow::Result<String> {
    let data = std::fs::read_to_string(path)?;
    let config: serde_json::Value = serde_json::from_str(&data)?;
    let cfg: StaticApiKeyConfig = serde_json::from_value(config)?;

    // Validate the new hash is parseable.
    PasswordHash::new(&cfg.key_hash)
        .map_err(|e| anyhow::anyhow!("invalid argon2id hash: {e}"))?;

    Ok(cfg.key_hash)
}

// Allow tests to construct provider instances without going through init().
#[cfg(test)]
impl StaticApiKeyProvider {
    pub(crate) fn new_for_test(key_hash: String) -> Self {
        Self {
            key_hash: Arc::new(RwLock::new(key_hash)),
            config_path: PathBuf::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Hash of "test-secret-key" for use in tests.
    /// Generated with: argon2 password_hash using the argon2 crate.
    fn test_hash() -> String {
        use argon2::password_hash::SaltString;
        use argon2::PasswordHasher;
        let salt = SaltString::from_b64("dGVzdHNhbHR2YWx1ZQ").unwrap();
        let hash = Argon2::default()
            .hash_password(b"test-secret-key", &salt)
            .unwrap();
        hash.to_string()
    }

    fn bearer_header(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", format!("Bearer {token}").parse().unwrap());
        headers
    }

    #[tokio::test]
    async fn test_valid_key_allows() {
        let provider = StaticApiKeyProvider::new_for_test(test_hash());
        let headers = bearer_header("test-secret-key");
        match provider.authenticate(&headers).await {
            AuthDecision::Allow(user) => {
                assert_eq!(user.sub, "apikey");
                assert!(user.name.is_none());
            }
            AuthDecision::Deny { reason, .. } => panic!("expected Allow, got Deny: {reason}"),
        }
    }

    #[tokio::test]
    async fn test_wrong_key_denies() {
        let provider = StaticApiKeyProvider::new_for_test(test_hash());
        let headers = bearer_header("wrong-key");
        match provider.authenticate(&headers).await {
            AuthDecision::Deny { status, .. } => {
                assert_eq!(status, StatusCode::UNAUTHORIZED);
            }
            AuthDecision::Allow(_) => panic!("expected Deny"),
        }
    }

    #[tokio::test]
    async fn test_missing_auth_header_denies() {
        let provider = StaticApiKeyProvider::new_for_test(test_hash());
        let headers = HeaderMap::new();
        match provider.authenticate(&headers).await {
            AuthDecision::Deny { status, .. } => {
                assert_eq!(status, StatusCode::UNAUTHORIZED);
            }
            AuthDecision::Allow(_) => panic!("expected Deny"),
        }
    }

    #[tokio::test]
    async fn test_malformed_header_denies() {
        let provider = StaticApiKeyProvider::new_for_test(test_hash());
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Basic abc123".parse().unwrap());
        match provider.authenticate(&headers).await {
            AuthDecision::Deny { status, .. } => {
                assert_eq!(status, StatusCode::UNAUTHORIZED);
            }
            AuthDecision::Allow(_) => panic!("expected Deny"),
        }
    }

    #[tokio::test]
    async fn test_empty_bearer_denies() {
        let provider = StaticApiKeyProvider::new_for_test(test_hash());
        let headers = bearer_header("");
        match provider.authenticate(&headers).await {
            AuthDecision::Deny { status, .. } => {
                assert_eq!(status, StatusCode::UNAUTHORIZED);
            }
            AuthDecision::Allow(_) => panic!("expected Deny"),
        }
    }

    #[tokio::test]
    async fn test_invalid_hash_in_config_fails_init() {
        let config = serde_json::json!({
            "provider": "static_api_key",
            "key_hash": "not-a-valid-hash"
        });
        let result = StaticApiKeyProvider::init(&config).await;
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(
            err_msg.contains("invalid argon2id hash"),
            "error should mention invalid hash, got: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_hot_reload_picks_up_new_hash() {
        use argon2::password_hash::SaltString;
        use argon2::PasswordHasher;

        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("auth.json");

        // Write initial config with hash of "old-key".
        let salt = SaltString::from_b64("dGVzdHNhbHR2YWx1ZQ").unwrap();
        let old_hash = Argon2::default()
            .hash_password(b"old-key", &salt)
            .unwrap()
            .to_string();
        let new_salt = SaltString::from_b64("bmV3dGVzdHNhbHR2YQ").unwrap();
        let new_hash = Argon2::default()
            .hash_password(b"new-key", &new_salt)
            .unwrap()
            .to_string();

        std::fs::write(
            &config_path,
            serde_json::json!({
                "provider": "static_api_key",
                "key_hash": old_hash
            })
            .to_string(),
        )
        .unwrap();

        let provider = StaticApiKeyProvider {
            key_hash: Arc::new(RwLock::new(old_hash)),
            config_path: config_path.clone(),
        };

        // Old key works.
        let headers = bearer_header("old-key");
        assert!(matches!(
            provider.authenticate(&headers).await,
            AuthDecision::Allow(_)
        ));

        // Write new config with hash of "new-key".
        // Ensure mtime differs (some filesystems have 1s granularity).
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(
            &config_path,
            serde_json::json!({
                "provider": "static_api_key",
                "key_hash": new_hash.clone()
            })
            .to_string(),
        )
        .unwrap();

        // Simulate what background() does: reload hash.
        let reloaded = reload_hash(&config_path).unwrap();
        assert_eq!(reloaded, new_hash);
        *provider.key_hash.write().await = reloaded;

        // New key works, old key doesn't.
        let new_headers = bearer_header("new-key");
        assert!(matches!(
            provider.authenticate(&new_headers).await,
            AuthDecision::Allow(_)
        ));

        let old_headers = bearer_header("old-key");
        assert!(matches!(
            provider.authenticate(&old_headers).await,
            AuthDecision::Deny { .. }
        ));
    }
}
