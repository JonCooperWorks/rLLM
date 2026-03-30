// ===========================================================================
// Auth hook system — pluggable authentication for the API server.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Defines the auth provider trait and the axum middleware that runs it on
//   every HTTP request.  The trait has three hooks (init, request, background)
//   that together cover the full lifecycle of an auth provider.
//
// Why a hook system instead of just hardcoding OIDC:
//   Different orgs have different auth infrastructure.  Some use OIDC, some
//   use custom JWTs signed with internal CAs, some have proprietary token
//   formats.  The trait lets each org implement their own provider without
//   forking rLLM.  OIDC ships as the built-in default because it's the most
//   common standard.
//
// Why not dyn dispatch / async-trait crate:
//   The set of providers is known at compile time — you add a variant to
//   AuthProviderKind and a match arm.  Enum dispatch (like TlsMode in tls.rs)
//   avoids boxing, avoids the async-trait crate dependency, and makes the
//   possible states visible in the type system.
//
// Design: HTTP-close
//   The request hook takes &HeaderMap and returns Allow/Deny with an HTTP
//   StatusCode.  No middleware-specific abstractions, no framework coupling
//   beyond what axum already provides.  This keeps the trait implementable
//   by anyone who knows HTTP, not just someone who knows axum internals.
//
// Deployment modes:
//   1. Solo/dev (default) — no auth, localhost or SSH tunnel, single user.
//      AuthProviderKind::None, middleware is a no-op, no AuthUser in extensions.
//
//   2. Gateway + rLLM auth (defense-in-depth) — gateway authenticates the
//      end user, does token exchange, mints a scoped JWT.  rLLM validates
//      it via the OIDC provider.  Identity enforced end-to-end.  Even if
//      the network boundary is breached, unauthenticated requests are
//      rejected before reaching the inference engine.
//
// Data flow:
//   auth.json (--auth-config)
//     → serve() reads "provider" field, dispatches to init()
//     → init() fetches JWKs / loads keys / validates config
//     → provider stored in ServerState (shared via Arc)
//     → background() spawned on tokio (JWKS refresh, etc.)
//     → auth_middleware runs authenticate() on every request
//     → Allow → AuthUser inserted into request extensions
//     → handlers extract AuthUser, attach to WorkerRequest
//     → worker loop logs user identity in per-sequence metrics
//
// Module layout:
//   mod.rs   — AuthUser, AuthDecision, AuthProvider trait, AuthProviderKind,
//              auth_middleware, tests
//   oidc.rs  — OidcProvider: OIDC discovery, JWKS caching, JWT verification
//
// Related files:
//   api/mod.rs         — wires auth into ServerState and axum router
//   commands/serve.rs  — --auth-config CLI arg
//   docs/authentication.md — full design rationale and deployment guidance
// ===========================================================================

pub(crate) mod oidc;
pub(crate) mod static_api_key;

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::http::{HeaderMap, StatusCode};

use super::ServerState;

// ---------------------------------------------------------------------------
// Core types.
// ---------------------------------------------------------------------------

/// Authenticated user identity extracted from a request.
///
/// Intentionally minimal — the auth provider populates whatever fields the
/// token contains.  `sub` is the only required identifier (OIDC "subject"
/// claim, but any string works for non-OIDC providers).
///
/// This struct flows through the entire request lifecycle:
///   middleware → request extensions → handler → WorkerRequest → RequestContext
///   → per-sequence log line (audit trail)
#[derive(Clone, Debug)]
pub(crate) struct AuthUser {
    /// Subject identifier (OIDC `sub` claim, API key owner, etc.).
    pub sub: String,
    /// Optional display name or email — not used for logging or access
    /// control.  Available for providers that surface it from token claims.
    #[allow(dead_code)] // populated from token claims for future audit/logging use
    pub name: Option<String>,
}

/// Result of authenticating a single HTTP request.
///
/// Deliberately close to HTTP: Allow carries the user identity forward,
/// Deny carries a StatusCode and a human-readable reason that gets returned
/// as a JSON error body.  No framework-specific types — just http primitives.
pub(crate) enum AuthDecision {
    /// Request is allowed.  The AuthUser is inserted into axum request
    /// extensions so handlers can extract it.
    Allow(AuthUser),
    /// Request is denied.  The status code and reason are returned to the
    /// client as `{"error": {"message": reason, "type": "authentication_error"}}`.
    Deny { status: StatusCode, reason: String },
}

// ---------------------------------------------------------------------------
// Auth provider trait — the hook point for pluggable authentication.
//
// Why three hooks:
//
//   init()         — one-time setup.  OIDC needs to fetch the discovery
//                    document and JWKS before it can validate anything.
//                    Other providers might load keys from disk, connect to
//                    a token introspection endpoint, etc.  Async because
//                    network I/O is common.  Errors abort server startup.
//
//   authenticate() — per-request.  This is the hot path.  Should be fast:
//                    cached key lookup, signature verification, claim checks.
//                    No network I/O unless a key rotation forces a refresh.
//
//   background()   — optional maintenance.  OIDC refreshes JWKS every hour
//                    to handle key rotation.  Other providers might refresh
//                    CRLs, re-fetch introspection configs, etc.  Default is
//                    a no-op — providers that don't need background work
//                    just don't override it.
// ---------------------------------------------------------------------------

/// Auth provider trait.  Implementations must be Send + Sync + 'static
/// because they're shared across tokio tasks via Arc<ServerState>.
pub(crate) trait AuthProvider: Send + Sync + 'static {
    /// Hook 1: Init — parse config and set up the provider.
    ///
    /// Called once at startup with the full auth.json value.  The provider
    /// deserialises whatever fields it needs and ignores the rest.
    /// Returns the initialised provider or an error that aborts startup.
    fn init(
        config: &serde_json::Value,
    ) -> impl std::future::Future<Output = anyhow::Result<Self>> + Send
    where
        Self: Sized;

    /// Hook 2: Request — authenticate from HTTP headers.
    ///
    /// Called on every request (except /health).  Receives the raw HeaderMap
    /// and returns Allow(AuthUser) or Deny(StatusCode, reason).
    fn authenticate(
        &self,
        headers: &HeaderMap,
    ) -> impl std::future::Future<Output = AuthDecision> + Send;

    /// Hook 3: Background — optional long-running maintenance task.
    ///
    /// Spawned once after init on a tokio task.  Runs for the server's
    /// lifetime.  Default: no-op (providers that don't need background
    /// work skip it).
    fn background(self: Arc<Self>) -> impl std::future::Future<Output = ()> + Send {
        async {}
    }
}

// ---------------------------------------------------------------------------
// Enum dispatch — registry of compiled-in providers.
//
// Follows the TlsMode pattern in tls.rs: avoids dyn + async-trait crate,
// and since the set of providers is known at compile time, enum dispatch is
// simpler and has zero overhead.
//
// Adding a new provider:
//   1. Implement AuthProvider for your struct in a new file under auth/.
//   2. Add a variant here wrapping Arc<YourProvider>.
//   3. Add a match arm in authenticate() and spawn_background() below.
//   4. Add a match arm in the factory in api/mod.rs (the "oidc" => ... block).
// ---------------------------------------------------------------------------

pub(crate) enum AuthProviderKind {
    /// No auth — auth middleware is a no-op, requests pass through without
    /// an AuthUser.  Default when --auth-config is omitted.
    None,
    /// OIDC JWT validation.  Wrapped in Arc because the background task
    /// (JWKS refresh) and the request path share the same provider instance.
    Oidc(Arc<oidc::OidcProvider>),
    /// Static API key — argon2id hash comparison.  For personal inference
    /// servers where OIDC is overkill.  See static_api_key.rs for security
    /// gaps and design rationale.
    StaticApiKey(Arc<static_api_key::StaticApiKeyProvider>),
}

impl AuthProviderKind {
    /// Authenticate a request.  Only called when auth is configured —
    /// the middleware skips this entirely for AuthProviderKind::None.
    pub(crate) async fn authenticate(&self, headers: &HeaderMap) -> AuthDecision {
        match self {
            Self::None => unreachable!("authenticate() should not be called when auth is disabled"),
            Self::Oidc(provider) => provider.authenticate(headers).await,
            Self::StaticApiKey(provider) => provider.authenticate(headers).await,
        }
    }

    /// Spawn the provider's background task (if any).
    pub(crate) fn spawn_background(&self) {
        match self {
            Self::None => {}
            Self::Oidc(provider) => {
                tokio::spawn(Arc::clone(provider).background());
            }
            Self::StaticApiKey(provider) => {
                tokio::spawn(Arc::clone(provider).background());
            }
        }
    }

    /// Whether auth is enabled.
    pub(crate) fn is_enabled(&self) -> bool {
        !matches!(self, Self::None)
    }
}

// ---------------------------------------------------------------------------
// Axum middleware — runs the request hook on every HTTP request.
//
// Why a middleware layer instead of an axum extractor:
//   An extractor would require changing every handler's signature.  A Tower
//   middleware layer applies uniformly to all routes — same pattern as the
//   existing CorsLayer.  The /health endpoint is exempted inside the
//   middleware so health checks work without credentials.
//
// Layer ordering in the router:
//   .layer(auth_middleware)          ← inner: runs after CORS, before handler
//   .layer(CorsLayer::permissive()) ← outer: runs first on request, last on response
//
//   This ensures preflight OPTIONS requests get CORS headers even when auth
//   would deny them — browsers need the CORS response to proceed.
// ---------------------------------------------------------------------------

pub(crate) async fn auth_middleware(
    State(state): State<Arc<ServerState>>,
    mut request: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> Response {
    // Skip auth entirely when disabled or for health/metrics endpoints.
    let path = request.uri().path();
    if !state.auth.is_enabled() || path == "/health" || path == "/metrics" {
        return next.run(request).await;
    }

    match state.auth.authenticate(request.headers()).await {
        AuthDecision::Allow(user) => {
            // Store the AuthUser in request extensions so handlers can extract
            // it via `Option<Extension<AuthUser>>` without parsing headers again.
            request.extensions_mut().insert(user);
            next.run(request).await
        }
        AuthDecision::Deny { status, reason } => {
            // Return an OpenAI-style error JSON body so clients get a
            // machine-parseable error regardless of which API format they use.
            let body = serde_json::json!({
                "error": {
                    "message": reason,
                    "type": "authentication_error",
                }
            });
            (status, axum::Json(body)).into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_auth_is_not_enabled() {
        let provider = AuthProviderKind::None;
        assert!(!provider.is_enabled());
    }

    #[test]
    fn test_oidc_is_enabled() {
        let (_, jwks, _) = super::oidc::tests::test_rsa_keys_pub();
        let provider = AuthProviderKind::Oidc(Arc::new(
            super::oidc::OidcProvider::new_for_test(
                jwks,
                "https://issuer.example.com".into(),
                "test-audience".into(),
            ),
        ));
        assert!(provider.is_enabled());
    }

    #[test]
    fn test_static_api_key_is_enabled() {
        use argon2::password_hash::SaltString;
        use argon2::PasswordHasher;
        let salt = SaltString::from_b64("dGVzdHNhbHR2YWx1ZQ").unwrap();
        let hash = argon2::Argon2::default()
            .hash_password(b"test", &salt)
            .unwrap()
            .to_string();
        let provider = AuthProviderKind::StaticApiKey(Arc::new(
            static_api_key::StaticApiKeyProvider::new_for_test(hash),
        ));
        assert!(provider.is_enabled());
    }

    #[test]
    fn test_deny_carries_status_and_reason() {
        let decision = AuthDecision::Deny {
            status: StatusCode::UNAUTHORIZED,
            reason: "missing token".into(),
        };
        match decision {
            AuthDecision::Deny { status, reason } => {
                assert_eq!(status, StatusCode::UNAUTHORIZED);
                assert_eq!(reason, "missing token");
            }
            AuthDecision::Allow(_) => panic!("expected Deny"),
        }
    }

    #[test]
    fn test_allow_carries_user_identity() {
        let decision = AuthDecision::Allow(AuthUser {
            sub: "user-123".into(),
            name: Some("Alice".into()),
        });
        match decision {
            AuthDecision::Allow(user) => {
                assert_eq!(user.sub, "user-123");
                assert_eq!(user.name.as_deref(), Some("Alice"));
            }
            AuthDecision::Deny { .. } => panic!("expected Allow"),
        }
    }
}
