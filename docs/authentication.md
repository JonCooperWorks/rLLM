# Authentication

Optional, pluggable authentication for the rLLM API server.  Disabled by
default.  When enabled via `--auth-config`, rLLM validates tokens on every
request and logs the authenticated user identity alongside inference metrics.

---

## Why Auth is Optional

rLLM's primary deployment mode is single-user: localhost, SSH tunnel to a
rented GPU, or a trusted LAN.  In this mode, auth adds complexity with no
benefit — you're the only user and you're not going to configure an OAuth2
provider just to talk to your own inference server.

Auth becomes valuable when rLLM is part of a larger deployment — behind a
gateway that mints scoped tokens, or shared among a small team.  The hook
system exists so an org's existing auth infrastructure can be integrated
without forking rLLM.

## Why Not a Gateway-Only Approach

The [Production Considerations](production-considerations.md) doc describes
a gateway that handles all auth and forwards requests to a "dumb" inference
server.  That's the right architecture for large-scale deployments.  But it
leaves a gap:

- **No identity at the inference layer.**  If the network boundary is
  breached, anyone who can reach the inference port gets unlimited access.
- **No audit trail on the inference server.**  Token counts and latency are
  logged, but there's no record of *who* used them.

rLLM's auth hook system closes this gap.  The gateway authenticates the end
user, does token exchange, and mints a short-lived JWT scoped for rLLM.
rLLM validates the JWT — identity is enforced end-to-end.  This is
defense-in-depth, not a replacement for gateway auth.

## Why Not Built-In Sessions / OAuth Flows / Login Pages

rLLM is an inference API, not a web application.  It speaks HTTP JSON, not
HTML.  Implementing OAuth flows, session management, cookie handling, or
login pages would add significant complexity to a system whose job is to
turn prompts into completions as fast as possible.

The design philosophy: rLLM validates tokens.  Someone else mints them.
That "someone" might be a gateway, an identity provider, a CLI tool that
does `curl` to the OIDC token endpoint — rLLM doesn't care.  It just checks
the signature, expiry, issuer, and audience.

---

## Architecture

### The Trait

Authentication is built around the `AuthProvider` trait with three hooks:

```
Hook 1: init(config) → Result<Self>
  Called once at startup.  Receives the auth.json config.
  Fetches JWKs, loads keys, validates config, sets up caches.
  Errors abort server startup.

Hook 2: authenticate(headers) → Allow(User) | Deny(Status, Reason)
  Called on every HTTP request (except /health).
  Receives the raw HTTP headers.
  Returns Allow with the user identity or Deny with an HTTP status and reason.
  Hot path — should be fast.  No network I/O unless key rotation forces a refresh.

Hook 3: background(self: Arc<Self>) → ()
  Optional.  Spawned after init on a tokio task, runs for the server's lifetime.
  Handles maintenance: JWKS refresh, cache expiry, CRL updates.
  Default: no-op.
```

### Data Flow

When enabled (`--auth-config`):

```
auth.json
  → serve() reads "provider" field, dispatches to init()
  → init() fetches JWKs / loads keys / validates config
  → provider stored in Arc<ServerState>
  → background() spawned on tokio
  → auth_middleware runs authenticate() on every request
  → Allow → AuthUser inserted into request extensions
  → handler extracts AuthUser, attaches to WorkerRequest
  → worker loop logs user.sub in per-sequence metrics
```

When disabled (no `--auth-config`):

```
auth_middleware checks is_enabled() → false → passes request through unchanged
  → no AuthUser in extensions, no identity logged
```

There is no fallback path.  Auth is either fully on or fully off.

### Why Enum Dispatch Instead of dyn

The set of providers is known at compile time.  `AuthProviderKind` is an
enum with a variant per provider (`None`, `Oidc`).  This avoids `dyn`
dispatch, avoids the `async-trait` crate, and makes the possible states
visible in the type system.  Same pattern as `TlsMode` in `tls.rs`.

---

## Configuration

Auth is configured via a JSON file passed with `--auth-config`:

```bash
rllm serve --model ./my-model --auth-config auth.json
```

The file has a `"provider"` field that selects the implementation.  The rest
is provider-specific — the framework passes the entire JSON value to the
provider's `init()` hook.

### OIDC

```json
{
  "provider": "oidc",
  "issuer": "https://accounts.google.com",
  "audience": "my-rllm-instance"
}
```

- **issuer** — the OIDC issuer URL.  rLLM fetches
  `{issuer}/.well-known/openid-configuration` at startup to discover the
  JWKS endpoint.  Must match the `iss` claim in tokens.
- **audience** — the expected `aud` claim.  Typically the client ID or a
  service identifier the gateway uses when minting tokens.

### No Auth (Default)

When `--auth-config` is omitted, the auth middleware is a no-op — requests
pass straight through with no `AuthUser` in the request context.  No auth
headers are checked, no identity is logged, no token validation happens.
There is no fallback "anonymous" identity; auth simply doesn't exist.

On localhost (the default bind address), this just works.  On external
interfaces (`--host 0.0.0.0`), rLLM requires `--dangerous-no-auth` to
confirm you intentionally want to run without authentication — anyone who
can reach the port gets unlimited access.

---

## OIDC Provider

The built-in OIDC provider validates JWTs against an OpenID Connect issuer's
published signing keys.

### Startup

1. Parse `issuer` and `audience` from auth.json
2. GET `{issuer}/.well-known/openid-configuration` — extract `jwks_uri`,
   validate that `issuer` matches
3. GET `{jwks_uri}` — parse the JSON Web Key Set, cache it

If any step fails, the server refuses to start.

### Per-Request Validation

1. Extract `Authorization: Bearer <token>` from headers
2. Decode the JWT header (unverified) to get the `kid` (key ID)
3. Look up the `kid` in the cached JWKS
4. If not found, try one eager JWKS refresh (rate-limited to 1 per 30s)
5. Verify signature, `exp`, `iss`, `aud` via `jsonwebtoken::decode()`
6. Return `Allow(AuthUser { sub, name })` or `Deny(401, reason)`

The common case (key in cache, token valid) is one read lock and one
signature verification — microseconds, no network I/O.

### Key Rotation

A background task refreshes the JWKS every 60 minutes.  It fetches the new
key set, compares key IDs against the cache, and only acquires the write
lock if keys actually changed.  Routine refreshes where nothing rotated
never block request-path readers.

If a request arrives with a `kid` not in the cache (key rotated between
background refreshes), the request path does one eager refresh, rate-limited
to prevent an attacker from triggering refresh storms with fabricated `kid`
values.

### What We Validate

- Token signature (algorithm from the JWK — RS256, ES256, etc.)
- `exp` — token not expired
- `iss` — matches the configured issuer
- `aud` — matches the configured audience

### What We Do Not Validate

- **Authorization** — all authenticated users have equal access.  rLLM has
  no permission model; the gateway handles authorization.
- **Token revocation** — no introspection endpoint, no CRL checks.  Tokens
  are short-lived; revocation is a gateway concern.
- **Scopes or roles** — not relevant for an inference API.

---

## Auth Without TLS

When auth is enabled but TLS is disabled, rLLM prints a startup warning.
Bearer tokens, prompts, and completions are sent in plaintext.  An attacker
with network access can:

- **Intercept tokens** — steal a valid JWT and impersonate the user
- **Read traffic** — observe all prompts and completions
- **Modify requests/responses** — alter prompts or completions in transit

On localhost (the default bind address), this is safe — traffic never leaves
the machine.  rLLM allows plain HTTP on localhost without `--dangerous-no-tls`.
If binding to an external interface (`--host 0.0.0.0`), TLS is required
unless you explicitly pass `--dangerous-no-tls`.  For SSH tunnels, the tunnel
provides encryption — localhost on the remote machine is fine.

---

## Per-User Logging

When auth is enabled, the authenticated user identity is included in the
per-sequence completion log line:

```
seq 123  |  user-42  |  500 prompt (200 cached) + 150 gen  |  TTFT 45 ms  |  32.1 tok/s  |  4.67s  |  eos
```

Without auth, the user segment is omitted entirely — zero visual noise for
the default local-only case:

```
seq 123  |  500 prompt (200 cached) + 150 gen  |  TTFT 45 ms  |  32.1 tok/s  |  4.67s  |  eos
```

This gives inference-side audit logging without additional infrastructure.
The identity always comes from `AuthUser.sub` — the stable subject identifier
from the JWT.  Display names are unreliable for audit trails.

---

## Adding a Custom Provider

To integrate an org's specific auth infrastructure:

1. Create `src/api/auth/your_provider.rs` implementing `AuthProvider`
2. Add `pub(crate) mod your_provider;` to `src/api/auth/mod.rs`
3. Add a variant to `AuthProviderKind` (e.g., `Custom(Arc<YourProvider>)`)
4. Add match arms in `AuthProviderKind::authenticate()` and
   `AuthProviderKind::spawn_background()`
5. Add a match arm in the factory in `src/api/mod.rs` (the `match provider_name`
   block)

The trait is deliberately HTTP-close: `authenticate()` takes `&HeaderMap`
and returns `Allow(AuthUser)` or `Deny(StatusCode, reason)`.  No
framework-specific abstractions — if you know HTTP, you can implement it.

---

## Files

| File | What |
|------|------|
| `src/api/auth/mod.rs` | AuthProvider trait, AuthUser, AuthDecision, AuthProviderKind, middleware |
| `src/api/auth/oidc.rs` | OIDC provider: discovery, JWKS caching, JWT verification |
| `src/api/mod.rs` | Wires auth into ServerState and axum router |
| `src/commands/serve.rs` | `--auth-config` CLI arg |

---

See also: [Threat Model](threat-model.md) ·
[Production Considerations](production-considerations.md)
