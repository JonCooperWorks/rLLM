// ===========================================================================
// TLS support — manual certificates and Let's Encrypt (ACME).
//
// Two modes:
//
// 1. Manual: the user provides --cert and --private-key PEM files.
//    We load them into a rustls ServerConfig and serve TLS directly.
//
// 2. Let's Encrypt: the user provides --domain and --letsencrypt.
//    We use rustls-acme to automatically provision and renew certificates
//    via the TLS-ALPN-01 challenge (no separate port 80 needed).
//
// Both modes share the same accept loop: TCP accept → TLS handshake →
// serve each connection with hyper (bridging axum's tower service).
// ===========================================================================

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

/// TLS configuration mode, determined from CLI args.
pub(crate) enum TlsMode {
    /// No TLS — serve plain HTTP.
    None,
    /// Manual TLS — user-provided certificate and private key.
    Manual { cert: PathBuf, key: PathBuf },
    /// Let's Encrypt automatic TLS via ACME TLS-ALPN-01.
    LetsEncrypt {
        domain: String,
        email: Option<String>,
        cache_dir: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Manual TLS.
// ---------------------------------------------------------------------------

/// Serve HTTPS with user-provided certificate and private key files.
pub(crate) async fn serve_manual_tls(
    app: Router,
    addr: &str,
    cert_path: &Path,
    key_path: &Path,
) -> anyhow::Result<()> {
    // Load certificate chain from PEM file.
    let cert_file = std::fs::File::open(cert_path)?;
    let mut cert_reader = std::io::BufReader::new(cert_file);
    let certs: Vec<_> = rustls_pemfile::certs(&mut cert_reader).collect::<Result<_, _>>()?;
    anyhow::ensure!(
        !certs.is_empty(),
        "no certificates found in {}",
        cert_path.display()
    );

    // Load private key from PEM file.
    let key_file = std::fs::File::open(key_path)?;
    let mut key_reader = std::io::BufReader::new(key_file);
    let key = rustls_pemfile::private_key(&mut key_reader)?
        .ok_or_else(|| anyhow::anyhow!("no private key found in {}", key_path.display()))?;

    // Build rustls server config.
    let mut server_config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)?;
    server_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

    let acceptor = TlsAcceptor::from(Arc::new(server_config));
    let listener = TcpListener::bind(addr).await?;

    serve_tls_loop(listener, acceptor, app).await
}

// ---------------------------------------------------------------------------
// Let's Encrypt (ACME).
// ---------------------------------------------------------------------------

/// Serve HTTPS with Let's Encrypt automatic certificate provisioning.
///
/// Uses rustls-acme with TLS-ALPN-01 challenge validation — the ACME
/// challenge and regular TLS traffic share the same port.  Certificates
/// are cached on disk so restarts don't trigger re-issuance.
pub(crate) async fn serve_letsencrypt(
    app: Router,
    addr: &str,
    domain: &str,
    email: Option<&str>,
    cache_dir: &Path,
) -> anyhow::Result<()> {
    use rustls_acme::AcmeConfig;
    use rustls_acme::caches::DirCache;
    use tokio_stream::StreamExt;

    let mut acme = AcmeConfig::new([domain])
        .cache(DirCache::new(cache_dir.to_owned()))
        .directory_lets_encrypt(true);

    if let Some(email) = email {
        acme = acme.contact_push(format!("mailto:{email}"));
    }

    let state = acme.state();
    let resolver = state.resolver();

    // Spawn ACME event processor — drives certificate acquisition and renewal.
    // The state must be continuously polled for the ACME protocol to work.
    tokio::spawn(async move {
        let mut state = std::pin::pin!(state);
        loop {
            match state.next().await {
                Some(Ok(ok)) => eprintln!("acme: {ok:?}"),
                Some(Err(err)) => eprintln!("acme error: {err:?}"),
                None => break,
            }
        }
    });

    // Build rustls config with ACME cert resolver.
    // Include acme-tls/1 ALPN protocol for TLS-ALPN-01 challenge validation.
    let mut server_config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_cert_resolver(resolver);
    server_config.alpn_protocols =
        vec![b"h2".to_vec(), b"http/1.1".to_vec(), b"acme-tls/1".to_vec()];

    let acceptor = TlsAcceptor::from(Arc::new(server_config));
    let listener = TcpListener::bind(addr).await?;

    serve_tls_loop(listener, acceptor, app).await
}

// ---------------------------------------------------------------------------
// Shared TLS accept loop.
// ---------------------------------------------------------------------------

/// Accept loop shared by both manual and Let's Encrypt TLS modes.
///
/// For each incoming TCP connection: perform TLS handshake, then serve
/// the connection with hyper (bridging axum's tower::Service).
async fn serve_tls_loop(
    listener: TcpListener,
    acceptor: TlsAcceptor,
    app: Router,
) -> anyhow::Result<()> {
    use hyper_util::rt::{TokioExecutor, TokioIo};
    use tower_service::Service;

    loop {
        let (tcp_stream, remote_addr) = listener.accept().await?;
        let acceptor = acceptor.clone();
        let app = app.clone();

        tokio::spawn(async move {
            // TLS handshake.
            let tls_stream = match acceptor.accept(tcp_stream).await {
                Ok(stream) => stream,
                Err(err) => {
                    eprintln!("TLS handshake failed from {remote_addr}: {err}");
                    return;
                }
            };

            // Serve the connection via hyper, bridging to axum's Router.
            let stream = TokioIo::new(tls_stream);
            let hyper_service = hyper::service::service_fn(
                move |request: hyper::Request<hyper::body::Incoming>| {
                    let mut app = app.clone();
                    async move { app.call(request.map(axum::body::Body::new)).await }
                },
            );

            if let Err(err) = hyper_util::server::conn::auto::Builder::new(TokioExecutor::new())
                .serve_connection(stream, hyper_service)
                .await
            {
                eprintln!("error serving {remote_addr}: {err}");
            }
        });
    }
}
