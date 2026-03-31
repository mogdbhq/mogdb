mod middleware;
mod models;
mod routes;

use metrics::counter;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use mogdb_storage::Database;
use sqlx::PgPool;
use tracing::info;

/// Shared application state — cheaply cloneable (Pool is Arc-backed).
#[derive(Clone)]
pub struct AppState {
    pub pool: PgPool,
    pub prom_handle: PrometheusHandle,
    /// None = auth disabled (no API_KEY env var set).
    pub api_key: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env if present
    dotenvy::dotenv().ok();

    // Init tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mogdb=debug,info".into()),
        )
        .init();

    // Install Prometheus recorder — metrics are served by Axum at GET /metrics.
    let prom_handle = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder");

    // Connect to Postgres + run migrations
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let db = Database::connect(&database_url).await?;
    db.ping().await?;
    info!("MogDB connected to database");

    // API key (optional — if not set, auth is disabled)
    let api_key = std::env::var("API_KEY").ok();
    if api_key.is_none() {
        tracing::warn!("API_KEY not set — authentication disabled");
    }

    let state = AppState {
        pool: db.pool.clone(),
        prom_handle,
        api_key,
    };

    // Record startup metric
    counter!("mogdb_startups_total").increment(1);

    // Spawn background workers
    let workers = mogdb_workers::spawn_workers(db.pool.clone());

    // Start Axum HTTP server
    let api_addr: std::net::SocketAddr = std::env::var("API_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".to_string())
        .parse()
        .expect("API_ADDR must be a valid socket address");

    let router = routes::build_router(state);

    info!(%api_addr, "MogDB API listening");
    info!("  POST   /memories");
    info!("  GET    /memories/search?agent_id=&user_id=&q=");
    info!("  GET    /memories/:id?agent_id=");
    info!("  DELETE /memories/:id?agent_id=");
    info!("  GET    /health");
    info!("  GET    /metrics");

    let listener = tokio::net::TcpListener::bind(api_addr).await?;

    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
            info!("shutdown signal received");
        })
        .await?;

    workers.shutdown().await;

    Ok(())
}
