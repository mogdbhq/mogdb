use metrics::counter;
use mogdb_storage::Database;
use tracing::info;

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

    // Install Prometheus metrics recorder + HTTP listener
    // Prometheus scrapes http://<host>:9090/metrics every 15 s.
    // Override with METRICS_ADDR env var (e.g. "0.0.0.0:9091").
    // MogDB exposes /metrics at port 9100 by default.
    // Prometheus (port 9090) scrapes this endpoint every 15 s.
    // Override with METRICS_ADDR env var.
    let metrics_addr: std::net::SocketAddr = std::env::var("METRICS_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:9100".to_string())
        .parse()
        .expect("METRICS_ADDR must be a valid socket address");

    metrics_exporter_prometheus::PrometheusBuilder::new()
        .with_http_listener(metrics_addr)
        .install()
        .expect("failed to install Prometheus metrics recorder");

    info!(%metrics_addr, "Prometheus metrics available at http://{metrics_addr}/metrics");

    // Connect to Postgres
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let db = Database::connect(&database_url).await?;
    db.ping().await?;
    let pool = db.pool.clone();

    info!("MogDB connected to database");

    // Record a startup counter so Prometheus immediately has something to scrape.
    counter!("mogdb_startups_total").increment(1);

    // Spawn background workers (decay / forget / scanner / consolidation)
    let workers = mogdb_workers::spawn_workers(pool);

    info!("MogDB is running — press Ctrl-C to stop");

    // Wait for shutdown signal
    tokio::signal::ctrl_c().await?;
    info!("shutdown signal received — stopping workers");

    workers.shutdown().await;

    Ok(())
}
