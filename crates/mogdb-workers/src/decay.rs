/// Decay worker — runs every hour and applies exponential strength decay
/// to all active memories. Emits Prometheus metrics after each pass.
use metrics::counter;
use sqlx::PgPool;
use tokio::sync::watch;
use tokio::time::{interval, Duration};
use tracing::{error, info};

pub async fn run(pool: PgPool, mut shutdown: watch::Receiver<bool>, period: Duration) {
    let mut tick = interval(period);
    tick.tick().await; // skip the immediate first tick

    loop {
        tokio::select! {
            _ = tick.tick() => {
                match mogdb_storage::decay::run_decay_pass(&pool).await {
                    Ok(n) => {
                        counter!("mogdb_decay_runs_total").increment(1);
                        counter!("mogdb_decay_records_updated_total").increment(n);
                        info!(updated = n, "decay pass complete");
                    }
                    Err(e) => {
                        error!(error = %e, "decay pass failed");
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("decay worker shutting down");
                    break;
                }
            }
        }
    }
}
