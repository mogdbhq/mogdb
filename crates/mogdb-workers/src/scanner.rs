/// Injection defence scanner — runs every 30 minutes.
///
/// Scans active, non-quarantined memories for signs of prompt-injection payloads
/// (long external content, known injection markers) and quarantines any matches.
/// The actual heuristics live in `mogdb_storage::forget::quarantine_suspicious`.
use metrics::counter;
use sqlx::PgPool;
use tokio::sync::watch;
use tokio::time::{interval, Duration};
use tracing::{error, info};

pub async fn run(pool: PgPool, mut shutdown: watch::Receiver<bool>, period: Duration) {
    let mut tick = interval(period);
    tick.tick().await; // skip immediate tick

    loop {
        tokio::select! {
            _ = tick.tick() => {
                counter!("mogdb_scanner_runs_total").increment(1);
                match mogdb_storage::forget::quarantine_suspicious(&pool).await {
                    Ok(n) => {
                        counter!("mogdb_scanner_quarantined_total").increment(n);
                        if n > 0 {
                            info!(quarantined = n, "injection scanner quarantined memories");
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "injection scanner failed");
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("injection scanner shutting down");
                    break;
                }
            }
        }
    }
}
