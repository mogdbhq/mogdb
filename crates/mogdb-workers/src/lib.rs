/// MogDB background workers.
///
/// Call `spawn_workers(pool)` from `main.rs` to start all four workers as
/// independent Tokio tasks. Returns a `WorkerSet` you can `.shutdown()` on
/// ctrl-c.
pub mod consolidation;
pub mod decay;
pub mod error;
pub mod forget;
pub mod llm;
pub mod scanner;

use llm::OllamaLlm;
use sqlx::PgPool;
use tokio::sync::watch;
use tokio::time::Duration;
use tracing::info;

/// Intervals for each background worker.
struct WorkerIntervals {
    decay: Duration,
    forget: Duration,
    scanner: Duration,
    consolidation: Duration,
}

impl WorkerIntervals {
    /// Production defaults. Set `MOGDB_DEV_MODE=1` to use 30-second intervals
    /// so all workers fire quickly for local testing.
    fn from_env() -> Self {
        let dev = std::env::var("MOGDB_DEV_MODE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        if dev {
            info!("MOGDB_DEV_MODE=1 — all workers set to 30-second intervals");
            let fast = Duration::from_secs(30);
            Self {
                decay: fast,
                forget: fast,
                scanner: fast,
                consolidation: fast,
            }
        } else {
            Self {
                decay: Duration::from_secs(60 * 60),                  // 1 hour
                forget: Duration::from_secs(60 * 60 * 24),            // 24 hours
                scanner: Duration::from_secs(30 * 60),                // 30 minutes
                consolidation: Duration::from_secs(60 * 60 * 24 * 7), // 7 days
            }
        }
    }
}

/// Handle returned by `spawn_workers`. Call `shutdown()` to signal all workers
/// to stop, then `await join()` to wait for clean exit.
pub struct WorkerSet {
    handles: Vec<tokio::task::JoinHandle<()>>,
    shutdown_tx: watch::Sender<bool>,
}

impl WorkerSet {
    /// Signal all workers to stop and wait for them to finish.
    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);
        for handle in self.handles {
            let _ = handle.await;
        }
        info!("all workers stopped");
    }
}

/// Spawn all background workers and return a handle for graceful shutdown.
///
/// Workers started:
/// - **Decay** — hourly, applies exponential strength decay
/// - **Forget** — nightly, applies enabled ForgetPolicies
/// - **Scanner** — every 30 min, quarantines injection payloads
/// - **Consolidation** — weekly, merges episodic memories via LLM
pub fn spawn_workers(pool: PgPool) -> WorkerSet {
    let (shutdown_tx, _) = watch::channel(false);
    let mut handles = Vec::new();
    let intervals = WorkerIntervals::from_env();

    // Decay
    {
        let rx = shutdown_tx.subscribe();
        let p = pool.clone();
        let period = intervals.decay;
        handles.push(tokio::spawn(async move {
            decay::run(p, rx, period).await;
        }));
    }

    // Forget
    {
        let rx = shutdown_tx.subscribe();
        let p = pool.clone();
        let period = intervals.forget;
        handles.push(tokio::spawn(async move {
            forget::run(p, rx, period).await;
        }));
    }

    // Scanner
    {
        let rx = shutdown_tx.subscribe();
        let p = pool.clone();
        let period = intervals.scanner;
        handles.push(tokio::spawn(async move {
            scanner::run(p, rx, period).await;
        }));
    }

    // Consolidation — with Ollama LLM
    {
        let rx = shutdown_tx.subscribe();
        let p = pool.clone();
        let period = intervals.consolidation;
        let llm = OllamaLlm::from_env();
        handles.push(tokio::spawn(async move {
            consolidation::run(p, llm, rx, period).await;
        }));
    }

    info!("background workers spawned (decay/forget/scanner/consolidation)");

    WorkerSet {
        handles,
        shutdown_tx,
    }
}
