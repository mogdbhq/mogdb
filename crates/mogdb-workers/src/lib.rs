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
use tracing::info;

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

    // Decay — hourly
    {
        let rx = shutdown_tx.subscribe();
        let p = pool.clone();
        handles.push(tokio::spawn(async move {
            decay::run(p, rx).await;
        }));
    }

    // Forget — nightly
    {
        let rx = shutdown_tx.subscribe();
        let p = pool.clone();
        handles.push(tokio::spawn(async move {
            forget::run(p, rx).await;
        }));
    }

    // Scanner — every 30 min
    {
        let rx = shutdown_tx.subscribe();
        let p = pool.clone();
        handles.push(tokio::spawn(async move {
            scanner::run(p, rx).await;
        }));
    }

    // Consolidation — weekly, with Ollama LLM
    {
        let rx = shutdown_tx.subscribe();
        let p = pool.clone();
        let llm = OllamaLlm::from_env();
        handles.push(tokio::spawn(async move {
            consolidation::run(p, llm, rx).await;
        }));
    }

    info!("background workers spawned (decay/forget/scanner/consolidation)");

    WorkerSet {
        handles,
        shutdown_tx,
    }
}
