/// Forget worker — runs nightly and applies all enabled ForgetPolicies.
///
/// For each agent that has at least one enabled policy, the worker loads all
/// enabled policies and calls `apply_policy` for each one in sequence.
use metrics::counter;
use sqlx::PgPool;
use tokio::sync::watch;
use tokio::time::{interval, Duration};
use tracing::{error, info, warn};

pub async fn run(pool: PgPool, mut shutdown: watch::Receiver<bool>) {
    let mut tick = interval(Duration::from_secs(60 * 60 * 24)); // every 24 hours
    tick.tick().await; // skip immediate tick

    loop {
        tokio::select! {
            _ = tick.tick() => {
                run_once(&pool).await;
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("forget worker shutting down");
                    break;
                }
            }
        }
    }
}

async fn run_once(pool: &PgPool) {
    counter!("mogdb_forget_runs_total").increment(1);

    // Collect distinct agent IDs that have enabled policies
    let agents: Vec<(String,)> =
        match sqlx::query_as("SELECT DISTINCT agent_id FROM forget_policies WHERE enabled = TRUE")
            .fetch_all(pool)
            .await
        {
            Ok(rows) => rows,
            Err(e) => {
                error!(error = %e, "failed to list forget policy agents");
                return;
            }
        };

    let mut total_affected = 0u64;

    for (agent_id,) in &agents {
        let policies = match mogdb_storage::forget::list_enabled_policies(pool, agent_id).await {
            Ok(p) => p,
            Err(e) => {
                warn!(agent_id, error = %e, "failed to load policies for agent");
                continue;
            }
        };

        for policy in &policies {
            match mogdb_storage::forget::apply_policy(pool, policy).await {
                Ok(n) => {
                    total_affected += n;
                    if n > 0 {
                        info!(
                            policy = policy.name,
                            agent_id,
                            affected = n,
                            "policy applied"
                        );
                    }
                }
                Err(e) => {
                    error!(policy = policy.name, agent_id, error = %e, "policy apply failed");
                }
            }
        }
    }

    counter!("mogdb_forget_memories_acted_total").increment(total_affected);
    info!(total_affected, "forget pass complete");
}
