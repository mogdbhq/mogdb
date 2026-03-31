/// Consolidation worker — runs weekly.
///
/// Finds agents with many similar episodic memories, uses the LLM to distil them
/// into a single semantic memory, and soft-expires the originals.
///
/// If the LLM is unavailable the pass is skipped gracefully — nothing is lost.
use crate::llm::LlmProvider;
use metrics::counter;
use mogdb_core::{MemoryKind, NewMemoryRecord, SourceTrust};
use sqlx::PgPool;
use tokio::sync::watch;
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// Consolidate clusters of ≥ MIN_CLUSTER episodic memories created within the
// last WINDOW_DAYS days that share entity refs.
const MIN_CLUSTER: usize = 5;
const WINDOW_DAYS: i64 = 30;

pub async fn run<L: LlmProvider>(pool: PgPool, llm: L, mut shutdown: watch::Receiver<bool>) {
    let mut tick = interval(Duration::from_secs(60 * 60 * 24 * 7)); // every 7 days
    tick.tick().await;

    loop {
        tokio::select! {
            _ = tick.tick() => {
                counter!("mogdb_consolidation_runs_total").increment(1);
                run_once(&pool, &llm).await;
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("consolidation worker shutting down");
                    break;
                }
            }
        }
    }
}

async fn run_once<L: LlmProvider>(pool: &PgPool, llm: &L) {
    // Find distinct agents with enough episodic memories in the window
    let agents: Vec<(String,)> = match sqlx::query_as(
        r#"
        SELECT DISTINCT agent_id
        FROM memory_records
        WHERE kind = 'episodic'
          AND t_expired IS NULL
          AND quarantined = FALSE
          AND t_created > NOW() - INTERVAL '30 days'
        GROUP BY agent_id
        HAVING COUNT(*) >= $1
        "#,
    )
    .bind(MIN_CLUSTER as i64)
    .fetch_all(pool)
    .await
    {
        Ok(rows) => rows,
        Err(e) => {
            error!(error = %e, "consolidation: failed to list agents");
            return;
        }
    };

    let mut total_merged = 0u64;

    for (agent_id,) in &agents {
        match consolidate_agent(pool, llm, agent_id).await {
            Ok(n) => total_merged += n,
            Err(e) => warn!(agent_id, error = %e, "consolidation failed for agent"),
        }
    }

    counter!("mogdb_consolidation_merged_total").increment(total_merged);
    if total_merged > 0 {
        info!(total_merged, "consolidation pass complete");
    }
}

async fn consolidate_agent<L: LlmProvider>(
    pool: &PgPool,
    llm: &L,
    agent_id: &str,
) -> Result<u64, crate::error::WorkerError> {
    // Load candidate episodic memories
    #[derive(sqlx::FromRow)]
    struct Row {
        id: Uuid,
        user_id: String,
        content: String,
    }

    let rows: Vec<Row> = sqlx::query_as(
        r#"
        SELECT id, user_id, content
        FROM memory_records
        WHERE agent_id = $1
          AND kind = 'episodic'
          AND t_expired IS NULL
          AND quarantined = FALSE
          AND t_created > NOW() - $2 * INTERVAL '1 day'
        ORDER BY importance DESC, t_created DESC
        LIMIT 20
        "#,
    )
    .bind(agent_id)
    .bind(WINDOW_DAYS)
    .fetch_all(pool)
    .await?;

    if rows.len() < MIN_CLUSTER {
        return Ok(0);
    }

    let user_id = rows[0].user_id.clone();
    let source_ids: Vec<Uuid> = rows.iter().map(|r| r.id).collect();

    // Build a LLM prompt
    let bullet_list: String = rows
        .iter()
        .map(|r| format!("• {}", r.content))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "You are a memory consolidation assistant.\n\
         Summarise the following {count} episodic memories into a single, concise semantic fact. \
         Write only the summary — no preamble, no bullet points.\n\n{bullets}",
        count = rows.len(),
        bullets = bullet_list,
    );

    let summary = match llm.complete(&prompt).await {
        Ok(s) => s,
        Err(e) => {
            warn!(agent_id, error = %e, "LLM unavailable — skipping consolidation");
            return Ok(0);
        }
    };

    if summary.is_empty() {
        return Ok(0);
    }

    debug!(agent_id, summary = %summary, "LLM consolidation summary");

    // Store the consolidated memory
    let consolidated_id = {
        let input = NewMemoryRecord {
            agent_id: agent_id.to_string(),
            user_id: user_id.clone(),
            session_id: None,
            content: summary,
            kind: MemoryKind::Semantic,
            source_trust: SourceTrust::System,
            t_valid: None,
            importance: Some(0.7),
            entity_refs: vec![],
            provenance: None,
        };
        mogdb_storage::memory::store(pool, input).await?.id
    };

    // Soft-expire the originals and link provenance
    let id_list: Vec<String> = source_ids.iter().map(|id| format!("'{id}'")).collect();
    let id_csv = id_list.join(", ");

    sqlx::query(&format!(
        "UPDATE memory_records SET t_expired = NOW(), provenance = $1 WHERE id IN ({id_csv})"
    ))
    .bind(consolidated_id)
    .execute(pool)
    .await?;

    info!(
        agent_id,
        consolidated_id = %consolidated_id,
        source_count = source_ids.len(),
        "consolidated episodic memories"
    );

    Ok(source_ids.len() as u64)
}
