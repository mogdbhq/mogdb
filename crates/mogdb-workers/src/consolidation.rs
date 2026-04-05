/// Consolidation (reflect/synthesis) worker — runs weekly.
///
/// Phase 1: Find agent+user pairs with clusters of similar episodic memories.
/// Phase 2: Group memories by shared entities into coherent clusters.
/// Phase 3: Use the LLM to distil each cluster into a single semantic fact.
/// Phase 4: Extract entities from the summary and store as a semantic memory.
/// Phase 5: Soft-expire the originals and link provenance.
///
/// If the LLM is unavailable the pass is skipped gracefully — nothing is lost.
use crate::llm::LlmProvider;
use metrics::counter;
use mogdb_core::{MemoryKind, NewMemoryRecord, SourceTrust};
use sqlx::PgPool;
use std::collections::{HashMap, HashSet};
use tokio::sync::watch;
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Minimum memories in a cluster to trigger consolidation.
const MIN_CLUSTER: usize = 3;
/// Look back window for candidate memories.
const WINDOW_DAYS: i64 = 30;
/// Maximum memories per cluster sent to the LLM.
const MAX_CLUSTER_SIZE: usize = 15;

pub async fn run<L: LlmProvider>(
    pool: PgPool,
    llm: L,
    mut shutdown: watch::Receiver<bool>,
    period: Duration,
) {
    let mut tick = interval(period);
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
    // Find distinct agent+user pairs with enough episodic memories in the window
    let pairs: Vec<(String, String)> = match sqlx::query_as(
        r#"
        SELECT agent_id, user_id
        FROM memory_records
        WHERE kind = 'episodic'
          AND t_expired IS NULL
          AND quarantined = FALSE
          AND t_created > NOW() - INTERVAL '30 days'
        GROUP BY agent_id, user_id
        HAVING COUNT(*) >= $1
        "#,
    )
    .bind(MIN_CLUSTER as i64)
    .fetch_all(pool)
    .await
    {
        Ok(rows) => rows,
        Err(e) => {
            error!(error = %e, "consolidation: failed to list agent+user pairs");
            return;
        }
    };

    let mut total_merged = 0u64;

    for (agent_id, user_id) in &pairs {
        match consolidate_user(pool, llm, agent_id, user_id).await {
            Ok(n) => total_merged += n,
            Err(e) => warn!(agent_id, user_id, error = %e, "consolidation failed"),
        }
    }

    counter!("mogdb_consolidation_merged_total").increment(total_merged);
    if total_merged > 0 {
        info!(total_merged, "consolidation pass complete");
    }
}

#[derive(sqlx::FromRow)]
struct MemRow {
    id: Uuid,
    content: String,
    entity_refs: Vec<String>,
    #[allow(dead_code)] // Used for ORDER BY in SQL, not in Rust
    importance: f64,
}

/// Consolidate memories for a single agent+user pair.
///
/// Groups memories into entity-based clusters, then consolidates each cluster.
async fn consolidate_user<L: LlmProvider>(
    pool: &PgPool,
    llm: &L,
    agent_id: &str,
    user_id: &str,
) -> Result<u64, crate::error::WorkerError> {
    // Load candidate episodic memories
    let rows: Vec<MemRow> = sqlx::query_as(
        r#"
        SELECT id, content, entity_refs, importance
        FROM memory_records
        WHERE agent_id = $1
          AND user_id = $2
          AND kind = 'episodic'
          AND t_expired IS NULL
          AND quarantined = FALSE
          AND t_created > NOW() - $3 * INTERVAL '1 day'
        ORDER BY importance DESC, t_created DESC
        LIMIT 50
        "#,
    )
    .bind(agent_id)
    .bind(user_id)
    .bind(WINDOW_DAYS)
    .fetch_all(pool)
    .await?;

    if rows.len() < MIN_CLUSTER {
        return Ok(0);
    }

    // Group into entity-based clusters
    let clusters = cluster_by_entities(&rows);
    let mut total = 0u64;

    for cluster_ids in clusters {
        if cluster_ids.len() < MIN_CLUSTER {
            continue;
        }

        let cluster_rows: Vec<&MemRow> = cluster_ids
            .iter()
            .filter_map(|id| rows.iter().find(|r| r.id == *id))
            .take(MAX_CLUSTER_SIZE)
            .collect();

        match consolidate_cluster(pool, llm, agent_id, user_id, &cluster_rows).await {
            Ok(n) => total += n,
            Err(e) => warn!(agent_id, user_id, error = %e, "cluster consolidation failed"),
        }
    }

    Ok(total)
}

/// Group memories into clusters based on shared entity references.
///
/// Uses a simple greedy algorithm: memories sharing ≥1 entity are in the same cluster.
/// Memories with no entities go into a single "unclustered" group.
fn cluster_by_entities(rows: &[MemRow]) -> Vec<Vec<Uuid>> {
    // Union-Find style clustering: entity_name → cluster_id
    let mut entity_to_cluster: HashMap<String, usize> = HashMap::new();
    let mut clusters: Vec<HashSet<Uuid>> = Vec::new();

    for row in rows {
        if row.entity_refs.is_empty() {
            // No entities — put in a catch-all cluster (cluster 0)
            if clusters.is_empty() {
                clusters.push(HashSet::new());
            }
            clusters[0].insert(row.id);
            continue;
        }

        // Find which cluster any of this memory's entities already belong to
        let mut target_cluster: Option<usize> = None;
        for entity in &row.entity_refs {
            let lower = entity.to_lowercase();
            if let Some(&cid) = entity_to_cluster.get(&lower) {
                target_cluster = Some(cid);
                break;
            }
        }

        let cid = match target_cluster {
            Some(cid) => cid,
            None => {
                // New cluster
                let cid = clusters.len();
                clusters.push(HashSet::new());
                cid
            }
        };

        clusters[cid].insert(row.id);

        // Map all this memory's entities to the cluster
        for entity in &row.entity_refs {
            entity_to_cluster.insert(entity.to_lowercase(), cid);
        }
    }

    clusters
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect()
}

/// Consolidate a single cluster of memories into a semantic fact.
async fn consolidate_cluster<L: LlmProvider>(
    pool: &PgPool,
    llm: &L,
    agent_id: &str,
    user_id: &str,
    cluster: &[&MemRow],
) -> Result<u64, crate::error::WorkerError> {
    // Collect all entity refs from the cluster
    let mut all_entities: HashSet<String> = HashSet::new();
    for row in cluster {
        for entity in &row.entity_refs {
            all_entities.insert(entity.clone());
        }
    }

    // Build LLM prompt
    let bullet_list: String = cluster
        .iter()
        .map(|r| format!("• {}", r.content))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "You are a memory consolidation assistant.\n\
         Summarise the following {count} episodic memories into one or two concise semantic facts.\n\
         Preserve key details: names, tools, preferences, dates, and decisions.\n\
         Write only the summary — no preamble, no bullet points.\n\n\
         {bullets}",
        count = cluster.len(),
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

    debug!(agent_id, user_id, summary = %summary, "LLM consolidation summary");

    // Extract entities from the summary (keyword-based, fast)
    let summary_entities = mogdb_storage::extraction::extract_entities(&summary);
    let mut entity_refs: Vec<String> = all_entities.into_iter().collect();
    for ent in &summary_entities {
        if !entity_refs
            .iter()
            .any(|e| e.eq_ignore_ascii_case(&ent.name))
        {
            entity_refs.push(ent.name.clone());
        }
    }

    // Store the consolidated memory
    let input = NewMemoryRecord {
        agent_id: agent_id.to_string(),
        user_id: user_id.to_string(),
        session_id: None,
        content: summary,
        kind: MemoryKind::Semantic,
        source_trust: SourceTrust::System,
        t_valid: None,
        importance: Some(0.7),
        entity_refs,
        provenance: None,
    };
    let consolidated = mogdb_storage::memory::store(pool, input).await?;

    // Soft-expire the originals and link provenance
    let source_ids: Vec<Uuid> = cluster.iter().map(|r| r.id).collect();
    let id_list: Vec<String> = source_ids.iter().map(|id| format!("'{id}'")).collect();
    let id_csv = id_list.join(", ");

    sqlx::query(&format!(
        "UPDATE memory_records SET t_expired = NOW(), provenance = $1 WHERE id IN ({id_csv})"
    ))
    .bind(consolidated.id)
    .execute(pool)
    .await?;

    info!(
        agent_id,
        user_id,
        consolidated_id = %consolidated.id,
        source_count = source_ids.len(),
        "consolidated episodic cluster"
    );

    Ok(source_ids.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(id: Uuid, entities: Vec<&str>) -> MemRow {
        MemRow {
            id,
            content: "test".into(),
            entity_refs: entities.into_iter().map(String::from).collect(),
            importance: 0.5,
        }
    }

    #[test]
    fn cluster_groups_by_shared_entities() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        let d = Uuid::new_v4();

        let rows = vec![
            make_row(a, vec!["PostgreSQL", "Redis"]),
            make_row(b, vec!["Redis"]),         // shares Redis with a
            make_row(c, vec!["Docker", "K8s"]), // separate cluster
            make_row(d, vec!["K8s"]),           // shares K8s with c
        ];

        let clusters = cluster_by_entities(&rows);
        // Should produce 2 clusters: {a,b} and {c,d}
        assert_eq!(clusters.len(), 2);

        let cluster_a = clusters.iter().find(|c| c.contains(&a)).unwrap();
        assert!(cluster_a.contains(&b), "a and b share Redis");

        let cluster_c = clusters.iter().find(|cl| cl.contains(&c)).unwrap();
        assert!(cluster_c.contains(&d), "c and d share K8s");
    }

    #[test]
    fn cluster_empty_entities_grouped_together() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();

        let rows = vec![
            make_row(a, vec![]),
            make_row(b, vec![]),
            make_row(c, vec!["Redis"]),
        ];

        let clusters = cluster_by_entities(&rows);
        // Cluster 0: {a, b} (no entities), Cluster 1: {c}
        assert_eq!(clusters.len(), 2);
        let empty_cluster = clusters.iter().find(|c| c.contains(&a)).unwrap();
        assert!(empty_cluster.contains(&b));
    }

    #[test]
    fn cluster_single_memory_no_cluster() {
        let a = Uuid::new_v4();
        let rows = vec![make_row(a, vec!["Redis"])];
        let clusters = cluster_by_entities(&rows);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 1);
    }
}
