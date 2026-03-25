/// Query Planner — hybrid search across full-text, temporal, and graph indexes.
/// Results are merged using Reciprocal Rank Fusion (RRF).
use chrono::{DateTime, Utc};
use mogdb_core::{AuditAction, MemoryKind, MogError};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use crate::audit;

/// Search parameters — all optional filters compose together.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub agent_id: String,
    pub user_id: String,
    /// The natural language search query.
    pub query: String,
    /// Point-in-time query: return facts that were valid at this moment.
    /// If None, returns currently-valid facts only.
    pub as_of: Option<DateTime<Utc>>,
    /// Filter by memory kind.
    pub kind: Option<MemoryKind>,
    /// Minimum strength threshold (filters out decayed memories).
    pub min_strength: Option<f64>,
    /// Whether to expand results with related entities from the graph.
    pub include_graph: bool,
    /// Maximum results to return.
    pub limit: i32,
}

impl SearchQuery {
    pub fn new(
        agent_id: impl Into<String>,
        user_id: impl Into<String>,
        query: impl Into<String>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            user_id: user_id.into(),
            query: query.into(),
            as_of: None,
            kind: None,
            min_strength: None,
            include_graph: false,
            limit: 10,
        }
    }

    pub fn as_of(mut self, ts: DateTime<Utc>) -> Self {
        self.as_of = Some(ts);
        self
    }

    pub fn kind(mut self, kind: MemoryKind) -> Self {
        self.kind = Some(kind);
        self
    }

    pub fn min_strength(mut self, min: f64) -> Self {
        self.min_strength = Some(min);
        self
    }

    pub fn with_graph(mut self) -> Self {
        self.include_graph = true;
        self
    }

    pub fn limit(mut self, limit: i32) -> Self {
        self.limit = limit;
        self
    }
}

/// A single search result with scoring metadata.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: Uuid,
    pub content: String,
    pub kind: MemoryKind,
    pub importance: f64,
    pub strength: f64,
    pub t_valid: DateTime<Utc>,
    pub t_invalid: Option<DateTime<Utc>>,
    pub t_created: DateTime<Utc>,
    pub entity_refs: Vec<String>,
    /// The final fused relevance score (higher = more relevant).
    pub score: f64,
    /// Related entity context pulled from the graph (if include_graph was true).
    pub graph_context: Vec<GraphNode>,
}

/// An entity node returned as graph context.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub entity_name: String,
    pub entity_kind: String,
    pub relation: String,
    pub related_to: String,
}

/// Raw row from full-text search.
#[derive(Debug, FromRow)]
struct FtsRow {
    id: Uuid,
    content: String,
    kind: String,
    importance: f64,
    strength: f64,
    t_valid: DateTime<Utc>,
    t_invalid: Option<DateTime<Utc>>,
    t_created: DateTime<Utc>,
    entity_refs: Vec<String>,
    rank: f32,
}

/// Execute a hybrid search query.
pub async fn search(pool: &PgPool, query: SearchQuery) -> Result<Vec<SearchResult>, MogError> {
    // Build tsquery from the search text
    let terms = build_tsquery(&query.query);
    if terms.is_empty() {
        return Ok(vec![]);
    }

    // Step 1: Full-text search (BM25 via Postgres ts_rank)
    let fts_results = fts_search(pool, &query, &terms).await?;

    // Step 2: Temporal filtering is built into the SQL (see fts_search)

    // Step 3: Apply decay-weighted scoring + RRF
    let mut results = fuse_results(fts_results);

    // Step 4: Graph expansion (if requested)
    if query.include_graph {
        for result in &mut results {
            result.graph_context =
                expand_graph(pool, &query.agent_id, &query.user_id, &result.entity_refs).await?;
        }
    }

    // Step 5: Sort by final score descending, apply limit
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(query.limit as usize);

    // Step 6: Bump access_count for returned memories
    let ids: Vec<Uuid> = results.iter().map(|r| r.id).collect();
    touch_accessed(pool, &ids).await?;

    // Audit log
    audit::log(
        pool,
        AuditAction::Read,
        &query.agent_id,
        None,
        Some(&query.query),
        Some(results.len() as i32),
    )
    .await?;

    Ok(results)
}

/// Full-text search with temporal + kind + strength filtering baked into SQL.
async fn fts_search(
    pool: &PgPool,
    query: &SearchQuery,
    tsquery: &str,
) -> Result<Vec<FtsRow>, MogError> {
    // Build the temporal clause
    let temporal_clause = match query.as_of {
        Some(ref ts) => {
            format!("AND t_valid <= '{ts}' AND (t_invalid IS NULL OR t_invalid > '{ts}')")
        }
        None => "AND t_invalid IS NULL".to_string(),
    };

    let kind_clause = match &query.kind {
        Some(k) => format!("AND kind = '{}'::memory_kind", k),
        None => String::new(),
    };

    let strength_clause = match query.min_strength {
        Some(min) => format!("AND strength >= {min}"),
        None => String::new(),
    };

    let sql = format!(
        r#"
        SELECT
            id, content, kind::TEXT AS kind, importance, strength,
            t_valid, t_invalid, t_created, entity_refs,
            ts_rank_cd(to_tsvector('english', content), to_tsquery('english', $3)) AS rank
        FROM memory_records
        WHERE agent_id = $1
          AND user_id = $2
          AND t_expired IS NULL
          AND quarantined = false
          AND to_tsvector('english', content) @@ to_tsquery('english', $3)
          {temporal_clause}
          {kind_clause}
          {strength_clause}
        ORDER BY rank DESC
        LIMIT $4
        "#
    );

    let rows = sqlx::query_as::<_, FtsRow>(&sql)
        .bind(&query.agent_id)
        .bind(&query.user_id)
        .bind(tsquery)
        .bind(query.limit * 3) // over-fetch for fusion
        .fetch_all(pool)
        .await?;

    Ok(rows)
}

/// Apply Reciprocal Rank Fusion (RRF) + decay weighting to merge results.
/// RRF formula: score = sum(1 / (k + rank_i)) for each ranking source.
/// We also multiply by strength to bias toward fresh memories.
fn fuse_results(fts_results: Vec<FtsRow>) -> Vec<SearchResult> {
    const K: f64 = 60.0; // RRF constant — standard value from the literature

    fts_results
        .into_iter()
        .enumerate()
        .map(|(rank, row)| {
            // RRF score from full-text rank position
            let rrf_score = 1.0 / (K + rank as f64 + 1.0);

            // Weight by the FTS rank directly as a second signal
            let fts_weight = row.rank as f64;

            // Decay-weighted importance
            let decay_weight = row.importance * row.strength;

            // Final fused score
            let score = rrf_score + (fts_weight * 0.3) + (decay_weight * 0.4);

            let kind = match row.kind.as_str() {
                "episodic" => MemoryKind::Episodic,
                "semantic" => MemoryKind::Semantic,
                "procedural" => MemoryKind::Procedural,
                "working" => MemoryKind::Working,
                _ => MemoryKind::Episodic,
            };

            SearchResult {
                id: row.id,
                content: row.content,
                kind,
                importance: row.importance,
                strength: row.strength,
                t_valid: row.t_valid,
                t_invalid: row.t_invalid,
                t_created: row.t_created,
                entity_refs: row.entity_refs,
                score,
                graph_context: vec![],
            }
        })
        .collect()
}

/// Expand entity graph: for each entity referenced in a memory,
/// find related entities via 1-hop traversal.
async fn expand_graph(
    pool: &PgPool,
    agent_id: &str,
    user_id: &str,
    entity_refs: &[String],
) -> Result<Vec<GraphNode>, MogError> {
    if entity_refs.is_empty() {
        return Ok(vec![]);
    }

    // Find entities by name, then get their edges + connected entity names
    let rows = sqlx::query_as::<_, (String, String, String, String)>(
        r#"
        SELECT
            e_from.name AS entity_name,
            e_from.kind::TEXT AS entity_kind,
            edge.relation,
            e_to.name AS related_to
        FROM entities e_from
        JOIN entity_edges edge ON edge.from_id = e_from.id AND edge.t_invalid IS NULL
        JOIN entities e_to ON e_to.id = edge.to_id
        WHERE e_from.agent_id = $1
          AND e_from.user_id = $2
          AND LOWER(e_from.name) = ANY(
              SELECT LOWER(unnest) FROM unnest($3::TEXT[])
          )
        UNION
        SELECT
            e_to.name AS entity_name,
            e_to.kind::TEXT AS entity_kind,
            edge.relation,
            e_from.name AS related_to
        FROM entities e_to
        JOIN entity_edges edge ON edge.to_id = e_to.id AND edge.t_invalid IS NULL
        JOIN entities e_from ON e_from.id = edge.from_id
        WHERE e_to.agent_id = $1
          AND e_to.user_id = $2
          AND LOWER(e_to.name) = ANY(
              SELECT LOWER(unnest) FROM unnest($3::TEXT[])
          )
        LIMIT 20
        "#,
    )
    .bind(agent_id)
    .bind(user_id)
    .bind(entity_refs)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(
            |(entity_name, entity_kind, relation, related_to)| GraphNode {
                entity_name,
                entity_kind,
                relation,
                related_to,
            },
        )
        .collect())
}

/// Update access_count and last_accessed for retrieved memories.
async fn touch_accessed(pool: &PgPool, ids: &[Uuid]) -> Result<(), MogError> {
    if ids.is_empty() {
        return Ok(());
    }

    sqlx::query(
        "UPDATE memory_records SET access_count = access_count + 1, last_accessed = NOW() WHERE id = ANY($1)",
    )
    .bind(ids)
    .execute(pool)
    .await?;

    Ok(())
}

/// Build a tsquery string from natural language text.
/// Splits into words, removes stop words, joins with OR operator.
fn build_tsquery(text: &str) -> String {
    const STOP_WORDS: &[&str] = &[
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they", "them", "the", "a",
        "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may", "might", "can", "and", "but",
        "or", "not", "no", "so", "if", "then", "what", "in", "on", "at", "to", "for", "of", "with",
        "from", "by", "this", "that", "these", "those", "here", "there", "how", "when", "also",
        "just", "very", "really", "actually", "now", "last", "about", "which", "who", "where",
        "why",
    ];

    let terms: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| {
            let lower = w.to_lowercase();
            w.len() >= 2 && !STOP_WORDS.contains(&lower.as_str())
        })
        .map(|w| w.to_lowercase())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    terms.join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_tsquery_filters_stop_words() {
        let q = build_tsquery("what cloud does the user prefer");
        assert!(!q.contains("what"));
        assert!(!q.contains("does"));
        assert!(!q.contains("the"));
        assert!(q.contains("cloud"));
        assert!(q.contains("prefer"));
        assert!(q.contains("user"));
    }

    #[test]
    fn build_tsquery_empty_input() {
        assert!(build_tsquery("").is_empty());
        assert!(build_tsquery("the a an").is_empty());
    }

    #[test]
    fn build_tsquery_deduplicates() {
        let q = build_tsquery("cloud cloud cloud hosting hosting");
        let parts: Vec<&str> = q.split(" | ").collect();
        assert_eq!(parts.len(), 2, "should deduplicate: {q}");
    }

    #[test]
    fn fuse_results_scores_decrease() {
        let rows = vec![
            FtsRow {
                id: Uuid::new_v4(),
                content: "first".into(),
                kind: "semantic".into(),
                importance: 0.9,
                strength: 1.0,
                t_valid: Utc::now(),
                t_invalid: None,
                t_created: Utc::now(),
                entity_refs: vec![],
                rank: 0.8,
            },
            FtsRow {
                id: Uuid::new_v4(),
                content: "second".into(),
                kind: "semantic".into(),
                importance: 0.5,
                strength: 0.7,
                t_valid: Utc::now(),
                t_invalid: None,
                t_created: Utc::now(),
                entity_refs: vec![],
                rank: 0.3,
            },
        ];

        let results = fuse_results(rows);
        assert!(
            results[0].score > results[1].score,
            "first should score higher: {} vs {}",
            results[0].score,
            results[1].score
        );
    }
}
