/// Conflict detection — find existing memories that contradict an incoming fact
/// and invalidate them (set t_invalid = now, not delete).

use chrono::{DateTime, Utc};
use mogdb_core::MogError;
use sqlx::PgPool;
use uuid::Uuid;

use crate::audit;

/// A memory that potentially conflicts with an incoming fact.
#[derive(Debug)]
pub struct ConflictCandidate {
    pub id: Uuid,
    pub content: String,
    pub t_valid: DateTime<Utc>,
    pub importance: f64,
    pub rank: f32,
}

/// Search for active memories that might conflict with the incoming content.
/// Uses PostgreSQL full-text search to find topically related memories,
/// then returns them as candidates for the caller to evaluate.
pub async fn find_conflicts(
    pool: &PgPool,
    agent_id: &str,
    user_id: &str,
    content: &str,
    entity_refs: &[String],
) -> Result<Vec<ConflictCandidate>, MogError> {
    // Build a tsquery from the content
    let query_terms = extract_search_terms(content);
    if query_terms.is_empty() {
        return Ok(vec![]);
    }

    let tsquery = query_terms.join(" | ");

    // Find active memories that share entities OR are textually related
    let candidates = if entity_refs.is_empty() {
        // No entities — use full-text search only
        sqlx::query_as::<_, (Uuid, String, DateTime<Utc>, f64, f32)>(
            r#"
            SELECT id, content, t_valid, importance,
                   ts_rank(to_tsvector('english', content), to_tsquery('english', $3)) AS rank
            FROM memory_records
            WHERE agent_id = $1
              AND user_id = $2
              AND t_expired IS NULL
              AND t_invalid IS NULL
              AND quarantined = false
              AND to_tsvector('english', content) @@ to_tsquery('english', $3)
            ORDER BY rank DESC
            LIMIT 10
            "#,
        )
        .bind(agent_id)
        .bind(user_id)
        .bind(&tsquery)
        .fetch_all(pool)
        .await?
    } else {
        // Have entities — find memories that reference the same entities OR are textually related
        sqlx::query_as::<_, (Uuid, String, DateTime<Utc>, f64, f32)>(
            r#"
            SELECT id, content, t_valid, importance,
                   ts_rank(to_tsvector('english', content), to_tsquery('english', $3)) AS rank
            FROM memory_records
            WHERE agent_id = $1
              AND user_id = $2
              AND t_expired IS NULL
              AND t_invalid IS NULL
              AND quarantined = false
              AND (
                  entity_refs && $4
                  OR to_tsvector('english', content) @@ to_tsquery('english', $3)
              )
            ORDER BY rank DESC
            LIMIT 10
            "#,
        )
        .bind(agent_id)
        .bind(user_id)
        .bind(&tsquery)
        .bind(entity_refs)
        .fetch_all(pool)
        .await?
    };

    Ok(candidates
        .into_iter()
        .map(|(id, content, t_valid, importance, rank)| ConflictCandidate {
            id,
            content,
            t_valid,
            importance,
            rank,
        })
        .collect())
}

/// Check if two pieces of text likely represent contradicting facts.
/// Phase 1: keyword heuristic. Looks for negation patterns and substitution patterns.
pub fn is_contradicting(existing: &str, incoming: &str, shared_entities: &[String]) -> bool {
    if shared_entities.is_empty() {
        return false;
    }

    // Identical content is never a contradiction
    if existing == incoming {
        return false;
    }

    let ex_lower = existing.to_lowercase();
    let in_lower = incoming.to_lowercase();

    // Case-insensitive identical is also not a contradiction
    if ex_lower == in_lower {
        return false;
    }

    // Pattern: "uses X" vs "uses Y" (same verb, different object, same entity context)
    let preference_verbs = ["uses", "prefers", "likes", "wants", "runs", "deployed on"];
    for verb in preference_verbs {
        let ex_has = ex_lower.contains(verb);
        let in_has = in_lower.contains(verb);
        if ex_has && in_has {
            // Same verb used in both — likely a preference change
            return true;
        }
    }

    // Pattern: "switched from X" / "moved from X" in incoming implies old fact is invalid
    let switch_patterns = [
        "switched from",
        "moved from",
        "changed from",
        "migrated from",
        "no longer uses",
        "stopped using",
        "quit using",
    ];
    for pattern in switch_patterns {
        if in_lower.contains(pattern) {
            // Check if any shared entity is mentioned after the "from"
            for entity in shared_entities {
                if in_lower.contains(&format!("{} {}", pattern, entity.to_lowercase())) {
                    return true;
                }
            }
        }
    }

    // Pattern: negation of existing fact
    // "likes X" vs "doesn't like X" / "hates X"
    let negation_pairs = [
        ("likes", "doesn't like"),
        ("likes", "hates"),
        ("prefers", "doesn't prefer"),
        ("prefers", "avoids"),
        ("uses", "doesn't use"),
        ("uses", "stopped using"),
    ];
    for (pos, neg) in negation_pairs {
        if (ex_lower.contains(pos) && in_lower.contains(neg))
            || (ex_lower.contains(neg) && in_lower.contains(pos))
        {
            return true;
        }
    }

    false
}

/// Invalidate a list of conflicting memories.
pub async fn invalidate_conflicts(
    pool: &PgPool,
    agent_id: &str,
    conflict_ids: &[Uuid],
) -> Result<u64, MogError> {
    if conflict_ids.is_empty() {
        return Ok(0);
    }

    let mut total = 0u64;
    for id in conflict_ids {
        let affected = sqlx::query(
            "UPDATE memory_records SET t_invalid = NOW() WHERE id = $1 AND agent_id = $2 AND t_invalid IS NULL",
        )
        .bind(id)
        .bind(agent_id)
        .execute(pool)
        .await?
        .rows_affected();

        if affected > 0 {
            audit::log(
                pool,
                mogdb_core::AuditAction::Invalidate,
                agent_id,
                Some(*id),
                Some("conflict_detected"),
                None,
            )
            .await?;
            total += affected;
        }
    }

    Ok(total)
}

/// Extract meaningful search terms from text, filtering out stop words.
fn extract_search_terms(text: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "the", "a", "an", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can",
        "and", "but", "or", "not", "no", "so", "if", "then",
        "in", "on", "at", "to", "for", "of", "with", "from", "by",
        "this", "that", "these", "those", "here", "there",
        "also", "just", "very", "really", "actually", "now", "last",
    ];

    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| {
            let lower = w.to_lowercase();
            w.len() >= 3 && !STOP_WORDS.contains(&lower.as_str())
        })
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contradicts_preference_change() {
        assert!(is_contradicting(
            "user uses AWS",
            "user uses Google Cloud",
            &["AWS".to_string(), "Google Cloud".to_string()],
        ));
    }

    #[test]
    fn contradicts_switch_from() {
        assert!(is_contradicting(
            "user uses AWS for hosting",
            "user switched from AWS to GCP",
            &["AWS".to_string()],
        ));
    }

    #[test]
    fn contradicts_negation() {
        assert!(is_contradicting(
            "user likes dark mode",
            "user hates dark mode",
            &["dark mode".to_string()],
        ));
    }

    #[test]
    fn no_contradiction_without_shared_entities() {
        assert!(!is_contradicting(
            "user likes Python",
            "user likes Rust",
            &[],
        ));
    }

    #[test]
    fn no_contradiction_unrelated() {
        assert!(!is_contradicting(
            "user prefers dark mode",
            "user deployed on AWS",
            &["AWS".to_string()],
        ));
    }

    #[test]
    fn search_terms_exclude_stop_words() {
        let terms = extract_search_terms("I switched from AWS to Google Cloud last month");
        assert!(terms.contains(&"switched".to_string()));
        assert!(terms.contains(&"aws".to_string()));
        assert!(terms.contains(&"google".to_string()));
        assert!(!terms.contains(&"from".to_string()));
        assert!(!terms.contains(&"to".to_string()));
    }
}
