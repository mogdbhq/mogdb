/// Cross-encoder reranking — rescore RRF-fused results using an LLM.
///
/// After the 3-way RRF merge (FTS + vector + graph), the top candidates are
/// sent to a local LLM via Ollama. The model scores each (query, passage) pair
/// for relevance, and the results are reordered accordingly.
///
/// This is the 4th retrieval signal — it doesn't add new candidates, but
/// dramatically improves precision of the final ranking.
///
/// Enable by setting `MOGDB_RERANK=1`. Disabled by default (zero-cost when off).
use mogdb_core::MogError;
use tracing::{debug, warn};

use crate::search::SearchResult;

/// Whether reranking is enabled (reads MOGDB_RERANK env var once).
pub fn is_enabled() -> bool {
    std::env::var("MOGDB_RERANK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Reranker that uses Ollama's chat API for pointwise relevance scoring.
pub struct OllamaReranker {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaReranker {
    pub fn from_env() -> Self {
        let base_url =
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model =
            std::env::var("MOGDB_RERANK_MODEL").unwrap_or_else(|_| "dolphin-llama3:8b".to_string());
        Self {
            client: reqwest::Client::new(),
            base_url,
            model,
        }
    }

    /// Score each result's relevance to the query via a single LLM call.
    ///
    /// Returns a Vec of scores (0.0–10.0), one per input result, in the same order.
    /// On failure, returns Err and the caller should keep the original order.
    async fn score(&self, query: &str, results: &[SearchResult]) -> Result<Vec<f64>, MogError> {
        if results.is_empty() {
            return Ok(vec![]);
        }

        // Build numbered passage list (truncate each to keep prompt manageable)
        let mut passages = String::new();
        for (i, r) in results.iter().enumerate() {
            let truncated: String = r.content.chars().take(300).collect();
            passages.push_str(&format!("[{}] {}\n", i + 1, truncated));
        }

        let prompt = format!(
            r#"Rate how relevant each passage is to the query. Score 0 (irrelevant) to 10 (perfectly relevant).

Query: {query}

Passages:
{passages}
Output ONLY scores in this exact format, one per line:
1:SCORE
2:SCORE
...

Scores:"#
        );

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&serde_json::json!({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": false,
                "options": {"temperature": 0, "num_predict": 256}
            }))
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| MogError::External(format!("rerank request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(MogError::External(format!("rerank API {status}: {body}")));
        }

        #[derive(serde::Deserialize)]
        struct ChatResponse {
            message: ChatMessage,
        }
        #[derive(serde::Deserialize)]
        struct ChatMessage {
            content: String,
        }

        let body: ChatResponse = resp
            .json()
            .await
            .map_err(|e| MogError::External(format!("rerank response parse error: {e}")))?;

        parse_scores(&body.message.content, results.len())
    }
}

/// Parse "1:8\n2:3\n3:9\n..." format into a Vec of scores.
/// Falls back gracefully: missing entries get score 0.0.
fn parse_scores(text: &str, expected: usize) -> Result<Vec<f64>, MogError> {
    let mut scores = vec![0.0_f64; expected];
    let mut parsed = 0;

    for line in text.lines() {
        let line = line.trim();
        if let Some((idx_str, score_str)) = line.split_once(':') {
            let idx_str = idx_str.trim().trim_start_matches('[').trim_end_matches(']');
            if let (Ok(idx), Ok(score)) =
                (idx_str.parse::<usize>(), score_str.trim().parse::<f64>())
            {
                if idx >= 1 && idx <= expected {
                    scores[idx - 1] = score.clamp(0.0, 10.0);
                    parsed += 1;
                }
            }
        }
    }

    if parsed < expected / 2 {
        return Err(MogError::External(format!(
            "rerank: only parsed {parsed}/{expected} scores"
        )));
    }

    Ok(scores)
}

/// Rerank search results using the cross-encoder.
///
/// Takes the fused results, scores them with the LLM, and returns them
/// reordered by relevance score. The original RRF score is blended with
/// the reranker score so that retrieval signals aren't thrown away.
///
/// On any failure, returns the original results unchanged.
pub async fn rerank(query: &str, mut results: Vec<SearchResult>) -> Vec<SearchResult> {
    if !is_enabled() || results.len() <= 1 {
        return results;
    }

    let reranker = OllamaReranker::from_env();

    // Only rerank top candidates (no point scoring low-ranked ones)
    let rerank_count = results.len().min(20);

    match reranker.score(query, &results[..rerank_count]).await {
        Ok(scores) => {
            debug!(
                "reranker scored {} results: {:?}",
                scores.len(),
                &scores[..scores.len().min(5)]
            );

            // Blend: 40% reranker score (normalized to 0-1) + 60% original RRF score (normalized)
            let max_rrf = results
                .iter()
                .map(|r| r.score)
                .fold(0.0_f64, f64::max)
                .max(0.001);

            for (i, score) in scores.into_iter().enumerate() {
                let rerank_normalized = score / 10.0;
                let rrf_normalized = results[i].score / max_rrf;
                results[i].score = (rrf_normalized * 0.6) + (rerank_normalized * 0.4);
            }

            // Re-sort by blended score
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            results
        }
        Err(e) => {
            warn!("reranking failed, keeping RRF order: {e}");
            results
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_scores_basic() {
        let text = "1:8\n2:3\n3:9\n";
        let scores = parse_scores(text, 3).unwrap();
        assert_eq!(scores, vec![8.0, 3.0, 9.0]);
    }

    #[test]
    fn parse_scores_with_brackets() {
        let text = "[1]: 7\n[2]: 5\n[3]: 2\n";
        let scores = parse_scores(text, 3).unwrap();
        assert_eq!(scores, vec![7.0, 5.0, 2.0]);
    }

    #[test]
    fn parse_scores_with_noise() {
        // Model might add extra text — we should still extract the scores
        let text = "Here are the scores:\n1:8\n2:3\n3:9\nDone.";
        let scores = parse_scores(text, 3).unwrap();
        assert_eq!(scores, vec![8.0, 3.0, 9.0]);
    }

    #[test]
    fn parse_scores_clamps() {
        let text = "1:15\n2:-3\n";
        let scores = parse_scores(text, 2).unwrap();
        assert_eq!(scores, vec![10.0, 0.0]);
    }

    #[test]
    fn parse_scores_fails_if_too_few() {
        let text = "1:8\n";
        assert!(parse_scores(text, 5).is_err());
    }

    #[test]
    fn is_enabled_defaults_off() {
        // In test env, MOGDB_RERANK is not set
        // (can't assert false definitively since env might be set, but default is false)
        let val = std::env::var("MOGDB_RERANK").unwrap_or_default();
        if val.is_empty() {
            assert!(!is_enabled());
        }
    }
}
