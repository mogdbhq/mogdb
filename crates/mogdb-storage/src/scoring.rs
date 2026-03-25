/// Rule-based importance scoring for memory records.
/// Phase 1: keyword heuristics. Later phases add LLM-based scoring.

/// Score a piece of text for importance. Returns 0.0–1.0.
pub fn score_importance(content: &str, is_procedural: bool) -> f64 {
    if is_procedural {
        return 0.9;
    }

    let lower = content.to_lowercase();
    let mut score: f64 = 0.5;

    // Strong directive keywords → high importance
    const HIGH_SIGNALS: &[&str] = &[
        "always", "never", "must", "do not", "don't", "important",
        "critical", "require", "mandatory", "forbidden", "rule",
        "ensure", "absolutely", "strictly",
    ];

    // Preference keywords → medium-high importance
    const MEDIUM_SIGNALS: &[&str] = &[
        "prefer", "hate", "love", "like", "dislike", "want",
        "switched to", "moved to", "changed to", "use instead",
        "stop using", "started using", "favorite", "favourite",
    ];

    // Low-signal content → reduce importance
    const LOW_SIGNALS: &[&str] = &[
        "maybe", "might", "not sure", "i think", "possibly",
        "sometimes", "occasionally",
    ];

    for kw in HIGH_SIGNALS {
        if lower.contains(kw) {
            score += 0.3;
            break;
        }
    }

    for kw in MEDIUM_SIGNALS {
        if lower.contains(kw) {
            score += 0.15;
            break;
        }
    }

    for kw in LOW_SIGNALS {
        if lower.contains(kw) {
            score -= 0.15;
            break;
        }
    }

    // Longer content with substance tends to be more important
    let word_count = content.split_whitespace().count();
    if word_count < 4 {
        score -= 0.1; // very short = probably noise
    }

    score.clamp(0.1, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_always_high() {
        assert_eq!(score_importance("anything", true), 0.9);
    }

    #[test]
    fn directive_keywords_score_high() {
        let score = score_importance("Never call the production database directly", false);
        assert!(score >= 0.75, "got {score}");
    }

    #[test]
    fn preference_keywords_score_medium() {
        let score = score_importance("I prefer dark mode in my editor", false);
        assert!(score >= 0.6, "got {score}");
    }

    #[test]
    fn uncertain_language_scores_lower() {
        let score = score_importance("I think maybe we could try something", false);
        assert!(score <= 0.45, "got {score}");
    }

    #[test]
    fn neutral_content_scores_baseline() {
        let score = score_importance("The meeting is scheduled for Tuesday afternoon", false);
        assert!((0.4..=0.6).contains(&score), "got {score}");
    }

    #[test]
    fn switched_to_scores_medium() {
        let score = score_importance("I switched to Google Cloud last month", false);
        assert!(score >= 0.6, "got {score}");
    }
}
