/// Worst-to-best scenario tests for importance scoring.

#[cfg(test)]
mod tests {
    use crate::scoring::score_importance;

    // ==================== WORST CASE ====================

    #[test]
    fn worst_empty_string() {
        let score = score_importance("", false);
        // Should not panic, should return something low
        assert!(score >= 0.1 && score <= 1.0, "empty string: got {score}");
    }

    #[test]
    fn worst_single_character() {
        let score = score_importance("x", false);
        assert!(score <= 0.5, "single char should score low, got {score}");
    }

    #[test]
    fn worst_only_whitespace() {
        let score = score_importance("     \n\t  ", false);
        assert!(score <= 0.5, "whitespace-only should score low, got {score}");
    }

    #[test]
    fn worst_unicode_garbage() {
        let score = score_importance("🤖💀🔥 ñ ü ö 你好世界", false);
        // Should not panic, should return some score
        assert!(score >= 0.1 && score <= 1.0, "unicode: got {score}");
    }

    #[test]
    fn worst_extremely_long_input() {
        let content = "word ".repeat(10_000);
        let score = score_importance(&content, false);
        assert!(score >= 0.1 && score <= 1.0, "long input: got {score}");
    }

    #[test]
    fn worst_conflicting_signals_always_and_maybe() {
        // Has both high ("always") and low ("maybe") signals
        let score = score_importance("maybe you should always do this", false);
        // Both signals fire — net effect should be some middle ground
        assert!(score >= 0.3 && score <= 0.9, "conflicting signals: got {score}");
    }

    #[test]
    fn worst_keyword_inside_another_word() {
        // "must" is inside "mustard", "not" inside "nothing"
        // These WILL false-positive because we use contains() — documenting this known limitation
        let score_mustard = score_importance("I like mustard on my sandwich", false);
        let score_actual_must = score_importance("You must do this", false);
        // Both will score high due to "must" matching — this is a known limitation
        // At minimum neither should panic
        assert!(score_mustard >= 0.1, "mustard: got {score_mustard}");
        assert!(score_actual_must >= 0.1, "actual must: got {score_actual_must}");
    }

    #[test]
    fn worst_all_stop_words() {
        let score = score_importance("I am the one who is", false);
        assert!(score <= 0.5, "stop words only should be low, got {score}");
    }

    // ==================== MEDIUM CASE ====================

    #[test]
    fn medium_neutral_statement() {
        let score = score_importance("The deployment completed at 3pm yesterday", false);
        assert!((0.35..=0.6).contains(&score), "neutral statement: got {score}");
    }

    #[test]
    fn medium_question() {
        let score = score_importance("What database does the team use for logging?", false);
        assert!(score <= 0.6, "question should not score high, got {score}");
    }

    #[test]
    fn medium_casual_mention() {
        let score = score_importance("I sometimes use vim for quick edits", false);
        // "sometimes" is a LOW_SIGNAL — should reduce score
        assert!(score < 0.5, "casual mention should be lowered, got {score}");
    }

    #[test]
    fn medium_implicit_preference_no_keyword() {
        // This IS a preference but has no preference keywords
        let score = score_importance("I write all my code in Rust these days", false);
        assert!((0.35..=0.6).contains(&score), "implicit preference: got {score}");
    }

    // ==================== GOOD CASE ====================

    #[test]
    fn good_clear_preference() {
        let score = score_importance("I prefer using TypeScript over JavaScript for new projects", false);
        assert!(score >= 0.6, "clear preference: got {score}");
    }

    #[test]
    fn good_switched_to() {
        let score = score_importance("We switched to Kubernetes last quarter for all deployments", false);
        assert!(score >= 0.6, "switched to: got {score}");
    }

    #[test]
    fn good_procedural_flag_overrides_content() {
        // Content is weak but procedural flag should force 0.9
        let score = score_importance("just a note", true);
        assert_eq!(score, 0.9, "procedural flag should force 0.9, got {score}");
    }

    // ==================== BEST CASE ====================

    #[test]
    fn best_strong_directive() {
        let score = score_importance("Never push directly to main without code review", false);
        assert!(score >= 0.75, "strong directive: got {score}");
    }

    #[test]
    fn best_critical_rule() {
        let score = score_importance("You must always run the test suite before deploying to production", false);
        assert!(score >= 0.75, "critical rule: got {score}");
    }

    #[test]
    fn best_multiple_high_signals() {
        // "always" + "must" + "important" — all high signals but score capped at 1.0
        let score = score_importance("It is critically important that you must always verify credentials", false);
        assert!(score >= 0.75 && score <= 1.0, "multiple high signals: got {score}");
    }
}
