/// Worst-to-best scenario tests for entity extraction.

#[cfg(test)]
mod tests {
    use crate::extraction::{extract_entities, extract_relations};
    use mogdb_core::EntityKind;

    // ==================== WORST CASE ====================

    #[test]
    fn worst_empty_string() {
        let entities = extract_entities("");
        assert!(
            entities.is_empty(),
            "empty should give no entities: {:?}",
            entities
        );

        let relations = extract_relations("");
        assert!(
            relations.is_empty(),
            "empty should give no relations: {:?}",
            relations
        );
    }

    #[test]
    fn worst_all_lowercase_no_entities() {
        let entities = extract_entities("the quick brown fox jumps over the lazy dog");
        assert!(
            entities.is_empty(),
            "should find no entities: {:?}",
            entities
        );
    }

    #[test]
    fn worst_unicode_and_emoji() {
        let entities = extract_entities("使用 PostgreSQL 和 Redis 来构建 🚀🔥");
        // Should still find PostgreSQL and Redis even with unicode around them
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"PostgreSQL"),
            "should find PostgreSQL in unicode: {:?}",
            names
        );
        assert!(
            names.contains(&"Redis"),
            "should find Redis in unicode: {:?}",
            names
        );
    }

    #[test]
    fn worst_misleading_capitalization_sentence_starts() {
        // Every word is capitalized because it starts a sentence — should NOT extract common words
        let entities = extract_entities("Go home. The cat sat. It was warm.");
        // "Go" might match as a tool (golang? no, "golang" or "go" — "go" is not in our list at lowercase check)
        // Actually checking: "golang" is in the list but "go" is not
        // None of "home", "The", "cat", "It" should be entities
        for e in &entities {
            assert!(
                !["home", "cat", "warm"].contains(&e.name.to_lowercase().as_str()),
                "should not extract common words: {:?}",
                entities
            );
        }
    }

    #[test]
    fn worst_entity_name_embedded_in_word() {
        // "rust" is inside "frustration", "vim" is inside "victim"
        // This WILL false-positive — documenting known limitation
        let entities = extract_entities("My frustration with being a victim of circumstance");
        let names: Vec<String> = entities.iter().map(|e| e.name.to_lowercase()).collect();
        // Rust might match from "frustration" — this is a known limitation of substring matching
        // The test documents the behavior rather than asserting correctness
        // At minimum should not panic
        assert!(
            names.len() <= 3,
            "should not extract too many false positives: {:?}",
            names
        );
    }

    #[test]
    fn worst_extremely_long_input() {
        let content = "PostgreSQL ".repeat(5000);
        let entities = extract_entities(&content);
        // Should find PostgreSQL exactly once (deduped), not 5000 times
        let pg_count = entities.iter().filter(|e| e.name == "PostgreSQL").count();
        assert_eq!(
            pg_count, 1,
            "should deduplicate: found {} PostgreSQL entries",
            pg_count
        );
    }

    #[test]
    fn worst_relation_extraction_no_pattern_match() {
        let relations = extract_relations("The weather is nice today");
        assert!(
            relations.is_empty(),
            "unrelated text should give no relations: {:?}",
            relations
        );
    }

    #[test]
    fn worst_relation_pattern_at_end_of_string() {
        // Pattern exists but nothing after it
        let relations = extract_relations("I switched to");
        // Should handle gracefully — either empty result or empty object
        for r in &relations {
            assert!(
                r.2.is_empty() || r.2.len() < 50,
                "should handle truncated pattern: {:?}",
                relations
            );
        }
    }

    // ==================== MEDIUM CASE ====================

    #[test]
    fn medium_mixed_known_and_unknown_entities() {
        let entities =
            extract_entities("We use PostgreSQL on Project Mercury with the Zebra framework");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        // Should find PostgreSQL (known) and Project Mercury (proper noun)
        assert!(
            names.contains(&"PostgreSQL"),
            "should find PostgreSQL: {:?}",
            names
        );
        assert!(
            names.contains(&"Project Mercury"),
            "should find Project Mercury: {:?}",
            names
        );
    }

    #[test]
    fn medium_case_insensitive_known_tools() {
        let entities = extract_entities("we deployed on POSTGRESQL and used REDIS for caching");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"PostgreSQL"),
            "should canonicalize POSTGRESQL: {:?}",
            names
        );
        assert!(
            names.contains(&"Redis"),
            "should canonicalize REDIS: {:?}",
            names
        );
    }

    #[test]
    fn medium_overlapping_known_names() {
        // "google cloud" should match as one entity, not "google" separately
        let entities = extract_entities("We migrated everything to google cloud platform");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"Google Cloud"),
            "should find Google Cloud: {:?}",
            names
        );
    }

    #[test]
    fn medium_entity_with_punctuation() {
        let entities =
            extract_entities("Have you tried Next.js? It's better than React, honestly.");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"Next.js"),
            "should find Next.js: {:?}",
            names
        );
        assert!(names.contains(&"React"), "should find React: {:?}", names);
    }

    #[test]
    fn medium_relation_with_extra_words() {
        let relations = extract_relations("I recently started using Docker for all my deployments");
        assert!(
            !relations.is_empty(),
            "should find 'started using' relation: {:?}",
            relations
        );
        assert_eq!(relations[0].1, "uses");
    }

    #[test]
    fn medium_entity_kind_classification() {
        let entities = extract_entities("We use AWS and Python with Django");
        for e in &entities {
            match e.name.as_str() {
                "AWS" => assert_eq!(e.kind, EntityKind::System, "AWS should be System"),
                "Python" => assert_eq!(e.kind, EntityKind::Tool, "Python should be Tool"),
                "Django" => assert_eq!(e.kind, EntityKind::Tool, "Django should be Tool"),
                _ => {}
            }
        }
    }

    // ==================== GOOD CASE ====================

    #[test]
    fn good_multiple_entities_with_relations() {
        let entities =
            extract_entities("I switched from AWS to Google Cloud for our backend services");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"AWS"), "should find AWS: {:?}", names);
        assert!(
            names.contains(&"Google Cloud"),
            "should find Google Cloud: {:?}",
            names
        );

        let relations =
            extract_relations("I switched from AWS to Google Cloud for our backend services");
        let rel_types: Vec<&str> = relations.iter().map(|r| r.1.as_str()).collect();
        assert!(
            rel_types.contains(&"previously_used"),
            "should detect previously_used: {:?}",
            relations
        );
        assert!(
            rel_types.contains(&"uses"),
            "should detect uses: {:?}",
            relations
        );
    }

    #[test]
    fn good_proper_noun_multi_word() {
        let entities =
            extract_entities("The team is building Project Phoenix on the Apollo Server");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"Project Phoenix"),
            "should find Project Phoenix: {:?}",
            names
        );
        assert!(
            names.contains(&"Apollo Server"),
            "should find Apollo Server: {:?}",
            names
        );
    }

    // ==================== BEST CASE ====================

    #[test]
    fn best_clear_tech_stack_description() {
        let entities = extract_entities(
            "Our stack: PostgreSQL for data, Redis for caching, Docker for deployment, React for frontend"
        );
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"PostgreSQL"));
        assert!(names.contains(&"Redis"));
        assert!(names.contains(&"Docker"));
        assert!(names.contains(&"React"));
        assert_eq!(
            names.len(),
            4,
            "should find exactly 4 entities: {:?}",
            names
        );
    }

    #[test]
    fn best_clear_switch_with_relation() {
        let relations = extract_relations("We migrated from MySQL to PostgreSQL last quarter");
        assert!(
            relations.len() >= 2,
            "should find both from and to relations: {:?}",
            relations
        );

        let has_prev = relations
            .iter()
            .any(|r| r.1 == "previously_used" && r.2.contains("MySQL"));
        let has_uses = relations
            .iter()
            .any(|r| r.1 == "uses" && r.2.contains("PostgreSQL"));
        assert!(
            has_prev,
            "should detect MySQL as previously_used: {:?}",
            relations
        );
        assert!(
            has_uses,
            "should detect PostgreSQL as uses: {:?}",
            relations
        );
    }
}
