/// Rule-based entity extraction from text.
/// Phase 1: pattern + keyword heuristics. Later phases add LLM extraction.

use mogdb_core::EntityKind;

/// An extracted entity from text, before DB lookup/creation.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedEntity {
    pub name: String,
    pub kind: EntityKind,
}

/// Known technology/tool names — matched case-insensitively.
const KNOWN_TOOLS: &[(&str, EntityKind)] = &[
    // Cloud providers
    ("aws", EntityKind::System),
    ("amazon web services", EntityKind::System),
    ("google cloud", EntityKind::System),
    ("gcp", EntityKind::System),
    ("azure", EntityKind::System),
    ("digitalocean", EntityKind::System),
    ("vercel", EntityKind::System),
    ("cloudflare", EntityKind::System),
    ("heroku", EntityKind::System),
    ("netlify", EntityKind::System),
    // Databases
    ("postgresql", EntityKind::Tool),
    ("postgres", EntityKind::Tool),
    ("mysql", EntityKind::Tool),
    ("mongodb", EntityKind::Tool),
    ("redis", EntityKind::Tool),
    ("sqlite", EntityKind::Tool),
    ("duckdb", EntityKind::Tool),
    ("clickhouse", EntityKind::Tool),
    ("elasticsearch", EntityKind::Tool),
    ("neo4j", EntityKind::Tool),
    ("pinecone", EntityKind::Tool),
    ("qdrant", EntityKind::Tool),
    ("supabase", EntityKind::Tool),
    // Languages
    ("python", EntityKind::Tool),
    ("javascript", EntityKind::Tool),
    ("typescript", EntityKind::Tool),
    ("rust", EntityKind::Tool),
    ("golang", EntityKind::Tool),
    ("java", EntityKind::Tool),
    ("ruby", EntityKind::Tool),
    ("swift", EntityKind::Tool),
    ("kotlin", EntityKind::Tool),
    // Frameworks
    ("react", EntityKind::Tool),
    ("nextjs", EntityKind::Tool),
    ("next.js", EntityKind::Tool),
    ("vue", EntityKind::Tool),
    ("angular", EntityKind::Tool),
    ("django", EntityKind::Tool),
    ("flask", EntityKind::Tool),
    ("fastapi", EntityKind::Tool),
    ("express", EntityKind::Tool),
    ("rails", EntityKind::Tool),
    ("spring", EntityKind::Tool),
    ("laravel", EntityKind::Tool),
    // DevOps
    ("docker", EntityKind::Tool),
    ("kubernetes", EntityKind::Tool),
    ("k8s", EntityKind::Tool),
    ("terraform", EntityKind::Tool),
    ("ansible", EntityKind::Tool),
    ("github", EntityKind::System),
    ("gitlab", EntityKind::System),
    ("jenkins", EntityKind::Tool),
    ("datadog", EntityKind::Tool),
    ("grafana", EntityKind::Tool),
    // Editors
    ("vim", EntityKind::Tool),
    ("neovim", EntityKind::Tool),
    ("vscode", EntityKind::Tool),
    ("cursor", EntityKind::Tool),
    ("emacs", EntityKind::Tool),
    ("intellij", EntityKind::Tool),
    // AI
    ("openai", EntityKind::System),
    ("claude", EntityKind::Tool),
    ("chatgpt", EntityKind::Tool),
    ("langchain", EntityKind::Tool),
    ("llamaindex", EntityKind::Tool),
];

/// Extract entities from text using known-name matching and capitalized-word heuristics.
pub fn extract_entities(text: &str) -> Vec<ExtractedEntity> {
    let mut found: Vec<ExtractedEntity> = Vec::new();
    let lower = text.to_lowercase();

    // Pass 1: match known tools/systems (longest match first to avoid partial hits)
    let mut sorted_known: Vec<_> = KNOWN_TOOLS.iter().collect();
    sorted_known.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    for (name, kind) in sorted_known {
        if lower.contains(name) {
            let canonical = canonicalize(name);
            if !found.iter().any(|e| e.name.eq_ignore_ascii_case(&canonical)) {
                found.push(ExtractedEntity {
                    name: canonical,
                    kind: kind.clone(),
                });
            }
        }
    }

    // Pass 2: capitalized multi-word phrases (likely proper nouns / project names)
    // Match sequences like "Project Apollo", "Team Infra"
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut i = 0;
    while i < words.len() {
        if starts_with_upper(words[i]) && !is_sentence_start(i, &words) && !is_common_word(words[i]) {
            // Collect consecutive capitalized words
            let start = i;
            while i < words.len() && starts_with_upper(words[i]) && !is_common_word(words[i]) {
                i += 1;
            }
            if i - start >= 1 {
                let phrase: String = words[start..i].join(" ");
                let clean = phrase.trim_matches(|c: char| c.is_ascii_punctuation()).to_string();
                if clean.len() >= 2
                    && !found.iter().any(|e| e.name.eq_ignore_ascii_case(&clean))
                    && !is_known_tool_name(&clean)
                {
                    found.push(ExtractedEntity {
                        name: clean,
                        kind: EntityKind::Other,
                    });
                }
            }
        } else {
            i += 1;
        }
    }

    found
}

/// Extract simple relationships from text.
/// Returns (subject_hint, relation, object_hint) tuples.
pub fn extract_relations(text: &str) -> Vec<(String, String, String)> {
    let lower = text.to_lowercase();
    let mut relations = Vec::new();

    // Pattern: "X uses Y", "X switched to Y", "X moved to Y"
    let patterns: &[(&str, &str)] = &[
        ("switched to ", "uses"),
        ("moved to ", "uses"),
        ("changed to ", "uses"),
        ("migrated to ", "uses"),
        ("switched from ", "previously_used"),
        ("moved from ", "previously_used"),
        ("started using ", "uses"),
        ("stop using ", "stopped_using"),
        ("stopped using ", "stopped_using"),
    ];

    // Special compound pattern: "switched/moved/changed from X to Y"
    let from_to_patterns = ["switched from", "moved from", "changed from", "migrated from"];
    for pattern in from_to_patterns {
        if let Some(pos) = lower.find(pattern) {
            let after = &text[pos + pattern.len()..];
            if let Some(to_pos) = after.to_lowercase().find(" to ") {
                let from_obj = after[..to_pos].trim().to_string();
                let to_obj = after[to_pos + 4..]
                    .split(|c: char| c == '.' || c == ',' || c == ';' || c == '\n')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !from_obj.is_empty() {
                    relations.push(("_user".to_string(), "previously_used".to_string(), from_obj));
                }
                if !to_obj.is_empty() {
                    relations.push(("_user".to_string(), "uses".to_string(), to_obj));
                }
            } else {
                // No "to" clause — just the "from" part
                let from_obj = after
                    .split(|c: char| c == '.' || c == ',' || c == ';' || c == '\n')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !from_obj.is_empty() {
                    relations.push(("_user".to_string(), "previously_used".to_string(), from_obj));
                }
            }
        }
    }

    for (pattern, relation) in patterns {
        // Skip "from" patterns already handled above
        if pattern.contains("from") {
            continue;
        }
        if let Some(pos) = lower.find(pattern) {
            let after = &text[pos + pattern.len()..];
            let object: String = after
                .split(|c: char| c == '.' || c == ',' || c == ';' || c == '\n')
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            if !object.is_empty() {
                relations.push(("_user".to_string(), relation.to_string(), object));
            }
        }
    }

    relations
}

fn canonicalize(name: &str) -> String {
    match name {
        "aws" | "amazon web services" => "AWS".to_string(),
        "gcp" | "google cloud" => "Google Cloud".to_string(),
        "k8s" | "kubernetes" => "Kubernetes".to_string(),
        "postgres" | "postgresql" => "PostgreSQL".to_string(),
        "nextjs" | "next.js" => "Next.js".to_string(),
        other => {
            // Title case single words
            let mut chars = other.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().to_string() + chars.as_str(),
            }
        }
    }
}

fn starts_with_upper(word: &str) -> bool {
    word.chars().next().is_some_and(|c| c.is_uppercase())
}

fn is_sentence_start(idx: usize, words: &[&str]) -> bool {
    if idx == 0 {
        return true;
    }
    // Previous word ends with sentence-ending punctuation
    words[idx - 1].ends_with('.') || words[idx - 1].ends_with('!') || words[idx - 1].ends_with('?')
}

fn is_common_word(word: &str) -> bool {
    const COMMON: &[&str] = &[
        "I", "The", "This", "That", "It", "We", "They", "He", "She",
        "My", "Our", "Your", "His", "Her", "Its", "Their",
        "A", "An", "And", "But", "Or", "Not", "No", "Yes",
        "If", "When", "Then", "So", "As", "In", "On", "At", "To", "For",
        "Is", "Are", "Was", "Were", "Be", "Been", "Being",
        "Have", "Has", "Had", "Do", "Does", "Did",
        "Can", "Could", "Will", "Would", "Should", "May", "Might",
        "Also", "Just", "Only", "Very", "Really", "Actually",
    ];
    let clean = word.trim_matches(|c: char| c.is_ascii_punctuation());
    COMMON.iter().any(|c| c.eq_ignore_ascii_case(clean))
}

fn is_known_tool_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    KNOWN_TOOLS.iter().any(|(k, _)| *k == lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_known_tools() {
        let entities = extract_entities("I use PostgreSQL and Redis for my project");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"PostgreSQL"), "got {:?}", names);
        assert!(names.contains(&"Redis"), "got {:?}", names);
    }

    #[test]
    fn canonicalizes_aws() {
        let entities = extract_entities("We deployed on aws last year");
        assert!(entities.iter().any(|e| e.name == "AWS"));
    }

    #[test]
    fn extracts_capitalized_proper_nouns() {
        let entities = extract_entities("The team works on Project Apollo every sprint");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Project Apollo"), "got {:?}", names);
    }

    #[test]
    fn skips_common_words() {
        let entities = extract_entities("The quick brown fox jumps over the lazy dog");
        // Should not extract "The" as an entity
        assert!(entities.is_empty(), "got {:?}", entities);
    }

    #[test]
    fn extracts_cloud_switch() {
        let entities = extract_entities("I switched from AWS to Google Cloud");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"AWS"));
        assert!(names.contains(&"Google Cloud"));
    }

    #[test]
    fn extracts_relations() {
        let rels = extract_relations("I switched to Google Cloud last month");
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].1, "uses");
        assert!(rels[0].2.contains("Google Cloud"));
    }

    #[test]
    fn extracts_from_relations() {
        let rels = extract_relations("We switched from AWS to GCP");
        let relation_types: Vec<&str> = rels.iter().map(|r| r.1.as_str()).collect();
        assert!(relation_types.contains(&"previously_used"));
        assert!(relation_types.contains(&"uses"));
    }
}
