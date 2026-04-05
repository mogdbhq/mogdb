/// Entity extraction from text — keyword heuristics + optional LLM extraction.
///
/// The keyword extractor catches known tools/systems reliably.
/// The LLM extractor (via Ollama) catches implicit entities like "my manager",
/// novel tools, and complex relationships that keywords miss.
/// Results are merged: keywords first, then LLM additions that aren't duplicates.
use mogdb_core::EntityKind;
use tracing::{debug, warn};

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
            if !found
                .iter()
                .any(|e| e.name.eq_ignore_ascii_case(&canonical))
            {
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
        if starts_with_upper(words[i]) && !is_sentence_start(i, &words) && !is_common_word(words[i])
        {
            // Collect consecutive capitalized words
            let start = i;
            while i < words.len() && starts_with_upper(words[i]) && !is_common_word(words[i]) {
                i += 1;
            }
            if i - start >= 1 {
                let phrase: String = words[start..i].join(" ");
                let clean = phrase
                    .trim_matches(|c: char| c.is_ascii_punctuation())
                    .to_string();
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
    let from_to_patterns = [
        "switched from",
        "moved from",
        "changed from",
        "migrated from",
    ];
    for pattern in from_to_patterns {
        if let Some(pos) = lower.find(pattern) {
            let after = &text[pos + pattern.len()..];
            if let Some(to_pos) = after.to_lowercase().find(" to ") {
                let from_obj = after[..to_pos].trim().to_string();
                let to_obj = after[to_pos + 4..]
                    .split(['.', ',', ';', '\n'])
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
                    .split(['.', ',', ';', '\n'])
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
                .split(['.', ',', ';', '\n'])
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

// ---------------------------------------------------------------------------
// LLM-based entity extraction (via Ollama)
// ---------------------------------------------------------------------------

/// LLM extractor that uses Ollama's chat API.
pub struct OllamaExtractor {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaExtractor {
    pub fn from_env() -> Self {
        let base_url =
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model = std::env::var("MOGDB_EXTRACT_MODEL")
            .unwrap_or_else(|_| "dolphin-llama3:8b".to_string());
        Self {
            client: reqwest::Client::new(),
            base_url,
            model,
        }
    }

    /// Extract entities from text using the LLM.
    /// Returns extracted entities or an error (caller should fall back to keywords).
    async fn extract(&self, text: &str) -> Result<ExtractionResult, String> {
        let prompt = format!(
            r#"Extract named entities and relationships from this text.

Text: {text}

Output ONLY valid JSON in this exact format:
{{"entities":[{{"name":"EntityName","kind":"person|tool|system|concept|project|other"}}],"relations":[{{"subject":"SubjectName","relation":"uses|owns|works_on|manages|previously_used","object":"ObjectName"}}]}}

Rules:
- Include people (by name or role like "my manager", "my boss"), tools, systems, projects, concepts
- For implicit references like "my manager", use the role as the name (e.g. "manager")
- kind must be one of: person, tool, system, concept, project, other
- Only include relations explicitly stated in the text
- If no entities found, return {{"entities":[],"relations":[]}}

JSON:"#
        );

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&serde_json::json!({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": false,
                "options": {"temperature": 0, "num_predict": 512}
            }))
            .timeout(std::time::Duration::from_secs(15))
            .send()
            .await
            .map_err(|e| format!("ollama request failed: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("ollama API {}", resp.status()));
        }

        #[derive(serde::Deserialize)]
        struct ChatResp {
            message: ChatMsg,
        }
        #[derive(serde::Deserialize)]
        struct ChatMsg {
            content: String,
        }

        let body: ChatResp = resp.json().await.map_err(|e| format!("parse error: {e}"))?;

        parse_llm_extraction(&body.message.content)
    }
}

/// Entities + relations extracted from text.
type ExtractionResult = (Vec<ExtractedEntity>, Vec<(String, String, String)>);

/// Parse the LLM's JSON output into entities and relations.
fn parse_llm_extraction(text: &str) -> Result<ExtractionResult, String> {
    // Find JSON in the response (model might add extra text)
    let json_str = text
        .find('{')
        .and_then(|start| text.rfind('}').map(|end| &text[start..=end]))
        .ok_or("no JSON found in response")?;

    #[derive(serde::Deserialize)]
    struct LlmOutput {
        #[serde(default)]
        entities: Vec<LlmEntity>,
        #[serde(default)]
        relations: Vec<LlmRelation>,
    }
    #[derive(serde::Deserialize)]
    struct LlmEntity {
        name: String,
        #[serde(default = "default_kind")]
        kind: String,
    }
    #[derive(serde::Deserialize)]
    struct LlmRelation {
        subject: String,
        relation: String,
        object: String,
    }
    fn default_kind() -> String {
        "other".to_string()
    }

    let output: LlmOutput =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let entities: Vec<ExtractedEntity> = output
        .entities
        .into_iter()
        .filter(|e| !e.name.is_empty() && e.name.len() <= 100)
        .map(|e| ExtractedEntity {
            name: e.name,
            kind: match e.kind.as_str() {
                "person" => EntityKind::Person,
                "tool" => EntityKind::Tool,
                "system" => EntityKind::System,
                "concept" => EntityKind::Concept,
                "project" => EntityKind::Project,
                _ => EntityKind::Other,
            },
        })
        .collect();

    let relations: Vec<(String, String, String)> = output
        .relations
        .into_iter()
        .filter(|r| !r.subject.is_empty() && !r.object.is_empty())
        .map(|r| (r.subject, r.relation, r.object))
        .collect();

    Ok((entities, relations))
}

/// Extract entities using LLM, merged with keyword results.
///
/// Keyword extraction runs first (fast, reliable). Then the LLM adds any
/// entities it finds that keywords missed. On LLM failure, returns keyword
/// results only.
pub async fn extract_entities_with_llm(
    text: &str,
) -> (Vec<ExtractedEntity>, Vec<(String, String, String)>) {
    // Always start with keyword extraction (reliable baseline)
    let mut entities = extract_entities(text);
    let mut relations = extract_relations(text);

    // Try LLM extraction
    let extractor = OllamaExtractor::from_env();
    match extractor.extract(text).await {
        Ok((llm_entities, llm_relations)) => {
            debug!(
                "LLM extracted {} entities, {} relations",
                llm_entities.len(),
                llm_relations.len()
            );

            // Merge: add LLM entities that aren't already in keyword results
            for llm_ent in llm_entities {
                if !entities
                    .iter()
                    .any(|e| e.name.eq_ignore_ascii_case(&llm_ent.name))
                {
                    entities.push(llm_ent);
                }
            }

            // Merge: add LLM relations that aren't duplicates
            for llm_rel in llm_relations {
                let is_dup = relations.iter().any(|(s, r, o)| {
                    s.eq_ignore_ascii_case(&llm_rel.0)
                        && r == &llm_rel.1
                        && o.eq_ignore_ascii_case(&llm_rel.2)
                });
                if !is_dup {
                    relations.push(llm_rel);
                }
            }
        }
        Err(e) => {
            warn!("LLM entity extraction failed, using keywords only: {e}");
        }
    }

    (entities, relations)
}

// ---------------------------------------------------------------------------
// Atomic fact decomposition (via Ollama)
// ---------------------------------------------------------------------------

/// Whether fact decomposition is enabled (reads MOGDB_DECOMPOSE env var).
pub fn decompose_enabled() -> bool {
    std::env::var("MOGDB_DECOMPOSE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Decompose text into atomic facts using the LLM.
///
/// Takes a potentially long, multi-topic text and breaks it into
/// short, single-fact statements. Each fact is self-contained and
/// independently retrievable.
///
/// Returns the list of atomic fact strings, or the original text
/// as a single-element vec on failure.
pub async fn decompose_facts(text: &str) -> Vec<String> {
    // Skip very short text — already atomic
    if text.len() < 80 || text.split_whitespace().count() < 12 {
        return vec![text.to_string()];
    }

    let extractor = OllamaExtractor::from_env();

    let prompt = format!(
        r#"Break this text into atomic facts. Each fact should be a single, self-contained statement.

Text: {text}

Rules:
- Each fact should be one sentence, containing one piece of information
- Preserve names, dates, numbers, and specific details exactly
- Include who/what the fact is about (don't use pronouns without context)
- If the text mentions preferences, state them explicitly ("User prefers X")
- If the text mentions dates/times, include them in the fact
- Output one fact per line, no numbering, no bullets
- If the text is already a single fact, output it as-is

Facts:"#
    );

    let resp = match extractor
        .client
        .post(format!("{}/api/chat", extractor.base_url))
        .json(&serde_json::json!({
            "model": extractor.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": false,
            "options": {"temperature": 0, "num_predict": 1024}
        }))
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!("fact decomposition request failed: {e}");
            return vec![text.to_string()];
        }
    };

    if !resp.status().is_success() {
        warn!("fact decomposition API error: {}", resp.status());
        return vec![text.to_string()];
    }

    #[derive(serde::Deserialize)]
    struct ChatResp {
        message: ChatMsg,
    }
    #[derive(serde::Deserialize)]
    struct ChatMsg {
        content: String,
    }

    let body: ChatResp = match resp.json().await {
        Ok(b) => b,
        Err(e) => {
            warn!("fact decomposition parse error: {e}");
            return vec![text.to_string()];
        }
    };

    let facts: Vec<String> = body
        .message
        .content
        .lines()
        .map(|l| l.trim())
        .map(|l| {
            l.trim_start_matches(|c: char| c == '-' || c == '*' || c == '•' || c.is_ascii_digit())
                .trim_start_matches('.')
                .trim_start_matches(')')
                .trim()
                .to_string()
        })
        .filter(|l| l.len() >= 10 && l.len() <= 500)
        .collect();

    if facts.is_empty() {
        return vec![text.to_string()];
    }

    debug!("decomposed into {} atomic facts", facts.len());
    facts
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
        "I", "The", "This", "That", "It", "We", "They", "He", "She", "My", "Our", "Your", "His",
        "Her", "Its", "Their", "A", "An", "And", "But", "Or", "Not", "No", "Yes", "If", "When",
        "Then", "So", "As", "In", "On", "At", "To", "For", "Is", "Are", "Was", "Were", "Be",
        "Been", "Being", "Have", "Has", "Had", "Do", "Does", "Did", "Can", "Could", "Will",
        "Would", "Should", "May", "Might", "Also", "Just", "Only", "Very", "Really", "Actually",
    ];
    let clean = word.trim_matches(|c: char| c.is_ascii_punctuation());
    COMMON.iter().any(|c| c.eq_ignore_ascii_case(clean))
}

fn is_known_tool_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    KNOWN_TOOLS.iter().any(|(k, _)| *k == lower)
}

// ---------------------------------------------------------------------------
// Temporal expression extraction
// ---------------------------------------------------------------------------

use chrono::{DateTime, Datelike, Duration, NaiveDate, TimeZone, Utc};

/// A temporal reference extracted from text — either a point or a range.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalRef {
    /// Center of the temporal reference (midpoint for ranges).
    pub center: DateTime<Utc>,
    /// Half-width of the window in seconds. A point has radius 0.
    /// "last week" → center = 3.5 days ago, radius = 3.5 days.
    pub radius_secs: i64,
}

/// Extract temporal references from a query string.
///
/// Recognizes patterns like:
///   - "yesterday", "last week", "last month", "last year"
///   - "N days/weeks/months ago"
///   - "in January", "in March 2024"
///   - "this week", "this month"
///
/// Returns None if no temporal reference is found.
pub fn extract_temporal(text: &str) -> Option<TemporalRef> {
    let lower = text.to_lowercase();
    let now = Utc::now();

    // "yesterday"
    if lower.contains("yesterday") {
        let center = now - Duration::hours(24);
        return Some(TemporalRef {
            center,
            radius_secs: 43200, // 12h
        });
    }

    // "today"
    if lower.contains("today") || lower.contains("this morning") || lower.contains("this evening") {
        return Some(TemporalRef {
            center: now,
            radius_secs: 43200, // 12h
        });
    }

    // "N days/weeks/months/years ago"
    if let Some(temporal) = parse_n_units_ago(&lower, now) {
        return Some(temporal);
    }

    // "last week" / "last month" / "last year"
    if lower.contains("last week") {
        let center = now - Duration::days(7);
        return Some(TemporalRef {
            center,
            radius_secs: 3 * 86400, // ±3 days
        });
    }
    if lower.contains("last month") {
        let center = now - Duration::days(30);
        return Some(TemporalRef {
            center,
            radius_secs: 15 * 86400, // ±15 days
        });
    }
    if lower.contains("last year") {
        let center = now - Duration::days(365);
        return Some(TemporalRef {
            center,
            radius_secs: 182 * 86400, // ±6 months
        });
    }

    // "this week"
    if lower.contains("this week") {
        let center = now - Duration::days(3);
        return Some(TemporalRef {
            center,
            radius_secs: 3 * 86400,
        });
    }

    // "this month"
    if lower.contains("this month") {
        let center = now - Duration::days(15);
        return Some(TemporalRef {
            center,
            radius_secs: 15 * 86400,
        });
    }

    // "in January", "in February", ... optionally with year
    if let Some(temporal) = parse_in_month(&lower, now) {
        return Some(temporal);
    }

    None
}

/// Parse "N days/weeks/months/years ago" patterns.
fn parse_n_units_ago(text: &str, now: DateTime<Utc>) -> Option<TemporalRef> {
    // Match patterns like "3 days ago", "two weeks ago", "a month ago"
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| c.is_ascii_punctuation())
                .to_string()
        })
        .collect();

    for (i, word) in words.iter().enumerate() {
        if word == "ago" && i >= 2 {
            let unit = words[i - 1].as_str();
            let n_str = words[i - 2].as_str();
            let n: i64 = match n_str {
                "a" | "an" | "one" => 1,
                "two" => 2,
                "three" => 3,
                "four" => 4,
                "five" => 5,
                "six" => 6,
                "seven" => 7,
                "eight" => 8,
                "nine" => 9,
                "ten" => 10,
                s => s.parse().ok()?,
            };

            let (days, radius) = match unit {
                "day" | "days" => (n, 86400_i64 / 2),
                "week" | "weeks" => (n * 7, 3 * 86400),
                "month" | "months" => (n * 30, 15 * 86400),
                "year" | "years" => (n * 365, 182 * 86400),
                _ => return None,
            };

            let center = now - Duration::days(days);
            return Some(TemporalRef {
                center,
                radius_secs: radius,
            });
        }
    }
    None
}

/// Parse "in January", "in March 2024" patterns.
fn parse_in_month(text: &str, now: DateTime<Utc>) -> Option<TemporalRef> {
    const MONTHS: &[(&str, u32)] = &[
        ("january", 1),
        ("february", 2),
        ("march", 3),
        ("april", 4),
        ("may", 5),
        ("june", 6),
        ("july", 7),
        ("august", 8),
        ("september", 9),
        ("october", 10),
        ("november", 11),
        ("december", 12),
    ];

    for (name, month_num) in MONTHS {
        if let Some(pos) = text.find(name) {
            // Check for "in <month>" pattern
            let before = &text[..pos];
            if !before.ends_with("in ")
                && !before.ends_with("since ")
                && !before.ends_with("during ")
            {
                continue;
            }

            // Check for year after month name
            let after = &text[pos + name.len()..];
            let year = after
                .split_whitespace()
                .next()
                .and_then(|w| {
                    w.trim_matches(|c: char| !c.is_ascii_digit())
                        .parse::<i32>()
                        .ok()
                })
                .filter(|y| *y >= 2000 && *y <= 2100)
                .unwrap_or_else(|| {
                    // Default: if the month is in the future this year, use last year
                    if *month_num > now.month() {
                        now.year() - 1
                    } else {
                        now.year()
                    }
                });

            let date = NaiveDate::from_ymd_opt(year, *month_num, 15)?;
            let center = Utc.from_utc_datetime(&date.and_hms_opt(12, 0, 0)?);
            return Some(TemporalRef {
                center,
                radius_secs: 15 * 86400, // ±15 days (covers the month)
            });
        }
    }
    None
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

    // -----------------------------------------------------------------------
    // Temporal extraction tests
    // -----------------------------------------------------------------------

    #[test]
    fn temporal_yesterday() {
        let t = extract_temporal("What did we discuss yesterday?").unwrap();
        let age = Utc::now() - t.center;
        assert!(age.num_hours() >= 12 && age.num_hours() <= 36);
    }

    #[test]
    fn temporal_last_week() {
        let t = extract_temporal("What happened last week?").unwrap();
        let age = Utc::now() - t.center;
        assert!(age.num_days() >= 5 && age.num_days() <= 10);
    }

    #[test]
    fn temporal_n_days_ago() {
        let t = extract_temporal("What did the user say 3 days ago?").unwrap();
        let age = Utc::now() - t.center;
        assert!(age.num_days() >= 2 && age.num_days() <= 4);
    }

    #[test]
    fn temporal_two_months_ago() {
        let t = extract_temporal("What were we working on two months ago?").unwrap();
        let age = Utc::now() - t.center;
        assert!(age.num_days() >= 50 && age.num_days() <= 70);
    }

    #[test]
    fn temporal_in_month() {
        let t = extract_temporal("What database was the user using in january?").unwrap();
        assert_eq!(t.center.month(), 1);
    }

    #[test]
    fn temporal_no_reference() {
        assert!(extract_temporal("What is the user's favorite color?").is_none());
    }

    #[test]
    fn temporal_this_week() {
        let t = extract_temporal("What did we cover this week?").unwrap();
        let age = Utc::now() - t.center;
        assert!(age.num_days() <= 7);
    }

    // -----------------------------------------------------------------------
    // LLM extraction JSON parser tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_llm_basic_json() {
        let json = r#"{"entities":[{"name":"PostgreSQL","kind":"tool"},{"name":"Alice","kind":"person"}],"relations":[{"subject":"Alice","relation":"uses","object":"PostgreSQL"}]}"#;
        let (ents, rels) = parse_llm_extraction(json).unwrap();
        assert_eq!(ents.len(), 2);
        assert_eq!(ents[0].name, "PostgreSQL");
        assert_eq!(ents[0].kind, EntityKind::Tool);
        assert_eq!(ents[1].name, "Alice");
        assert_eq!(ents[1].kind, EntityKind::Person);
        assert_eq!(rels.len(), 1);
        assert_eq!(
            rels[0],
            ("Alice".into(), "uses".into(), "PostgreSQL".into())
        );
    }

    #[test]
    fn parse_llm_with_surrounding_text() {
        let text = r#"Here are the extracted entities:
{"entities":[{"name":"Redis","kind":"tool"}],"relations":[]}
That's all."#;
        let (ents, _) = parse_llm_extraction(text).unwrap();
        assert_eq!(ents.len(), 1);
        assert_eq!(ents[0].name, "Redis");
    }

    #[test]
    fn parse_llm_empty_results() {
        let json = r#"{"entities":[],"relations":[]}"#;
        let (ents, rels) = parse_llm_extraction(json).unwrap();
        assert!(ents.is_empty());
        assert!(rels.is_empty());
    }

    #[test]
    fn parse_llm_filters_empty_names() {
        let json = r#"{"entities":[{"name":"","kind":"tool"},{"name":"Valid","kind":"other"}],"relations":[]}"#;
        let (ents, _) = parse_llm_extraction(json).unwrap();
        assert_eq!(ents.len(), 1);
        assert_eq!(ents[0].name, "Valid");
    }
}
