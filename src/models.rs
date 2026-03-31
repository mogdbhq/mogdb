/// JSON request / response types for the MogDB HTTP API.
use chrono::{DateTime, Utc};
use mogdb_core::{MemoryKind, SourceTrust};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// POST /memories
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct IngestBody {
    pub agent_id: String,
    pub user_id: String,
    pub content: String,
    /// "episodic" | "semantic" | "procedural" | "working"  (default: episodic)
    pub kind: Option<String>,
    /// "agent" | "user" | "system" | "external"  (default: agent)
    pub source_trust: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct IngestResponse {
    pub id: Uuid,
    pub importance: f64,
    pub conflicts_invalidated: u64,
    pub entities_touched: Vec<String>,
    pub quarantined: bool,
}

// ---------------------------------------------------------------------------
// GET /memories/search
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SearchParams {
    pub agent_id: String,
    pub user_id: String,
    pub q: String,
    /// Filter by kind.
    pub kind: Option<String>,
    /// Maximum results (default 10).
    pub limit: Option<i32>,
    /// Point-in-time: ISO 8601 datetime.
    pub as_of: Option<DateTime<Utc>>,
    /// Minimum decay strength 0.0–1.0.
    pub min_strength: Option<f64>,
    /// Include entity graph context in results.
    pub include_graph: Option<bool>,
}

// ---------------------------------------------------------------------------
// GET /api/graph
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    /// "memory" | "entity"
    pub node_type: String,
    /// e.g. "episodic" / "semantic" / "person" / "tool"
    pub kind: String,
    /// 0.0–1.0 decay strength (memories); 1.0 for entities
    pub strength: f64,
    /// 0.0–1.0 importance (memories); 0.5 for entities
    pub importance: f64,
    /// Full content — only populated for memories
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
}

#[derive(Debug, Serialize)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

// ---------------------------------------------------------------------------
// Error response
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: String,
}

// ---------------------------------------------------------------------------
// Helpers — parse kind / source_trust from strings
// ---------------------------------------------------------------------------

pub fn parse_kind(s: &str) -> Result<MemoryKind, String> {
    match s.to_lowercase().as_str() {
        "episodic" => Ok(MemoryKind::Episodic),
        "semantic" => Ok(MemoryKind::Semantic),
        "procedural" => Ok(MemoryKind::Procedural),
        "working" => Ok(MemoryKind::Working),
        other => Err(format!(
            "unknown kind '{other}' — expected episodic|semantic|procedural|working"
        )),
    }
}

pub fn parse_source_trust(s: &str) -> Result<SourceTrust, String> {
    match s.to_lowercase().as_str() {
        "agent" => Ok(SourceTrust::Agent),
        "user" => Ok(SourceTrust::User),
        "system" => Ok(SourceTrust::System),
        "external" => Ok(SourceTrust::External),
        other => Err(format!(
            "unknown source_trust '{other}' — expected agent|user|system|external"
        )),
    }
}
