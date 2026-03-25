use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The type of memory — controls decay rate and retrieval priority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "memory_kind", rename_all = "lowercase")]
pub enum MemoryKind {
    /// A specific past event or interaction. Medium decay.
    Episodic,
    /// A distilled, time-independent fact or preference. Slow decay.
    Semantic,
    /// A task-relevant rule or workflow pattern. Very slow decay.
    Procedural,
    /// A temporary note for the current session. Fast decay.
    Working,
}

impl std::fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryKind::Episodic => write!(f, "episodic"),
            MemoryKind::Semantic => write!(f, "semantic"),
            MemoryKind::Procedural => write!(f, "procedural"),
            MemoryKind::Working => write!(f, "working"),
        }
    }
}

impl std::fmt::Display for SourceTrust {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceTrust::Agent => write!(f, "agent"),
            SourceTrust::User => write!(f, "user"),
            SourceTrust::System => write!(f, "system"),
            SourceTrust::External => write!(f, "external"),
        }
    }
}

impl std::fmt::Display for AuditAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditAction::Read => write!(f, "read"),
            AuditAction::Write => write!(f, "write"),
            AuditAction::Forget => write!(f, "forget"),
            AuditAction::Invalidate => write!(f, "invalidate"),
            AuditAction::Quarantine => write!(f, "quarantine"),
        }
    }
}

impl std::fmt::Display for EntityKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntityKind::Person => write!(f, "person"),
            EntityKind::System => write!(f, "system"),
            EntityKind::Concept => write!(f, "concept"),
            EntityKind::Project => write!(f, "project"),
            EntityKind::Tool => write!(f, "tool"),
            EntityKind::Other => write!(f, "other"),
        }
    }
}

/// Trust level of the source that produced this memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "source_trust", rename_all = "lowercase")]
pub enum SourceTrust {
    /// Directly from the agent or user. Always trusted.
    Agent,
    /// From the user directly.
    User,
    /// System-generated (background workers, consolidation).
    System,
    /// From an external source (web, email, APIs). Goes to quarantine.
    External,
}

/// A single memory record — the core unit stored in MogDB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: Uuid,

    // Identity
    pub agent_id: String,
    pub user_id: String,
    pub session_id: Option<String>,

    // Content
    pub content: String,
    pub kind: MemoryKind,
    pub source_trust: SourceTrust,

    // Bi-temporal timestamps
    /// When MogDB stored this record.
    pub t_created: DateTime<Utc>,
    /// When MogDB soft-deleted this record (None = still active in DB).
    pub t_expired: Option<DateTime<Utc>>,
    /// When this fact became true in the real world.
    pub t_valid: DateTime<Utc>,
    /// When this fact stopped being true in the real world (None = still true).
    pub t_invalid: Option<DateTime<Utc>>,

    // Salience / decay
    /// LLM or rule-based importance score. 0.0 to 1.0.
    pub importance: f64,
    /// Decays over time based on importance + recency. 0.0 to 1.0.
    pub strength: f64,
    pub access_count: i32,
    pub last_accessed: Option<DateTime<Utc>>,

    // Graph links
    /// Entity IDs referenced in this memory.
    pub entity_refs: Vec<String>,
    /// Source record ID if this was derived by consolidation.
    pub provenance: Option<Uuid>,

    // Quarantine flag (for external sources pending review)
    pub quarantined: bool,
}

/// Used to create a new memory — only requires the fields the caller provides.
/// Defaults are applied in the storage layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMemoryRecord {
    pub agent_id: String,
    pub user_id: String,
    pub session_id: Option<String>,
    pub content: String,
    pub kind: MemoryKind,
    pub source_trust: SourceTrust,
    /// When this fact became true. Defaults to now() if not provided.
    pub t_valid: Option<DateTime<Utc>>,
    /// Caller-provided importance hint. 0.0–1.0. Defaults to 0.5.
    pub importance: Option<f64>,
    pub entity_refs: Vec<String>,
    pub provenance: Option<Uuid>,
}

impl NewMemoryRecord {
    pub fn new(agent_id: impl Into<String>, user_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            user_id: user_id.into(),
            session_id: None,
            content: content.into(),
            kind: MemoryKind::Episodic,
            source_trust: SourceTrust::Agent,
            t_valid: None,
            importance: None,
            entity_refs: vec![],
            provenance: None,
        }
    }

    pub fn kind(mut self, kind: MemoryKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn importance(mut self, score: f64) -> Self {
        self.importance = Some(score);
        self
    }

    pub fn session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
}

/// A named entity referenced across memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: Uuid,
    pub agent_id: String,
    pub user_id: String,
    pub name: String,
    pub kind: EntityKind,
    pub attributes: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "entity_kind", rename_all = "lowercase")]
pub enum EntityKind {
    Person,
    System,
    Concept,
    Project,
    Tool,
    Other,
}

/// A directed, typed, bi-temporal relationship between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityEdge {
    pub id: Uuid,
    pub from_id: Uuid,
    pub to_id: Uuid,
    /// e.g. "uses", "owns", "is_part_of", "worked_on"
    pub relation: String,
    pub weight: f64,
    pub t_valid: DateTime<Utc>,
    pub t_invalid: Option<DateTime<Utc>>,
    pub source_memory: Option<Uuid>,
}

/// Every read and write is recorded here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLog {
    pub id: Uuid,
    pub ts: DateTime<Utc>,
    pub action: AuditAction,
    pub actor: String,
    pub memory_id: Option<Uuid>,
    pub query_text: Option<String>,
    pub result_count: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "audit_action", rename_all = "lowercase")]
pub enum AuditAction {
    Read,
    Write,
    Forget,
    Invalidate,
    Quarantine,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_memory_record_defaults() {
        let m = NewMemoryRecord::new("agent-1", "user-1", "user prefers dark mode");
        assert_eq!(m.agent_id, "agent-1");
        assert_eq!(m.user_id, "user-1");
        assert_eq!(m.content, "user prefers dark mode");
        assert_eq!(m.kind, MemoryKind::Episodic);
        assert_eq!(m.source_trust, SourceTrust::Agent);
        assert!(m.importance.is_none());
        assert!(m.session_id.is_none());
    }

    #[test]
    fn new_memory_record_builder() {
        let m = NewMemoryRecord::new("agent-1", "user-1", "never call prod directly")
            .kind(MemoryKind::Procedural)
            .importance(1.0)
            .session("sess-abc");
        assert_eq!(m.kind, MemoryKind::Procedural);
        assert_eq!(m.importance, Some(1.0));
        assert_eq!(m.session_id, Some("sess-abc".to_string()));
    }

    #[test]
    fn memory_kind_display() {
        assert_eq!(MemoryKind::Episodic.to_string(), "episodic");
        assert_eq!(MemoryKind::Procedural.to_string(), "procedural");
    }
}
