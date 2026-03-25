use chrono::{DateTime, Utc};
use mogdb_core::{AuditAction, MogError, MemoryKind, MemoryRecord, NewMemoryRecord, SourceTrust};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use crate::audit;

/// Raw DB row — enum columns stored as TEXT so sqlx maps without a live DB at compile time.
#[derive(Debug, FromRow)]
struct MemoryRow {
    id: Uuid,
    agent_id: String,
    user_id: String,
    session_id: Option<String>,
    content: String,
    kind: String,
    source_trust: String,
    t_created: DateTime<Utc>,
    t_expired: Option<DateTime<Utc>>,
    t_valid: DateTime<Utc>,
    t_invalid: Option<DateTime<Utc>>,
    importance: f64,
    strength: f64,
    access_count: i32,
    last_accessed: Option<DateTime<Utc>>,
    entity_refs: Vec<String>,
    provenance: Option<Uuid>,
    quarantined: bool,
}

impl TryFrom<MemoryRow> for MemoryRecord {
    type Error = MogError;

    fn try_from(row: MemoryRow) -> Result<Self, Self::Error> {
        Ok(MemoryRecord {
            id: row.id,
            agent_id: row.agent_id,
            user_id: row.user_id,
            session_id: row.session_id,
            content: row.content,
            kind: parse_kind(&row.kind)?,
            source_trust: parse_source_trust(&row.source_trust)?,
            t_created: row.t_created,
            t_expired: row.t_expired,
            t_valid: row.t_valid,
            t_invalid: row.t_invalid,
            importance: row.importance,
            strength: row.strength,
            access_count: row.access_count,
            last_accessed: row.last_accessed,
            entity_refs: row.entity_refs,
            provenance: row.provenance,
            quarantined: row.quarantined,
        })
    }
}

fn parse_kind(s: &str) -> Result<MemoryKind, MogError> {
    match s {
        "episodic"   => Ok(MemoryKind::Episodic),
        "semantic"   => Ok(MemoryKind::Semantic),
        "procedural" => Ok(MemoryKind::Procedural),
        "working"    => Ok(MemoryKind::Working),
        other        => Err(MogError::InvalidInput(format!("unknown memory kind: {other}"))),
    }
}

fn parse_source_trust(s: &str) -> Result<SourceTrust, MogError> {
    match s {
        "agent"    => Ok(SourceTrust::Agent),
        "user"     => Ok(SourceTrust::User),
        "system"   => Ok(SourceTrust::System),
        "external" => Ok(SourceTrust::External),
        other      => Err(MogError::InvalidInput(format!("unknown source trust: {other}"))),
    }
}

const SELECT_FIELDS: &str = r#"
    id, agent_id, user_id, session_id,
    content, kind::TEXT AS kind, source_trust::TEXT AS source_trust,
    t_created, t_expired, t_valid, t_invalid,
    importance, strength, access_count, last_accessed,
    entity_refs, provenance, quarantined
"#;

/// Store a new memory record. Returns the fully populated MemoryRecord.
pub async fn store(pool: &PgPool, input: NewMemoryRecord) -> Result<MemoryRecord, MogError> {
    let id = Uuid::new_v4();
    let now = Utc::now();
    let t_valid = input.t_valid.unwrap_or(now);
    let importance = input.importance.unwrap_or(0.5).clamp(0.0, 1.0);
    let quarantined = input.source_trust == SourceTrust::External;

    sqlx::query(
        r#"
        INSERT INTO memory_records (
            id, agent_id, user_id, session_id,
            content, kind, source_trust,
            t_created, t_expired, t_valid, t_invalid,
            importance, strength, access_count, last_accessed,
            entity_refs, provenance, quarantined
        ) VALUES (
            $1, $2, $3, $4,
            $5, $6::memory_kind, $7::source_trust,
            $8, NULL, $9, NULL,
            $10, $10, 0, NULL,
            $11, $12, $13
        )
        "#,
    )
    .bind(id)
    .bind(&input.agent_id)
    .bind(&input.user_id)
    .bind(&input.session_id)
    .bind(&input.content)
    .bind(input.kind.to_string())
    .bind(input.source_trust.to_string())
    .bind(now)
    .bind(t_valid)
    .bind(importance)
    .bind(&input.entity_refs)
    .bind(input.provenance)
    .bind(quarantined)
    .execute(pool)
    .await?;

    audit::log(pool, AuditAction::Write, &input.agent_id, Some(id), None, None).await?;

    fetch_by_id(pool, id, &input.agent_id).await
}

/// Fetch a single memory record by ID, scoped to agent.
pub async fn fetch_by_id(pool: &PgPool, id: Uuid, agent_id: &str) -> Result<MemoryRecord, MogError> {
    let sql = format!("SELECT {} FROM memory_records WHERE id = $1 AND agent_id = $2", SELECT_FIELDS);

    let row = sqlx::query_as::<_, MemoryRow>(&sql)
        .bind(id)
        .bind(agent_id)
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| MogError::NotFound(id.to_string()))?;

    MemoryRecord::try_from(row)
}

/// Soft-delete a memory by setting t_expired = now().
pub async fn expire(pool: &PgPool, id: Uuid, agent_id: &str) -> Result<(), MogError> {
    let rows_affected = sqlx::query(
        "UPDATE memory_records SET t_expired = NOW() WHERE id = $1 AND agent_id = $2 AND t_expired IS NULL",
    )
    .bind(id)
    .bind(agent_id)
    .execute(pool)
    .await?
    .rows_affected();

    if rows_affected == 0 {
        return Err(MogError::NotFound(id.to_string()));
    }

    audit::log(pool, AuditAction::Forget, agent_id, Some(id), None, None).await?;
    Ok(())
}

/// Invalidate a memory in the world — sets t_invalid = now().
pub async fn invalidate(pool: &PgPool, id: Uuid, agent_id: &str) -> Result<(), MogError> {
    let rows_affected = sqlx::query(
        "UPDATE memory_records SET t_invalid = NOW() WHERE id = $1 AND agent_id = $2 AND t_invalid IS NULL",
    )
    .bind(id)
    .bind(agent_id)
    .execute(pool)
    .await?
    .rows_affected();

    if rows_affected == 0 {
        return Err(MogError::NotFound(id.to_string()));
    }

    audit::log(pool, AuditAction::Invalidate, agent_id, Some(id), None, None).await?;
    Ok(())
}

/// List active (non-expired, non-quarantined) memories for a user, newest first.
pub async fn list_active(
    pool: &PgPool,
    agent_id: &str,
    user_id: &str,
    limit: i64,
) -> Result<Vec<MemoryRecord>, MogError> {
    let sql = format!(
        "SELECT {} FROM memory_records WHERE agent_id = $1 AND user_id = $2 AND t_expired IS NULL AND quarantined = false ORDER BY t_created DESC LIMIT $3",
        SELECT_FIELDS
    );

    let rows = sqlx::query_as::<_, MemoryRow>(&sql)
        .bind(agent_id)
        .bind(user_id)
        .bind(limit)
        .fetch_all(pool)
        .await?;

    rows.into_iter().map(MemoryRecord::try_from).collect()
}
