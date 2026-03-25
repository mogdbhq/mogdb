/// Entity graph storage — create/lookup entities and relationship edges.
use chrono::{DateTime, Utc};
use mogdb_core::{Entity, EntityEdge, EntityKind, MogError};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

#[derive(Debug, FromRow)]
struct EntityRow {
    id: Uuid,
    agent_id: String,
    user_id: String,
    name: String,
    kind: String,
    attributes: serde_json::Value,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

impl TryFrom<EntityRow> for Entity {
    type Error = MogError;

    fn try_from(row: EntityRow) -> Result<Self, Self::Error> {
        Ok(Entity {
            id: row.id,
            agent_id: row.agent_id,
            user_id: row.user_id,
            name: row.name,
            kind: parse_entity_kind(&row.kind)?,
            attributes: row.attributes,
            created_at: row.created_at,
            updated_at: row.updated_at,
        })
    }
}

fn parse_entity_kind(s: &str) -> Result<EntityKind, MogError> {
    match s {
        "person" => Ok(EntityKind::Person),
        "system" => Ok(EntityKind::System),
        "concept" => Ok(EntityKind::Concept),
        "project" => Ok(EntityKind::Project),
        "tool" => Ok(EntityKind::Tool),
        "other" => Ok(EntityKind::Other),
        o => Err(MogError::InvalidInput(format!("unknown entity kind: {o}"))),
    }
}

/// Find or create an entity by name (case-insensitive, scoped to agent+user).
/// Returns the entity's UUID.
pub async fn upsert(
    pool: &PgPool,
    agent_id: &str,
    user_id: &str,
    name: &str,
    kind: &EntityKind,
) -> Result<Uuid, MogError> {
    // Try to find existing
    let existing = sqlx::query_as::<_, (Uuid,)>(
        "SELECT id FROM entities WHERE agent_id = $1 AND user_id = $2 AND LOWER(name) = LOWER($3)",
    )
    .bind(agent_id)
    .bind(user_id)
    .bind(name)
    .fetch_optional(pool)
    .await?;

    if let Some((id,)) = existing {
        // Update the updated_at timestamp
        sqlx::query("UPDATE entities SET updated_at = NOW() WHERE id = $1")
            .bind(id)
            .execute(pool)
            .await?;
        return Ok(id);
    }

    // Create new
    let id = Uuid::new_v4();
    sqlx::query(
        r#"
        INSERT INTO entities (id, agent_id, user_id, name, kind, attributes, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5::entity_kind, '{}'::jsonb, NOW(), NOW())
        "#,
    )
    .bind(id)
    .bind(agent_id)
    .bind(user_id)
    .bind(name)
    .bind(kind.to_string())
    .execute(pool)
    .await?;

    Ok(id)
}

/// Create a directed relationship edge between two entities.
pub async fn create_edge(
    pool: &PgPool,
    from_id: Uuid,
    to_id: Uuid,
    relation: &str,
    source_memory: Option<Uuid>,
) -> Result<Uuid, MogError> {
    let id = Uuid::new_v4();
    sqlx::query(
        r#"
        INSERT INTO entity_edges (id, from_id, to_id, relation, weight, t_valid, t_invalid, source_memory)
        VALUES ($1, $2, $3, $4, 1.0, NOW(), NULL, $5)
        "#,
    )
    .bind(id)
    .bind(from_id)
    .bind(to_id)
    .bind(relation)
    .bind(source_memory)
    .execute(pool)
    .await?;

    Ok(id)
}

/// Invalidate all edges of a specific relation between two entities.
pub async fn invalidate_edges(
    pool: &PgPool,
    from_id: Uuid,
    to_id: Uuid,
    relation: &str,
) -> Result<u64, MogError> {
    let affected = sqlx::query(
        "UPDATE entity_edges SET t_invalid = NOW() WHERE from_id = $1 AND to_id = $2 AND relation = $3 AND t_invalid IS NULL",
    )
    .bind(from_id)
    .bind(to_id)
    .bind(relation)
    .execute(pool)
    .await?
    .rows_affected();

    Ok(affected)
}

/// Fetch an entity by ID.
pub async fn fetch_by_id(pool: &PgPool, id: Uuid) -> Result<Entity, MogError> {
    let row = sqlx::query_as::<_, EntityRow>(
        "SELECT id, agent_id, user_id, name, kind::TEXT AS kind, attributes, created_at, updated_at FROM entities WHERE id = $1",
    )
    .bind(id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| MogError::NotFound(id.to_string()))?;

    Entity::try_from(row)
}

/// Find entity by name (scoped to agent+user).
pub async fn find_by_name(
    pool: &PgPool,
    agent_id: &str,
    user_id: &str,
    name: &str,
) -> Result<Option<Entity>, MogError> {
    let row = sqlx::query_as::<_, EntityRow>(
        "SELECT id, agent_id, user_id, name, kind::TEXT AS kind, attributes, created_at, updated_at FROM entities WHERE agent_id = $1 AND user_id = $2 AND LOWER(name) = LOWER($3)",
    )
    .bind(agent_id)
    .bind(user_id)
    .bind(name)
    .fetch_optional(pool)
    .await?;

    match row {
        Some(r) => Ok(Some(Entity::try_from(r)?)),
        None => Ok(None),
    }
}

/// Get all edges from an entity (active only).
pub async fn get_edges_from(pool: &PgPool, entity_id: Uuid) -> Result<Vec<EntityEdge>, MogError> {
    let rows = sqlx::query_as::<_, (Uuid, Uuid, Uuid, String, f64, DateTime<Utc>, Option<DateTime<Utc>>, Option<Uuid>)>(
        "SELECT id, from_id, to_id, relation, weight, t_valid, t_invalid, source_memory FROM entity_edges WHERE from_id = $1 AND t_invalid IS NULL",
    )
    .bind(entity_id)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(
            |(id, from_id, to_id, relation, weight, t_valid, t_invalid, source_memory)| {
                EntityEdge {
                    id,
                    from_id,
                    to_id,
                    relation,
                    weight,
                    t_valid,
                    t_invalid,
                    source_memory,
                }
            },
        )
        .collect())
}
