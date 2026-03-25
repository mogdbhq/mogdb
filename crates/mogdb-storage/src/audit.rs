use mogdb_core::{AuditAction, MogError};
use sqlx::PgPool;
use uuid::Uuid;

/// Append an immutable audit log entry.
pub async fn log(
    pool: &PgPool,
    action: AuditAction,
    actor: &str,
    memory_id: Option<Uuid>,
    query_text: Option<&str>,
    result_count: Option<i32>,
) -> Result<(), MogError> {
    sqlx::query(
        r#"
        INSERT INTO audit_log (id, ts, action, actor, memory_id, query_text, result_count)
        VALUES (gen_random_uuid(), NOW(), $1::audit_action, $2, $3, $4, $5)
        "#,
    )
    .bind(action.to_string())
    .bind(actor)
    .bind(memory_id)
    .bind(query_text)
    .bind(result_count)
    .execute(pool)
    .await?;

    Ok(())
}
