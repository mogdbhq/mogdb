/// Forget policy CRUD and policy application.
///
/// Policies are stored in `forget_policies` and applied by the nightly ForgetWorker.
use mogdb_core::{ForgetPolicy, MogError, PolicyAction, PolicyCondition};
use sqlx::PgPool;
use tracing::debug;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

/// Insert a new forget policy, returning the created record.
pub async fn create_policy(
    pool: &PgPool,
    agent_id: &str,
    name: &str,
    description: Option<&str>,
    condition: &PolicyCondition,
    action: &PolicyAction,
) -> Result<ForgetPolicy, MogError> {
    let condition_json = serde_json::to_value(condition)
        .map_err(|e| MogError::External(format!("failed to serialize condition: {e}")))?;
    let action_str = action.to_string();

    let row = sqlx::query_as::<_, PolicyRow>(
        r#"
        INSERT INTO forget_policies (agent_id, name, description, condition, action)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id, agent_id, name, description, enabled, condition, action, created_at, updated_at
        "#,
    )
    .bind(agent_id)
    .bind(name)
    .bind(description)
    .bind(&condition_json)
    .bind(&action_str)
    .fetch_one(pool)
    .await?;

    row.into_domain()
}

/// List all enabled forget policies for an agent.
pub async fn list_enabled_policies(
    pool: &PgPool,
    agent_id: &str,
) -> Result<Vec<ForgetPolicy>, MogError> {
    let rows = sqlx::query_as::<_, PolicyRow>(
        r#"
        SELECT id, agent_id, name, description, enabled, condition, action, created_at, updated_at
        FROM forget_policies
        WHERE agent_id = $1 AND enabled = TRUE
        ORDER BY created_at
        "#,
    )
    .bind(agent_id)
    .fetch_all(pool)
    .await?;

    rows.into_iter().map(|r| r.into_domain()).collect()
}

/// Delete a policy by ID (returns true if it existed).
pub async fn delete_policy(pool: &PgPool, policy_id: Uuid) -> Result<bool, MogError> {
    let result = sqlx::query("DELETE FROM forget_policies WHERE id = $1")
        .bind(policy_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

// ---------------------------------------------------------------------------
// Policy application
// ---------------------------------------------------------------------------

/// Apply a single forget policy against the memory_records table.
/// Returns the number of memories acted upon.
pub async fn apply_policy(pool: &PgPool, policy: &ForgetPolicy) -> Result<u64, MogError> {
    let c = &policy.condition;

    // Build parameterised query fragments. We use COALESCE / IS NULL tricks so
    // that NULL parameters simply skip the corresponding filter.
    let kind_str: Option<String> = c.kind.as_ref().map(|k| k.to_string());
    let strength_below: Option<f64> = c.strength_below;
    let older_than_days: Option<f64> = c.older_than_days.map(|d| d as f64);
    let access_count_below: Option<i32> = c.access_count_below;

    let rows_affected = match &policy.action {
        PolicyAction::Expire => sqlx::query(
            r#"
                UPDATE memory_records
                SET t_expired = NOW()
                WHERE agent_id = $1
                  AND t_expired IS NULL
                  AND quarantined = FALSE
                  AND ($2::TEXT IS NULL OR kind::TEXT = $2)
                  AND ($3::FLOAT8 IS NULL OR strength < $3)
                  AND ($4::FLOAT8 IS NULL OR EXTRACT(EPOCH FROM (NOW() - t_created)) / 86400.0 > $4)
                  AND ($5::INT IS NULL OR access_count < $5)
                "#,
        )
        .bind(&policy.agent_id)
        .bind(&kind_str)
        .bind(strength_below)
        .bind(older_than_days)
        .bind(access_count_below)
        .execute(pool)
        .await?
        .rows_affected(),
        PolicyAction::Invalidate => sqlx::query(
            r#"
                UPDATE memory_records
                SET t_invalid = NOW()
                WHERE agent_id = $1
                  AND t_invalid IS NULL
                  AND t_expired IS NULL
                  AND quarantined = FALSE
                  AND ($2::TEXT IS NULL OR kind::TEXT = $2)
                  AND ($3::FLOAT8 IS NULL OR strength < $3)
                  AND ($4::FLOAT8 IS NULL OR EXTRACT(EPOCH FROM (NOW() - t_created)) / 86400.0 > $4)
                  AND ($5::INT IS NULL OR access_count < $5)
                "#,
        )
        .bind(&policy.agent_id)
        .bind(&kind_str)
        .bind(strength_below)
        .bind(older_than_days)
        .bind(access_count_below)
        .execute(pool)
        .await?
        .rows_affected(),
        PolicyAction::Quarantine => sqlx::query(
            r#"
                UPDATE memory_records
                SET quarantined = TRUE
                WHERE agent_id = $1
                  AND quarantined = FALSE
                  AND t_expired IS NULL
                  AND ($2::TEXT IS NULL OR kind::TEXT = $2)
                  AND ($3::FLOAT8 IS NULL OR strength < $3)
                  AND ($4::FLOAT8 IS NULL OR EXTRACT(EPOCH FROM (NOW() - t_created)) / 86400.0 > $4)
                  AND ($5::INT IS NULL OR access_count < $5)
                "#,
        )
        .bind(&policy.agent_id)
        .bind(&kind_str)
        .bind(strength_below)
        .bind(older_than_days)
        .bind(access_count_below)
        .execute(pool)
        .await?
        .rows_affected(),
    };

    debug!(
        policy = policy.name,
        action = %policy.action,
        affected = rows_affected,
        "policy applied"
    );

    Ok(rows_affected)
}

// ---------------------------------------------------------------------------
// Injection defence helper
// ---------------------------------------------------------------------------

/// Quarantine memories that show signs of prompt-injection payloads.
/// Heuristics checked:
///   • Very long content (> 4000 chars) from external sources
///   • Contains instruction-injection markers
/// Returns number of memories quarantined.
pub async fn quarantine_suspicious(pool: &PgPool) -> Result<u64, MogError> {
    let result = sqlx::query(
        r#"
        UPDATE memory_records
        SET quarantined = TRUE
        WHERE quarantined = FALSE
          AND t_expired IS NULL
          AND (
              -- Suspiciously long external content
              (source_trust = 'external' AND LENGTH(content) > 4000)
              OR
              -- Classic injection markers (case-insensitive)
              content ILIKE '%ignore previous instructions%'
              OR content ILIKE '%disregard all prior%'
              OR content ILIKE '%system prompt%'
              OR content ILIKE '%you are now%'
              OR content ILIKE '%act as%DAN%'
              OR content ILIKE '%<|im_start|>%'
              OR content ILIKE '%[INST]%'
          )
        "#,
    )
    .execute(pool)
    .await?;

    Ok(result.rows_affected())
}

// ---------------------------------------------------------------------------
// Internal row type for sqlx
// ---------------------------------------------------------------------------

#[derive(sqlx::FromRow)]
struct PolicyRow {
    id: Uuid,
    agent_id: String,
    name: String,
    description: Option<String>,
    enabled: bool,
    condition: serde_json::Value,
    action: String,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
}

impl PolicyRow {
    fn into_domain(self) -> Result<ForgetPolicy, MogError> {
        let condition: PolicyCondition = serde_json::from_value(self.condition)
            .map_err(|e| MogError::External(format!("invalid policy condition JSON: {e}")))?;

        let action = match self.action.as_str() {
            "expire" => PolicyAction::Expire,
            "invalidate" => PolicyAction::Invalidate,
            "quarantine" => PolicyAction::Quarantine,
            other => {
                return Err(MogError::External(format!(
                    "unknown policy action '{other}'"
                )))
            }
        };

        Ok(ForgetPolicy {
            id: self.id,
            agent_id: self.agent_id,
            name: self.name,
            description: self.description,
            enabled: self.enabled,
            condition,
            action,
            created_at: self.created_at,
            updated_at: self.updated_at,
        })
    }
}
