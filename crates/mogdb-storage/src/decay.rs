/// Decay engine — applies exponential strength decay to all active memories.
///
/// Decay rates (fraction of strength lost per day):
///   working    — 0.50  (fast, session-scoped notes)
///   episodic   — 0.10  (event memories fade over weeks)
///   semantic   — 0.02  (facts are durable, decay over months)
///   procedural — 0.01  (procedures are very durable)
///
/// Formula: strength = MAX(0, importance × exp(−rate × days_since_access))
use mogdb_core::MogError;
use sqlx::PgPool;
use tracing::debug;

/// Run one decay pass across all active, non-quarantined memories.
/// Returns the number of records updated.
pub async fn run_decay_pass(pool: &PgPool) -> Result<u64, MogError> {
    let result = sqlx::query(
        r#"
        UPDATE memory_records
        SET strength = GREATEST(0.0,
            importance * EXP(
                -CASE kind
                    WHEN 'working'    THEN 0.50
                    WHEN 'episodic'   THEN 0.10
                    WHEN 'semantic'   THEN 0.02
                    WHEN 'procedural' THEN 0.01
                    ELSE 0.10
                END
                * EXTRACT(EPOCH FROM (NOW() - COALESCE(last_accessed, t_created))) / 86400.0
            )
        )
        WHERE t_expired IS NULL
          AND quarantined = FALSE
        "#,
    )
    .execute(pool)
    .await?;

    let n = result.rows_affected();
    debug!(updated = n, "decay pass complete");
    Ok(n)
}

/// Count of active memories per agent — used to update the Prometheus gauge.
pub async fn count_active(pool: &PgPool, agent_id: &str) -> Result<i64, MogError> {
    let row: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM memory_records WHERE agent_id = $1 AND t_expired IS NULL AND quarantined = FALSE",
    )
    .bind(agent_id)
    .fetch_one(pool)
    .await?;
    Ok(row.0)
}
