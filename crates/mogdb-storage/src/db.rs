use sqlx::postgres::{PgPool, PgPoolOptions};
use std::time::Duration;
use tracing::info;

/// Central database handle. Clone freely — it's backed by a connection pool.
#[derive(Clone, Debug)]
pub struct Database {
    pub pool: PgPool,
}

impl Database {
    /// Connect to Postgres and run all pending migrations.
    pub async fn connect(database_url: &str) -> Result<Self, sqlx::Error> {
        info!("connecting to database");

        let pool = PgPoolOptions::new()
            .max_connections(20)
            .min_connections(2)
            .acquire_timeout(Duration::from_secs(5))
            .connect(database_url)
            .await?;

        info!("running migrations");
        sqlx::migrate!("../../migrations").run(&pool).await?;

        info!("database ready");
        Ok(Self { pool })
    }

    /// Ping the database — useful for health checks.
    pub async fn ping(&self) -> Result<(), sqlx::Error> {
        sqlx::query("SELECT 1").execute(&self.pool).await?;
        Ok(())
    }
}
