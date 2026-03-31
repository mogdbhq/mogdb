use thiserror::Error;

#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),

    #[error("storage error: {0}")]
    Storage(#[from] mogdb_core::MogError),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("serialization error: {0}")]
    Json(#[from] serde_json::Error),
}
