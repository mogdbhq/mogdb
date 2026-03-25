use thiserror::Error;

#[derive(Debug, Error)]
pub enum MogError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("record not found: {0}")]
    NotFound(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("conflict detected: {0}")]
    Conflict(String),
}
