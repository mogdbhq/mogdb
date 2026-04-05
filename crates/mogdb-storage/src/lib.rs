pub mod audit;
pub mod conflict;
pub mod db;
pub mod decay;
pub mod embedding;
pub mod entity;
pub mod extraction;
pub mod forget;
pub mod memory;
pub mod pipeline;
pub mod reranking;
pub mod scoring;
pub mod search;

#[cfg(test)]
mod conflict_test;
#[cfg(test)]
mod extraction_test;
#[cfg(test)]
mod scoring_test;

pub use db::Database;
pub use embedding::{EmbeddingProvider, OllamaEmbeddings};
pub use pipeline::{ingest, ingest_with_embedder, IngestResult};
pub use search::{search_hybrid, SearchQuery, SearchResult};
