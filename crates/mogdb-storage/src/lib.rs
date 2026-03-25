pub mod audit;
pub mod conflict;
pub mod db;
pub mod entity;
pub mod extraction;
pub mod memory;
pub mod pipeline;
pub mod scoring;

#[cfg(test)]
mod scoring_test;
#[cfg(test)]
mod extraction_test;
#[cfg(test)]
mod conflict_test;

pub use db::Database;
pub use pipeline::{ingest, IngestResult};
