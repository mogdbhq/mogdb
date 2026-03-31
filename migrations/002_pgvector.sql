-- MogDB Phase 2: pgvector HNSW index for semantic/vector search.
-- Requires the pgvector extension to be installed in Postgres.
-- Install: https://github.com/pgvector/pgvector

CREATE EXTENSION IF NOT EXISTS vector;

-- 1024-dimensional embedding column.
-- Matches mxbai-embed-large (default) and bge-m3 via Ollama.
-- For nomic-embed-text use 768; for OpenAI text-embedding-3-small use 1536.
-- NULL means the memory was stored without an embedding provider configured.
ALTER TABLE memory_records ADD COLUMN embedding vector(1024);

-- HNSW index for approximate nearest neighbor search using cosine distance.
-- m=16 (connections per layer) and ef_construction=64 are the standard defaults.
-- Only indexes rows that have an embedding (WHERE clause is implicit — sparse index).
CREATE INDEX idx_memory_embedding_hnsw ON memory_records
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
