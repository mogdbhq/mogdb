-- MogDB initial schema
-- Phase 0: core tables, types, indexes

-- ---------------------------------------------------------------------------
-- Custom ENUM types
-- ---------------------------------------------------------------------------

CREATE TYPE memory_kind AS ENUM ('episodic', 'semantic', 'procedural', 'working');
CREATE TYPE source_trust AS ENUM ('agent', 'user', 'system', 'external');
CREATE TYPE audit_action AS ENUM ('read', 'write', 'forget', 'invalidate', 'quarantine');
CREATE TYPE entity_kind  AS ENUM ('person', 'system', 'concept', 'project', 'tool', 'other');

-- ---------------------------------------------------------------------------
-- memory_records — the core table
-- ---------------------------------------------------------------------------

CREATE TABLE memory_records (
    -- Identity
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id    TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    session_id  TEXT,

    -- Content
    content     TEXT NOT NULL,
    kind        memory_kind NOT NULL DEFAULT 'episodic',
    source_trust source_trust NOT NULL DEFAULT 'agent',

    -- Bi-temporal timestamps
    t_created   TIMESTAMPTZ NOT NULL DEFAULT NOW(),   -- when stored in MogDB
    t_expired   TIMESTAMPTZ,                          -- when soft-deleted in MogDB (NULL = active)
    t_valid     TIMESTAMPTZ NOT NULL DEFAULT NOW(),   -- when true in the real world
    t_invalid   TIMESTAMPTZ,                          -- when stopped being true (NULL = still true)

    -- Salience / decay
    importance   FLOAT NOT NULL DEFAULT 0.5 CHECK (importance >= 0.0 AND importance <= 1.0),
    strength     FLOAT NOT NULL DEFAULT 1.0 CHECK (strength   >= 0.0 AND strength   <= 1.0),
    access_count INT   NOT NULL DEFAULT 0,
    last_accessed TIMESTAMPTZ,

    -- Graph links
    entity_refs  TEXT[]  NOT NULL DEFAULT '{}',
    provenance   UUID REFERENCES memory_records(id) ON DELETE SET NULL,

    -- Security
    quarantined  BOOLEAN NOT NULL DEFAULT FALSE
);

-- Tenant-scoped lookup (most queries filter by agent + user)
CREATE INDEX idx_memory_agent_user     ON memory_records (agent_id, user_id);
-- Active memories only (no t_expired, no quarantine)
CREATE INDEX idx_memory_active         ON memory_records (agent_id, user_id) WHERE t_expired IS NULL AND quarantined = FALSE;
-- Bi-temporal queries
CREATE INDEX idx_memory_t_valid        ON memory_records (t_valid, t_invalid);
-- Decay worker scans by strength
CREATE INDEX idx_memory_strength       ON memory_records (strength) WHERE t_expired IS NULL;
-- Full-text search (will be extended with BM25 in Phase 2)
CREATE INDEX idx_memory_content_fts    ON memory_records USING GIN (to_tsvector('english', content));
-- Entity refs lookup
CREATE INDEX idx_memory_entity_refs    ON memory_records USING GIN (entity_refs);

-- ---------------------------------------------------------------------------
-- entities — named things referenced across memories
-- ---------------------------------------------------------------------------

CREATE TABLE entities (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id    TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    kind        entity_kind NOT NULL DEFAULT 'other',
    attributes  JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_entity_agent_user ON entities (agent_id, user_id);
CREATE INDEX idx_entity_name       ON entities (agent_id, user_id, name);
CREATE UNIQUE INDEX idx_entity_unique ON entities (agent_id, user_id, LOWER(name));

-- ---------------------------------------------------------------------------
-- entity_edges — directed, typed, bi-temporal relationships
-- ---------------------------------------------------------------------------

CREATE TABLE entity_edges (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_id      UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    to_id        UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation     TEXT NOT NULL,
    weight       FLOAT NOT NULL DEFAULT 1.0,
    t_valid      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    t_invalid    TIMESTAMPTZ,
    source_memory UUID REFERENCES memory_records(id) ON DELETE SET NULL
);

CREATE INDEX idx_edge_from      ON entity_edges (from_id);
CREATE INDEX idx_edge_to        ON entity_edges (to_id);
CREATE INDEX idx_edge_relation  ON entity_edges (relation);
CREATE INDEX idx_edge_t_valid   ON entity_edges (t_valid, t_invalid);

-- ---------------------------------------------------------------------------
-- audit_log — immutable record of every read and write
-- ---------------------------------------------------------------------------

CREATE TABLE audit_log (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ts           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action       audit_action NOT NULL,
    actor        TEXT NOT NULL,
    memory_id    UUID,
    query_text   TEXT,
    result_count INT
);

-- Fast lookup by actor or memory
CREATE INDEX idx_audit_actor     ON audit_log (actor, ts DESC);
CREATE INDEX idx_audit_memory_id ON audit_log (memory_id) WHERE memory_id IS NOT NULL;
