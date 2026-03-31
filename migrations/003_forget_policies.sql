-- MogDB Phase 3: forget policies
-- Declarative rules that describe which memories should be expired, invalidated,
-- or quarantined by the nightly ForgetWorker.

CREATE TABLE forget_policies (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id    TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,

    -- PolicyCondition serialised as JSONB
    -- { "kind": "episodic"|null, "strength_below": 0.2|null,
    --   "older_than_days": 30|null, "access_count_below": 2|null }
    condition   JSONB NOT NULL DEFAULT '{}',

    -- PolicyAction: "expire" | "invalidate" | "quarantine"
    action      TEXT NOT NULL DEFAULT 'expire',

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_forget_policy_agent   ON forget_policies (agent_id);
CREATE INDEX idx_forget_policy_enabled ON forget_policies (agent_id) WHERE enabled = TRUE;
