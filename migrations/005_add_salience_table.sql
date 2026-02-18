-- Salience state for FSRS-based reinforcement scoring
-- Created lazily: row inserted on first reinforce or first search that references the memory
CREATE TABLE IF NOT EXISTS memory_salience (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    -- FSRS state
    stability   REAL NOT NULL DEFAULT 1.0,   -- days until 90% retrievability; grows with reinforcement
    difficulty  REAL NOT NULL DEFAULT 5.0,   -- FSRS difficulty [1,10]; affects stability growth rate
    -- Tracking
    reinforcement_count   INTEGER NOT NULL DEFAULT 0,
    last_reinforced_at    TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
