-- Migration 007: Add memory_consolidations junction table
-- Supports Phase 6.1 Plan 03: Non-destructive memory consolidation
-- Tracks provenance of consolidated memories — originals are preserved with links.

CREATE TABLE IF NOT EXISTS memory_consolidations (
    id TEXT PRIMARY KEY NOT NULL,
    consolidated_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    original_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    similarity_score REAL NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (consolidated_id, original_id)
);

-- Index for looking up all originals that fed into a consolidated memory
CREATE INDEX IF NOT EXISTS idx_consolidations_consolidated ON memory_consolidations(consolidated_id);

-- Index for checking whether a given memory has been consolidated into another
CREATE INDEX IF NOT EXISTS idx_consolidations_original ON memory_consolidations(original_id);
