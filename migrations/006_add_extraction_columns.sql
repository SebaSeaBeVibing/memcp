-- Migration 006: Add extraction columns, consolidation flags, and GIN indexes
-- Supports Phase 6.1 Search Enrichment (Plans 01-03):
--   - Extraction columns for entity/fact extraction pipeline (Plan 02)
--   - Consolidation flag columns for memory deduplication (Plan 03)
--   - GIN indexes for fast JSONB containment search in symbolic search leg (Plan 01)

-- Extraction columns
ALTER TABLE memories ADD COLUMN IF NOT EXISTS extracted_entities JSONB;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS extracted_facts JSONB;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS extraction_status TEXT NOT NULL DEFAULT 'pending';

-- Consolidation flag columns
-- Simpler approach than junction table: avoids extra join at query time
ALTER TABLE memories ADD COLUMN IF NOT EXISTS is_consolidated_original BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS consolidated_into TEXT REFERENCES memories(id);

-- GIN indexes for fast containment search on extracted entities/facts
-- fastupdate=off: read-heavy workload — memories written once, queried repeatedly
CREATE INDEX IF NOT EXISTS idx_memories_entities
    ON memories USING GIN (extracted_entities jsonb_path_ops)
    WITH (fastupdate=off);

CREATE INDEX IF NOT EXISTS idx_memories_facts
    ON memories USING GIN (extracted_facts jsonb_path_ops)
    WITH (fastupdate=off);

CREATE INDEX IF NOT EXISTS idx_memories_extraction_status
    ON memories(extraction_status);

CREATE INDEX IF NOT EXISTS idx_memories_consolidated
    ON memories(is_consolidated_original)
    WHERE is_consolidated_original = TRUE;
