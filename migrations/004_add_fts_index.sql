-- Full-text search GIN expression index on memory content
-- Uses fastupdate=off for read-heavy workload (memories written once, queried repeatedly)
CREATE INDEX IF NOT EXISTS idx_memories_fts
    ON memories
    USING GIN (to_tsvector('english', content))
    WITH (fastupdate=off);
