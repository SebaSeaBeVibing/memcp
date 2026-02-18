-- Add embedding status to memories table
ALTER TABLE memories ADD COLUMN embedding_status TEXT NOT NULL DEFAULT 'pending';

-- Separate embeddings table for model migration support
-- Each row stores one embedding vector for one memory with model metadata
CREATE TABLE IF NOT EXISTS memory_embeddings (
    id TEXT PRIMARY KEY NOT NULL,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    embedding vector,
    is_current BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

-- Index for looking up embeddings by memory
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_memory_id ON memory_embeddings(memory_id);
-- Index for model migration queries (find all embeddings for a model, filter by current)
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_model_current ON memory_embeddings(model_name, model_version, is_current);
-- Index for embedding status queries (backfill, stats)
CREATE INDEX IF NOT EXISTS idx_memories_embedding_status ON memories(embedding_status);
