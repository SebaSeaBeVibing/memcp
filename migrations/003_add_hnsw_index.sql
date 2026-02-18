-- Add HNSW index for approximate nearest neighbor (ANN) vector search
--
-- This index is used by ORDER BY embedding <=> $1 ASC LIMIT N queries in search_similar().
-- Requires pgvector 0.8.0+ for iterative scan support (SET hnsw.iterative_scan).
--
-- Index parameters:
--   m = 16: number of bidirectional links per node (higher = better recall, more memory)
--   ef_construction = 64: candidate list size during index build (higher = better quality, slower build)
--
-- vector_cosine_ops: optimizes for cosine distance (<=>), matches semantic similarity use case.
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_hnsw
    ON memory_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
