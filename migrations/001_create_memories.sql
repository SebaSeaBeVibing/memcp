CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY NOT NULL,
    content TEXT NOT NULL,
    type_hint TEXT NOT NULL DEFAULT 'fact',
    source TEXT NOT NULL DEFAULT 'default',
    tags TEXT,                          -- JSON array stored as TEXT
    created_at TEXT NOT NULL,           -- ISO-8601 UTC
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_type_hint ON memories(type_hint);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC);
