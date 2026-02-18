/// PostgreSQL-backed implementation of MemoryStore
///
/// Uses sqlx with PgPool for connection pooling and production-grade persistence.
/// Supports optional migration execution on startup.

use async_trait::async_trait;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use chrono::{DateTime, Utc};
use sqlx::{
    postgres::{PgPool, PgPoolOptions, PgRow},
    Row,
};
use std::time::Duration;
use uuid::Uuid;

use crate::errors::MemcpError;
use crate::store::{CreateMemory, ListFilter, ListResult, Memory, MemoryStore, UpdateMemory};

/// PostgreSQL-backed memory store using sqlx connection pool.
pub struct PostgresMemoryStore {
    pool: PgPool,
}

impl PostgresMemoryStore {
    /// Create a new PostgresMemoryStore, connecting to the PostgreSQL database at database_url.
    ///
    /// Configures a production-ready connection pool with sensible defaults.
    /// If run_migrations is true, automatically runs pending migrations on startup.
    pub async fn new(database_url: &str, run_migrations: bool) -> Result<Self, MemcpError> {
        let pool = PgPoolOptions::new()
            .max_connections(10)         // good default for single-server MCP stdio
            .min_connections(1)          // keep at least one warm connection
            .idle_timeout(Duration::from_secs(300))    // 5 min idle cleanup
            .max_lifetime(Duration::from_secs(1800))   // 30 min max connection age
            .connect(database_url)
            .await
            .map_err(|e| MemcpError::Storage(format!("Failed to connect to database: {}", e)))?;

        if run_migrations {
            sqlx::migrate!("./migrations")
                .run(&pool)
                .await
                .map_err(|e| MemcpError::Storage(format!("Migration failed: {}", e)))?;
        }

        Ok(PostgresMemoryStore { pool })
    }
}

/// Encode a pagination cursor from created_at and id.
fn encode_cursor(created_at: &DateTime<Utc>, id: &str) -> String {
    let raw = format!("{}|{}", created_at.to_rfc3339(), id);
    URL_SAFE_NO_PAD.encode(raw.as_bytes())
}

/// Decode a pagination cursor back into (created_at, id).
fn decode_cursor(cursor: &str) -> Result<(DateTime<Utc>, String), MemcpError> {
    let bytes = URL_SAFE_NO_PAD
        .decode(cursor)
        .map_err(|e| MemcpError::Validation {
            message: format!("Invalid cursor encoding: {}", e),
            field: Some("cursor".to_string()),
        })?;
    let raw = String::from_utf8(bytes).map_err(|e| MemcpError::Validation {
        message: format!("Invalid cursor content: {}", e),
        field: Some("cursor".to_string()),
    })?;
    let mut parts = raw.splitn(2, '|');
    let ts_str = parts.next().ok_or_else(|| MemcpError::Validation {
        message: "Cursor missing timestamp".to_string(),
        field: Some("cursor".to_string()),
    })?;
    let id_str = parts.next().ok_or_else(|| MemcpError::Validation {
        message: "Cursor missing id".to_string(),
        field: Some("cursor".to_string()),
    })?;
    let created_at = ts_str
        .parse::<DateTime<Utc>>()
        .map_err(|e| MemcpError::Validation {
            message: format!("Cursor timestamp parse error: {}", e),
            field: Some("cursor".to_string()),
        })?;
    Ok((created_at, id_str.to_string()))
}

/// Map a sqlx PgRow to a Memory struct.
///
/// PostgreSQL native types map directly:
/// - TIMESTAMPTZ -> DateTime<Utc> (no string parsing)
/// - JSONB -> Option<serde_json::Value> (no string parsing)
fn row_to_memory(row: &PgRow) -> Result<Memory, MemcpError> {
    Ok(Memory {
        id: row.try_get("id").map_err(|e| MemcpError::Storage(e.to_string()))?,
        content: row.try_get("content").map_err(|e| MemcpError::Storage(e.to_string()))?,
        type_hint: row.try_get("type_hint").map_err(|e| MemcpError::Storage(e.to_string()))?,
        source: row.try_get("source").map_err(|e| MemcpError::Storage(e.to_string()))?,
        tags: row.try_get("tags").map_err(|e| MemcpError::Storage(e.to_string()))?,
        created_at: row.try_get("created_at").map_err(|e| MemcpError::Storage(e.to_string()))?,
        updated_at: row.try_get("updated_at").map_err(|e| MemcpError::Storage(e.to_string()))?,
        last_accessed_at: row.try_get("last_accessed_at").map_err(|e| MemcpError::Storage(e.to_string()))?,
        access_count: row.try_get("access_count").map_err(|e| MemcpError::Storage(e.to_string()))?,
        embedding_status: row.try_get("embedding_status").map_err(|e| MemcpError::Storage(e.to_string()))?,
    })
}

#[async_trait]
impl MemoryStore for PostgresMemoryStore {
    async fn store(&self, input: CreateMemory) -> Result<Memory, MemcpError> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();

        // Convert tags Vec<String> to serde_json::Value for JSONB binding
        let tags_json: Option<serde_json::Value> = input
            .tags
            .as_ref()
            .map(|t| serde_json::json!(t));

        sqlx::query(
            "INSERT INTO memories (id, content, type_hint, source, tags, created_at, updated_at, access_count, embedding_status) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, 0, 'pending')",
        )
        .bind(&id)
        .bind(&input.content)
        .bind(&input.type_hint)
        .bind(&input.source)
        .bind(&tags_json)     // JSONB — bind serde_json::Value directly
        .bind(&now)           // TIMESTAMPTZ — bind DateTime<Utc> directly
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| MemcpError::Storage(format!("Failed to insert memory: {}", e)))?;

        Ok(Memory {
            id,
            content: input.content,
            type_hint: input.type_hint,
            source: input.source,
            tags: tags_json,
            created_at: now,
            updated_at: now,
            last_accessed_at: None,
            access_count: 0,
            embedding_status: "pending".to_string(),
        })
    }

    async fn get(&self, id: &str) -> Result<Memory, MemcpError> {
        let row = sqlx::query(
            "SELECT id, content, type_hint, source, tags, created_at, updated_at, last_accessed_at, access_count, embedding_status \
             FROM memories WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| MemcpError::Storage(e.to_string()))?
        .ok_or_else(|| MemcpError::NotFound { id: id.to_string() })?;

        let memory = row_to_memory(&row)?;

        // Fire-and-forget touch to update access stats
        let _ = self.touch(id).await;

        Ok(memory)
    }

    async fn update(&self, id: &str, input: UpdateMemory) -> Result<Memory, MemcpError> {
        // Verify the memory exists first
        let row = sqlx::query("SELECT id FROM memories WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| MemcpError::Storage(e.to_string()))?;

        if row.is_none() {
            return Err(MemcpError::NotFound { id: id.to_string() });
        }

        let now = Utc::now();

        // Build dynamic SET clause with numbered PostgreSQL parameters
        let mut param_idx: u32 = 1;
        let mut sets: Vec<String> = Vec::new();

        // updated_at is always set
        sets.push(format!("updated_at = ${}", param_idx));
        param_idx += 1;

        if input.content.is_some() {
            sets.push(format!("content = ${}", param_idx));
            param_idx += 1;
        }
        if input.type_hint.is_some() {
            sets.push(format!("type_hint = ${}", param_idx));
            param_idx += 1;
        }
        if input.source.is_some() {
            sets.push(format!("source = ${}", param_idx));
            param_idx += 1;
        }
        if input.tags.is_some() {
            sets.push(format!("tags = ${}", param_idx));
            param_idx += 1;
        }

        let sql = format!(
            "UPDATE memories SET {} WHERE id = ${}",
            sets.join(", "),
            param_idx
        );

        let mut q = sqlx::query(&sql).bind(&now); // $1 = updated_at
        if let Some(ref content) = input.content {
            q = q.bind(content);
        }
        if let Some(ref type_hint) = input.type_hint {
            q = q.bind(type_hint);
        }
        if let Some(ref source) = input.source {
            q = q.bind(source);
        }
        if let Some(ref tags) = input.tags {
            // Convert Vec<String> to serde_json::Value for JSONB
            let tags_json = serde_json::json!(tags);
            q = q.bind(tags_json);
        }
        q = q.bind(id); // final $N = id

        q.execute(&self.pool)
            .await
            .map_err(|e| MemcpError::Storage(format!("Failed to update memory: {}", e)))?;

        // Re-fetch and return the updated record
        let updated_row = sqlx::query(
            "SELECT id, content, type_hint, source, tags, created_at, updated_at, last_accessed_at, access_count, embedding_status \
             FROM memories WHERE id = $1",
        )
        .bind(id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| MemcpError::Storage(e.to_string()))?;

        row_to_memory(&updated_row)
    }

    async fn delete(&self, id: &str) -> Result<(), MemcpError> {
        let result = sqlx::query("DELETE FROM memories WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| MemcpError::Storage(e.to_string()))?;

        if result.rows_affected() == 0 {
            return Err(MemcpError::NotFound { id: id.to_string() });
        }

        Ok(())
    }

    async fn list(&self, filter: ListFilter) -> Result<ListResult, MemcpError> {
        let limit = filter.limit.min(100).max(1);

        // Build WHERE clause with numbered PostgreSQL parameters
        let mut conditions: Vec<String> = Vec::new();
        let mut param_idx: u32 = 1;
        let mut cursor_created_at: Option<DateTime<Utc>> = None;
        let mut cursor_id: Option<String> = None;

        if filter.type_hint.is_some() {
            conditions.push(format!("type_hint = ${}", param_idx));
            param_idx += 1;
        }
        if filter.source.is_some() {
            conditions.push(format!("source = ${}", param_idx));
            param_idx += 1;
        }
        if filter.created_after.is_some() {
            conditions.push(format!("created_at > ${}", param_idx));
            param_idx += 1;
        }
        if filter.created_before.is_some() {
            conditions.push(format!("created_at < ${}", param_idx));
            param_idx += 1;
        }
        if filter.updated_after.is_some() {
            conditions.push(format!("updated_at > ${}", param_idx));
            param_idx += 1;
        }
        if filter.updated_before.is_some() {
            conditions.push(format!("updated_at < ${}", param_idx));
            param_idx += 1;
        }
        if let Some(ref cursor) = filter.cursor {
            let (ca, cid) = decode_cursor(cursor)?;
            cursor_created_at = Some(ca);
            cursor_id = Some(cid);
            // Cursor comparison uses 3 params: created_at < $N OR (created_at = $N+1 AND id > $N+2)
            conditions.push(format!(
                "(created_at < ${} OR (created_at = ${} AND id > ${}))",
                param_idx, param_idx + 1, param_idx + 2
            ));
            param_idx += 3;
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let sql = format!(
            "SELECT id, content, type_hint, source, tags, created_at, updated_at, last_accessed_at, access_count, embedding_status \
             FROM memories {} ORDER BY created_at DESC, id ASC LIMIT ${}",
            where_clause, param_idx
        );

        let mut q = sqlx::query(&sql);
        if let Some(ref th) = filter.type_hint {
            q = q.bind(th);
        }
        if let Some(ref src) = filter.source {
            q = q.bind(src);
        }
        if let Some(ref ca) = filter.created_after {
            q = q.bind(ca);
        }
        if let Some(ref cb) = filter.created_before {
            q = q.bind(cb);
        }
        if let Some(ref ua) = filter.updated_after {
            q = q.bind(ua);
        }
        if let Some(ref ub) = filter.updated_before {
            q = q.bind(ub);
        }
        if let Some(ref ca) = cursor_created_at {
            let cid = cursor_id.as_deref().unwrap_or("");
            // Bind 3 times for the cursor comparison: $N, $N+1 (same value), $N+2
            q = q.bind(ca).bind(ca).bind(cid.to_string());
        }
        // Fetch one extra to determine if there are more pages
        q = q.bind(limit + 1);

        let rows = q
            .fetch_all(&self.pool)
            .await
            .map_err(|e| MemcpError::Storage(e.to_string()))?;

        let has_more = rows.len() as i64 > limit;
        let take = if has_more { limit as usize } else { rows.len() };
        let mut memories = Vec::with_capacity(take);

        for row in rows.iter().take(take) {
            memories.push(row_to_memory(row)?);
        }

        let next_cursor = if has_more {
            memories.last().map(|m| encode_cursor(&m.created_at, &m.id))
        } else {
            None
        };

        Ok(ListResult {
            memories,
            next_cursor,
        })
    }

    async fn count_matching(&self, filter: &ListFilter) -> Result<u64, MemcpError> {
        let mut conditions: Vec<String> = Vec::new();
        let mut param_idx: u32 = 1;

        if filter.type_hint.is_some() {
            conditions.push(format!("type_hint = ${}", param_idx));
            param_idx += 1;
        }
        if filter.source.is_some() {
            conditions.push(format!("source = ${}", param_idx));
            param_idx += 1;
        }
        if filter.created_after.is_some() {
            conditions.push(format!("created_at > ${}", param_idx));
            param_idx += 1;
        }
        if filter.created_before.is_some() {
            conditions.push(format!("created_at < ${}", param_idx));
            param_idx += 1;
        }
        if filter.updated_after.is_some() {
            conditions.push(format!("updated_at > ${}", param_idx));
            param_idx += 1;
        }
        if filter.updated_before.is_some() {
            conditions.push(format!("updated_at < ${}", param_idx));
            let _ = param_idx + 1; // suppress unused increment warning
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let sql = format!("SELECT COUNT(*) as count FROM memories {}", where_clause);

        let mut q = sqlx::query(&sql);
        if let Some(ref th) = filter.type_hint {
            q = q.bind(th);
        }
        if let Some(ref src) = filter.source {
            q = q.bind(src);
        }
        if let Some(ref ca) = filter.created_after {
            q = q.bind(ca);
        }
        if let Some(ref cb) = filter.created_before {
            q = q.bind(cb);
        }
        if let Some(ref ua) = filter.updated_after {
            q = q.bind(ua);
        }
        if let Some(ref ub) = filter.updated_before {
            q = q.bind(ub);
        }

        let row = q
            .fetch_one(&self.pool)
            .await
            .map_err(|e| MemcpError::Storage(e.to_string()))?;

        let count: i64 = row.try_get("count").map_err(|e| MemcpError::Storage(e.to_string()))?;
        Ok(count as u64)
    }

    async fn delete_matching(&self, filter: &ListFilter) -> Result<u64, MemcpError> {
        let mut conditions: Vec<String> = Vec::new();
        let mut param_idx: u32 = 1;

        if filter.type_hint.is_some() {
            conditions.push(format!("type_hint = ${}", param_idx));
            param_idx += 1;
        }
        if filter.source.is_some() {
            conditions.push(format!("source = ${}", param_idx));
            param_idx += 1;
        }
        if filter.created_after.is_some() {
            conditions.push(format!("created_at > ${}", param_idx));
            param_idx += 1;
        }
        if filter.created_before.is_some() {
            conditions.push(format!("created_at < ${}", param_idx));
            param_idx += 1;
        }
        if filter.updated_after.is_some() {
            conditions.push(format!("updated_at > ${}", param_idx));
            param_idx += 1;
        }
        if filter.updated_before.is_some() {
            conditions.push(format!("updated_at < ${}", param_idx));
            let _ = param_idx + 1; // suppress unused increment warning
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let sql = format!("DELETE FROM memories {}", where_clause);

        let mut q = sqlx::query(&sql);
        if let Some(ref th) = filter.type_hint {
            q = q.bind(th);
        }
        if let Some(ref src) = filter.source {
            q = q.bind(src);
        }
        if let Some(ref ca) = filter.created_after {
            q = q.bind(ca);
        }
        if let Some(ref cb) = filter.created_before {
            q = q.bind(cb);
        }
        if let Some(ref ua) = filter.updated_after {
            q = q.bind(ua);
        }
        if let Some(ref ub) = filter.updated_before {
            q = q.bind(ub);
        }

        let result = q
            .execute(&self.pool)
            .await
            .map_err(|e| MemcpError::Storage(e.to_string()))?;

        Ok(result.rows_affected())
    }

    async fn touch(&self, id: &str) -> Result<(), MemcpError> {
        let now = Utc::now();
        // Silently ignore if id doesn't exist (fire-and-forget)
        let _ = sqlx::query(
            "UPDATE memories SET last_accessed_at = $1, access_count = access_count + 1 WHERE id = $2",
        )
        .bind(&now) // TIMESTAMPTZ — bind DateTime<Utc> directly
        .bind(id)
        .execute(&self.pool)
        .await;

        Ok(())
    }
}

impl PostgresMemoryStore {
    /// Insert a new embedding record for a memory.
    pub async fn insert_embedding(
        &self,
        id: &str,
        memory_id: &str,
        model_name: &str,
        model_version: &str,
        dimension: i32,
        embedding: &pgvector::Vector,
        is_current: bool,
    ) -> Result<(), MemcpError> {
        let now = Utc::now();
        sqlx::query(
            "INSERT INTO memory_embeddings \
             (id, memory_id, model_name, model_version, dimension, embedding, is_current, created_at, updated_at) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
        )
        .bind(id)
        .bind(memory_id)
        .bind(model_name)
        .bind(model_version)
        .bind(dimension)
        .bind(embedding)
        .bind(is_current)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| MemcpError::Storage(format!("Failed to insert embedding: {}", e)))?;

        Ok(())
    }

    /// Update the embedding_status field on a memory (internal metadata — does not update updated_at).
    pub async fn update_embedding_status(
        &self,
        memory_id: &str,
        status: &str,
    ) -> Result<(), MemcpError> {
        sqlx::query("UPDATE memories SET embedding_status = $1 WHERE id = $2")
            .bind(status)
            .bind(memory_id)
            .execute(&self.pool)
            .await
            .map_err(|e| MemcpError::Storage(format!("Failed to update embedding status: {}", e)))?;

        Ok(())
    }

    /// Retrieve memories that need embedding (status 'pending' or 'failed'), ordered oldest first.
    pub async fn get_pending_memories(&self, limit: i64) -> Result<Vec<crate::store::Memory>, MemcpError> {
        let rows = sqlx::query(
            "SELECT id, content, type_hint, source, tags, created_at, updated_at, last_accessed_at, access_count, embedding_status \
             FROM memories WHERE embedding_status IN ('pending', 'failed') \
             ORDER BY created_at ASC LIMIT $1",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| MemcpError::Storage(e.to_string()))?;

        rows.iter().map(row_to_memory).collect()
    }

    /// Return embedding statistics grouped by status and by model.
    ///
    /// Returns:
    /// ```json
    /// { "by_status": { "pending": N, "complete": N, "failed": N },
    ///   "by_model": [ { "model_name": ..., "model_version": ..., "is_current": true, "count": N } ] }
    /// ```
    pub async fn embedding_stats(&self) -> Result<serde_json::Value, MemcpError> {
        // Query 1: counts by embedding_status
        let status_rows = sqlx::query(
            "SELECT embedding_status, COUNT(*) as count FROM memories GROUP BY embedding_status",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| MemcpError::Storage(e.to_string()))?;

        let mut by_status = serde_json::Map::new();
        for row in &status_rows {
            let status: String = row
                .try_get("embedding_status")
                .map_err(|e| MemcpError::Storage(e.to_string()))?;
            let count: i64 = row
                .try_get("count")
                .map_err(|e| MemcpError::Storage(e.to_string()))?;
            by_status.insert(status, serde_json::json!(count));
        }

        // Query 2: counts by model
        let model_rows = sqlx::query(
            "SELECT model_name, model_version, is_current, COUNT(*) as count \
             FROM memory_embeddings GROUP BY model_name, model_version, is_current",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| MemcpError::Storage(e.to_string()))?;

        let mut by_model: Vec<serde_json::Value> = Vec::new();
        for row in &model_rows {
            let model_name: String = row
                .try_get("model_name")
                .map_err(|e| MemcpError::Storage(e.to_string()))?;
            let model_version: String = row
                .try_get("model_version")
                .map_err(|e| MemcpError::Storage(e.to_string()))?;
            let is_current: bool = row
                .try_get("is_current")
                .map_err(|e| MemcpError::Storage(e.to_string()))?;
            let count: i64 = row
                .try_get("count")
                .map_err(|e| MemcpError::Storage(e.to_string()))?;
            by_model.push(serde_json::json!({
                "model_name": model_name,
                "model_version": model_version,
                "is_current": is_current,
                "count": count,
            }));
        }

        Ok(serde_json::json!({
            "by_status": by_status,
            "by_model": by_model,
        }))
    }

    /// Mark ALL current embeddings as stale (used when switching to a new embedding model).
    ///
    /// Sets is_current = false on all memory_embeddings, and resets embedding_status = 'pending'
    /// on all affected memories so the backfill can re-embed them with the new model.
    /// Returns the count of embeddings marked stale.
    pub async fn mark_all_embeddings_stale(&self) -> Result<u64, MemcpError> {
        // Step 1: mark all current embeddings stale and collect affected memory_ids
        let rows = sqlx::query(
            "UPDATE memory_embeddings SET is_current = false, updated_at = NOW() \
             WHERE is_current = true RETURNING memory_id",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| MemcpError::Storage(format!("Failed to mark embeddings stale: {}", e)))?;

        let count = rows.len() as u64;

        if count > 0 {
            // Step 2: collect memory_ids and reset their embedding_status to 'pending'
            let memory_ids: Vec<String> = rows
                .iter()
                .filter_map(|r| r.try_get::<String, _>("memory_id").ok())
                .collect();

            sqlx::query(
                "UPDATE memories SET embedding_status = 'pending' WHERE id = ANY($1)",
            )
            .bind(&memory_ids)
            .execute(&self.pool)
            .await
            .map_err(|e| {
                MemcpError::Storage(format!("Failed to reset memory embedding_status: {}", e))
            })?;
        }

        Ok(count)
    }

    /// Return the underlying PgPool so embedding pipeline can share the connection pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}
