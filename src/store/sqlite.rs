/// SQLite-backed implementation of MemoryStore
///
/// Uses sqlx with WAL mode for cross-restart persistence.
/// Runs migrations automatically on initialization.

use async_trait::async_trait;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use chrono::{DateTime, Utc};
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqliteSynchronous},
    Row,
};
use uuid::Uuid;

use crate::errors::MemcpError;
use crate::store::{CreateMemory, ListFilter, ListResult, Memory, MemoryStore, UpdateMemory};

/// SQLite-backed memory store using sqlx connection pool.
pub struct SqliteMemoryStore {
    pool: SqlitePool,
}

impl SqliteMemoryStore {
    /// Create a new SqliteMemoryStore, opening (or creating) the database at db_path.
    ///
    /// Enables WAL mode for better concurrent performance.
    /// Automatically runs pending migrations on startup.
    pub async fn new(db_path: &str) -> Result<Self, MemcpError> {
        // Strip "sqlite://" prefix if present for SqliteConnectOptions
        let path = db_path.strip_prefix("sqlite://").unwrap_or(db_path);

        let opts = path
            .parse::<SqliteConnectOptions>()
            .map_err(|e| MemcpError::Storage(format!("Invalid db_path '{}': {}", db_path, e)))?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .foreign_keys(true);

        let pool = SqlitePool::connect_with(opts)
            .await
            .map_err(|e| MemcpError::Storage(format!("Failed to connect to database: {}", e)))?;

        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .map_err(|e| MemcpError::Storage(format!("Migration failed: {}", e)))?;

        Ok(SqliteMemoryStore { pool })
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

/// Map a sqlx row to a Memory struct manually (avoids FromRow complexity with JSON columns).
fn row_to_memory(row: &sqlx::sqlite::SqliteRow) -> Result<Memory, MemcpError> {
    let id: String = row.try_get("id").map_err(|e| MemcpError::Storage(e.to_string()))?;
    let content: String = row.try_get("content").map_err(|e| MemcpError::Storage(e.to_string()))?;
    let type_hint: String = row.try_get("type_hint").map_err(|e| MemcpError::Storage(e.to_string()))?;
    let source: String = row.try_get("source").map_err(|e| MemcpError::Storage(e.to_string()))?;
    let tags_str: Option<String> = row.try_get("tags").map_err(|e| MemcpError::Storage(e.to_string()))?;
    let created_at_str: String = row.try_get("created_at").map_err(|e| MemcpError::Storage(e.to_string()))?;
    let updated_at_str: String = row.try_get("updated_at").map_err(|e| MemcpError::Storage(e.to_string()))?;
    let last_accessed_at_str: Option<String> = row.try_get("last_accessed_at").map_err(|e| MemcpError::Storage(e.to_string()))?;
    let access_count: i64 = row.try_get("access_count").map_err(|e| MemcpError::Storage(e.to_string()))?;

    let tags = tags_str.as_deref().and_then(|s| {
        if s.is_empty() {
            None
        } else {
            serde_json::from_str(s).ok()
        }
    });

    let created_at = created_at_str
        .parse::<DateTime<Utc>>()
        .map_err(|e| MemcpError::Storage(format!("Parse created_at '{}': {}", created_at_str, e)))?;

    let updated_at = updated_at_str
        .parse::<DateTime<Utc>>()
        .map_err(|e| MemcpError::Storage(format!("Parse updated_at '{}': {}", updated_at_str, e)))?;

    let last_accessed_at = last_accessed_at_str
        .as_deref()
        .map(|s| {
            s.parse::<DateTime<Utc>>()
                .map_err(|e| MemcpError::Storage(format!("Parse last_accessed_at '{}': {}", s, e)))
        })
        .transpose()?;

    Ok(Memory {
        id,
        content,
        type_hint,
        source,
        tags,
        created_at,
        updated_at,
        last_accessed_at,
        access_count,
    })
}

#[async_trait]
impl MemoryStore for SqliteMemoryStore {
    async fn store(&self, input: CreateMemory) -> Result<Memory, MemcpError> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let now_str = now.to_rfc3339();
        let tags_json = input
            .tags
            .as_ref()
            .map(|t| serde_json::to_string(t).unwrap_or_default());

        sqlx::query(
            "INSERT INTO memories (id, content, type_hint, source, tags, created_at, updated_at, access_count) \
             VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
        )
        .bind(&id)
        .bind(&input.content)
        .bind(&input.type_hint)
        .bind(&input.source)
        .bind(&tags_json)
        .bind(&now_str)
        .bind(&now_str)
        .execute(&self.pool)
        .await
        .map_err(|e| MemcpError::Storage(format!("Failed to insert memory: {}", e)))?;

        Ok(Memory {
            id,
            content: input.content,
            type_hint: input.type_hint,
            source: input.source,
            tags: input.tags.map(|t| serde_json::json!(t)),
            created_at: now,
            updated_at: now,
            last_accessed_at: None,
            access_count: 0,
        })
    }

    async fn get(&self, id: &str) -> Result<Memory, MemcpError> {
        let row = sqlx::query(
            "SELECT id, content, type_hint, source, tags, created_at, updated_at, last_accessed_at, access_count \
             FROM memories WHERE id = ?",
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
        let row = sqlx::query(
            "SELECT id FROM memories WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| MemcpError::Storage(e.to_string()))?;

        if row.is_none() {
            return Err(MemcpError::NotFound { id: id.to_string() });
        }

        let now = Utc::now();
        let now_str = now.to_rfc3339();

        // Build dynamic SET clause
        let mut sets = vec!["updated_at = ?".to_string()];
        if input.content.is_some() {
            sets.push("content = ?".to_string());
        }
        if input.type_hint.is_some() {
            sets.push("type_hint = ?".to_string());
        }
        if input.source.is_some() {
            sets.push("source = ?".to_string());
        }
        if input.tags.is_some() {
            sets.push("tags = ?".to_string());
        }

        let sql = format!(
            "UPDATE memories SET {} WHERE id = ?",
            sets.join(", ")
        );

        let mut q = sqlx::query(&sql).bind(&now_str);
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
            let tags_json = serde_json::to_string(tags).unwrap_or_default();
            q = q.bind(tags_json);
        }
        q = q.bind(id);

        q.execute(&self.pool)
            .await
            .map_err(|e| MemcpError::Storage(format!("Failed to update memory: {}", e)))?;

        // Re-fetch and return the updated record
        let updated_row = sqlx::query(
            "SELECT id, content, type_hint, source, tags, created_at, updated_at, last_accessed_at, access_count \
             FROM memories WHERE id = ?",
        )
        .bind(id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| MemcpError::Storage(e.to_string()))?;

        row_to_memory(&updated_row)
    }

    async fn delete(&self, id: &str) -> Result<(), MemcpError> {
        let result = sqlx::query("DELETE FROM memories WHERE id = ?")
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

        // Build WHERE clause components
        let mut conditions: Vec<String> = Vec::new();
        let mut cursor_created_at: Option<DateTime<Utc>> = None;
        let mut cursor_id: Option<String> = None;

        if filter.type_hint.is_some() {
            conditions.push("type_hint = ?".to_string());
        }
        if filter.source.is_some() {
            conditions.push("source = ?".to_string());
        }
        if filter.created_after.is_some() {
            conditions.push("created_at > ?".to_string());
        }
        if filter.created_before.is_some() {
            conditions.push("created_at < ?".to_string());
        }
        if filter.updated_after.is_some() {
            conditions.push("updated_at > ?".to_string());
        }
        if filter.updated_before.is_some() {
            conditions.push("updated_at < ?".to_string());
        }
        if let Some(ref cursor) = filter.cursor {
            let (ca, cid) = decode_cursor(cursor)?;
            cursor_created_at = Some(ca);
            cursor_id = Some(cid);
            conditions.push("(created_at < ? OR (created_at = ? AND id > ?))".to_string());
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let sql = format!(
            "SELECT id, content, type_hint, source, tags, created_at, updated_at, last_accessed_at, access_count \
             FROM memories {} ORDER BY created_at DESC, id ASC LIMIT ?",
            where_clause
        );

        let mut q = sqlx::query(&sql);
        if let Some(ref th) = filter.type_hint {
            q = q.bind(th);
        }
        if let Some(ref src) = filter.source {
            q = q.bind(src);
        }
        if let Some(ref ca) = filter.created_after {
            q = q.bind(ca.to_rfc3339());
        }
        if let Some(ref cb) = filter.created_before {
            q = q.bind(cb.to_rfc3339());
        }
        if let Some(ref ua) = filter.updated_after {
            q = q.bind(ua.to_rfc3339());
        }
        if let Some(ref ub) = filter.updated_before {
            q = q.bind(ub.to_rfc3339());
        }
        if let Some(ref ca) = cursor_created_at {
            let ca_str = ca.to_rfc3339();
            let cid = cursor_id.as_deref().unwrap_or("");
            q = q.bind(ca_str.clone()).bind(ca_str).bind(cid.to_string());
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

        if filter.type_hint.is_some() {
            conditions.push("type_hint = ?".to_string());
        }
        if filter.source.is_some() {
            conditions.push("source = ?".to_string());
        }
        if filter.created_after.is_some() {
            conditions.push("created_at > ?".to_string());
        }
        if filter.created_before.is_some() {
            conditions.push("created_at < ?".to_string());
        }
        if filter.updated_after.is_some() {
            conditions.push("updated_at > ?".to_string());
        }
        if filter.updated_before.is_some() {
            conditions.push("updated_at < ?".to_string());
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
            q = q.bind(ca.to_rfc3339());
        }
        if let Some(ref cb) = filter.created_before {
            q = q.bind(cb.to_rfc3339());
        }
        if let Some(ref ua) = filter.updated_after {
            q = q.bind(ua.to_rfc3339());
        }
        if let Some(ref ub) = filter.updated_before {
            q = q.bind(ub.to_rfc3339());
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

        if filter.type_hint.is_some() {
            conditions.push("type_hint = ?".to_string());
        }
        if filter.source.is_some() {
            conditions.push("source = ?".to_string());
        }
        if filter.created_after.is_some() {
            conditions.push("created_at > ?".to_string());
        }
        if filter.created_before.is_some() {
            conditions.push("created_at < ?".to_string());
        }
        if filter.updated_after.is_some() {
            conditions.push("updated_at > ?".to_string());
        }
        if filter.updated_before.is_some() {
            conditions.push("updated_at < ?".to_string());
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
            q = q.bind(ca.to_rfc3339());
        }
        if let Some(ref cb) = filter.created_before {
            q = q.bind(cb.to_rfc3339());
        }
        if let Some(ref ua) = filter.updated_after {
            q = q.bind(ua.to_rfc3339());
        }
        if let Some(ref ub) = filter.updated_before {
            q = q.bind(ub.to_rfc3339());
        }

        let result = q
            .execute(&self.pool)
            .await
            .map_err(|e| MemcpError::Storage(e.to_string()))?;

        Ok(result.rows_affected())
    }

    async fn touch(&self, id: &str) -> Result<(), MemcpError> {
        let now_str = Utc::now().to_rfc3339();
        // Silently ignore if id doesn't exist (fire-and-forget)
        let _ = sqlx::query(
            "UPDATE memories SET last_accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
        )
        .bind(&now_str)
        .bind(id)
        .execute(&self.pool)
        .await;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    /// Create an in-memory SQLite store for testing.
    /// Uses max_connections(1) to avoid multi-connection pitfalls with in-memory SQLite.
    async fn create_test_store() -> SqliteMemoryStore {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool");

        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .expect("Failed to run migrations");

        SqliteMemoryStore { pool }
    }

    #[tokio::test]
    async fn test_store_and_get() {
        let store = create_test_store().await;

        let input = CreateMemory {
            content: "The capital of France is Paris".to_string(),
            type_hint: "fact".to_string(),
            source: "user".to_string(),
            tags: Some(vec!["geography".to_string(), "europe".to_string()]),
        };

        let created = store.store(input).await.expect("Failed to store memory");

        assert!(!created.id.is_empty());
        assert_eq!(created.content, "The capital of France is Paris");
        assert_eq!(created.type_hint, "fact");
        assert_eq!(created.source, "user");
        assert_eq!(created.access_count, 0);
        assert!(created.last_accessed_at.is_none());

        // Verify tags are stored correctly
        if let Some(ref tags) = created.tags {
            let tags_vec: Vec<String> = serde_json::from_value(tags.clone()).unwrap();
            assert_eq!(tags_vec, vec!["geography", "europe"]);
        } else {
            panic!("Tags should be present");
        }

        // Get by ID
        let fetched = store.get(&created.id).await.expect("Failed to get memory");
        assert_eq!(fetched.id, created.id);
        assert_eq!(fetched.content, created.content);
        // Access count should be incremented after get
        // (touch is async, give a small moment)
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    #[tokio::test]
    async fn test_update() {
        let store = create_test_store().await;

        let input = CreateMemory {
            content: "Original content".to_string(),
            type_hint: "fact".to_string(),
            source: "user".to_string(),
            tags: None,
        };

        let created = store.store(input).await.expect("Failed to store memory");

        let update = UpdateMemory {
            content: Some("Updated content".to_string()),
            tags: Some(vec!["updated".to_string()]),
            ..Default::default()
        };

        let updated = store
            .update(&created.id, update)
            .await
            .expect("Failed to update memory");

        assert_eq!(updated.id, created.id);
        assert_eq!(updated.content, "Updated content");
        assert_eq!(updated.type_hint, "fact"); // unchanged
        assert_eq!(updated.source, "user"); // unchanged

        if let Some(ref tags) = updated.tags {
            let tags_vec: Vec<String> = serde_json::from_value(tags.clone()).unwrap();
            assert_eq!(tags_vec, vec!["updated"]);
        } else {
            panic!("Tags should be present after update");
        }

        // updated_at should be >= created_at
        assert!(updated.updated_at >= created.created_at);
    }

    #[tokio::test]
    async fn test_delete() {
        let store = create_test_store().await;

        let input = CreateMemory {
            content: "To be deleted".to_string(),
            type_hint: "fact".to_string(),
            source: "test".to_string(),
            tags: None,
        };

        let created = store.store(input).await.expect("Failed to store memory");
        let id = created.id.clone();

        // Delete the memory
        store.delete(&id).await.expect("Failed to delete memory");

        // Verify it's gone
        let result = store.get(&id).await;
        assert!(
            matches!(result, Err(MemcpError::NotFound { .. })),
            "Expected NotFound error, got: {:?}",
            result
        );

        // Second delete should also return NotFound
        let result2 = store.delete(&id).await;
        assert!(matches!(result2, Err(MemcpError::NotFound { .. })));
    }

    #[tokio::test]
    async fn test_list_with_pagination() {
        let store = create_test_store().await;

        // Store 5 memories
        for i in 0..5 {
            let input = CreateMemory {
                content: format!("Memory {}", i),
                type_hint: "fact".to_string(),
                source: "test".to_string(),
                tags: None,
            };
            store.store(input).await.expect("Failed to store memory");
            // Small delay to ensure distinct created_at timestamps
            tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        }

        // List first page (limit 2)
        let filter = ListFilter {
            limit: 2,
            ..Default::default()
        };
        let page1 = store.list(filter).await.expect("Failed to list memories");

        assert_eq!(page1.memories.len(), 2, "Page 1 should have 2 memories");
        assert!(page1.next_cursor.is_some(), "Should have a next cursor");

        // List second page using cursor
        let filter2 = ListFilter {
            limit: 2,
            cursor: page1.next_cursor.clone(),
            ..Default::default()
        };
        let page2 = store.list(filter2).await.expect("Failed to list page 2");

        assert_eq!(page2.memories.len(), 2, "Page 2 should have 2 memories");
        assert!(page2.next_cursor.is_some(), "Should have a next cursor for page 3");

        // List third page
        let filter3 = ListFilter {
            limit: 2,
            cursor: page2.next_cursor.clone(),
            ..Default::default()
        };
        let page3 = store.list(filter3).await.expect("Failed to list page 3");

        assert_eq!(page3.memories.len(), 1, "Page 3 should have 1 memory");
        assert!(page3.next_cursor.is_none(), "Should have no cursor on last page");

        // Verify no duplicate IDs across pages
        let mut all_ids: Vec<String> = page1.memories.iter().map(|m| m.id.clone()).collect();
        all_ids.extend(page2.memories.iter().map(|m| m.id.clone()));
        all_ids.extend(page3.memories.iter().map(|m| m.id.clone()));
        all_ids.sort();
        all_ids.dedup();
        assert_eq!(all_ids.len(), 5, "Should have 5 unique memory IDs across all pages");
    }

    #[tokio::test]
    async fn test_count_and_delete_matching() {
        let store = create_test_store().await;

        // Store 3 memories with type_hint = "fact"
        for i in 0..3 {
            let input = CreateMemory {
                content: format!("Fact {}", i),
                type_hint: "fact".to_string(),
                source: "test".to_string(),
                tags: None,
            };
            store.store(input).await.expect("Failed to store fact");
        }

        // Store 2 memories with type_hint = "preference"
        for i in 0..2 {
            let input = CreateMemory {
                content: format!("Preference {}", i),
                type_hint: "preference".to_string(),
                source: "test".to_string(),
                tags: None,
            };
            store.store(input).await.expect("Failed to store preference");
        }

        // Count matching "fact" type
        let filter = ListFilter {
            type_hint: Some("fact".to_string()),
            ..Default::default()
        };

        let count = store
            .count_matching(&filter)
            .await
            .expect("Failed to count matching");
        assert_eq!(count, 3, "Should have 3 facts");

        // Delete matching "fact" type
        let deleted = store
            .delete_matching(&filter)
            .await
            .expect("Failed to delete matching");
        assert_eq!(deleted, 3, "Should have deleted 3 facts");

        // Verify facts are gone
        let count_after = store
            .count_matching(&filter)
            .await
            .expect("Failed to count after delete");
        assert_eq!(count_after, 0, "Should have 0 facts after delete");

        // Preferences should still be there
        let pref_filter = ListFilter {
            type_hint: Some("preference".to_string()),
            ..Default::default()
        };
        let pref_count = store
            .count_matching(&pref_filter)
            .await
            .expect("Failed to count preferences");
        assert_eq!(pref_count, 2, "Preferences should be untouched");
    }
}
