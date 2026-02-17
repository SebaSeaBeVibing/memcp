/// Memory store abstraction layer
///
/// Provides the MemoryStore trait and associated types for CRUD operations on memories.
/// The trait abstraction enables multiple database backends — currently PostgreSQL.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::errors::MemcpError;

pub mod postgres;

/// Represents a stored memory with all rich metadata fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// Unique identifier (UUID v4)
    pub id: String,
    /// The actual memory content
    pub content: String,
    /// Classification hint: "fact", "preference", "instruction", etc.
    pub type_hint: String,
    /// Origin source: "user", "assistant", "system", etc.
    pub source: String,
    /// Optional tags for categorization (stored as JSONB in PostgreSQL)
    pub tags: Option<serde_json::Value>,
    /// When the memory was created
    pub created_at: DateTime<Utc>,
    /// When the memory was last modified
    pub updated_at: DateTime<Utc>,
    /// When the memory was last accessed via get()
    pub last_accessed_at: Option<DateTime<Utc>>,
    /// Number of times memory has been retrieved
    pub access_count: i64,
}

/// Input type for creating a new memory.
///
/// The store generates id, timestamps, and access_count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateMemory {
    /// The memory content (required)
    pub content: String,
    /// Classification hint (default: "fact")
    #[serde(default = "default_type_hint")]
    pub type_hint: String,
    /// Origin source (default: "default")
    #[serde(default = "default_source")]
    pub source: String,
    /// Optional tags for categorization
    pub tags: Option<Vec<String>>,
}

fn default_type_hint() -> String {
    "fact".to_string()
}

fn default_source() -> String {
    "default".to_string()
}

/// Input type for partially updating an existing memory.
///
/// All fields are optional — only non-None fields are updated.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UpdateMemory {
    /// New content (optional)
    pub content: Option<String>,
    /// New type hint (optional)
    pub type_hint: Option<String>,
    /// New source (optional)
    pub source: Option<String>,
    /// New tags (optional, replaces existing tags)
    pub tags: Option<Vec<String>>,
}

/// Filter criteria for listing memories with cursor-based pagination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListFilter {
    /// Filter by type_hint (exact match)
    pub type_hint: Option<String>,
    /// Filter by source (exact match)
    pub source: Option<String>,
    /// Filter memories created after this timestamp
    pub created_after: Option<DateTime<Utc>>,
    /// Filter memories created before this timestamp
    pub created_before: Option<DateTime<Utc>>,
    /// Filter memories updated after this timestamp
    pub updated_after: Option<DateTime<Utc>>,
    /// Filter memories updated before this timestamp
    pub updated_before: Option<DateTime<Utc>>,
    /// Maximum number of memories to return (default: 20, max: 100)
    pub limit: i64,
    /// Cursor from previous page for pagination
    pub cursor: Option<String>,
}

impl Default for ListFilter {
    fn default() -> Self {
        ListFilter {
            type_hint: None,
            source: None,
            created_after: None,
            created_before: None,
            updated_after: None,
            updated_before: None,
            limit: 20,
            cursor: None,
        }
    }
}

/// Result of a list operation with optional pagination cursor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResult {
    /// The retrieved memories
    pub memories: Vec<Memory>,
    /// Cursor for next page (None if no more results)
    pub next_cursor: Option<String>,
}

/// Core abstraction for memory persistence operations.
///
/// All implementations must be Send + Sync to support concurrent access.
/// The trait uses async_trait to enable async methods in trait definitions.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Store a new memory and return the created record.
    async fn store(&self, input: CreateMemory) -> Result<Memory, MemcpError>;

    /// Retrieve a memory by ID.
    ///
    /// Also increments access_count and updates last_accessed_at via touch().
    async fn get(&self, id: &str) -> Result<Memory, MemcpError>;

    /// Update an existing memory (partial update).
    ///
    /// Only non-None fields in UpdateMemory are applied.
    async fn update(&self, id: &str, input: UpdateMemory) -> Result<Memory, MemcpError>;

    /// Delete a memory by ID.
    ///
    /// Returns NotFound error if the memory doesn't exist.
    async fn delete(&self, id: &str) -> Result<(), MemcpError>;

    /// List memories with optional filtering and cursor-based pagination.
    async fn list(&self, filter: ListFilter) -> Result<ListResult, MemcpError>;

    /// Count memories matching the given filter (for two-step bulk delete confirmation).
    async fn count_matching(&self, filter: &ListFilter) -> Result<u64, MemcpError>;

    /// Delete all memories matching the given filter.
    ///
    /// Returns the number of deleted memories.
    async fn delete_matching(&self, filter: &ListFilter) -> Result<u64, MemcpError>;

    /// Update last_accessed_at and increment access_count for a memory.
    ///
    /// Silently ignores if the ID doesn't exist (fire-and-forget semantics).
    async fn touch(&self, id: &str) -> Result<(), MemcpError>;
}
