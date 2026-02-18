/// Memory store abstraction layer
///
/// Provides the MemoryStore trait and associated types for CRUD operations on memories.
/// The trait abstraction enables multiple database backends — currently PostgreSQL.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use pgvector::Vector;
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
    /// Embedding generation status: "pending", "complete", or "failed"
    /// Use EmbeddingStatus enum (in embedding module) for type-safe pipeline logic.
    pub embedding_status: String,
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

/// Filter criteria for vector similarity search with optional date and tag filters.
///
/// OFFSET-based pagination is used (not keyset-based) because ORDER BY embedding distance
/// doesn't have a stable keyset property — distances change with the query vector.
#[derive(Debug, Clone)]
pub struct SearchFilter {
    /// The embedded query vector to search against (callers always set this explicitly)
    pub query_embedding: Vector,
    /// Maximum number of results to return (default: 10, max: 100)
    pub limit: i64,
    /// Number of results to skip for pagination (default: 0)
    pub offset: i64,
    /// Filter memories created after this timestamp
    pub created_after: Option<DateTime<Utc>>,
    /// Filter memories created before this timestamp
    pub created_before: Option<DateTime<Utc>>,
    /// Filter memories that have ALL specified tags (containment match)
    pub tags: Option<Vec<String>>,
}

impl Default for SearchFilter {
    fn default() -> Self {
        SearchFilter {
            // Callers always set query_embedding explicitly — this is a non-meaningful default
            query_embedding: Vector::from(vec![0.0f32; 384]),
            limit: 10,
            offset: 0,
            created_after: None,
            created_before: None,
            tags: None,
        }
    }
}

/// A single search result containing the matched memory and its cosine similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// The matched memory with all metadata
    pub memory: Memory,
    /// Cosine similarity score in [0.0, 1.0] — higher is more similar
    pub similarity: f64,
}

/// Paginated results from a vector similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The matched memories with similarity scores, ordered by similarity descending
    pub hits: Vec<SearchHit>,
    /// Total number of embedded memories matching the filters (ignoring limit/offset)
    pub total_matches: u64,
    /// Base64-encoded offset for fetching the next page (None if no more results)
    pub next_cursor: Option<String>,
    /// Whether there are more results beyond the current page
    pub has_more: bool,
}

/// Encode a search pagination cursor from an offset value.
///
/// Search cursors are OFFSET-based (not keyset-based like list_memories cursors)
/// because vector distance ordering doesn't have a stable keyset property.
pub fn encode_search_cursor(offset: i64) -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
    URL_SAFE_NO_PAD.encode(offset.to_string().as_bytes())
}

/// Decode a search pagination cursor back into an offset value.
pub fn decode_search_cursor(cursor: &str) -> Result<i64, MemcpError> {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
    let bytes = URL_SAFE_NO_PAD.decode(cursor).map_err(|e| MemcpError::Validation {
        message: format!("Invalid search cursor encoding: {}", e),
        field: Some("cursor".to_string()),
    })?;
    let raw = String::from_utf8(bytes).map_err(|e| MemcpError::Validation {
        message: format!("Invalid search cursor content: {}", e),
        field: Some("cursor".to_string()),
    })?;
    raw.parse::<i64>().map_err(|_| MemcpError::Validation {
        message: "Invalid search cursor value".to_string(),
        field: Some("cursor".to_string()),
    })
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
