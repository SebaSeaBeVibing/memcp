use rmcp::{
    ServerHandler,
    tool,
    model::{
        ServerCapabilities, Implementation, ProtocolVersion, CallToolResult,
        RawResource, ListResourcesResult, ReadResourceResult, ResourceContents,
        ReadResourceRequestParams, AnnotateAble,
    },
    handler::server::wrapper::Parameters,
    service::{RequestContext, RoleServer},
    ErrorData as McpError,
};
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use chrono::DateTime;

use crate::errors::MemcpError;
use crate::store::{CreateMemory, ListFilter, Memory, MemoryStore, UpdateMemory};

pub struct MemoryService {
    store: Arc<dyn MemoryStore + Send + Sync>,
    start_time: Instant,
}

impl MemoryService {
    pub fn new(store: Arc<dyn MemoryStore + Send + Sync>) -> Self {
        Self {
            store,
            start_time: Instant::now(),
        }
    }

    fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
}

// Parameter structs

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct StoreMemoryParams {
    /// The memory content to store (required)
    pub content: String,
    /// Classification hint: "fact", "preference", "instruction", etc. (default: "fact")
    pub type_hint: Option<String>,
    /// Origin source: "user", "assistant", "system", etc. (default: "default")
    pub source: Option<String>,
    /// Optional tags for categorization
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct GetMemoryParams {
    /// Memory ID to retrieve (required)
    pub id: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct UpdateMemoryParams {
    /// Memory ID to update (required)
    pub id: String,
    /// New content (optional)
    pub content: Option<String>,
    /// New classification hint (optional)
    pub type_hint: Option<String>,
    /// New origin source (optional)
    pub source: Option<String>,
    /// New tags, replaces existing (optional)
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct DeleteMemoryParams {
    /// Memory ID to delete (required)
    pub id: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct BulkDeleteMemoriesParams {
    /// Filter by type_hint (optional)
    pub type_hint: Option<String>,
    /// Filter by source (optional)
    pub source: Option<String>,
    /// Delete memories created after this ISO-8601 timestamp (optional)
    pub created_after: Option<String>,
    /// Delete memories created before this ISO-8601 timestamp (optional)
    pub created_before: Option<String>,
    /// Delete memories updated after this ISO-8601 timestamp (optional)
    pub updated_after: Option<String>,
    /// Delete memories updated before this ISO-8601 timestamp (optional)
    pub updated_before: Option<String>,
    /// Set to true to confirm deletion (default: false — returns count only)
    #[serde(default)]
    pub confirm: bool,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ListMemoriesParams {
    /// Filter by type_hint (optional)
    pub type_hint: Option<String>,
    /// Filter by source (optional)
    pub source: Option<String>,
    /// Filter memories created after this ISO-8601 timestamp (optional)
    pub created_after: Option<String>,
    /// Filter memories created before this ISO-8601 timestamp (optional)
    pub created_before: Option<String>,
    /// Filter memories updated after this ISO-8601 timestamp (optional)
    pub updated_after: Option<String>,
    /// Filter memories updated before this ISO-8601 timestamp (optional)
    pub updated_before: Option<String>,
    /// Maximum results to return (1-100, default: 20)
    pub limit: Option<u32>,
    /// Cursor from previous page for pagination (optional)
    pub cursor: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct SearchMemoryParams {
    /// Natural language search query (required)
    pub query: String,
    /// Maximum number of results to return (1-100, default: 10)
    pub limit: Option<u32>,
}

// Helper: convert MemcpError to CallToolResult with isError: true
fn store_error_to_result(err: MemcpError) -> CallToolResult {
    match err {
        MemcpError::NotFound { id } => {
            CallToolResult::structured_error(json!({
                "isError": true,
                "error": format!("Memory not found: {}", id),
                "hint": "Use list_memories to find available memory IDs"
            }))
        }
        MemcpError::Validation { message, field } => {
            let mut obj = json!({
                "isError": true,
                "error": message,
            });
            if let Some(f) = field {
                obj["field"] = json!(f);
            }
            CallToolResult::structured_error(obj)
        }
        MemcpError::Storage(msg) => {
            CallToolResult::structured_error(json!({
                "isError": true,
                "error": format!("Storage error: {}", msg)
            }))
        }
        other => {
            CallToolResult::structured_error(json!({
                "isError": true,
                "error": other.to_string()
            }))
        }
    }
}

// Helper: parse optional ISO-8601 string to DateTime<Utc>
fn parse_datetime(s: &str, field: &str) -> Result<chrono::DateTime<chrono::Utc>, CallToolResult> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .map_err(|_| {
            CallToolResult::structured_error(json!({
                "isError": true,
                "error": format!("Invalid datetime format for '{}': expected ISO-8601 (e.g. 2026-02-17T00:00:00Z)", field),
                "field": field
            }))
        })
}

// Tool implementations
#[rmcp::tool_router]
impl MemoryService {
    #[tool(description = "Store a new memory with content, type hint, source, and tags. Returns the created memory with its ID.")]
    async fn store_memory(
        &self,
        Parameters(params): Parameters<StoreMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            tool = "store_memory",
            type_hint = ?params.type_hint,
            source = ?params.source,
            "Tool called"
        );

        if params.content.trim().is_empty() {
            return Ok(CallToolResult::structured_error(json!({
                "isError": true,
                "error": "Field 'content' is required and cannot be empty",
                "field": "content"
            })));
        }

        let input = CreateMemory {
            content: params.content,
            type_hint: params.type_hint.unwrap_or_else(|| "fact".to_string()),
            source: params.source.unwrap_or_else(|| "default".to_string()),
            tags: params.tags,
        };

        match self.store.store(input).await {
            Ok(memory) => Ok(CallToolResult::structured(json!({
                "id": memory.id,
                "content": memory.content,
                "type_hint": memory.type_hint,
                "source": memory.source,
                "tags": memory.tags,
                "created_at": memory.created_at.to_rfc3339(),
                "updated_at": memory.updated_at.to_rfc3339(),
                "access_count": memory.access_count,
                "hint": "Use get_memory with this ID to retrieve, or update_memory to modify"
            }))),
            Err(e) => Ok(store_error_to_result(e)),
        }
    }

    #[tool(description = "Retrieve a specific memory by ID. Also updates access count and last accessed timestamp.")]
    async fn get_memory(
        &self,
        Parameters(params): Parameters<GetMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            tool = "get_memory",
            id = %params.id,
            "Tool called"
        );

        if params.id.trim().is_empty() {
            return Ok(CallToolResult::structured_error(json!({
                "isError": true,
                "error": "Field 'id' is required and cannot be empty",
                "field": "id"
            })));
        }

        match self.store.get(&params.id).await {
            Ok(memory) => Ok(CallToolResult::structured(json!({
                "id": memory.id,
                "content": memory.content,
                "type_hint": memory.type_hint,
                "source": memory.source,
                "tags": memory.tags,
                "created_at": memory.created_at.to_rfc3339(),
                "updated_at": memory.updated_at.to_rfc3339(),
                "last_accessed_at": memory.last_accessed_at.map(|dt| dt.to_rfc3339()),
                "access_count": memory.access_count,
                "hint": "Use update_memory to modify or delete_memory to remove"
            }))),
            Err(e) => Ok(store_error_to_result(e)),
        }
    }

    #[tool(description = "Update an existing memory's content, type hint, source, or tags. At least one field must be provided.")]
    async fn update_memory(
        &self,
        Parameters(params): Parameters<UpdateMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            tool = "update_memory",
            id = %params.id,
            has_content = params.content.is_some(),
            has_type_hint = params.type_hint.is_some(),
            has_source = params.source.is_some(),
            has_tags = params.tags.is_some(),
            "Tool called"
        );

        if params.id.trim().is_empty() {
            return Ok(CallToolResult::structured_error(json!({
                "isError": true,
                "error": "Field 'id' is required and cannot be empty",
                "field": "id"
            })));
        }

        if params.content.is_none()
            && params.type_hint.is_none()
            && params.source.is_none()
            && params.tags.is_none()
        {
            return Ok(CallToolResult::structured_error(json!({
                "isError": true,
                "error": "At least one of 'content', 'type_hint', 'source', or 'tags' must be provided"
            })));
        }

        let input = UpdateMemory {
            content: params.content,
            type_hint: params.type_hint,
            source: params.source,
            tags: params.tags,
        };

        match self.store.update(&params.id, input).await {
            Ok(memory) => Ok(CallToolResult::structured(json!({
                "id": memory.id,
                "content": memory.content,
                "type_hint": memory.type_hint,
                "source": memory.source,
                "tags": memory.tags,
                "created_at": memory.created_at.to_rfc3339(),
                "updated_at": memory.updated_at.to_rfc3339(),
                "access_count": memory.access_count,
                "hint": "Use get_memory to re-read or delete_memory to remove"
            }))),
            Err(e) => Ok(store_error_to_result(e)),
        }
    }

    #[tool(description = "Delete a single memory by ID. This is permanent and cannot be undone.")]
    async fn delete_memory(
        &self,
        Parameters(params): Parameters<DeleteMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            tool = "delete_memory",
            id = %params.id,
            "Tool called"
        );

        if params.id.trim().is_empty() {
            return Ok(CallToolResult::structured_error(json!({
                "isError": true,
                "error": "Field 'id' is required and cannot be empty",
                "field": "id"
            })));
        }

        match self.store.delete(&params.id).await {
            Ok(()) => Ok(CallToolResult::structured(json!({
                "deleted": true,
                "id": params.id,
                "hint": "Memory permanently removed. Use store_memory to create new memories."
            }))),
            Err(e) => Ok(store_error_to_result(e)),
        }
    }

    #[tool(description = "Bulk delete memories by filter. First call (confirm: false) returns the count. Second call (confirm: true) performs deletion.")]
    async fn bulk_delete_memories(
        &self,
        Parameters(params): Parameters<BulkDeleteMemoriesParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            tool = "bulk_delete_memories",
            confirm = params.confirm,
            type_hint = ?params.type_hint,
            source = ?params.source,
            "Tool called"
        );

        // Parse optional datetime strings
        let created_after = if let Some(ref s) = params.created_after {
            match parse_datetime(s, "created_after") {
                Ok(dt) => Some(dt),
                Err(result) => return Ok(result),
            }
        } else {
            None
        };

        let created_before = if let Some(ref s) = params.created_before {
            match parse_datetime(s, "created_before") {
                Ok(dt) => Some(dt),
                Err(result) => return Ok(result),
            }
        } else {
            None
        };

        let updated_after = if let Some(ref s) = params.updated_after {
            match parse_datetime(s, "updated_after") {
                Ok(dt) => Some(dt),
                Err(result) => return Ok(result),
            }
        } else {
            None
        };

        let updated_before = if let Some(ref s) = params.updated_before {
            match parse_datetime(s, "updated_before") {
                Ok(dt) => Some(dt),
                Err(result) => return Ok(result),
            }
        } else {
            None
        };

        let filter = ListFilter {
            type_hint: params.type_hint,
            source: params.source,
            created_after,
            created_before,
            updated_after,
            updated_before,
            ..ListFilter::default()
        };

        if !params.confirm {
            match self.store.count_matching(&filter).await {
                Ok(count) => Ok(CallToolResult::structured(json!({
                    "matched": count,
                    "deleted": false,
                    "hint": format!("Call bulk_delete_memories again with confirm: true to delete these {} memories", count)
                }))),
                Err(e) => Ok(store_error_to_result(e)),
            }
        } else {
            match self.store.delete_matching(&filter).await {
                Ok(count) => Ok(CallToolResult::structured(json!({
                    "deleted": count,
                    "confirmed": true,
                    "hint": "Bulk deletion complete. Use list_memories to verify."
                }))),
                Err(e) => Ok(store_error_to_result(e)),
            }
        }
    }

    #[tool(description = "List memories with optional filters and cursor-based pagination.")]
    async fn list_memories(
        &self,
        Parameters(params): Parameters<ListMemoriesParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            tool = "list_memories",
            type_hint = ?params.type_hint,
            source = ?params.source,
            limit = ?params.limit,
            has_cursor = params.cursor.is_some(),
            "Tool called"
        );

        let limit = params.limit.unwrap_or(20).clamp(1, 100);

        // Parse optional datetime strings
        let created_after = if let Some(ref s) = params.created_after {
            match parse_datetime(s, "created_after") {
                Ok(dt) => Some(dt),
                Err(result) => return Ok(result),
            }
        } else {
            None
        };

        let created_before = if let Some(ref s) = params.created_before {
            match parse_datetime(s, "created_before") {
                Ok(dt) => Some(dt),
                Err(result) => return Ok(result),
            }
        } else {
            None
        };

        let updated_after = if let Some(ref s) = params.updated_after {
            match parse_datetime(s, "updated_after") {
                Ok(dt) => Some(dt),
                Err(result) => return Ok(result),
            }
        } else {
            None
        };

        let updated_before = if let Some(ref s) = params.updated_before {
            match parse_datetime(s, "updated_before") {
                Ok(dt) => Some(dt),
                Err(result) => return Ok(result),
            }
        } else {
            None
        };

        let filter = ListFilter {
            type_hint: params.type_hint,
            source: params.source,
            created_after,
            created_before,
            updated_after,
            updated_before,
            limit: limit as i64,
            cursor: params.cursor,
        };

        match self.store.list(filter).await {
            Ok(result) => {
                let memories: Vec<serde_json::Value> = result
                    .memories
                    .iter()
                    .map(|m| {
                        json!({
                            "id": m.id,
                            "content": m.content,
                            "type_hint": m.type_hint,
                            "source": m.source,
                            "tags": m.tags,
                            "created_at": m.created_at.to_rfc3339(),
                            "updated_at": m.updated_at.to_rfc3339(),
                            "access_count": m.access_count,
                        })
                    })
                    .collect();

                let count = memories.len();
                let has_more = result.next_cursor.is_some();

                Ok(CallToolResult::structured(json!({
                    "memories": memories,
                    "count": count,
                    "next_cursor": result.next_cursor,
                    "has_more": has_more,
                    "hint": "Use next_cursor value in cursor parameter to get next page"
                })))
            }
            Err(e) => Ok(store_error_to_result(e)),
        }
    }

    #[tool(description = "Search is not yet implemented. Currently returns sample results. Full semantic search coming in a future update. Use list_memories to browse all memories.")]
    async fn search_memory(
        &self,
        Parameters(params): Parameters<SearchMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            tool = "search_memory",
            query = %params.query,
            limit = ?params.limit,
            "Tool called (stub)"
        );

        if params.query.trim().is_empty() {
            return Ok(CallToolResult::structured_error(json!({
                "isError": true,
                "error": "Field 'query' is required and cannot be empty",
                "field": "query"
            })));
        }

        let limit = params.limit.unwrap_or(10).clamp(1, 100);

        // Stub response - search not yet implemented
        Ok(CallToolResult::structured(json!({
            "results": [],
            "count": 0,
            "query": params.query,
            "limit": limit,
            "note": "Search is not yet implemented. Full semantic search coming in a future update.",
            "hint": "Use list_memories to browse all memories, or get_memory to retrieve by ID"
        })))
    }

    #[tool(description = "Check server health and status")]
    async fn health_check(
        &self,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(tool = "health_check", "Tool called");

        let response = json!({
            "status": "ok",
            "version": env!("CARGO_PKG_VERSION"),
            "uptime_seconds": self.uptime_seconds(),
        });

        Ok(CallToolResult::structured(response))
    }
}

// Helper: format a slice of memories into human-readable text for resource consumption
fn format_memories_text(memories: &[Memory]) -> String {
    if memories.is_empty() {
        return String::new();
    }
    memories
        .iter()
        .map(|m| {
            format!(
                "---\n[{}] {}\nCreated: {} | Source: {} | Accessed: {} times\n---",
                m.type_hint,
                m.content,
                m.created_at.to_rfc3339(),
                m.source,
                m.access_count
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ServerHandler implementation
#[rmcp::tool_handler(router = Self::tool_router())]
impl ServerHandler for MemoryService {
    fn get_info(&self) -> rmcp::model::InitializeResult {
        rmcp::model::InitializeResult {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
            server_info: Implementation {
                name: "memcp".to_string(),
                title: None,
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: Some("High-performance MCP memory server with persistent SQLite storage".to_string()),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Memory server for AI agents. Tools: store_memory, get_memory, search_memory, update_memory, delete_memory, bulk_delete_memories, list_memories, health_check. Resources: memory://session-primer (recent memories), memory://user-profile (preferences).".to_string()
            ),
        }
    }

    async fn list_resources(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult {
            meta: None,
            resources: vec![
                RawResource {
                    uri: "memory://session-primer".to_string(),
                    name: "session-primer".to_string(),
                    title: Some("Session Memory Primer".to_string()),
                    description: Some("Recent memories for session context".to_string()),
                    mime_type: Some("text/plain".to_string()),
                    size: None,
                    icons: None,
                    meta: None,
                }
                .no_annotation(),
                RawResource {
                    uri: "memory://user-profile".to_string(),
                    name: "user-profile".to_string(),
                    title: Some("User Profile".to_string()),
                    description: Some("User preferences and persistent facts".to_string()),
                    mime_type: Some("text/plain".to_string()),
                    size: None,
                    icons: None,
                    meta: None,
                }
                .no_annotation(),
            ],
            next_cursor: None,
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        match request.uri.as_str() {
            "memory://session-primer" => {
                let filter = ListFilter {
                    limit: 20,
                    ..Default::default()
                };
                let result = self
                    .store
                    .list(filter)
                    .await
                    .map_err(|e| McpError::resource_not_found(e.to_string(), None))?;

                let text = if result.memories.is_empty() {
                    "No memories stored yet. Use store_memory to add your first memory.".to_string()
                } else {
                    format_memories_text(&result.memories)
                };

                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text(text, request.uri)],
                })
            }
            "memory://user-profile" => {
                let filter = ListFilter {
                    type_hint: Some("preference".to_string()),
                    limit: 50,
                    ..Default::default()
                };
                let result = self
                    .store
                    .list(filter)
                    .await
                    .map_err(|e| McpError::resource_not_found(e.to_string(), None))?;

                let text = if result.memories.is_empty() {
                    "No user preferences stored yet. Use store_memory with type_hint: 'preference' to add preferences.".to_string()
                } else {
                    format_memories_text(&result.memories)
                };

                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text(text, request.uri)],
                })
            }
            uri => Err(McpError::resource_not_found(
                format!("Resource not found: {}", uri),
                None,
            )),
        }
    }
}
