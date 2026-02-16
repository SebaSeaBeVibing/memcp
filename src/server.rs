use rmcp::{
    ServerHandler,
    tool,
    model::{ServerCapabilities, Implementation, ProtocolVersion, CallToolResult},
    handler::server::wrapper::Parameters,
    ErrorData as McpError,
};
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use serde_json::json;
use uuid::Uuid;
use chrono::Utc;
use std::time::Instant;

pub struct MemoryService {
    start_time: Instant,
}

impl MemoryService {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
}

// Helper function for agent_id validation
fn is_valid_agent_id(agent_id: &str) -> bool {
    !agent_id.is_empty()
        && agent_id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_')
}

// Parameter structs
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct StoreMemoryParams {
    /// The memory content to store
    pub content: String,
    /// Agent identifier (defaults to "default")
    #[serde(default = "default_agent_id")]
    pub agent_id: String,
    /// Optional tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct SearchMemoryParams {
    /// Natural language search query
    pub query: String,
    /// Agent identifier (defaults to "default")
    #[serde(default = "default_agent_id")]
    pub agent_id: String,
    /// Maximum number of results to return (1-100, default 10)
    #[serde(default = "default_limit")]
    pub limit: u32,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct UpdateMemoryParams {
    /// Memory ID to update
    pub id: String,
    /// New content (optional)
    pub content: Option<String>,
    /// New tags (optional)
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct DeleteMemoryParams {
    /// Memory ID to delete
    pub id: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ListMemoriesParams {
    /// Agent identifier (defaults to "default")
    #[serde(default = "default_agent_id")]
    pub agent_id: String,
    /// Maximum number of results to return (1-100, default 20)
    #[serde(default = "default_limit_20")]
    pub limit: u32,
    /// Offset for pagination (default 0)
    #[serde(default)]
    pub offset: u32,
}

fn default_agent_id() -> String {
    "default".to_string()
}

fn default_limit() -> u32 {
    10
}

fn default_limit_20() -> u32 {
    20
}

// Tool implementations
#[rmcp::tool_router]
impl MemoryService {
    #[tool(description = "Store a new memory. Returns a unique memory ID.")]
    async fn store_memory(
        &self,
        Parameters(params): Parameters<StoreMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            tool = "store_memory",
            agent_id = %params.agent_id,
            tags = ?params.tags,
            "Tool called"
        );

        // Validate content
        if params.content.trim().is_empty() {
            let error = json!({
                "error": "Field 'content' is required and cannot be empty"
            });
            return Ok(CallToolResult::structured_error(error));
        }

        // Validate agent_id
        if !is_valid_agent_id(&params.agent_id) {
            let error = json!({
                "error": "'agent_id' must contain only alphanumeric characters, hyphens, and underscores"
            });
            return Ok(CallToolResult::structured_error(error));
        }

        // Generate mock response
        let memory_id = Uuid::new_v4().to_string();
        let timestamp = Utc::now().to_rfc3339();

        let response = json!({
            "id": memory_id,
            "agent_id": params.agent_id,
            "content": params.content,
            "tags": params.tags,
            "tag_count": params.tags.len(),
            "created_at": timestamp,
        });

        Ok(CallToolResult::structured(response))
    }

    #[tool(description = "Search memories by natural language query. Returns ranked results.")]
    async fn search_memory(
        &self,
        Parameters(params): Parameters<SearchMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            tool = "search_memory",
            agent_id = %params.agent_id,
            query = %params.query,
            limit = params.limit,
            "Tool called"
        );

        // Validate query
        if params.query.trim().is_empty() {
            let error = json!({
                "error": "Field 'query' is required and cannot be empty"
            });
            return Ok(CallToolResult::structured_error(error));
        }

        // Validate limit
        if params.limit < 1 || params.limit > 100 {
            let error = json!({
                "error": "Field 'limit' must be between 1 and 100"
            });
            return Ok(CallToolResult::structured_error(error));
        }

        // Generate mock response with 2 sample memories
        let timestamp = Utc::now().to_rfc3339();
        let memories = vec![
            json!({
                "id": Uuid::new_v4().to_string(),
                "content": format!("Sample memory matching '{}'", params.query),
                "agent_id": params.agent_id,
                "relevance_score": 0.92,
                "created_at": timestamp,
            }),
            json!({
                "id": Uuid::new_v4().to_string(),
                "content": format!("Another memory related to '{}'", params.query),
                "agent_id": params.agent_id,
                "relevance_score": 0.78,
                "created_at": timestamp,
            }),
        ];

        let response = json!({
            "results": memories,
            "count": 2,
            "query": params.query,
        });

        Ok(CallToolResult::structured(response))
    }

    #[tool(description = "Update an existing memory's content or tags.")]
    async fn update_memory(
        &self,
        Parameters(params): Parameters<UpdateMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            tool = "update_memory",
            id = %params.id,
            has_content = params.content.is_some(),
            has_tags = params.tags.is_some(),
            "Tool called"
        );

        // Validate id
        if params.id.trim().is_empty() {
            let error = json!({
                "error": "Field 'id' is required and cannot be empty"
            });
            return Ok(CallToolResult::structured_error(error));
        }

        // Validate at least one field is provided
        if params.content.is_none() && params.tags.is_none() {
            let error = json!({
                "error": "At least one of 'content' or 'tags' must be provided"
            });
            return Ok(CallToolResult::structured_error(error));
        }

        // Generate mock response
        let timestamp = Utc::now().to_rfc3339();
        let response = json!({
            "id": params.id,
            "content": params.content.unwrap_or_else(|| "Original content".to_string()),
            "tags": params.tags.unwrap_or_default(),
            "updated_at": timestamp,
        });

        Ok(CallToolResult::structured(response))
    }

    #[tool(description = "Delete a memory by ID.")]
    async fn delete_memory(
        &self,
        Parameters(params): Parameters<DeleteMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            tool = "delete_memory",
            id = %params.id,
            "Tool called"
        );

        // Validate id
        if params.id.trim().is_empty() {
            let error = json!({
                "error": "Field 'id' is required and cannot be empty"
            });
            return Ok(CallToolResult::structured_error(error));
        }

        // Generate mock response
        let response = json!({
            "deleted": true,
            "id": params.id,
        });

        Ok(CallToolResult::structured(response))
    }

    #[tool(description = "List memories with pagination. Filter by agent_id.")]
    async fn list_memories(
        &self,
        Parameters(params): Parameters<ListMemoriesParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            tool = "list_memories",
            agent_id = %params.agent_id,
            limit = params.limit,
            offset = params.offset,
            "Tool called"
        );

        // Validate limit
        if params.limit < 1 || params.limit > 100 {
            let error = json!({
                "error": "Field 'limit' must be between 1 and 100"
            });
            return Ok(CallToolResult::structured_error(error));
        }

        // Generate mock response with 3 sample memories
        let timestamp = Utc::now().to_rfc3339();
        let memories = vec![
            json!({
                "id": Uuid::new_v4().to_string(),
                "content": "Sample memory 1",
                "agent_id": params.agent_id,
                "tags": ["example", "mock"],
                "created_at": timestamp,
            }),
            json!({
                "id": Uuid::new_v4().to_string(),
                "content": "Sample memory 2",
                "agent_id": params.agent_id,
                "tags": ["test"],
                "created_at": timestamp,
            }),
            json!({
                "id": Uuid::new_v4().to_string(),
                "content": "Sample memory 3",
                "agent_id": params.agent_id,
                "tags": [],
                "created_at": timestamp,
            }),
        ];

        let response = json!({
            "memories": memories,
            "total_count": 3,
            "limit": params.limit,
            "offset": params.offset,
            "has_more": false,
        });

        Ok(CallToolResult::structured(response))
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

// ServerHandler implementation
#[rmcp::tool_handler(router = Self::tool_router())]
impl ServerHandler for MemoryService {
    fn get_info(&self) -> rmcp::model::InitializeResult {
        rmcp::model::InitializeResult {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation {
                name: "memcp".to_string(),
                title: None,
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: Some("High-performance MCP memory server".to_string()),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Memory server for AI agents. Tools: store_memory, search_memory, update_memory, delete_memory, list_memories, health_check. Use agent_id to namespace memories per agent (defaults to 'default').".to_string()
            ),
        }
    }
}
