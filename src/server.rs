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
use std::time::{Duration, Instant};
use chrono::DateTime;
use chrono::Utc;
use crate::query_intelligence::{RankedCandidate, temporal::parse_temporal_hint};

use crate::config::SalienceConfig;
use crate::embedding::{EmbeddingJob, EmbeddingProvider};
use crate::errors::MemcpError;
use crate::extraction::ExtractionJob;
use crate::search::{SalienceScorer, ScoredHit};
use crate::search::salience::SalienceInput;
use crate::store::{CreateMemory, ListFilter, Memory, MemoryStore, UpdateMemory};

pub struct MemoryService {
    store: Arc<dyn MemoryStore + Send + Sync>,
    pipeline: Option<crate::embedding::pipeline::EmbeddingPipeline>,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    pg_store: Option<Arc<crate::store::postgres::PostgresMemoryStore>>,
    salience_config: SalienceConfig,
    start_time: Instant,
    extraction_pipeline: Option<crate::extraction::pipeline::ExtractionPipeline>,
    qi_expansion_provider: Option<Arc<dyn crate::query_intelligence::QueryIntelligenceProvider + Send + Sync>>,
    qi_reranking_provider: Option<Arc<dyn crate::query_intelligence::QueryIntelligenceProvider + Send + Sync>>,
    qi_config: crate::config::QueryIntelligenceConfig,
}

impl MemoryService {
    pub fn new(
        store: Arc<dyn MemoryStore + Send + Sync>,
        pipeline: Option<crate::embedding::pipeline::EmbeddingPipeline>,
        embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
        pg_store: Option<Arc<crate::store::postgres::PostgresMemoryStore>>,
        salience_config: SalienceConfig,
        extraction_pipeline: Option<crate::extraction::pipeline::ExtractionPipeline>,
        qi_expansion_provider: Option<Arc<dyn crate::query_intelligence::QueryIntelligenceProvider + Send + Sync>>,
        qi_reranking_provider: Option<Arc<dyn crate::query_intelligence::QueryIntelligenceProvider + Send + Sync>>,
        qi_config: crate::config::QueryIntelligenceConfig,
    ) -> Self {
        Self {
            store,
            pipeline,
            embedding_provider,
            pg_store,
            salience_config,
            start_time: Instant::now(),
            extraction_pipeline,
            qi_expansion_provider,
            qi_reranking_provider,
            qi_config,
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
pub struct ReinforceMemoryParams {
    /// Memory ID to reinforce (required)
    pub id: String,
    /// Reinforcement strength: "good" (default) for standard reinforcement, "easy" for stronger boost
    #[serde(default = "default_rating")]
    pub rating: Option<String>,
}

fn default_rating() -> Option<String> {
    Some("good".to_string())
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct SearchMemoryParams {
    /// Natural language query — find memories by meaning, not exact words (required)
    pub query: String,
    /// Maximum results to return (1-100, default: 10)
    pub limit: Option<u32>,
    /// Return only memories created after this ISO-8601 timestamp (optional)
    pub created_after: Option<String>,
    /// Return only memories created before this ISO-8601 timestamp (optional)
    pub created_before: Option<String>,
    /// Filter by tags — return only memories with ALL specified tags (optional)
    pub tags: Option<Vec<String>>,
    /// Cursor from previous page for pagination (optional)
    pub cursor: Option<String>,
    /// Weight for BM25 keyword search path (0.0 to disable, 1.0 = default, >1.0 = emphasize).
    /// Controls how much exact keyword matches influence results.
    pub bm25_weight: Option<f64>,
    /// Weight for vector semantic search path (0.0 to disable, 1.0 = default, >1.0 = emphasize).
    /// Controls how much meaning similarity influences results.
    pub vector_weight: Option<f64>,
    /// Weight for symbolic metadata search path (0.0 to disable, 1.0 = default, >1.0 = emphasize).
    /// Controls how much tag/type/source matches influence results.
    pub symbolic_weight: Option<f64>,
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
            Ok(memory) => {
                // Enqueue background embedding job (non-blocking)
                if let Some(ref pipeline) = self.pipeline {
                    let text = crate::embedding::build_embedding_text(&memory.content, &memory.tags);
                    pipeline.enqueue(EmbeddingJob {
                        memory_id: memory.id.clone(),
                        text,
                        attempt: 0,
                    });
                }
                // Enqueue background extraction job (non-blocking)
                if let Some(ref extraction_pipeline) = self.extraction_pipeline {
                    extraction_pipeline.enqueue(ExtractionJob {
                        memory_id: memory.id.clone(),
                        content: memory.content.clone(),
                        attempt: 0,
                    });
                }
                Ok(CallToolResult::structured(json!({
                    "id": memory.id,
                    "content": memory.content,
                    "type_hint": memory.type_hint,
                    "source": memory.source,
                    "tags": memory.tags,
                    "created_at": memory.created_at.to_rfc3339(),
                    "updated_at": memory.updated_at.to_rfc3339(),
                    "access_count": memory.access_count,
                    "embedding_status": memory.embedding_status,
                    "hint": "Use get_memory with this ID to retrieve, or update_memory to modify"
                })))
            }
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
            Ok(memory) => {
                // Implicit salience bump on direct retrieval (fire-and-forget, not on search results)
                if let Some(ref pg_store) = self.pg_store {
                    let store = pg_store.clone();
                    let id = params.id.clone();
                    tokio::spawn(async move {
                        if let Err(e) = store.touch_salience(&id).await {
                            tracing::warn!("Failed to touch salience for {}: {}", id, e);
                        }
                    });
                }
                Ok(CallToolResult::structured(json!({
                    "id": memory.id,
                    "content": memory.content,
                    "type_hint": memory.type_hint,
                    "source": memory.source,
                    "tags": memory.tags,
                    "created_at": memory.created_at.to_rfc3339(),
                    "updated_at": memory.updated_at.to_rfc3339(),
                    "last_accessed_at": memory.last_accessed_at.map(|dt| dt.to_rfc3339()),
                    "access_count": memory.access_count,
                    "embedding_status": memory.embedding_status,
                    "hint": "Use update_memory to modify or delete_memory to remove"
                })))
            }
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

        // Track if content or tags changed — determines if re-embedding is needed
        let content_changed = params.content.is_some();
        let tags_changed = params.tags.is_some();

        let input = UpdateMemory {
            content: params.content,
            type_hint: params.type_hint,
            source: params.source,
            tags: params.tags,
        };

        match self.store.update(&params.id, input).await {
            Ok(memory) => {
                // Re-embed when content or tags change (tags are part of the embedding text)
                if content_changed || tags_changed {
                    if let Some(ref pipeline) = self.pipeline {
                        let text = crate::embedding::build_embedding_text(&memory.content, &memory.tags);
                        pipeline.enqueue(EmbeddingJob {
                            memory_id: memory.id.clone(),
                            text,
                            attempt: 0,
                        });
                    }
                }
                // Re-extract when content changes (extraction is content-only, not tags)
                if content_changed {
                    if let Some(ref extraction_pipeline) = self.extraction_pipeline {
                        // Reset extraction status to pending, then enqueue
                        if let Some(ref pg_store) = self.pg_store {
                            let store = pg_store.clone();
                            let id = memory.id.clone();
                            tokio::spawn(async move {
                                if let Err(e) = store.update_extraction_status(&id, "pending").await {
                                    tracing::warn!("Failed to reset extraction status for {}: {}", id, e);
                                }
                            });
                        }
                        extraction_pipeline.enqueue(ExtractionJob {
                            memory_id: memory.id.clone(),
                            content: memory.content.clone(),
                            attempt: 0,
                        });
                    }
                }
                Ok(CallToolResult::structured(json!({
                    "id": memory.id,
                    "content": memory.content,
                    "type_hint": memory.type_hint,
                    "source": memory.source,
                    "tags": memory.tags,
                    "created_at": memory.created_at.to_rfc3339(),
                    "updated_at": memory.updated_at.to_rfc3339(),
                    "access_count": memory.access_count,
                    "embedding_status": memory.embedding_status,
                    "hint": "Use get_memory to re-read or delete_memory to remove"
                })))
            }
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
                            "embedding_status": m.embedding_status,
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

    #[tool(description = "Search memories using both keyword matching and semantic similarity for best results. Use this when you want to find memories related to a concept, topic, or question. Results are ranked by salience score combining recency, access frequency, semantic relevance, and reinforcement. For browsing all memories or filtering by type/source, use list_memories instead.")]
    async fn search_memory(
        &self,
        Parameters(params): Parameters<SearchMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            tool = "search_memory",
            query = %params.query,
            limit = ?params.limit,
            has_cursor = params.cursor.is_some(),
            "Tool called"
        );

        // 1. Validate query
        if params.query.trim().is_empty() {
            return Ok(CallToolResult::structured_error(json!({
                "isError": true,
                "error": "Field 'query' is required and cannot be empty",
                "field": "query"
            })));
        }

        // 2. Validate limit
        let limit = params.limit.unwrap_or(10).clamp(1, 100);

        // 3. Get concrete PostgresMemoryStore reference (required for hybrid search)
        let pg_store = match &self.pg_store {
            Some(s) => s,
            None => {
                return Ok(CallToolResult::structured_error(json!({
                    "isError": true,
                    "error": "Search requires PostgreSQL backend",
                    "hint": "Use list_memories to browse memories"
                })));
            }
        };

        // 4. Query Intelligence: expansion (if enabled)
        let qi_start = Instant::now();
        let qi_budget = Duration::from_millis(self.qi_config.latency_budget_ms);

        let (search_query, qi_time_range) = if let Some(ref provider) = self.qi_expansion_provider {
            let expansion_budget = qi_budget * 6 / 10; // 60% for expansion
            match tokio::time::timeout(expansion_budget, provider.expand(&params.query)).await {
                Ok(Ok(expanded)) => {
                    tracing::info!(
                        variants = expanded.variants.len(),
                        has_time_range = expanded.time_range.is_some(),
                        "Query expanded"
                    );
                    // Use first variant as the search query (best formulation)
                    let best_query = expanded.variants.into_iter().next().unwrap_or_else(|| params.query.clone());
                    (best_query, expanded.time_range)
                }
                Ok(Err(e)) => {
                    tracing::warn!(error = %e, "Query expansion failed, using original query");
                    (params.query.clone(), None)
                }
                Err(_) => {
                    tracing::warn!(elapsed_ms = ?qi_start.elapsed().as_millis(), "Query expansion timed out, using original query");
                    (params.query.clone(), None)
                }
            }
        } else {
            // No LLM expansion — try deterministic temporal fallback
            let time_range = parse_temporal_hint(&params.query, Utc::now());
            (params.query.clone(), time_range)
        };

        // 5. Optionally embed the search_query (graceful degradation to BM25-only if no provider)
        let query_embedding: Option<pgvector::Vector> = if let Some(ref provider) = self.embedding_provider {
            match provider.embed(&search_query).await {
                Ok(vec) => Some(pgvector::Vector::from(vec)),
                Err(e) => {
                    tracing::warn!("Failed to embed search query, falling back to BM25-only: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // 6. Parse optional datetime params
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

        // 7. Convert weight params to per-leg k values for RRF fusion.
        //    Formula: k = base_k / weight (lower k = more top-result influence).
        //    weight=0.0 → None (skip leg entirely).
        //    weight=None → default k (1.0 = no change to base_k).
        const BM25_BASE_K: f64 = 60.0;
        const VECTOR_BASE_K: f64 = 60.0;
        const SYMBOLIC_BASE_K: f64 = 40.0;

        let bm25_k = match params.bm25_weight {
            Some(w) if w == 0.0 => None,          // disabled
            Some(w) => Some(BM25_BASE_K / w),     // weight=2.0 → k=30.0 (stronger influence)
            None => Some(BM25_BASE_K),             // default
        };
        let vector_k = match params.vector_weight {
            Some(w) if w == 0.0 => None,
            Some(w) => Some(VECTOR_BASE_K / w),
            None => Some(VECTOR_BASE_K),
        };
        let symbolic_k = match params.symbolic_weight {
            Some(w) if w == 0.0 => None,
            Some(w) => Some(SYMBOLIC_BASE_K / w),
            None => Some(SYMBOLIC_BASE_K),
        };

        // Validate that at least one search path is enabled
        if bm25_k.is_none() && vector_k.is_none() && symbolic_k.is_none() {
            return Ok(CallToolResult::structured_error(json!({
                "isError": true,
                "error": "At least one search path must be enabled (bm25_weight, vector_weight, or symbolic_weight must be non-zero)",
            })));
        }

        // 8. Call hybrid_search — BM25 + vector + symbolic with three-way RRF fusion.
        // Note: cursor-based pagination not applied at this level; salience re-ranking
        // must happen on the full result set before we can paginate meaningfully.
        let tags_slice: Option<Vec<String>> = params.tags.clone();
        let raw_hits = match pg_store.hybrid_search(
            &search_query,
            query_embedding.as_ref(),
            limit as i64,
            created_after,
            created_before,
            tags_slice.as_deref(),
            bm25_k,
            vector_k,
            symbolic_k,
        ).await {
            Ok(hits) => hits,
            Err(e) => return Ok(store_error_to_result(e)),
        };

        // 9. Fetch salience data for all result IDs
        let ids: Vec<String> = raw_hits.iter().map(|h| h.memory.id.clone()).collect();
        let salience_data = match pg_store.get_salience_data(&ids).await {
            Ok(data) => data,
            Err(e) => return Ok(store_error_to_result(e)),
        };

        // 10. Build ScoredHit vec for salience re-ranking
        let mut scored_hits: Vec<ScoredHit> = raw_hits
            .into_iter()
            .map(|hit| ScoredHit {
                memory: hit.memory,
                rrf_score: hit.rrf_score,
                salience_score: 0.0, // populated by rank()
                match_source: hit.match_source,
                breakdown: None,     // populated by rank() when debug_scoring=true
            })
            .collect();

        // 11. Build SalienceInput for each hit (parallel order to scored_hits)
        let salience_inputs: Vec<SalienceInput> = scored_hits
            .iter()
            .map(|hit| {
                let row = salience_data
                    .get(&hit.memory.id)
                    .cloned()
                    .unwrap_or_default();
                let days_since_reinforced = row.last_reinforced_at
                    .map(|dt| {
                        let duration = Utc::now().signed_duration_since(dt);
                        (duration.num_seconds() as f64 / 86_400.0).max(0.0)
                    })
                    .unwrap_or(365.0); // 1 year default for never-reinforced memories
                SalienceInput {
                    stability: row.stability,
                    days_since_reinforced,
                }
            })
            .collect();

        // 12. Apply salience re-ranking
        let scorer = SalienceScorer::new(&self.salience_config);
        scorer.rank(&mut scored_hits, &salience_inputs);

        // 12.5 Apply temporal soft boost if time range extracted
        if let Some(ref time_range) = qi_time_range {
            for hit in &mut scored_hits {
                let created = hit.memory.created_at;
                let in_range = match (time_range.after, time_range.before) {
                    (Some(after), Some(before)) => created >= after && created <= before,
                    (Some(after), None) => created >= after,
                    (None, Some(before)) => created <= before,
                    (None, None) => false,
                };
                if in_range {
                    hit.salience_score *= 2.0; // 2x boost for in-range memories (soft boost, not filter)
                }
            }
            // Re-sort by boosted salience score
            scored_hits.sort_by(|a, b| b.salience_score.partial_cmp(&a.salience_score).unwrap_or(std::cmp::Ordering::Equal));
        }

        // 12.75 LLM re-ranking (if enabled and budget remaining)
        if let Some(ref provider) = self.qi_reranking_provider {
            let remaining = qi_budget.saturating_sub(qi_start.elapsed());
            if remaining > Duration::from_millis(100) { // Only attempt if >100ms remains
                // Take top 10 for re-ranking (locked decision)
                let top_n = scored_hits.len().min(10);
                let candidates: Vec<RankedCandidate> = scored_hits[..top_n]
                    .iter()
                    .enumerate()
                    .map(|(i, hit)| {
                        let content = if hit.memory.content.len() > self.qi_config.rerank_content_chars {
                            hit.memory.content[..self.qi_config.rerank_content_chars].to_string()
                        } else {
                            hit.memory.content.clone()
                        };
                        RankedCandidate {
                            id: hit.memory.id.clone(),
                            content,
                            current_rank: i + 1,
                        }
                    })
                    .collect();

                match tokio::time::timeout(remaining, provider.rerank(&params.query, &candidates)).await {
                    Ok(Ok(ranked)) => {
                        tracing::info!(ranked_count = ranked.len(), "LLM re-ranking applied");
                        // Blend: 0.7 * llm_rank_score + 0.3 * salience_score (normalized)
                        // llm_rank_score = 1.0 / (1.0 + llm_rank as f64)
                        let max_salience = scored_hits.iter().map(|h| h.salience_score).fold(f64::MIN, f64::max);
                        let min_salience = scored_hits.iter().map(|h| h.salience_score).fold(f64::MAX, f64::min);
                        let salience_range = (max_salience - min_salience).max(1e-6);

                        for hit in scored_hits[..top_n].iter_mut() {
                            if let Some(r) = ranked.iter().find(|r| r.id == hit.memory.id) {
                                let llm_score = 1.0 / (1.0 + r.llm_rank as f64);
                                let norm_salience = (hit.salience_score - min_salience) / salience_range;
                                hit.salience_score = 0.7 * llm_score + 0.3 * norm_salience;
                            }
                        }
                        // Re-sort top_n portion only
                        scored_hits[..top_n].sort_by(|a, b| b.salience_score.partial_cmp(&a.salience_score).unwrap_or(std::cmp::Ordering::Equal));
                    }
                    Ok(Err(e)) => {
                        tracing::warn!(error = %e, "LLM re-ranking failed, keeping salience order");
                    }
                    Err(_) => {
                        tracing::warn!(elapsed_ms = ?qi_start.elapsed().as_millis(), "LLM re-ranking timed out, keeping salience order");
                    }
                }
            } else {
                tracing::debug!(remaining_ms = ?remaining.as_millis(), "Skipping re-ranking — insufficient budget remaining");
            }
        }

        // 13. Format results
        let count = scored_hits.len();
        let results: Vec<serde_json::Value> = scored_hits.iter().map(|hit| {
            let mut obj = json!({
                "id": hit.memory.id,
                "content": hit.memory.content,
                "type_hint": hit.memory.type_hint,
                "source": hit.memory.source,
                "tags": hit.memory.tags,
                "created_at": hit.memory.created_at.to_rfc3339(),
                "updated_at": hit.memory.updated_at.to_rfc3339(),
                "access_count": hit.memory.access_count,
                "relevance_score": (hit.salience_score * 1000.0).round() / 1000.0,
                "match_source": hit.match_source,
                "rrf_score": (hit.rrf_score * 10000.0).round() / 10000.0,
            });
            // Add score breakdown when debug_scoring is enabled
            if let Some(ref bd) = hit.breakdown {
                obj["score_breakdown"] = json!({
                    "recency": (bd.recency * 1000.0).round() / 1000.0,
                    "access": (bd.access * 1000.0).round() / 1000.0,
                    "semantic": (bd.semantic * 1000.0).round() / 1000.0,
                    "reinforcement": (bd.reinforcement * 1000.0).round() / 1000.0,
                });
            }
            obj
        }).collect();

        // 14. Build final response JSON
        let mut response = json!({
            "memories": results,
            "total_results": count,
            "query": params.query,
            "has_more": false,
        });

        if count == 0 {
            response["hint"] = json!("No memories matched your query. Try broader search terms or use list_memories to browse all memories.");
        }

        Ok(CallToolResult::structured(response))
    }

    #[tool(description = "Reinforce a memory to boost its salience in future searches. Use when a memory is particularly relevant or important. Reinforcing a faded memory produces a stronger boost than reinforcing a recently accessed one (spaced repetition). Rating: 'good' (default) for standard reinforcement, 'easy' for extra-strong boost.")]
    async fn reinforce_memory(
        &self,
        Parameters(params): Parameters<ReinforceMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            tool = "reinforce_memory",
            id = %params.id,
            rating = ?params.rating,
            "Tool called"
        );

        if params.id.trim().is_empty() {
            return Ok(CallToolResult::structured_error(json!({
                "isError": true,
                "error": "Field 'id' is required and cannot be empty",
                "field": "id"
            })));
        }

        // Verify memory exists
        match self.store.get(&params.id).await {
            Err(MemcpError::NotFound { .. }) => {
                return Ok(CallToolResult::structured_error(json!({
                    "isError": true,
                    "error": format!("Memory not found: {}", params.id),
                    "hint": "Use list_memories to find available memory IDs"
                })));
            }
            Err(e) => return Ok(store_error_to_result(e)),
            Ok(_) => {}
        }

        // Validate and normalize rating
        let rating = params.rating.as_deref().unwrap_or("good");
        let rating = if rating == "easy" { "easy" } else { "good" };

        // Get concrete pg_store reference
        let pg_store = match &self.pg_store {
            Some(s) => s,
            None => {
                return Ok(CallToolResult::structured_error(json!({
                    "isError": true,
                    "error": "Reinforcement requires PostgreSQL backend"
                })));
            }
        };

        match pg_store.reinforce_salience(&params.id, rating).await {
            Ok(row) => Ok(CallToolResult::structured(json!({
                "id": params.id,
                "stability": row.stability,
                "reinforcement_count": row.reinforcement_count,
                "message": format!(
                    "Memory reinforced. Stability: {:.1} days, reinforcements: {}",
                    row.stability, row.reinforcement_count
                )
            }))),
            Err(e) => Ok(store_error_to_result(e)),
        }
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
                description: Some("High-performance MCP memory server with persistent PostgreSQL storage with semantic search".to_string()),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Memory server for AI agents. Tools: store_memory, get_memory, search_memory, update_memory, delete_memory, bulk_delete_memories, list_memories, health_check, reinforce_memory. Resources: memory://session-primer (recent memories), memory://user-profile (preferences).".to_string()
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
