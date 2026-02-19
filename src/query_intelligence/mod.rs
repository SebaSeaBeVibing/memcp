/// Query intelligence provider trait and supporting types
///
/// Provides a pluggable interface for LLM-based query expansion and re-ranking.
/// Supports Ollama (local, default, no API key) and OpenAI-compatible APIs.
///
/// Both features are disabled by default — set expansion_enabled or reranking_enabled
/// in QueryIntelligenceConfig to opt in.

pub mod temporal;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use thiserror::Error;

use crate::errors::MemcpError;

/// Errors that can occur during query intelligence operations.
#[derive(Debug, Error)]
pub enum QueryIntelligenceError {
    /// Inference or JSON parse failure
    #[error("Query intelligence generation error: {0}")]
    Generation(String),

    /// API provider returned an HTTP error
    #[error("API error (status {status}): {message}")]
    Api { status: u16, message: String },

    /// Provider not configured (e.g., missing API key or model)
    #[error("Provider not configured: {0}")]
    NotConfigured(String),

    /// Operation exceeded latency budget
    #[error("Query intelligence timeout: {0}")]
    Timeout(String),
}

impl From<QueryIntelligenceError> for MemcpError {
    fn from(e: QueryIntelligenceError) -> Self {
        MemcpError::Internal(e.to_string())
    }
}

/// A time range filter derived from temporal hints in a query.
#[derive(Debug, Clone)]
pub struct TimeRange {
    /// Lower bound (inclusive): memories after this timestamp
    pub after: Option<DateTime<Utc>>,
    /// Upper bound (inclusive): memories before this timestamp
    pub before: Option<DateTime<Utc>>,
}

/// Result of expanding a query via LLM.
#[derive(Debug, Clone)]
pub struct ExpandedQuery {
    /// Alternative phrasings of the query (2–3 variants, may exclude original)
    pub variants: Vec<String>,
    /// Optional time range extracted from temporal hints in the query
    pub time_range: Option<TimeRange>,
}

/// A candidate memory for re-ranking.
#[derive(Debug, Clone)]
pub struct RankedCandidate {
    /// Unique memory ID
    pub id: String,
    /// Memory content (may be truncated per rerank_content_chars config)
    pub content: String,
    /// Current rank in the retrieval result list (1-indexed, lower = more relevant)
    pub current_rank: usize,
}

/// A re-ranked memory result from the LLM.
#[derive(Debug, Clone)]
pub struct RankedResult {
    /// Memory ID
    pub id: String,
    /// New rank assigned by LLM (1-indexed, lower = more relevant)
    pub llm_rank: usize,
}

/// Core trait for LLM-based query expansion and candidate re-ranking.
///
/// Implementations must be Send + Sync to support use in async contexts
/// and across thread boundaries (e.g., Arc<dyn QueryIntelligenceProvider>).
#[async_trait]
pub trait QueryIntelligenceProvider: Send + Sync {
    /// Expand a query into variants and extract any temporal hints.
    async fn expand(&self, query: &str) -> Result<ExpandedQuery, QueryIntelligenceError>;

    /// Re-rank retrieved candidates, returning them in LLM-preferred order.
    async fn rerank(
        &self,
        query: &str,
        candidates: &[RankedCandidate],
    ) -> Result<Vec<RankedResult>, QueryIntelligenceError>;

    /// Return the model name identifier used by this provider.
    fn model_name(&self) -> &str;
}

/// Build the query expansion prompt.
///
/// Instructs the LLM to act as an AI assistant searching its own memory bank,
/// generate 2–3 query variants, and extract any temporal hints.
pub fn build_expansion_prompt(query: &str, current_date: &str) -> String {
    format!(
        "You are helping an AI assistant search its own memory bank.\n\
         Today's date: {current_date}\n\n\
         Given the search query below, do two things:\n\
         1. Generate 2-3 alternative phrasings that would help retrieve relevant memories \
            (you may discard the original if a variant is clearly better).\n\
         2. If the query contains a temporal hint (e.g. 'last week', 'yesterday', \
            'after 2024-01-01'), extract it as a time range with ISO-8601 after/before fields.\n\n\
         Output only valid JSON matching the provided schema. Do not add commentary.\n\n\
         Query: {query}"
    )
}

/// Build the re-ranking prompt.
///
/// Instructs the LLM to re-order candidate memories by relevance to the query.
pub fn build_reranking_prompt(query: &str, candidates_json: &str) -> String {
    format!(
        "You are helping an AI assistant search its own memory bank.\n\
         Given the search query and a list of candidate memories below, \
         re-order the candidates from most relevant to least relevant.\n\n\
         Output only valid JSON matching the provided schema: \
         {{\"ranked_ids\": [\"id1\", \"id2\", ...]}}. \
         Include ALL candidate IDs. Do not add commentary.\n\n\
         Query: {query}\n\n\
         Candidates:\n{candidates_json}"
    )
}

/// JSON schema for expansion output.
///
/// `variants` is required; `time_range` is optional with optional after/before fields.
pub fn expansion_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "variants": {
                "type": "array",
                "items": { "type": "string" },
                "maxItems": 3,
                "description": "Alternative phrasings of the original query"
            },
            "time_range": {
                "type": "object",
                "properties": {
                    "after": {
                        "type": "string",
                        "description": "ISO-8601 datetime lower bound (inclusive)"
                    },
                    "before": {
                        "type": "string",
                        "description": "ISO-8601 datetime upper bound (inclusive)"
                    }
                }
            }
        },
        "required": ["variants"]
    })
}

/// JSON schema for re-ranking output.
///
/// `ranked_ids` must contain all candidate IDs, most relevant first.
pub fn reranking_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "ranked_ids": {
                "type": "array",
                "items": { "type": "string" },
                "description": "All candidate IDs ordered from most to least relevant"
            }
        },
        "required": ["ranked_ids"]
    })
}
