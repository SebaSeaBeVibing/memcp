/// Extraction provider trait and supporting types
///
/// Provides a pluggable interface for entity and fact extraction from memory content.
/// Supports Ollama (local, default, no API key) and OpenAI API.

pub mod ollama;
pub mod openai;
pub mod pipeline;

use async_trait::async_trait;
use thiserror::Error;

use crate::errors::MemcpError;

/// Errors that can occur during extraction operations.
#[derive(Debug, Error)]
pub enum ExtractionError {
    /// Model initialization failure
    #[error("Model initialization error: {0}")]
    ModelInit(String),

    /// Extraction generation failure (inference error or parse error)
    #[error("Extraction generation error: {0}")]
    Generation(String),

    /// API provider returned an HTTP error
    #[error("API error (status {status}): {message}")]
    Api { status: u16, message: String },

    /// Provider not configured (e.g., missing API key)
    #[error("Provider not configured: {0}")]
    NotConfigured(String),
}

impl From<ExtractionError> for MemcpError {
    fn from(e: ExtractionError) -> Self {
        MemcpError::Internal(e.to_string())
    }
}

/// Result of extracting entities and facts from memory content.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// Named entities found: people, places, dates, tools, projects, concepts, preferences
    pub entities: Vec<String>,
    /// Key facts: specific assertions, preferences, relationships, or instructions
    pub facts: Vec<String>,
}

/// A pending extraction job for a memory.
#[derive(Debug, Clone)]
pub struct ExtractionJob {
    /// The memory ID to extract entities/facts for
    pub memory_id: String,
    /// The text content to extract from
    pub content: String,
    /// Current attempt number (for retry logic)
    pub attempt: u8,
}

/// Build the extraction prompt for a given content string.
pub fn build_extraction_prompt(content: &str) -> String {
    format!(
        "Extract named entities and key facts from the following text.\n\
         Entities: people, places, dates, tools, projects, concepts, preferences.\n\
         Facts: specific assertions, preferences, relationships, or instructions stated.\n\
         Be comprehensive. Output only JSON matching the provided schema.\n\n\
         Text:\n{}",
        content
    )
}

/// Core trait for extracting entities and facts from text.
///
/// Implementations must be Send + Sync to support use in async contexts
/// and across thread boundaries (e.g., Arc<dyn ExtractionProvider>).
#[async_trait]
pub trait ExtractionProvider: Send + Sync {
    /// Extract entities and facts from the given content.
    async fn extract(&self, content: &str) -> Result<ExtractionResult, ExtractionError>;

    /// Return the model name identifier used by this provider.
    fn model_name(&self) -> &str;
}
