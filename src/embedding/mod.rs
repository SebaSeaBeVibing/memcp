/// Embedding provider trait and supporting types
///
/// Provides a pluggable interface for text embedding generation.
/// Supports local fastembed models (default, no API key) and OpenAI API.

pub mod local;
pub mod openai;
pub mod pipeline;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use thiserror::Error;

/// Status of embedding generation for a memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingStatus {
    Pending,
    Complete,
    Failed,
}

impl fmt::Display for EmbeddingStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbeddingStatus::Pending => write!(f, "pending"),
            EmbeddingStatus::Complete => write!(f, "complete"),
            EmbeddingStatus::Failed => write!(f, "failed"),
        }
    }
}

impl FromStr for EmbeddingStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(EmbeddingStatus::Pending),
            "complete" => Ok(EmbeddingStatus::Complete),
            "failed" => Ok(EmbeddingStatus::Failed),
            other => Err(format!("Unknown embedding status: {}", other)),
        }
    }
}

/// Errors that can occur during embedding operations.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// fastembed model initialization failure
    #[error("Model initialization error: {0}")]
    ModelInit(String),

    /// Embedding generation failure (inference error)
    #[error("Embedding generation error: {0}")]
    Generation(String),

    /// API provider returned an HTTP error
    #[error("API error (status {status}): {message}")]
    Api { status: u16, message: String },

    /// Provider not configured (e.g., missing API key)
    #[error("Provider not configured: {0}")]
    NotConfigured(String),
}

/// A pending embedding job for a memory.
#[derive(Debug, Clone)]
pub struct EmbeddingJob {
    /// The memory ID to generate an embedding for
    pub memory_id: String,
    /// The text content to embed
    pub text: String,
    /// Current attempt number (for retry logic)
    pub attempt: u8,
}

/// Concatenate memory content and tags into a single string for embedding.
/// Tags are appended space-separated after the content.
pub fn build_embedding_text(content: &str, tags: &Option<serde_json::Value>) -> String {
    let mut text = content.to_string();
    if let Some(tags_val) = tags {
        if let Some(arr) = tags_val.as_array() {
            let tag_strs: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
            if !tag_strs.is_empty() {
                text.push(' ');
                text.push_str(&tag_strs.join(" "));
            }
        }
    }
    text
}

/// Core trait for embedding text into fixed-dimension float vectors.
///
/// Implementations must be Send + Sync to support use in async contexts
/// and across thread boundaries (e.g., Arc<dyn EmbeddingProvider>).
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding vector for the given text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Return the model name identifier (e.g., "all-MiniLM-L6-v2").
    fn model_name(&self) -> &str;

    /// Return the dimension of the embedding vectors produced by this model.
    fn dimension(&self) -> usize;
}
