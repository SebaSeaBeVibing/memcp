/// OpenAI embedding provider
///
/// Calls the OpenAI Embeddings API using reqwest.
/// Supports text-embedding-3-small (1536 dimensions) by default.
/// Requires MEMCP_EMBEDDING__OPENAI_API_KEY env var or openai_api_key in config.

use async_trait::async_trait;

use super::{EmbeddingError, EmbeddingProvider};

/// Request body for OpenAI Embeddings API
#[derive(serde::Serialize)]
struct EmbedRequest {
    input: String,
    model: String,
}

/// Response from OpenAI Embeddings API
#[derive(serde::Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

/// Single embedding result from OpenAI
#[derive(serde::Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

/// OpenAI-backed embedding provider.
///
/// Uses text-embedding-3-small (1536 dimensions) by default.
/// Requires a valid API key — validate on construction, not at embed time.
pub struct OpenAIEmbeddingProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dim: usize,
}

impl OpenAIEmbeddingProvider {
    /// Create a new OpenAIEmbeddingProvider.
    ///
    /// # Arguments
    /// * `api_key` - OpenAI API key (must be non-empty)
    ///
    /// # Errors
    /// Returns `EmbeddingError::NotConfigured` if api_key is empty.
    pub fn new(api_key: String) -> Result<Self, EmbeddingError> {
        if api_key.trim().is_empty() {
            return Err(EmbeddingError::NotConfigured(
                "OpenAI API key is required when using the openai embedding provider. \
                 Set MEMCP_EMBEDDING__OPENAI_API_KEY or openai_api_key in memcp.toml"
                    .to_string(),
            ));
        }

        Ok(OpenAIEmbeddingProvider {
            client: reqwest::Client::new(),
            api_key,
            model: "text-embedding-3-small".to_string(),
            dim: 1536,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let request = EmbedRequest {
            input: text.to_string(),
            model: self.model.clone(),
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::Generation(format!("HTTP request failed: {}", e)))?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(EmbeddingError::Api {
                status,
                message: body,
            });
        }

        let embed_response: EmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::Generation(format!("Failed to parse API response: {}", e)))?;

        embed_response
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| EmbeddingError::Generation("API returned empty embedding list".to_string()))
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}
