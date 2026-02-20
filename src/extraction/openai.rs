/// OpenAI extraction provider
///
/// Calls the OpenAI Chat Completions API with json_object response format.
/// Uses gpt-4o-mini by default — requires MEMCP_EXTRACTION__OPENAI_API_KEY.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{ExtractionError, ExtractionProvider, ExtractionResult, build_extraction_prompt};

/// Request body for OpenAI Chat Completions API
#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    response_format: ResponseFormat,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

/// Response from OpenAI Chat Completions API
#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

/// Parsed extraction result from model output
#[derive(Deserialize)]
struct ExtractionOutput {
    #[serde(default)]
    entities: Vec<String>,
    #[serde(default)]
    facts: Vec<String>,
}

/// OpenAI-backed extraction provider.
///
/// Uses the chat completions API with json_object response format.
/// Requires a valid OpenAI API key.
pub struct OpenAIExtractionProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    max_content_chars: usize,
}

impl OpenAIExtractionProvider {
    /// Create a new OpenAIExtractionProvider.
    ///
    /// # Arguments
    /// * `api_key` - OpenAI API key (must be non-empty)
    /// * `model` - Model name (default: "gpt-4o-mini")
    /// * `max_content_chars` - Maximum content length before truncation
    ///
    /// # Errors
    /// Returns `ExtractionError::NotConfigured` if api_key is empty.
    pub fn new(api_key: String, model: String, max_content_chars: usize) -> Result<Self, ExtractionError> {
        if api_key.trim().is_empty() {
            return Err(ExtractionError::NotConfigured(
                "OpenAI API key is required when using the openai extraction provider. \
                 Set MEMCP_EXTRACTION__OPENAI_API_KEY in the environment"
                    .to_string(),
            ));
        }

        Ok(OpenAIExtractionProvider {
            client: reqwest::Client::new(),
            api_key,
            model,
            max_content_chars,
        })
    }
}

#[async_trait]
impl ExtractionProvider for OpenAIExtractionProvider {
    async fn extract(&self, content: &str) -> Result<ExtractionResult, ExtractionError> {
        // Truncate content if too long
        let truncated_content = if content.len() > self.max_content_chars {
            tracing::warn!(
                original_len = content.len(),
                truncated_to = self.max_content_chars,
                "Content truncated for extraction"
            );
            &content[..self.max_content_chars]
        } else {
            content
        };

        let prompt = build_extraction_prompt(truncated_content);

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            response_format: ResponseFormat {
                format_type: "json_object".to_string(),
            },
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| ExtractionError::Generation(format!("HTTP request failed: {}", e)))?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(ExtractionError::Api { status, message: body });
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .map_err(|e| ExtractionError::Generation(format!("Failed to parse OpenAI response: {}", e)))?;

        let content_str = chat_response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| ExtractionError::Generation("OpenAI returned empty choices list".to_string()))?;

        let output: ExtractionOutput = serde_json::from_str(&content_str)
            .map_err(|e| ExtractionError::Generation(format!(
                "Failed to parse extraction JSON from model output: {} (content: {})",
                e, &content_str
            )))?;

        Ok(ExtractionResult {
            entities: output.entities,
            facts: output.facts,
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
