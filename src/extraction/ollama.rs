/// Ollama extraction provider
///
/// Calls the Ollama /api/chat endpoint with structured JSON output schema.
/// Uses llama3.2:3b by default — no API key required for self-hosted deployments.
/// Supports MEMCP_EXTRACTION__OLLAMA_MODEL and MEMCP_EXTRACTION__OLLAMA_BASE_URL.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{ExtractionError, ExtractionProvider, ExtractionResult, build_extraction_prompt};

/// Request body for Ollama /api/chat with structured output
#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    options: OllamaOptions,
    format: serde_json::Value,
}

#[derive(Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
}

/// Response from Ollama /api/chat
#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaResponseMessage,
}

#[derive(Deserialize)]
struct OllamaResponseMessage {
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

/// Ollama-backed extraction provider.
///
/// Uses the /api/chat endpoint with structured JSON output (format field).
/// Truncates content to max_content_chars to avoid context overflow.
pub struct OllamaExtractionProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
    max_content_chars: usize,
}

impl OllamaExtractionProvider {
    /// Create a new OllamaExtractionProvider.
    ///
    /// # Arguments
    /// * `base_url` - Ollama server base URL (e.g., "http://localhost:11434")
    /// * `model` - Model name (e.g., "llama3.2:3b")
    /// * `max_content_chars` - Maximum content length before truncation (default: 1500)
    pub fn new(base_url: String, model: String, max_content_chars: usize) -> Self {
        OllamaExtractionProvider {
            client: reqwest::Client::new(),
            base_url,
            model,
            max_content_chars,
        }
    }
}

/// JSON schema for structured extraction output
fn extraction_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {"type": "string"}
            },
            "facts": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["entities", "facts"]
    })
}

#[async_trait]
impl ExtractionProvider for OllamaExtractionProvider {
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

        let request = OllamaChatRequest {
            model: self.model.clone(),
            messages: vec![OllamaMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            stream: false,
            options: OllamaOptions { temperature: 0.0 },
            format: extraction_schema(),
        };

        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .client
            .post(&url)
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

        let chat_response: OllamaChatResponse = response
            .json()
            .await
            .map_err(|e| ExtractionError::Generation(format!("Failed to parse Ollama response: {}", e)))?;

        // The content field is a JSON string — parse it into ExtractionOutput
        let output: ExtractionOutput = serde_json::from_str(&chat_response.message.content)
            .map_err(|e| ExtractionError::Generation(format!(
                "Failed to parse extraction JSON from model output: {} (content: {})",
                e, &chat_response.message.content
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
