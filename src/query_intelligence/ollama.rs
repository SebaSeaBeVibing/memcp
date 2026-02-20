/// Ollama query intelligence provider
///
/// Calls the Ollama /api/chat endpoint with structured JSON output schema.
/// Supports both query expansion (with temporal hint extraction) and candidate re-ranking.
/// No API key required — designed for self-hosted Ollama deployments.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{
    ExpandedQuery, QueryIntelligenceError, QueryIntelligenceProvider, RankedCandidate,
    RankedResult, TimeRange, build_expansion_prompt, build_reranking_prompt, expansion_schema,
    reranking_schema,
};

// --- HTTP request/response structs (local — mirrors extraction/ollama.rs pattern) ---

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

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaResponseMessage,
}

#[derive(Deserialize)]
struct OllamaResponseMessage {
    content: String,
}

// --- JSON output structs ---

/// Parsed query expansion output from LLM
#[derive(Deserialize)]
struct ExpandedQueryOutput {
    #[serde(default)]
    variants: Vec<String>,
    time_range: Option<TimeRangeOutput>,
}

#[derive(Deserialize)]
struct TimeRangeOutput {
    after: Option<String>,
    before: Option<String>,
}

/// Parsed re-ranking output from LLM
#[derive(Deserialize)]
struct RerankOutput {
    #[serde(default)]
    ranked_ids: Vec<String>,
}

// --- Provider ---

/// Ollama-backed query intelligence provider.
///
/// Uses /api/chat with structured JSON output (format field) for both
/// query expansion and result re-ranking.
pub struct OllamaQueryIntelligenceProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaQueryIntelligenceProvider {
    /// Create a new OllamaQueryIntelligenceProvider.
    ///
    /// # Arguments
    /// * `base_url` - Ollama server base URL (e.g., "http://localhost:11434")
    /// * `model` - Model name (e.g., "llama3.2:3b")
    pub fn new(base_url: String, model: String) -> Self {
        OllamaQueryIntelligenceProvider {
            client: reqwest::Client::new(),
            base_url,
            model,
        }
    }

    /// POST to Ollama /api/chat with a given prompt and schema, return content string.
    async fn chat(
        &self,
        prompt: String,
        schema: serde_json::Value,
    ) -> Result<String, QueryIntelligenceError> {
        let request = OllamaChatRequest {
            model: self.model.clone(),
            messages: vec![OllamaMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            stream: false,
            options: OllamaOptions { temperature: 0.0 },
            format: schema,
        };

        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                QueryIntelligenceError::Generation(format!("HTTP request failed: {}", e))
            })?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(QueryIntelligenceError::Api { status, message: body });
        }

        let chat_response: OllamaChatResponse = response.json().await.map_err(|e| {
            QueryIntelligenceError::Generation(format!("Failed to parse Ollama response: {}", e))
        })?;

        Ok(chat_response.message.content)
    }
}

/// Parse an optional RFC-3339 string to DateTime<Utc>, returning None on failure.
fn parse_datetime_opt(s: Option<String>) -> Option<DateTime<Utc>> {
    s.and_then(|raw| {
        DateTime::parse_from_rfc3339(&raw)
            .ok()
            .map(|dt| dt.with_timezone(&Utc))
    })
}

#[async_trait]
impl QueryIntelligenceProvider for OllamaQueryIntelligenceProvider {
    async fn expand(&self, query: &str) -> Result<ExpandedQuery, QueryIntelligenceError> {
        let current_date = Utc::now().format("%Y-%m-%d").to_string();
        let prompt = build_expansion_prompt(query, &current_date);

        let content = self.chat(prompt, expansion_schema()).await?;

        let output: ExpandedQueryOutput = serde_json::from_str(&content).map_err(|e| {
            QueryIntelligenceError::Generation(format!(
                "Failed to parse expansion JSON from model output: {} (content: {})",
                e, &content
            ))
        })?;

        if output.variants.is_empty() {
            return Err(QueryIntelligenceError::Generation(
                "LLM returned no query variants".to_string(),
            ));
        }

        let time_range = output.time_range.map(|tr| TimeRange {
            after: parse_datetime_opt(tr.after),
            before: parse_datetime_opt(tr.before),
        });

        Ok(ExpandedQuery {
            variants: output.variants,
            time_range,
        })
    }

    async fn rerank(
        &self,
        query: &str,
        candidates: &[RankedCandidate],
    ) -> Result<Vec<RankedResult>, QueryIntelligenceError> {
        // Serialize candidates as JSON array for prompt
        let candidates_json = {
            let arr: Vec<serde_json::Value> = candidates
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "id": c.id,
                        "content": c.content,
                        "rank": c.current_rank
                    })
                })
                .collect();
            serde_json::to_string(&arr).map_err(|e| {
                QueryIntelligenceError::Generation(format!(
                    "Failed to serialize candidates: {}",
                    e
                ))
            })?
        };

        let prompt = build_reranking_prompt(query, &candidates_json);
        let content = self.chat(prompt, reranking_schema()).await?;

        let output: RerankOutput = serde_json::from_str(&content).map_err(|e| {
            QueryIntelligenceError::Generation(format!(
                "Failed to parse rerank JSON from model output: {} (content: {})",
                e, &content
            ))
        })?;

        // Build a set of valid candidate IDs for defensive filtering
        let valid_ids: std::collections::HashSet<&str> =
            candidates.iter().map(|c| c.id.as_str()).collect();

        // Convert to RankedResult — llm_rank is 1-based index position
        let results: Vec<RankedResult> = output
            .ranked_ids
            .into_iter()
            .filter(|id| valid_ids.contains(id.as_str()))
            .enumerate()
            .map(|(idx, id)| RankedResult {
                id,
                llm_rank: idx + 1,
            })
            .collect();

        Ok(results)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
