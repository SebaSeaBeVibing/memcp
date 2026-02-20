/// OpenAI-compatible query intelligence provider
///
/// Calls any OpenAI-compatible Chat Completions API with json_object response format.
/// The base_url is configurable — supports OpenAI, Kimi Code API, and any compatible endpoint.
/// Requires an API key.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{
    ExpandedQuery, QueryIntelligenceError, QueryIntelligenceProvider, RankedCandidate,
    RankedResult, TimeRange, build_expansion_prompt, build_reranking_prompt,
};

// --- HTTP request/response structs (local — mirrors extraction/openai.rs pattern) ---

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

/// OpenAI-compatible query intelligence provider.
///
/// Uses the chat completions API with json_object response format.
/// base_url is configurable — not hardcoded — enabling Kimi Code API and other
/// OpenAI-compatible endpoints in addition to api.openai.com.
pub struct OpenAIQueryIntelligenceProvider {
    client: reqwest::Client,
    /// Configurable base URL — supports Kimi and other OpenAI-compatible APIs
    base_url: String,
    api_key: String,
    model: String,
}

impl OpenAIQueryIntelligenceProvider {
    /// Create a new OpenAIQueryIntelligenceProvider.
    ///
    /// # Arguments
    /// * `base_url` - API base URL (e.g., "https://api.openai.com/v1" or Kimi endpoint)
    /// * `api_key` - API key (must be non-empty)
    /// * `model` - Model name (e.g., "gpt-4o-mini")
    ///
    /// # Errors
    /// Returns `QueryIntelligenceError::NotConfigured` if api_key is empty.
    pub fn new(
        base_url: String,
        api_key: String,
        model: String,
    ) -> Result<Self, QueryIntelligenceError> {
        if api_key.trim().is_empty() {
            return Err(QueryIntelligenceError::NotConfigured(
                "OpenAI API key is required when using the openai query intelligence provider. \
                 Set the api_key in QueryIntelligenceConfig"
                    .to_string(),
            ));
        }

        Ok(OpenAIQueryIntelligenceProvider {
            client: reqwest::Client::new(),
            base_url,
            api_key,
            model,
        })
    }

    /// POST to {base_url}/chat/completions with json_object response format.
    async fn chat(&self, prompt: String) -> Result<String, QueryIntelligenceError> {
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

        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
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

        let chat_response: ChatResponse = response.json().await.map_err(|e| {
            QueryIntelligenceError::Generation(format!("Failed to parse OpenAI response: {}", e))
        })?;

        let content = chat_response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| {
                QueryIntelligenceError::Generation(
                    "OpenAI returned empty choices list".to_string(),
                )
            })?;

        Ok(content)
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
impl QueryIntelligenceProvider for OpenAIQueryIntelligenceProvider {
    async fn expand(&self, query: &str) -> Result<ExpandedQuery, QueryIntelligenceError> {
        let current_date = Utc::now().format("%Y-%m-%d").to_string();
        let prompt = build_expansion_prompt(query, &current_date);

        let content = self.chat(prompt).await?;

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
        let content = self.chat(prompt).await?;

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
