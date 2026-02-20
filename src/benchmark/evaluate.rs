/// GPT-4o API calling functions for the LongMemEval benchmark pipeline.
///
/// Provides answer generation from retrieved memories and binary judge scoring
/// with exponential backoff retry on rate limits (429) and server errors (5xx).

use reqwest::Client;
use serde_json::json;
use std::time::Duration;

use crate::store::Memory;

use super::prompts;

const OPENAI_CHAT_URL: &str = "https://api.openai.com/v1/chat/completions";
const JUDGE_MODEL: &str = "gpt-4o-2024-08-06";
const ANSWER_MODEL: &str = "gpt-4o-2024-08-06";
const MAX_RETRIES: u32 = 5;

/// Generate an answer from retrieved memories using GPT-4o.
pub async fn generate_answer(
    client: &Client,
    api_key: &str,
    question: &str,
    question_date: &str,
    retrieved_memories: &[Memory],
) -> Result<String, anyhow::Error> {
    let prompt = prompts::build_answer_prompt(question, question_date, retrieved_memories);

    let body = json!({
        "model": ANSWER_MODEL,
        "temperature": 0,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": prompt}]
    });

    let response_text = call_openai_with_retry(client, api_key, &body).await?;
    Ok(response_text)
}

/// Judge whether the hypothesis correctly answers the question using GPT-4o.
/// Returns true if the answer is judged correct.
pub async fn judge_answer(
    client: &Client,
    api_key: &str,
    question: &str,
    ground_truth: &str,
    hypothesis: &str,
    is_abstention: bool,
) -> Result<bool, anyhow::Error> {
    let prompt = if is_abstention {
        prompts::build_abstention_judge_prompt(question, hypothesis)
    } else {
        prompts::build_judge_prompt(question, ground_truth, hypothesis)
    };

    let body = json!({
        "model": JUDGE_MODEL,
        "temperature": 0,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": prompt}]
    });

    let response_text = call_openai_with_retry(client, api_key, &body).await?;
    Ok(response_text.to_lowercase().contains("yes"))
}

/// Call OpenAI API with exponential backoff retry on rate limits (429) and server errors (5xx).
async fn call_openai_with_retry(
    client: &Client,
    api_key: &str,
    body: &serde_json::Value,
) -> Result<String, anyhow::Error> {
    for attempt in 0..MAX_RETRIES {
        let resp = client
            .post(OPENAI_CHAT_URL)
            .bearer_auth(api_key)
            .json(body)
            .send()
            .await?;

        let status = resp.status();

        if status.is_success() {
            let json: serde_json::Value = resp.json().await?;
            let text = json["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            return Ok(text);
        }

        if status.as_u16() == 429 || status.is_server_error() {
            let delay = Duration::from_secs(2u64.pow(attempt));
            tracing::warn!(
                attempt = attempt + 1,
                status = %status,
                delay_secs = delay.as_secs(),
                "OpenAI API error, retrying with backoff"
            );
            tokio::time::sleep(delay).await;
            continue;
        }

        // Non-retryable error
        let error_body = resp.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!(
            "OpenAI API error {}: {}",
            status,
            error_body
        ));
    }

    Err(anyhow::anyhow!(
        "OpenAI API failed after {} retries",
        MAX_RETRIES
    ))
}
