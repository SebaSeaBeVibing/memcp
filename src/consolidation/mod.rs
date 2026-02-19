/// Memory consolidation module.
///
/// Non-destructive memory deduplication pipeline:
/// 1. After a memory is embedded, check pgvector for similar memories.
/// 2. If similarity exceeds threshold (default 0.92), synthesize a consolidated memory via LLM.
/// 3. Link originals to the consolidated memory via the memory_consolidations table.
/// 4. Mark originals as `is_consolidated_original = TRUE` so search suppresses them.
///
/// Consolidation is triggered via an mpsc channel from the embedding pipeline.
/// The background worker processes jobs asynchronously — store_memory never blocks.

pub mod similarity;

use std::sync::Arc;
use tokio::sync::mpsc;

use crate::config::ConsolidationConfig;
use crate::store::postgres::PostgresMemoryStore;
use similarity::find_similar_memories;

/// A pending consolidation job.
///
/// Created by the embedding pipeline after successful embedding storage.
#[derive(Debug, Clone)]
pub struct ConsolidationJob {
    /// The memory ID that was just embedded.
    pub memory_id: String,
    /// The embedding vector (used directly for similarity search).
    pub embedding: pgvector::Vector,
    /// The content of the newly embedded memory (for synthesis).
    pub content: String,
}

/// Background consolidation worker.
///
/// Receives jobs from the embedding pipeline via mpsc channel.
/// For each job: checks similarity, and if matches found, calls Ollama/OpenAI to synthesize
/// a consolidated memory, then creates the consolidation record atomically.
pub struct ConsolidationWorker {
    sender: mpsc::Sender<ConsolidationJob>,
}

impl ConsolidationWorker {
    /// Create a new ConsolidationWorker and spawn the background task.
    ///
    /// - `store`: PostgresMemoryStore for DB operations.
    /// - `config`: ConsolidationConfig (threshold, max group size).
    /// - `ollama_base_url`: Ollama base URL for synthesis (e.g., "http://localhost:11434").
    /// - `ollama_model`: Model to use for synthesis (e.g., "llama3.2:3b").
    /// - `capacity`: Bounded channel capacity (recommended: 500).
    pub fn new(
        store: Arc<PostgresMemoryStore>,
        config: ConsolidationConfig,
        ollama_base_url: String,
        ollama_model: String,
        capacity: usize,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<ConsolidationJob>(capacity);

        let client = reqwest::Client::new();

        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                let pool = store.pool();

                // Find similar memories above threshold
                let similar = match find_similar_memories(
                    pool,
                    &job.memory_id,
                    &job.embedding,
                    config.similarity_threshold,
                    config.max_consolidation_group as i64,
                )
                .await
                {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(
                            memory_id = %job.memory_id,
                            error = %e,
                            "Similarity search failed during consolidation check"
                        );
                        continue;
                    }
                };

                if similar.is_empty() {
                    tracing::debug!(
                        memory_id = %job.memory_id,
                        "No similar memories found — skipping consolidation"
                    );
                    continue;
                }

                tracing::info!(
                    memory_id = %job.memory_id,
                    similar_count = similar.len(),
                    "Similar memories found — consolidating"
                );

                // Collect all contents for synthesis
                let mut all_contents: Vec<&str> = vec![job.content.as_str()];
                for s in &similar {
                    all_contents.push(s.content.as_str());
                }

                // Synthesize consolidated content via LLM (fallback: concatenation)
                let synthesized = match synthesize_memories(
                    &client,
                    &ollama_base_url,
                    &ollama_model,
                    &all_contents,
                )
                .await
                {
                    Ok(text) => text,
                    Err(e) => {
                        tracing::warn!(
                            memory_id = %job.memory_id,
                            error = %e,
                            "LLM synthesis failed — using concatenation fallback"
                        );
                        concatenate_memories(&all_contents)
                    }
                };

                // Collect source IDs and similarity scores (new memory gets similarity 1.0)
                let mut source_ids: Vec<String> = vec![job.memory_id.clone()];
                let mut similarities: Vec<f64> = vec![1.0];
                for s in &similar {
                    source_ids.push(s.memory_id.clone());
                    similarities.push(s.similarity);
                }

                // Atomically create consolidated memory + links + mark originals
                match store.create_consolidated_memory(&synthesized, &source_ids, &similarities).await {
                    Ok(consolidated_id) => {
                        tracing::info!(
                            consolidated_id = %consolidated_id,
                            source_count = source_ids.len(),
                            "Memory consolidation complete"
                        );
                    }
                    Err(e) => {
                        // UNIQUE constraint violation = already consolidated — safe to ignore
                        let msg = e.to_string();
                        if msg.contains("duplicate key") || msg.contains("unique") || msg.contains("23505") {
                            tracing::debug!(
                                memory_id = %job.memory_id,
                                "Consolidation already exists (idempotent) — skipping"
                            );
                        } else {
                            tracing::error!(
                                memory_id = %job.memory_id,
                                error = %e,
                                "Failed to create consolidated memory"
                            );
                        }
                    }
                }
            }
        });

        ConsolidationWorker { sender: tx }
    }

    /// Return a clone of the underlying sender for use in the embedding pipeline.
    pub fn sender(&self) -> mpsc::Sender<ConsolidationJob> {
        self.sender.clone()
    }
}

/// Build the synthesis prompt for LLM consolidation.
fn build_synthesis_prompt(contents: &[&str]) -> String {
    let mut prompt = "Synthesize these related memories into one comprehensive memory. \
        Preserve all unique facts, preferences, and specific details. \
        Do not add information not present in the originals. \
        Write a single cohesive paragraph.\n\n"
        .to_string();
    for (i, content) in contents.iter().enumerate() {
        prompt.push_str(&format!("Memory {}:\n{}\n\n", i + 1, content));
    }
    prompt.push_str("Synthesized memory:");
    prompt
}

/// Concatenate memories as a fallback when LLM synthesis fails.
fn concatenate_memories(contents: &[&str]) -> String {
    contents
        .iter()
        .enumerate()
        .map(|(i, c)| format!("Memory {}:\n{}", i + 1, c))
        .collect::<Vec<_>>()
        .join("\n---\n")
}

/// Synthesis error for internal use.
#[derive(Debug)]
enum SynthesisError {
    Http(String),
    Parse(String),
}

impl std::fmt::Display for SynthesisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SynthesisError::Http(e) => write!(f, "HTTP error: {}", e),
            SynthesisError::Parse(e) => write!(f, "Parse error: {}", e),
        }
    }
}

/// Ollama request for free-form synthesis (no format schema — want plain text).
#[derive(serde::Serialize)]
struct OllamaSynthesisRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    options: OllamaOptions,
}

#[derive(serde::Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(serde::Serialize)]
struct OllamaOptions {
    temperature: f32,
}

#[derive(serde::Deserialize)]
struct OllamaSynthesisResponse {
    message: OllamaResponseMessage,
}

#[derive(serde::Deserialize)]
struct OllamaResponseMessage {
    content: String,
}

/// Call Ollama for free-form text synthesis of consolidated memory.
///
/// No `format` field (unlike extraction) — we want plain text, not structured JSON.
/// Falls through to `Err` on any failure; caller falls back to concatenation.
async fn synthesize_memories(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    contents: &[&str],
) -> Result<String, SynthesisError> {
    let prompt = build_synthesis_prompt(contents);

    let request = OllamaSynthesisRequest {
        model: model.to_string(),
        messages: vec![OllamaMessage {
            role: "user".to_string(),
            content: prompt,
        }],
        stream: false,
        options: OllamaOptions { temperature: 0.2 },
    };

    let url = format!("{}/api/chat", base_url);

    let response = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .map_err(|e| SynthesisError::Http(format!("Request failed: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "unknown error".to_string());
        return Err(SynthesisError::Http(format!("Status {}: {}", status, body)));
    }

    let chat_response: OllamaSynthesisResponse = response
        .json()
        .await
        .map_err(|e| SynthesisError::Parse(format!("Failed to parse Ollama response: {}", e)))?;

    let text = chat_response.message.content.trim().to_string();
    if text.is_empty() {
        return Err(SynthesisError::Parse("Empty synthesis response".to_string()));
    }

    Ok(text)
}

