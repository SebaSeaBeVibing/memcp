/// Benchmark runner orchestrator for the LongMemEval evaluation pipeline.
///
/// Runs the full per-question pipeline: truncate -> ingest -> search -> generate -> score.
/// Supports checkpoint/resume so interrupted runs can continue from where they left off.
/// Config matrix enables comparison of search weight configurations.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;

use crate::embedding::pipeline::EmbeddingPipeline;
use crate::embedding::EmbeddingProvider;
use crate::store::postgres::PostgresMemoryStore;

use super::dataset::LongMemEvalQuestion;
use super::{evaluate, BenchmarkConfig, BenchmarkState, QuestionResult};
use super::ingest::ingest_question;

/// Run benchmark for a single configuration across all questions.
///
/// For each question:
/// 1. Truncate all data (clean slate for database isolation)
/// 2. Ingest question's haystack sessions as memories (with temporal timestamps)
/// 3. Search using configured weights (BM25/vector/symbolic via hybrid_search)
/// 4. Generate answer from retrieved memories via GPT-4o
/// 5. Judge answer correctness via GPT-4o (binary yes/no)
/// 6. Save checkpoint after each question (for resume support)
///
/// Returns Vec of QuestionResult for all questions processed.
pub async fn run_benchmark(
    questions: &[LongMemEvalQuestion],
    config: &BenchmarkConfig,
    store: Arc<PostgresMemoryStore>,
    pipeline: &EmbeddingPipeline,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    openai_api_key: &str,
    checkpoint_path: &std::path::Path,
    resume_state: Option<BenchmarkState>,
) -> Result<Vec<QuestionResult>, anyhow::Error> {
    let client = Client::new();

    // Initialize or restore state from resume checkpoint
    let mut state = resume_state.unwrap_or_else(|| BenchmarkState {
        config_name: config.name.clone(),
        completed_question_ids: Vec::new(),
        results: Vec::new(),
        started_at: chrono::Utc::now(),
    });

    // O(1) lookup for already-completed questions
    let completed: HashSet<String> = state.completed_question_ids.iter().cloned().collect();

    // Progress bar showing question id and ETA
    let pb = ProgressBar::new(questions.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{pos}/{len}] {msg} [{elapsed_precise} / {eta_precise}]")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );

    // Advance progress bar to reflect already-completed questions from resume
    pb.set_position(completed.len() as u64);

    for question in questions {
        // Skip already-completed questions (resume support)
        if completed.contains(&question.question_id) {
            continue;
        }

        pb.set_message(question.question_id.clone());

        let start = Instant::now();

        // Step 1: Clean slate — truncate all memories for database isolation per question
        store.truncate_all().await?;

        // Step 2: Ingest haystack sessions as memories with temporal timestamps
        ingest_question(question, &store, pipeline).await?;

        // Step 3: Search with configured weights
        // Map config weights to hybrid_search k parameters:
        // weight > 0.0 → Some(k) enables the leg; 0.0 → None disables it
        let bm25_k = if config.bm25_weight > 0.0 {
            Some(60.0f64)
        } else {
            None
        };
        let vector_k = if config.vector_weight > 0.0 {
            Some(60.0f64)
        } else {
            None
        };
        let symbolic_k = if config.symbolic_weight > 0.0 {
            Some(40.0f64)
        } else {
            None
        };

        // Embed the question for vector search leg; fall back to BM25-only if embedding fails
        let query_embedding = if vector_k.is_some() {
            match embedding_provider.embed(&question.question).await {
                Ok(vec) => Some(pgvector::Vector::from(vec)),
                Err(e) => {
                    tracing::warn!(
                        question_id = %question.question_id,
                        error = %e,
                        "Failed to embed question — falling back to BM25-only"
                    );
                    None
                }
            }
        } else {
            None
        };

        let hits = store
            .hybrid_search(
                &question.question,
                query_embedding.as_ref(),
                20,    // fetch 20 candidates from fused results
                None,  // no date filters for benchmark
                None,
                None,  // no tag filters
                bm25_k,
                vector_k,
                symbolic_k,
            )
            .await?;

        // Take top 10 memories for answer generation (fits context window)
        let memories: Vec<_> = hits.into_iter().take(10).map(|h| h.memory).collect();
        let retrieved_count = memories.len();

        // Step 4: Generate answer from retrieved memories via GPT-4o
        let hypothesis = evaluate::generate_answer(
            &client,
            openai_api_key,
            &question.question,
            &question.question_date,
            &memories,
        )
        .await?;

        // Step 5: Judge answer correctness via GPT-4o (binary yes/no)
        let correct = evaluate::judge_answer(
            &client,
            openai_api_key,
            &question.question,
            &question.answer_text(),
            &hypothesis,
            question.is_abstention(),
        )
        .await?;

        let latency_ms = start.elapsed().as_millis() as u64;

        // Build result
        let result = QuestionResult {
            question_id: question.question_id.clone(),
            question_type: question.question_type.clone(),
            is_abstention: question.is_abstention(),
            correct,
            hypothesis,
            ground_truth: question.answer_text(),
            retrieved_count,
            latency_ms,
        };

        // Update checkpoint state
        state.completed_question_ids.push(question.question_id.clone());
        state.results.push(result.clone());

        // Save checkpoint after each question so interrupted runs can resume
        save_checkpoint(&state, checkpoint_path)?;

        pb.inc(1);
    }

    pb.finish_with_message("done");

    Ok(state.results)
}

/// Save benchmark state as JSON to the given path for checkpoint/resume support.
fn save_checkpoint(state: &BenchmarkState, path: &std::path::Path) -> Result<(), anyhow::Error> {
    let json = serde_json::to_string_pretty(state)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load a benchmark checkpoint from disk. Returns None if the file does not exist.
pub fn load_checkpoint(path: &std::path::Path) -> Result<Option<BenchmarkState>, anyhow::Error> {
    if path.exists() {
        let json = std::fs::read_to_string(path)?;
        let state: BenchmarkState = serde_json::from_str(&json)?;
        Ok(Some(state))
    } else {
        Ok(None)
    }
}
