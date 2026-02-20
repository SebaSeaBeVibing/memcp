/// Benchmark module for LongMemEval evaluation pipeline.
///
/// Provides dataset types, ingestion logic, and shared result types
/// for benchmarking the memcp search pipeline against the LongMemEval dataset.

pub mod dataset;
pub mod evaluate;
pub mod ingest;
pub mod prompts;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Configuration for a benchmark run. Controls search weights and QI features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub name: String,
    pub bm25_weight: f64,
    pub vector_weight: f64,
    pub symbolic_weight: f64,
    pub qi_expansion: bool,
    pub qi_reranking: bool,
}

/// Predefined configurations for comparison runs.
pub fn default_configs() -> Vec<BenchmarkConfig> {
    vec![
        BenchmarkConfig {
            name: "vector-only".into(),
            bm25_weight: 0.0,
            vector_weight: 1.0,
            symbolic_weight: 0.0,
            qi_expansion: false,
            qi_reranking: false,
        },
        BenchmarkConfig {
            name: "hybrid".into(),
            bm25_weight: 1.0,
            vector_weight: 1.0,
            symbolic_weight: 1.0,
            qi_expansion: false,
            qi_reranking: false,
        },
        BenchmarkConfig {
            name: "hybrid+qi".into(),
            bm25_weight: 1.0,
            vector_weight: 1.0,
            symbolic_weight: 1.0,
            qi_expansion: true,
            qi_reranking: true,
        },
    ]
}

/// Result for a single benchmark question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionResult {
    pub question_id: String,
    pub question_type: String,
    pub is_abstention: bool,
    pub correct: bool,
    pub hypothesis: String,
    pub ground_truth: String,
    pub retrieved_count: usize,
    pub latency_ms: u64,
}

/// Checkpoint state for resumable benchmark runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkState {
    pub config_name: String,
    pub completed_question_ids: Vec<String>,
    pub results: Vec<QuestionResult>,
    pub started_at: DateTime<Utc>,
}
