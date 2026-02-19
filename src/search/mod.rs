pub mod salience;

// Re-export key types for convenience
pub use salience::{SalienceScorer, ScoredHit, ScoreBreakdown};

use crate::store::Memory;

/// A raw fused search hit before salience re-ranking.
///
/// Produced by hybrid_search() on PostgresMemoryStore.
/// The salience re-ranking step (in server.rs) converts these to ScoredHit.
#[derive(Debug, Clone)]
pub struct HybridRawHit {
    pub memory: Memory,
    /// Reciprocal Rank Fusion score (sum of 1/(k + rank) over all legs)
    pub rrf_score: f64,
    /// Which legs this result appeared in.
    ///
    /// Source bit flags: 1=bm25, 2=vector, 4=symbolic. Combined values:
    /// - "all_three" (7): appeared in all three legs
    /// - "vector_symbolic" (6): vector + symbolic
    /// - "bm25_symbolic" (5): bm25 + symbolic
    /// - "hybrid" (3): bm25 + vector (legacy two-leg name preserved)
    /// - "symbolic_only" (4): symbolic only
    /// - "vector_only" (2): vector only
    /// - "bm25_only" (1): bm25 only
    pub match_source: String,
}

/// Fuse BM25, vector, and symbolic ranked lists via Reciprocal Rank Fusion (RRF).
///
/// RRF score for each document = sum of 1/(k_i + rank_i) over each retrieval leg i.
/// Documents appearing in multiple legs score higher than single-leg results.
///
/// Per-leg k values control top-result influence (lower k = more top-result influence):
/// - Default: bm25_k=60.0, vector_k=60.0 (research default)
/// - symbolic_k=40.0 (lower = exact metadata matches have stronger signal)
///
/// Passing an empty slice for any leg gracefully omits that leg from fusion.
///
/// # Arguments
/// - `bm25_ranks`: (id, rank) pairs from search_bm25 — rank is 1-based position
/// - `vector_ranks`: (id, rank) pairs from search_similar — rank is 1-based position
/// - `symbolic_ranks`: (id, rank) pairs from search_symbolic — rank is 1-based position
/// - `bm25_k`: RRF smoothing constant for BM25 leg
/// - `vector_k`: RRF smoothing constant for vector leg
/// - `symbolic_k`: RRF smoothing constant for symbolic leg
///
/// # Returns
/// Vec of (id, rrf_score, match_source) sorted by rrf_score descending.
pub fn rrf_fuse(
    bm25_ranks: &[(String, i64)],
    vector_ranks: &[(String, i64)],
    symbolic_ranks: &[(String, i64)],
    bm25_k: f64,
    vector_k: f64,
    symbolic_k: f64,
) -> Vec<(String, f64, String)> {
    use std::collections::HashMap;

    // Track RRF score and which legs each ID appeared in (bit flags: 1=bm25, 2=vector, 4=symbolic)
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut sources: HashMap<String, u8> = HashMap::new();

    for (id, rank) in bm25_ranks {
        *scores.entry(id.clone()).or_default() += 1.0 / (bm25_k + *rank as f64);
        *sources.entry(id.clone()).or_default() |= 1;
    }
    for (id, rank) in vector_ranks {
        *scores.entry(id.clone()).or_default() += 1.0 / (vector_k + *rank as f64);
        *sources.entry(id.clone()).or_default() |= 2;
    }
    for (id, rank) in symbolic_ranks {
        *scores.entry(id.clone()).or_default() += 1.0 / (symbolic_k + *rank as f64);
        *sources.entry(id.clone()).or_default() |= 4;
    }

    let mut result: Vec<(String, f64, String)> = scores
        .into_iter()
        .map(|(id, score)| {
            let source_bits = sources.get(&id).copied().unwrap_or(0);
            let source = match source_bits {
                7 => "all_three".to_string(),
                6 => "vector_symbolic".to_string(),
                5 => "bm25_symbolic".to_string(),
                3 => "hybrid".to_string(),       // bm25 + vector (legacy name preserved)
                4 => "symbolic_only".to_string(),
                2 => "vector_only".to_string(),
                1 => "bm25_only".to_string(),
                _ => "unknown".to_string(),
            };
            (id, score, source)
        })
        .collect();

    // Sort by RRF score descending (higher = more relevant)
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}
