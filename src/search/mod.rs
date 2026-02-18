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
    /// Which legs this result appeared in: "hybrid", "bm25_only", or "vector_only"
    pub match_source: String,
}

/// Fuse BM25 and vector ranked lists via Reciprocal Rank Fusion (RRF).
///
/// RRF score for each document = sum of 1/(k + rank_i) over each retrieval leg i.
/// Documents appearing in both legs score higher than single-leg results.
///
/// # Arguments
/// - `bm25_ranks`: (id, rank) pairs from search_bm25 — rank is 1-based position
/// - `vector_ranks`: (id, rank) pairs from search_similar — rank is 1-based position
/// - `k`: smoothing constant (default 60.0 from research; reduces impact of top-1 dominance)
///
/// # Returns
/// Vec of (id, rrf_score, match_source) sorted by rrf_score descending.
pub fn rrf_fuse(
    bm25_ranks: &[(String, i64)],
    vector_ranks: &[(String, i64)],
    k: f64,
) -> Vec<(String, f64, String)> {
    use std::collections::HashMap;

    // Track RRF score and which legs each ID appeared in (bit flags: 1=bm25, 2=vector)
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut sources: HashMap<String, u8> = HashMap::new();

    for (id, rank) in bm25_ranks {
        *scores.entry(id.clone()).or_default() += 1.0 / (k + *rank as f64);
        *sources.entry(id.clone()).or_default() |= 1;
    }
    for (id, rank) in vector_ranks {
        *scores.entry(id.clone()).or_default() += 1.0 / (k + *rank as f64);
        *sources.entry(id.clone()).or_default() |= 2;
    }

    let mut result: Vec<(String, f64, String)> = scores
        .into_iter()
        .map(|(id, score)| {
            let source_bits = sources.get(&id).copied().unwrap_or(0);
            let source = match source_bits {
                3 => "hybrid".to_string(),
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
