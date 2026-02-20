/// Similarity search for consolidation candidate finding.
///
/// Queries pgvector for memories with cosine similarity above a threshold.
/// Excludes the source memory itself and any memories already marked as originals
/// (to avoid cascading consolidations).

use crate::errors::MemcpError;

/// A memory that is similar enough to be a consolidation candidate.
#[derive(Debug, Clone)]
pub struct SimilarMemory {
    /// The memory ID of the similar memory.
    pub memory_id: String,
    /// Cosine similarity score (0.0–1.0, higher = more similar).
    pub similarity: f64,
    /// Content of the similar memory (for LLM synthesis).
    pub content: String,
}

/// Find memories similar to the given embedding above the specified threshold.
///
/// Excludes:
/// - The memory itself (`memory_id != $2`)
/// - Memories already marked as consolidated originals (`is_consolidated_original = FALSE`)
/// - Memories that haven't been embedded yet (`embedding_status = 'complete'`)
///
/// Returns at most `limit` results, ordered by descending similarity.
pub async fn find_similar_memories(
    pool: &sqlx::PgPool,
    memory_id: &str,
    embedding: &pgvector::Vector,
    threshold: f64,
    limit: i64,
) -> Result<Vec<SimilarMemory>, MemcpError> {
    let rows = sqlx::query(
        "SELECT me.memory_id,
            (1 - (me.embedding <=> $1)) AS cosine_similarity,
            m.content
         FROM memory_embeddings me
         JOIN memories m ON m.id = me.memory_id
         WHERE me.is_current = TRUE
           AND m.embedding_status = 'complete'
           AND m.is_consolidated_original = FALSE
           AND me.memory_id != $2
           AND (1 - (me.embedding <=> $1)) >= $3
         ORDER BY cosine_similarity DESC
         LIMIT $4",
    )
    .bind(embedding)
    .bind(memory_id)
    .bind(threshold)
    .bind(limit)
    .fetch_all(pool)
    .await
    .map_err(|e| MemcpError::Storage(format!("Similarity search failed: {}", e)))?;

    let mut results = Vec::with_capacity(rows.len());
    for row in &rows {
        use sqlx::Row;
        let mid: String = row.try_get("memory_id").map_err(|e| MemcpError::Storage(e.to_string()))?;
        let sim: f64 = row.try_get("cosine_similarity").map_err(|e| MemcpError::Storage(e.to_string()))?;
        let content: String = row.try_get("content").map_err(|e| MemcpError::Storage(e.to_string()))?;
        results.push(SimilarMemory { memory_id: mid, similarity: sim, content });
    }

    Ok(results)
}
