/// Async embedding pipeline with bounded mpsc channel and background worker.
///
/// Non-blocking design: store_memory never waits for embedding completion.
/// Failed embeddings are retried up to 3 times with exponential backoff (1s, 2s, 4s),
/// then marked as failed for backfill on next startup.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::{EmbeddingJob, EmbeddingProvider, build_embedding_text};
use crate::consolidation::ConsolidationJob;
use crate::store::MemoryStore;
use crate::store::postgres::PostgresMemoryStore;

/// Async embedding pipeline: enqueues jobs onto a bounded mpsc channel and
/// processes them in a background tokio task.
pub struct EmbeddingPipeline {
    sender: mpsc::Sender<EmbeddingJob>,
    /// Count of jobs currently in-flight (enqueued but not yet completed).
    /// Used by flush() to block until the pipeline drains.
    pending_count: Arc<AtomicUsize>,
}

impl EmbeddingPipeline {
    /// Create a new EmbeddingPipeline and spawn the background worker.
    ///
    /// - `provider`: The embedding provider to call for each job.
    /// - `store`: The PostgresMemoryStore for storing embeddings and updating status.
    /// - `capacity`: Bounded channel capacity (recommended: 1000).
    /// - `consolidation_sender`: Optional channel to the consolidation worker. When provided,
    ///   each successfully embedded memory triggers a consolidation check via this channel.
    pub fn new(
        provider: Arc<dyn EmbeddingProvider>,
        store: Arc<PostgresMemoryStore>,
        capacity: usize,
        consolidation_sender: Option<mpsc::Sender<ConsolidationJob>>,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<EmbeddingJob>(capacity);
        // Clone tx for retry re-sends inside the worker
        let retry_tx = tx.clone();

        // Shared counter tracking jobs currently in-flight (enqueued but not completed).
        let pending_count = Arc::new(AtomicUsize::new(0));
        let worker_pending = Arc::clone(&pending_count);

        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                let text = job.text.clone();
                match provider.embed(&text).await {
                    Ok(vector) => {
                        let embedding = pgvector::Vector::from(vector);
                        let emb_id = Uuid::new_v4().to_string();
                        let model = provider.model_name().to_string();
                        let dim = provider.dimension() as i32;
                        if let Err(e) = store
                            .insert_embedding(&emb_id, &job.memory_id, &model, "v1", dim, &embedding, true)
                            .await
                        {
                            tracing::error!(
                                memory_id = %job.memory_id,
                                error = %e,
                                "Failed to store embedding"
                            );
                            // Storage error is not retryable — mark as failed
                            let _ = store.update_embedding_status(&job.memory_id, "failed").await;
                            worker_pending.fetch_sub(1, Ordering::Relaxed);
                        } else {
                            let _ = store.update_embedding_status(&job.memory_id, "complete").await;
                            tracing::debug!(memory_id = %job.memory_id, "Embedding complete");

                            // Trigger consolidation check after successful embedding.
                            // Consolidation requires the embedding to exist first (for cosine similarity).
                            // try_send is non-blocking — if the channel is full, skip consolidation for
                            // this memory (not critical, backfill does not apply here).
                            if let Some(ref consolidation_tx) = consolidation_sender {
                                // Fetch the memory content for synthesis if consolidation triggers
                                match store.get(&job.memory_id).await {
                                    Ok(memory) => {
                                        let _ = consolidation_tx.try_send(ConsolidationJob {
                                            memory_id: job.memory_id.clone(),
                                            embedding: embedding.clone(),
                                            content: memory.content,
                                        });
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            memory_id = %job.memory_id,
                                            error = %e,
                                            "Failed to fetch memory for consolidation job — skipping"
                                        );
                                    }
                                }
                            }
                            worker_pending.fetch_sub(1, Ordering::Relaxed);
                        }
                    }
                    Err(e) if job.attempt < 3 => {
                        tracing::warn!(
                            memory_id = %job.memory_id,
                            attempt = job.attempt + 1,
                            error = %e,
                            "Embedding failed, retrying"
                        );
                        // Exponential backoff: 1s, 2s, 4s
                        let delay = Duration::from_secs(2u64.pow(job.attempt as u32));
                        tokio::time::sleep(delay).await;
                        // Re-enqueue with incremented attempt (pending_count stays the same — job continues)
                        let _ = retry_tx.try_send(EmbeddingJob {
                            attempt: job.attempt + 1,
                            ..job
                        });
                    }
                    Err(e) => {
                        tracing::error!(
                            memory_id = %job.memory_id,
                            attempts = 3,
                            error = %e,
                            "Embedding failed after 3 retries, marking as failed"
                        );
                        let _ = store.update_embedding_status(&job.memory_id, "failed").await;
                        worker_pending.fetch_sub(1, Ordering::Relaxed);
                    }
                }
            }
        });

        EmbeddingPipeline { sender: tx, pending_count }
    }

    /// Enqueue an embedding job (non-blocking).
    ///
    /// Uses try_send — if the channel is full, the job is dropped and a warning is logged.
    /// The backfill process will pick up missed memories on next startup.
    pub fn enqueue(&self, job: EmbeddingJob) {
        self.pending_count.fetch_add(1, Ordering::Relaxed);
        if let Err(_) = self.sender.try_send(job) {
            // Job dropped — decrement since no worker will process it
            self.pending_count.fetch_sub(1, Ordering::Relaxed);
            tracing::warn!(
                "Embedding queue full — memory stored, embedding deferred to backfill"
            );
        }
    }

    /// Return a clone of the underlying mpsc sender (for use with the backfill function).
    pub fn sender(&self) -> mpsc::Sender<EmbeddingJob> {
        self.sender.clone()
    }

    /// Wait until all enqueued embedding jobs have completed (success or failure).
    /// Polls pending count every 100ms. Used by benchmark to ensure all embeddings
    /// are complete before running search.
    pub async fn flush(&self) {
        loop {
            let pending = self.pending_count.load(Ordering::Relaxed);
            if pending == 0 {
                break;
            }
            tracing::debug!(pending, "Waiting for embedding pipeline to flush");
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }
}

/// Queue all pending/failed memories for re-embedding.
///
/// Queries the store in batches of 100 and enqueues each memory on the pipeline channel.
/// Returns the total count of memories queued.
pub async fn backfill(
    store: &PostgresMemoryStore,
    sender: &mpsc::Sender<EmbeddingJob>,
) -> u64 {
    let mut total_queued: u64 = 0;

    loop {
        let pending = match store.get_pending_memories(100).await {
            Ok(memories) => memories,
            Err(e) => {
                tracing::error!(error = %e, "Failed to fetch pending memories for backfill");
                break;
            }
        };

        if pending.is_empty() {
            break;
        }

        let batch_size = pending.len() as u64;
        for memory in pending {
            let text = build_embedding_text(&memory.content, &memory.tags);
            let job = EmbeddingJob {
                memory_id: memory.id,
                text,
                attempt: 0,
            };
            if let Err(_) = sender.try_send(job) {
                tracing::warn!("Embedding queue full during backfill — some memories deferred");
                // Channel is full — stop trying this batch; next startup will continue
                return total_queued;
            }
        }

        total_queued += batch_size;

        // If we got fewer than 100, we've exhausted the pending set
        if batch_size < 100 {
            break;
        }
    }

    if total_queued > 0 {
        tracing::info!(count = total_queued, "Queued memories for embedding backfill");
    }

    total_queued
}
