/// Async extraction pipeline with bounded mpsc channel and background worker.
///
/// Non-blocking design: store_memory never waits for extraction completion.
/// Failed extractions are retried up to 3 times with exponential backoff (1s, 2s, 4s),
/// then marked as failed.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

use super::{ExtractionJob, ExtractionProvider};
use crate::store::postgres::PostgresMemoryStore;

/// Async extraction pipeline: enqueues jobs onto a bounded mpsc channel and
/// processes them in a background tokio task.
pub struct ExtractionPipeline {
    sender: mpsc::Sender<ExtractionJob>,
}

impl ExtractionPipeline {
    /// Create a new ExtractionPipeline and spawn the background worker.
    ///
    /// - `provider`: The extraction provider to call for each job.
    /// - `store`: The PostgresMemoryStore for storing results and updating status.
    /// - `capacity`: Bounded channel capacity (recommended: 1000).
    pub fn new(
        provider: Arc<dyn ExtractionProvider>,
        store: Arc<PostgresMemoryStore>,
        capacity: usize,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<ExtractionJob>(capacity);
        let retry_tx = tx.clone();

        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                let content = job.content.clone();
                match provider.extract(&content).await {
                    Ok(result) => {
                        if let Err(e) = store
                            .update_extraction_results(
                                &job.memory_id,
                                &result.entities,
                                &result.facts,
                            )
                            .await
                        {
                            tracing::error!(
                                memory_id = %job.memory_id,
                                error = %e,
                                "Failed to store extraction results"
                            );
                            let _ = store.update_extraction_status(&job.memory_id, "failed").await;
                        } else {
                            let _ = store.update_extraction_status(&job.memory_id, "complete").await;
                            tracing::debug!(
                                memory_id = %job.memory_id,
                                entities = result.entities.len(),
                                facts = result.facts.len(),
                                "Extraction complete"
                            );
                        }
                    }
                    Err(e) if job.attempt < 3 => {
                        tracing::warn!(
                            memory_id = %job.memory_id,
                            attempt = job.attempt + 1,
                            error = %e,
                            "Extraction failed, retrying"
                        );
                        // Exponential backoff: 1s, 2s, 4s
                        let delay = Duration::from_secs(2u64.pow(job.attempt as u32));
                        tokio::time::sleep(delay).await;
                        let _ = retry_tx.try_send(ExtractionJob {
                            attempt: job.attempt + 1,
                            ..job
                        });
                    }
                    Err(e) => {
                        tracing::error!(
                            memory_id = %job.memory_id,
                            attempts = 3,
                            error = %e,
                            "Extraction failed after 3 retries, marking as failed"
                        );
                        let _ = store.update_extraction_status(&job.memory_id, "failed").await;
                    }
                }
            }
        });

        ExtractionPipeline { sender: tx }
    }

    /// Enqueue an extraction job (non-blocking).
    ///
    /// Uses try_send — if the channel is full, the job is dropped and a warning is logged.
    /// The backfill process will pick up missed memories on next startup.
    pub fn enqueue(&self, job: ExtractionJob) {
        if let Err(_) = self.sender.try_send(job) {
            tracing::warn!(
                "Extraction queue full — memory stored, extraction deferred to backfill"
            );
        }
    }

    /// Return a clone of the underlying mpsc sender (for use with the backfill function).
    pub fn sender(&self) -> mpsc::Sender<ExtractionJob> {
        self.sender.clone()
    }
}
