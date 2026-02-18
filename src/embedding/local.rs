/// Local embedding provider using fastembed
///
/// Provides offline embedding generation using all-MiniLM-L6-v2 (384 dimensions).
/// No API key required — model weights are downloaded and cached locally.
/// All CPU-bound fastembed calls are wrapped in spawn_blocking to avoid blocking async runtime.

use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::task;

use super::{EmbeddingError, EmbeddingProvider};

/// Local embedding provider backed by fastembed.
///
/// Uses all-MiniLM-L6-v2 model (384 dimensions) as the default.
/// fastembed is synchronous, so embed() uses spawn_blocking internally.
pub struct LocalEmbeddingProvider {
    model: Arc<Mutex<TextEmbedding>>,
    name: String,
    dim: usize,
}

impl LocalEmbeddingProvider {
    /// Create a new LocalEmbeddingProvider, downloading model weights if not cached.
    ///
    /// # Arguments
    /// * `cache_dir` - Directory to cache model weights (fastembed downloads on first use)
    pub async fn new(cache_dir: &str) -> Result<Self, EmbeddingError> {
        let cache_path = PathBuf::from(cache_dir);

        let te = task::spawn_blocking(move || {
            TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                    .with_cache_dir(cache_path)
                    .with_show_download_progress(true),
            )
        })
        .await
        .map_err(|e| EmbeddingError::ModelInit(e.to_string()))?
        .map_err(|e| EmbeddingError::ModelInit(e.to_string()))?;

        Ok(LocalEmbeddingProvider {
            model: Arc::new(Mutex::new(te)),
            name: "all-MiniLM-L6-v2".to_string(),
            dim: 384,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for LocalEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let model = Arc::clone(&self.model);
        let text = text.to_string();

        task::spawn_blocking(move || {
            let mut model = model.lock().unwrap();
            let mut embeddings = model
                .embed(vec![text], None)
                .map_err(|e| EmbeddingError::Generation(e.to_string()))?;

            embeddings
                .pop()
                .ok_or_else(|| EmbeddingError::Generation("fastembed returned empty result".to_string()))
        })
        .await
        .map_err(|e| EmbeddingError::Generation(format!("spawn_blocking panicked: {}", e)))?
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}
