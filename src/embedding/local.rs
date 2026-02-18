/// Local embedding provider using fastembed
///
/// Provides offline embedding generation using all-MiniLM-L6-v2 (384 dimensions).
/// No API key required — model weights are downloaded and cached locally.
/// All CPU-bound fastembed calls are wrapped in spawn_blocking to avoid blocking async runtime.

use async_trait::async_trait;
use std::path::PathBuf;
use tokio::task;

use super::{EmbeddingError, EmbeddingProvider};

/// Local embedding provider backed by fastembed.
///
/// Uses all-MiniLM-L6-v2 model (384 dimensions) as the default.
/// fastembed is synchronous, so embed() uses spawn_blocking internally.
pub struct LocalEmbeddingProvider {
    // Placeholder: fastembed TextEmbedding wrapped in Mutex for thread safety
    // Will be Arc<std::sync::Mutex<TextEmbedding>> once fastembed is added as dependency
    _cache_dir: PathBuf,
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

        // Ensure cache directory exists
        task::spawn_blocking({
            let cache_path = cache_path.clone();
            move || {
                std::fs::create_dir_all(&cache_path)
                    .map_err(|e| EmbeddingError::ModelInit(format!("Failed to create cache dir: {}", e)))
            }
        })
        .await
        .map_err(|e| EmbeddingError::ModelInit(e.to_string()))??;

        Ok(LocalEmbeddingProvider {
            _cache_dir: cache_path,
            name: "all-MiniLM-L6-v2".to_string(),
            dim: 384,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for LocalEmbeddingProvider {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Stub: returns zero vector until fastembed dependency is available
        // Task 2 will replace this with actual fastembed inference
        Ok(vec![0.0f32; self.dim])
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}
