/// Configuration management using figment
///
/// Loads configuration with this precedence (highest wins):
/// 1. Defaults (hardcoded)
/// 2. TOML file: memcp.toml (in working directory)
/// 3. Environment variables: DATABASE_URL (standard PostgreSQL convention)
/// 4. Environment variables: prefixed MEMCP_ (e.g., MEMCP_LOG_LEVEL=debug)

use figment::{
    Figment,
    providers::{Env, Format, Toml, Serialized},
};
use serde::{Deserialize, Serialize};
use crate::errors::MemcpError;

/// Configuration for the search subsystem.
///
/// BM25 backend selection is explicit — having ParadeDB installed does NOT auto-switch.
/// Nested env var overrides use double underscores:
///   MEMCP_SEARCH__BM25_BACKEND=paradedb
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// BM25 backend: "native" (PostgreSQL tsvector, default) or "paradedb" (pg_search extension)
    /// Default: "native" — no extension required for self-hosted deployments
    #[serde(default = "default_bm25_backend")]
    pub bm25_backend: String,
}

fn default_bm25_backend() -> String {
    "native".to_string()
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            bm25_backend: default_bm25_backend(),
        }
    }
}

/// Configuration for the salience scoring subsystem.
///
/// Weights control how much each dimension contributes to the final salience score.
/// All four weights should ideally sum to 1.0 (they are not automatically normalized).
/// Nested env var overrides use double underscores:
///   MEMCP_SALIENCE__W_RECENCY=0.30
///   MEMCP_SALIENCE__DEBUG_SCORING=true
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalienceConfig {
    /// Weight for recency dimension (default: 0.25)
    #[serde(default = "default_w_recency")]
    pub w_recency: f64,
    /// Weight for access frequency dimension (default: 0.15)
    #[serde(default = "default_w_access")]
    pub w_access: f64,
    /// Weight for semantic relevance dimension (default: 0.45)
    #[serde(default = "default_w_semantic")]
    pub w_semantic: f64,
    /// Weight for reinforcement strength dimension (default: 0.15)
    #[serde(default = "default_w_reinforce")]
    pub w_reinforce: f64,
    /// Exponential recency decay rate (default: 0.01, ~70-day half-life)
    #[serde(default = "default_recency_lambda")]
    pub recency_lambda: f64,
    /// Enable debug scoring output (shows dimension breakdown in results)
    #[serde(default)]
    pub debug_scoring: bool,
}

fn default_w_recency() -> f64 { 0.25 }
fn default_w_access() -> f64 { 0.15 }
fn default_w_semantic() -> f64 { 0.45 }
fn default_w_reinforce() -> f64 { 0.15 }
fn default_recency_lambda() -> f64 { 0.01 }

impl Default for SalienceConfig {
    fn default() -> Self {
        SalienceConfig {
            w_recency: default_w_recency(),
            w_access: default_w_access(),
            w_semantic: default_w_semantic(),
            w_reinforce: default_w_reinforce(),
            recency_lambda: default_recency_lambda(),
            debug_scoring: false,
        }
    }
}

/// Configuration for the extraction pipeline subsystem.
///
/// Provider selection is explicit — "ollama" is the default (local, no API key needed).
/// Nested env var overrides use double underscores:
///   MEMCP_EXTRACTION__PROVIDER=openai
///   MEMCP_EXTRACTION__OPENAI_API_KEY=sk-...
///   MEMCP_EXTRACTION__ENABLED=false
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Which provider to use: "ollama" (local, default) or "openai"
    #[serde(default = "default_extraction_provider")]
    pub provider: String,

    /// Ollama server base URL
    #[serde(default = "default_ollama_base_url")]
    pub ollama_base_url: String,

    /// Ollama model for extraction
    #[serde(default = "default_ollama_model")]
    pub ollama_model: String,

    /// OpenAI API key — only required when provider = "openai"
    #[serde(default)]
    pub openai_api_key: Option<String>,

    /// OpenAI model for extraction
    #[serde(default = "default_openai_extraction_model")]
    pub openai_model: String,

    /// Whether extraction is enabled (default: true). Set to false to skip extraction entirely.
    #[serde(default = "default_extraction_enabled")]
    pub enabled: bool,

    /// Maximum content characters to send for extraction (truncated beyond this)
    #[serde(default = "default_max_content_chars")]
    pub max_content_chars: usize,
}

fn default_extraction_provider() -> String {
    "ollama".to_string()
}

fn default_ollama_base_url() -> String {
    "http://localhost:11434".to_string()
}

fn default_ollama_model() -> String {
    "llama3.2:3b".to_string()
}

fn default_openai_extraction_model() -> String {
    "gpt-4o-mini".to_string()
}

fn default_extraction_enabled() -> bool {
    true
}

fn default_max_content_chars() -> usize {
    1500
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        ExtractionConfig {
            provider: default_extraction_provider(),
            ollama_base_url: default_ollama_base_url(),
            ollama_model: default_ollama_model(),
            openai_api_key: None,
            openai_model: default_openai_extraction_model(),
            enabled: default_extraction_enabled(),
            max_content_chars: default_max_content_chars(),
        }
    }
}

/// Configuration for the memory consolidation subsystem.
///
/// When enabled, new memories trigger a pgvector similarity check after embedding.
/// If any existing memories exceed the threshold, they are auto-merged via LLM synthesis.
/// Nested env var overrides use double underscores:
///   MEMCP_CONSOLIDATION__ENABLED=false
///   MEMCP_CONSOLIDATION__SIMILARITY_THRESHOLD=0.92
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    /// Whether consolidation is enabled (default: true).
    /// Set to false to disable automatic merging.
    #[serde(default = "default_consolidation_enabled")]
    pub enabled: bool,

    /// Cosine similarity threshold above which memories are merged (default: 0.92).
    /// Range: 0.0–1.0. Higher values require tighter similarity before merging.
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f64,

    /// Maximum number of originals merged into a single consolidated memory (default: 5).
    #[serde(default = "default_max_consolidation_group")]
    pub max_consolidation_group: usize,
}

fn default_consolidation_enabled() -> bool { true }
fn default_similarity_threshold() -> f64 { 0.92 }
fn default_max_consolidation_group() -> usize { 5 }

impl Default for ConsolidationConfig {
    fn default() -> Self {
        ConsolidationConfig {
            enabled: default_consolidation_enabled(),
            similarity_threshold: default_similarity_threshold(),
            max_consolidation_group: default_max_consolidation_group(),
        }
    }
}

/// Configuration for the embedding provider subsystem.
///
/// Provider selection is explicit — having an API key does NOT auto-switch from local.
/// Nested env var overrides use double underscores:
///   MEMCP_EMBEDDING__PROVIDER=openai
///   MEMCP_EMBEDDING__OPENAI_API_KEY=sk-...
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Which provider to use: "local" (fastembed) or "openai"
    /// Default: "local" — no API key required for self-hosted deployments
    #[serde(default = "default_embedding_provider")]
    pub provider: String,

    /// OpenAI API key — only required when provider = "openai"
    #[serde(default)]
    pub openai_api_key: Option<String>,

    /// Directory for caching model weights (fastembed downloads)
    /// Default: platform cache dir + "/memcp/models", fallback to /tmp/memcp_models
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,
}

fn default_embedding_provider() -> String {
    "local".to_string()
}

fn default_cache_dir() -> String {
    dirs::cache_dir()
        .map(|p| p.join("memcp").join("models").to_string_lossy().into_owned())
        .unwrap_or_else(|| "/tmp/memcp_models".to_string())
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig {
            provider: default_embedding_provider(),
            openai_api_key: None,
            cache_dir: default_cache_dir(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Log level: trace, debug, info, warn, error
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// Optional file path for log output (in addition to stderr)
    #[serde(default)]
    pub log_file: Option<String>,

    /// PostgreSQL database URL.
    /// Configurable via DATABASE_URL or MEMCP_DATABASE_URL env var, or database_url in memcp.toml.
    #[serde(default = "default_database_url")]
    pub database_url: String,

    /// Embedding provider configuration.
    /// Existing configs without [embedding] section still work (serde default applied).
    #[serde(default)]
    pub embedding: EmbeddingConfig,

    /// Search subsystem configuration.
    /// Existing configs without [search] section still work (serde default applied).
    #[serde(default)]
    pub search: SearchConfig,

    /// Salience scoring configuration.
    /// Existing configs without [salience] section still work (serde default applied).
    #[serde(default)]
    pub salience: SalienceConfig,

    /// Extraction pipeline configuration.
    /// Existing configs without [extraction] section still work (serde default applied).
    #[serde(default)]
    pub extraction: ExtractionConfig,

    /// Memory consolidation configuration.
    /// Existing configs without [consolidation] section still work (serde default applied).
    #[serde(default)]
    pub consolidation: ConsolidationConfig,
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_database_url() -> String {
    "postgres://memcp:memcp@localhost:5432/memcp".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Config {
            log_level: default_log_level(),
            log_file: None,
            database_url: default_database_url(),
            embedding: EmbeddingConfig::default(),
            search: SearchConfig::default(),
            salience: SalienceConfig::default(),
            extraction: ExtractionConfig::default(),
            consolidation: ConsolidationConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from defaults, TOML file, and environment variables
    ///
    /// Environment variables override TOML file values.
    /// DATABASE_URL is checked first (standard PostgreSQL convention),
    /// then MEMCP_DATABASE_URL, then database_url in memcp.toml.
    pub fn load() -> Result<Config, MemcpError> {
        Figment::new()
            .merge(Serialized::defaults(Config::default()))
            .merge(Toml::file("memcp.toml"))
            // Standard DATABASE_URL env var (highest priority for database config)
            .merge(Env::raw().only(&["DATABASE_URL"]).map(|_| "database_url".into()))
            // MEMCP_-prefixed env vars (includes MEMCP_DATABASE_URL, MEMCP_LOG_LEVEL, etc.)
            // Double underscore handles nested: MEMCP_EMBEDDING__PROVIDER=openai
            .merge(Env::prefixed("MEMCP_"))
            .extract()
            .map_err(|e| MemcpError::Config(format!("Failed to load config: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = Config::default();
        assert_eq!(config.log_level, "info");
        assert_eq!(config.log_file, None);
        assert_eq!(config.database_url, "postgres://memcp:memcp@localhost:5432/memcp");
        assert_eq!(config.embedding.provider, "local");
        assert_eq!(config.embedding.openai_api_key, None);
        assert_eq!(config.search.bm25_backend, "native");
    }
}
