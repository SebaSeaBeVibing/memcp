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
    }
}
