/// Configuration management using figment
///
/// Loads configuration with this precedence (highest wins):
/// 1. Defaults (hardcoded)
/// 2. TOML file: memcp.toml (in working directory)
/// 3. Environment variables: prefixed MEMCP_ (e.g., MEMCP_LOG_LEVEL=debug)

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

    /// SQLite database path. Supports sqlite:// URI scheme.
    /// Configurable via MEMCP_DB_PATH env var or db_path in memcp.toml.
    #[serde(default = "default_db_path")]
    pub db_path: String,
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_db_path() -> String {
    "sqlite://memcp.db".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Config {
            log_level: default_log_level(),
            log_file: None,
            db_path: default_db_path(),
        }
    }
}

impl Config {
    /// Load configuration from defaults, TOML file, and environment variables
    ///
    /// Environment variables override TOML file values.
    /// Example: MEMCP_LOG_LEVEL=debug overrides log_level in memcp.toml
    pub fn load() -> Result<Config, MemcpError> {
        Figment::new()
            .merge(Serialized::defaults(Config::default()))
            .merge(Toml::file("memcp.toml"))
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
        assert_eq!(config.db_path, "sqlite://memcp.db");
    }
}
