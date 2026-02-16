/// Structured logging setup using tracing
///
/// CRITICAL: Writes to stderr ONLY (never stdout) to avoid corrupting JSON-RPC stream.
/// Auto-detects format: human-readable with ANSI colors when stderr is a terminal,
/// structured JSON when piped/redirected.

use std::io::IsTerminal;
use tracing_subscriber::{
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
};
use crate::config::Config;

/// Initialize tracing subscriber with stderr-only output
///
/// Format auto-detection:
/// - Terminal: human-readable with ANSI colors
/// - Pipe/redirect: structured JSON
///
/// Log level from config.log_level (default: info)
/// RUST_LOG env var can override at runtime
pub fn init_logging(config: &Config) {
    // Build env filter from config, with RUST_LOG override
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.log_level));

    // Auto-detect format based on stderr terminal status
    let stderr_is_terminal = std::io::stderr().is_terminal();

    if stderr_is_terminal {
        // Human-readable format with ANSI colors for terminal
        tracing_subscriber::registry()
            .with(env_filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(std::io::stderr)
                    .with_ansi(true)
            )
            .init();
    } else {
        // Structured JSON format for pipes/redirects
        tracing_subscriber::registry()
            .with(env_filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(std::io::stderr)
                    .json()
            )
            .init();
    }

    // TODO: Add file output layer if config.log_file is set
    // This requires layering a file appender on top of the stderr layer
    // For Phase 1, stderr-only is sufficient
    if config.log_file.is_some() {
        tracing::warn!(
            "log_file configuration is not yet implemented, logging to stderr only"
        );
    }
}
