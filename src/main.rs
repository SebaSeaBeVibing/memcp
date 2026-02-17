use anyhow::Result;
use std::sync::Arc;
use memcp::config::Config;
use memcp::logging;
use memcp::server::MemoryService;
use memcp::store::sqlite::SqliteMemoryStore;
use rmcp::ServiceExt;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Load configuration
    let config = Config::load().unwrap_or_else(|e| {
        eprintln!("Config error (using defaults): {}", e);
        Config::default()
    });

    // 2. Initialize logging FIRST (before any other output)
    // CRITICAL: logging goes to stderr only — stdout is reserved for JSON-RPC
    logging::init_logging(&config);

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        "memcp server starting"
    );

    // 3. Initialize persistent SQLite store
    let store = SqliteMemoryStore::new(&config.db_path)
        .await
        .expect("Failed to initialize database");

    tracing::info!(db_path = %config.db_path, "SQLite store initialized");

    // 4. Create service with store
    let service = MemoryService::new(Arc::new(store));

    // 5. Serve via stdio transport
    let (stdin, stdout) = rmcp::transport::io::stdio();
    let server = service.serve((stdin, stdout)).await?;

    tracing::info!("memcp server running — awaiting tool calls via stdio");

    // 6. Wait for shutdown (client disconnects or signal)
    server.waiting().await?;

    tracing::info!("memcp server stopped");

    Ok(())
}
