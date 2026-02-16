use anyhow::Result;
use memcp::config::Config;
use memcp::logging;
use memcp::server::MemoryService;
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

    // 3. Create service
    let service = MemoryService::new();

    // 4. Serve via stdio transport
    let (stdin, stdout) = rmcp::transport::io::stdio();
    let server = service.serve((stdin, stdout)).await?;

    tracing::info!("memcp server running — awaiting tool calls via stdio");

    // 5. Wait for shutdown (client disconnects or signal)
    server.waiting().await?;

    tracing::info!("memcp server stopped");

    Ok(())
}
