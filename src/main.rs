use anyhow::Result;
use clap::{Parser, Subcommand};
use std::sync::Arc;
use memcp::config::Config;
use memcp::logging;
use memcp::server::MemoryService;
use memcp::store::postgres::PostgresMemoryStore;
use rmcp::ServiceExt;

#[derive(Parser)]
#[command(name = "memcp", version, about = "High-performance MCP memory server")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Skip automatic database migration on startup
    #[arg(long)]
    skip_migrate: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run database migrations and exit
    Migrate,
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Parse CLI args
    let cli = Cli::parse();

    // 2. Load configuration
    let config = Config::load().unwrap_or_else(|e| {
        eprintln!("Config error (using defaults): {}", e);
        Config::default()
    });

    // 3. Initialize logging FIRST (before any other output)
    // CRITICAL: logging goes to stderr only — stdout is reserved for JSON-RPC
    logging::init_logging(&config);

    // 4. Handle subcommands
    match cli.command {
        Some(Commands::Migrate) => {
            tracing::info!("Running database migrations...");
            // run_migrations=true, just connect and migrate then exit
            let _store = PostgresMemoryStore::new(&config.database_url, true)
                .await
                .expect("Failed to connect and run migrations");
            println!("Migrations completed successfully.");
            return Ok(());
        }
        None => {
            // Default: start the MCP server
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                "memcp server starting"
            );

            // 5. Initialize PostgreSQL store
            let run_migrations = !cli.skip_migrate;
            let store = PostgresMemoryStore::new(&config.database_url, run_migrations)
                .await
                .expect("Failed to initialize database");

            tracing::info!(database_url = %config.database_url, "PostgreSQL store initialized");

            // 6. Create service with store
            let service = MemoryService::new(Arc::new(store));

            // 7. Serve via stdio transport
            let (stdin, stdout) = rmcp::transport::io::stdio();
            let server = service.serve((stdin, stdout)).await?;

            tracing::info!("memcp server running — awaiting tool calls via stdio");

            // 8. Wait for shutdown (client disconnects or signal)
            server.waiting().await?;

            tracing::info!("memcp server stopped");
        }
    }

    Ok(())
}
