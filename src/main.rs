use anyhow::Result;
use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::time::Duration;
use memcp::config::Config;
use memcp::embedding::EmbeddingProvider;
use memcp::embedding::local::LocalEmbeddingProvider;
use memcp::embedding::openai::OpenAIEmbeddingProvider;
use memcp::embedding::pipeline::{EmbeddingPipeline, backfill};
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
    /// Embedding management operations
    Embed {
        #[command(subcommand)]
        action: EmbedAction,
    },
}

#[derive(Subcommand)]
enum EmbedAction {
    /// Queue all un-embedded or failed memories for re-embedding
    Backfill,
    /// Show embedding statistics (counts by model, pending, failed)
    Stats,
    /// Switch to a new embedding model (marks current embeddings as stale)
    SwitchModel {
        /// New model name to switch to (e.g., "text-embedding-3-small")
        #[arg(long)]
        model: String,
        /// Show what would happen without making changes
        #[arg(long)]
        dry_run: bool,
    },
}

/// Create the embedding provider based on configuration.
async fn create_embedding_provider(config: &Config) -> Result<Arc<dyn EmbeddingProvider + Send + Sync>> {
    match config.embedding.provider.as_str() {
        "openai" => {
            let api_key = config.embedding.openai_api_key.clone()
                .ok_or_else(|| anyhow::anyhow!(
                    "OpenAI API key required when provider is 'openai'. \
                     Set MEMCP_EMBEDDING__OPENAI_API_KEY or embedding.openai_api_key in memcp.toml"
                ))?;
            Ok(Arc::new(OpenAIEmbeddingProvider::new(api_key)?))
        }
        "local" | _ => {
            Ok(Arc::new(LocalEmbeddingProvider::new(&config.embedding.cache_dir).await?))
        }
    }
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

        Some(Commands::Embed { action }) => {
            let store = Arc::new(
                PostgresMemoryStore::new(&config.database_url, true)
                    .await
                    .expect("Failed to connect to database"),
            );

            match action {
                EmbedAction::Backfill => {
                    println!("Starting embedding backfill...");
                    let provider = create_embedding_provider(&config).await?;
                    let pipeline = EmbeddingPipeline::new(provider, store.clone(), 1000);
                    let count = backfill(&store, &pipeline.sender()).await;
                    println!("Queued {} memories for embedding.", count);
                    // Wait briefly for some embeddings to process
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    let stats = store.embedding_stats().await?;
                    println!("Current stats: {}", serde_json::to_string_pretty(&stats)?);
                }
                EmbedAction::Stats => {
                    let stats = store.embedding_stats().await?;
                    println!("{}", serde_json::to_string_pretty(&stats)?);
                }
                EmbedAction::SwitchModel { model, dry_run } => {
                    let stats = store.embedding_stats().await?;

                    if dry_run {
                        println!("DRY RUN — Switch model to '{}'", model);
                        println!("Current embedding stats:");
                        println!("{}", serde_json::to_string_pretty(&stats)?);
                        println!("\nThis would:");
                        println!("  - Mark all current embeddings as stale (is_current = false)");
                        println!("  - Set embedding_status = 'pending' for all affected memories");
                        println!("  - New embeddings will use model '{}' on next backfill", model);
                        println!("\nRun without --dry-run to apply.");
                    } else {
                        println!("Switching to model '{}'...", model);
                        let stale_count = store.mark_all_embeddings_stale().await?;
                        println!("Marked {} embeddings as stale.", stale_count);
                        println!("Run 'memcp embed backfill' to generate new embeddings with the new model.");
                    }
                }
            }
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
            let store = Arc::new(
                PostgresMemoryStore::new(&config.database_url, run_migrations)
                    .await
                    .expect("Failed to initialize database"),
            );

            tracing::info!(database_url = %config.database_url, "PostgreSQL store initialized");

            // 6. Create embedding provider and pipeline
            let provider = create_embedding_provider(&config).await
                .expect("Failed to initialize embedding provider");
            let provider_for_search = provider.clone();  // Clone for MemoryService search
            let pipeline = EmbeddingPipeline::new(provider, store.clone(), 1000);

            // 7. Run startup backfill — queue any un-embedded memories from previous runs
            let queued = backfill(&store, &pipeline.sender()).await;
            if queued > 0 {
                tracing::info!(count = queued, "Startup backfill queued memories for embedding");
            }

            // 8. Create service with store, pipeline, and embedding provider for search
            let pg_store_for_search = store.clone();
            let service = MemoryService::new(
                store as Arc<dyn memcp::store::MemoryStore + Send + Sync>,
                Some(pipeline),
                Some(provider_for_search),
                Some(pg_store_for_search),
            );

            // 9. Serve via stdio transport
            let (stdin, stdout) = rmcp::transport::io::stdio();
            let server = service.serve((stdin, stdout)).await?;

            tracing::info!("memcp server running — awaiting tool calls via stdio");

            // 10. Wait for shutdown (client disconnects or signal)
            server.waiting().await?;

            tracing::info!("memcp server stopped");
        }
    }

    Ok(())
}
