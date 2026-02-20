/// Benchmark CLI binary for LongMemEval evaluation.
///
/// Runs the full benchmark pipeline: load dataset → ingest → search → generate → score.
/// Supports single config or "all" for comparison across vector-only / hybrid / hybrid+qi.
/// CI integration via --subset (stratified sample) and --min-accuracy (exit code threshold).

use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

use memcp::benchmark::dataset::load_dataset;
use memcp::benchmark::report;
use memcp::benchmark::runner::{load_checkpoint, run_benchmark};
use memcp::benchmark::report::BenchmarkReport;
use memcp::benchmark::default_configs;
use memcp::embedding::local::LocalEmbeddingProvider;
use memcp::embedding::pipeline::EmbeddingPipeline;
use memcp::store::postgres::PostgresMemoryStore;

#[derive(Parser)]
#[command(name = "memcp-benchmark", about = "LongMemEval benchmark runner for memcp")]
struct Cli {
    /// Path to LongMemEval dataset JSON
    #[arg(long, default_value = "data/longmemeval/longmemeval_s_cleaned.json")]
    dataset: PathBuf,

    /// Search configuration: "vector-only", "hybrid", "hybrid+qi", or "all" for comparison
    #[arg(long, default_value = "hybrid")]
    config: String,

    /// Run only first N questions (for CI speed). Preserves category distribution via truncation sorted by question_id.
    #[arg(long)]
    subset: Option<usize>,

    /// Minimum overall accuracy to pass (CI threshold, e.g. 0.60 for 60%)
    #[arg(long)]
    min_accuracy: Option<f64>,

    /// Output directory for results
    #[arg(long, default_value = "data/longmemeval/results")]
    output_dir: PathBuf,

    /// Resume from checkpoint if available
    #[arg(long)]
    resume: bool,

    /// OpenAI API key (can also be set via OPENAI_API_KEY env var)
    #[arg(long, env = "OPENAI_API_KEY")]
    openai_api_key: String,

    /// Database URL (can also be set via DATABASE_URL env var)
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // 1. Parse CLI args
    let cli = Cli::parse();

    // 2. Initialize tracing (stdout, info level)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // 3. Load dataset
    tracing::info!(path = %cli.dataset.display(), "Loading dataset");
    let mut questions = load_dataset(&cli.dataset)?;
    tracing::info!(count = questions.len(), "Dataset loaded");

    // 4. Apply subset if specified (sort by question_id for reproducibility, then truncate)
    if let Some(n) = cli.subset {
        questions.sort_by(|a, b| a.question_id.cmp(&b.question_id));
        questions.truncate(n);
        tracing::info!(subset = n, "Applied subset — using {} questions", questions.len());
    }

    // 5. Print summary: total questions, per-category counts
    let mut category_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for q in &questions {
        *category_counts.entry(q.category().to_string()).or_insert(0) += 1;
    }
    println!("=== LongMemEval Benchmark ===");
    println!("Dataset: {}", cli.dataset.display());
    println!("Questions: {}", questions.len());
    println!("Per-category counts:");
    for cat in &[
        "information_extraction",
        "multi_session",
        "temporal_reasoning",
        "knowledge_update",
        "abstention",
    ] {
        let count = category_counts.get(*cat).copied().unwrap_or(0);
        println!("  {:<25} {}", format!("{}:", cat), count);
    }
    println!();

    // 6. Create output directory
    std::fs::create_dir_all(&cli.output_dir)?;

    // 7. Initialize database connection and store (with migrations)
    tracing::info!(database_url = %cli.database_url, "Connecting to database");
    let store = Arc::new(
        PostgresMemoryStore::new(&cli.database_url, true).await?,
    );
    tracing::info!("Database ready");

    // 8. Initialize local embedding provider and pipeline
    // Benchmark uses local fastembed provider (no API key needed, deterministic)
    tracing::info!("Initializing local embedding provider");
    let embedding_provider: Arc<dyn memcp::embedding::EmbeddingProvider + Send + Sync> =
        Arc::new(LocalEmbeddingProvider::new(".fastembed_cache").await?);

    // No consolidation sender for benchmark (consolidation is MCP live-trigger only)
    let pipeline = EmbeddingPipeline::new(embedding_provider.clone(), store.clone(), 1000, None);

    // 9. Determine configs to run
    let all_configs = default_configs();
    let configs_to_run: Vec<_> = if cli.config == "all" {
        all_configs.iter().collect()
    } else {
        let found = all_configs
            .iter()
            .find(|c| c.name == cli.config)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Unknown config '{}'. Valid options: vector-only, hybrid, hybrid+qi, all",
                    cli.config
                )
            })?;
        vec![found]
    };

    // 10. Run each config
    let mut reports: Vec<BenchmarkReport> = Vec::new();

    for config in &configs_to_run {
        println!("--- Running config: {} ---", config.name);

        let checkpoint_path = cli.output_dir.join(format!("{}_checkpoint.json", config.name));

        // Load checkpoint if --resume and file exists
        let resume_state = if cli.resume {
            match load_checkpoint(&checkpoint_path) {
                Ok(Some(state)) => {
                    tracing::info!(
                        config = %config.name,
                        completed = state.completed_question_ids.len(),
                        "Resuming from checkpoint"
                    );
                    Some(state)
                }
                Ok(None) => {
                    tracing::info!(config = %config.name, "No checkpoint found — starting fresh");
                    None
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to load checkpoint — starting fresh");
                    None
                }
            }
        } else {
            None
        };

        // Run the benchmark
        let results = run_benchmark(
            &questions,
            config,
            store.clone(),
            &pipeline,
            embedding_provider.clone(),
            &cli.openai_api_key,
            &checkpoint_path,
            resume_state,
        )
        .await?;

        // Generate report
        let report = report::generate_report(&config.name, &results);

        // Print report
        report::print_report(&report);
        println!();

        // Save report JSON
        let report_path = cli.output_dir.join(format!("{}_report.json", config.name));
        report::save_report(&report, &report_path)?;
        tracing::info!(path = %report_path.display(), "Report saved");

        reports.push(report);
    }

    // 11. If multiple configs ran, print comparison
    if reports.len() > 1 {
        report::print_comparison(&reports);
        println!();
    }

    // 12. CI threshold check
    if let Some(threshold) = cli.min_accuracy {
        // Check the last config's overall accuracy (or single config if not "all")
        let last_report = reports.last().expect("At least one report must exist");
        if last_report.overall_accuracy < threshold {
            eprintln!(
                "FAIL: overall accuracy {:.1}% < threshold {:.1}%",
                last_report.overall_accuracy * 100.0,
                threshold * 100.0
            );
            std::process::exit(1);
        } else {
            println!(
                "PASS: overall accuracy {:.1}% >= threshold {:.1}%",
                last_report.overall_accuracy * 100.0,
                threshold * 100.0
            );
        }
    }

    Ok(())
}
