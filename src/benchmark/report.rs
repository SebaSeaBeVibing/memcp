/// Benchmark reporting module for LongMemEval evaluation results.
///
/// Generates per-category accuracy metrics, comparison tables across configurations,
/// and JSON output for cross-run comparison.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::QuestionResult;

/// Per-category accuracy metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryMetrics {
    pub accuracy: f64,
    pub total: usize,
    pub correct: usize,
}

/// Full benchmark report for a single configuration run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub config_name: String,
    pub timestamp: DateTime<Utc>,
    pub overall_accuracy: f64,
    pub task_averaged_accuracy: f64,
    pub categories: HashMap<String, CategoryMetrics>,
    pub total_questions: usize,
    pub total_correct: usize,
    pub mean_latency_ms: u64,
    pub p95_latency_ms: u64,
}

/// Map a raw question_type string to its normalized category name.
///
/// Mirrors the logic of LongMemEvalQuestion::category() in dataset.rs.
fn map_category(question_type: &str, is_abstention: bool) -> &'static str {
    if is_abstention {
        return "abstention";
    }
    match question_type {
        "single-session-user"
        | "single-session-assistant"
        | "single-session-preference" => "information_extraction",
        "multi-session" => "multi_session",
        "temporal-reasoning" => "temporal_reasoning",
        "knowledge-update" => "knowledge_update",
        _ => "unknown",
    }
}

/// Generate a BenchmarkReport from a set of QuestionResults.
pub fn generate_report(config_name: &str, results: &[QuestionResult]) -> BenchmarkReport {
    let total_questions = results.len();

    // Group results by category
    let mut category_map: HashMap<String, (usize, usize)> = HashMap::new(); // (total, correct)
    let mut total_correct = 0usize;
    let mut latencies: Vec<u64> = Vec::with_capacity(results.len());

    for r in results {
        let cat = map_category(&r.question_type, r.is_abstention);
        let entry = category_map.entry(cat.to_string()).or_insert((0, 0));
        entry.0 += 1;
        if r.correct {
            entry.1 += 1;
            total_correct += 1;
        }
        latencies.push(r.latency_ms);
    }

    // Build CategoryMetrics map
    let categories: HashMap<String, CategoryMetrics> = category_map
        .into_iter()
        .map(|(cat, (total, correct))| {
            let accuracy = if total > 0 {
                correct as f64 / total as f64
            } else {
                0.0
            };
            (cat, CategoryMetrics { accuracy, total, correct })
        })
        .collect();

    // Overall accuracy = total_correct / total_questions
    let overall_accuracy = if total_questions > 0 {
        total_correct as f64 / total_questions as f64
    } else {
        0.0
    };

    // Task-averaged accuracy = mean of per-category accuracies (official LongMemEval metric)
    let task_averaged_accuracy = if categories.is_empty() {
        0.0
    } else {
        let sum: f64 = categories.values().map(|m| m.accuracy).sum();
        sum / categories.len() as f64
    };

    // Compute latency stats
    let mean_latency_ms = if latencies.is_empty() {
        0
    } else {
        latencies.iter().sum::<u64>() / latencies.len() as u64
    };

    let p95_latency_ms = if latencies.is_empty() {
        0
    } else {
        latencies.sort_unstable();
        let idx = ((0.95 * latencies.len() as f64).ceil() as usize).saturating_sub(1);
        latencies[idx.min(latencies.len() - 1)]
    };

    BenchmarkReport {
        config_name: config_name.to_string(),
        timestamp: Utc::now(),
        overall_accuracy,
        task_averaged_accuracy,
        categories,
        total_questions,
        total_correct,
        mean_latency_ms,
        p95_latency_ms,
    }
}

/// Print a formatted report to stdout.
pub fn print_report(report: &BenchmarkReport) {
    println!("=== Benchmark Report: {} ===", report.config_name);
    println!("Date: {}", report.timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
    println!("Questions: {}", report.total_questions);
    println!("Overall Accuracy: {:.1}%", report.overall_accuracy * 100.0);
    println!("Task-Averaged Accuracy: {:.1}%", report.task_averaged_accuracy * 100.0);
    println!();
    println!("Per-Category Breakdown:");

    // Print categories in a fixed order for readability
    let ordered_categories = [
        "information_extraction",
        "multi_session",
        "temporal_reasoning",
        "knowledge_update",
        "abstention",
    ];

    for cat in &ordered_categories {
        if let Some(m) = report.categories.get(*cat) {
            println!(
                "  {:<25}  {}/{} ({:.1}%)",
                format!("{}:", cat),
                m.correct,
                m.total,
                m.accuracy * 100.0
            );
        }
    }

    // Print any unexpected categories not in the standard list
    for (cat, m) in &report.categories {
        if !ordered_categories.contains(&cat.as_str()) {
            println!(
                "  {:<25}  {}/{} ({:.1}%)",
                format!("{}:", cat),
                m.correct,
                m.total,
                m.accuracy * 100.0
            );
        }
    }

    println!();
    println!(
        "Latency: mean={}ms, p95={}ms",
        report.mean_latency_ms, report.p95_latency_ms
    );
}

/// Print a side-by-side comparison of multiple reports.
pub fn print_comparison(reports: &[BenchmarkReport]) {
    if reports.is_empty() {
        return;
    }

    println!("=== Configuration Comparison ===");
    println!();

    // Build header
    let config_names: Vec<&str> = reports.iter().map(|r| r.config_name.as_str()).collect();
    let col_width = 12usize;
    let label_width = 21usize;

    // Header row
    let header_names: Vec<String> = config_names
        .iter()
        .map(|n| format!("{:>col_width$}", n, col_width = col_width))
        .collect();
    println!("{:<label_width$}| {}", "Category", header_names.join(" | "));

    // Separator
    let sep = format!(
        "{:-<label_width$}|-{}",
        "",
        vec![format!("{:-<col_width$}", "", col_width = col_width)]
            .iter()
            .chain(
                (1..reports.len())
                    .map(|_| format!("{:-<col_width$}", "", col_width = col_width))
                    .collect::<Vec<_>>()
                    .iter()
            )
            .cloned()
            .collect::<Vec<_>>()
            .join("-|-")
    );
    println!("{}", sep);

    // Category rows
    let ordered_categories = [
        "information_extraction",
        "multi_session",
        "temporal_reasoning",
        "knowledge_update",
        "abstention",
    ];

    for cat in &ordered_categories {
        let values: Vec<String> = reports
            .iter()
            .map(|r| {
                r.categories
                    .get(*cat)
                    .map(|m| format!("{:>col_width$.1}%", m.accuracy * 100.0, col_width = col_width - 1))
                    .unwrap_or_else(|| format!("{:>col_width$}", "N/A", col_width = col_width))
            })
            .collect();
        // Truncate category name to fit label column
        let label = if cat.len() > label_width - 1 {
            &cat[..label_width - 1]
        } else {
            cat
        };
        println!("{:<label_width$}| {}", label, values.join(" | "));
    }

    // Separator before totals
    println!("{}", sep);

    // Overall accuracy row
    let overall_values: Vec<String> = reports
        .iter()
        .map(|r| format!("{:>col_width$.1}%", r.overall_accuracy * 100.0, col_width = col_width - 1))
        .collect();
    println!("{:<label_width$}| {}", "Overall", overall_values.join(" | "));

    // Task-averaged accuracy row
    let task_avg_values: Vec<String> = reports
        .iter()
        .map(|r| format!("{:>col_width$.1}%", r.task_averaged_accuracy * 100.0, col_width = col_width - 1))
        .collect();
    println!("{:<label_width$}| {}", "Task-Averaged", task_avg_values.join(" | "));
}

/// Save report as JSON to a file path.
pub fn save_report(report: &BenchmarkReport, path: &std::path::Path) -> Result<(), anyhow::Error> {
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load a previously saved report from JSON.
pub fn load_report(path: &std::path::Path) -> Result<BenchmarkReport, anyhow::Error> {
    let json = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&json)?)
}
