/// LongMemEval dataset types for parsing and categorizing benchmark questions.
///
/// Matches the HuggingFace LongMemEval schema:
/// https://huggingface.co/datasets/xiaowu0162/LongMemEval

use serde::Deserialize;

/// A single question from the LongMemEval dataset.
///
/// Each question has a set of haystack sessions (the memory corpus) and
/// a ground-truth answer for evaluation.
#[derive(Debug, Deserialize)]
pub struct LongMemEvalQuestion {
    pub question_id: String,
    /// Question category: "single-session-user", "multi-session", "temporal-reasoning", etc.
    /// Kept as String for schema flexibility; use category() for normalized output.
    pub question_type: String,
    pub question: String,
    /// Answer may be a String or Number in the dataset — use answer_text() to normalize.
    pub answer: serde_json::Value,
    pub question_date: String,
    /// Session IDs for the haystack sessions (parallel to haystack_sessions).
    pub haystack_session_ids: Vec<String>,
    /// Date strings for the haystack sessions (parallel to haystack_sessions).
    pub haystack_dates: Vec<String>,
    /// The conversation sessions that form the memory haystack.
    pub haystack_sessions: Vec<Vec<Turn>>,
    /// Session IDs that contain the answer (subset of haystack_session_ids).
    pub answer_session_ids: Vec<String>,
}

impl LongMemEvalQuestion {
    /// Get answer as string regardless of JSON type (string or number).
    pub fn answer_text(&self) -> String {
        match &self.answer {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Number(n) => n.to_string(),
            other => other.to_string(),
        }
    }

    /// Whether this is an abstention question (question_id ends with _abs).
    ///
    /// Abstention questions test whether the system correctly refuses to answer
    /// when the answer is not present in the memory haystack.
    pub fn is_abstention(&self) -> bool {
        self.question_id.ends_with("_abs")
    }

    /// Map question_type string to normalized category name for reporting.
    ///
    /// Categories:
    /// - information_extraction: single-session questions (user/assistant/preference)
    /// - multi_session: requires joining information across sessions
    /// - temporal_reasoning: requires understanding time relationships
    /// - knowledge_update: tests superseded information handling
    /// - abstention: answer not present in haystack
    pub fn category(&self) -> &str {
        if self.is_abstention() {
            return "abstention";
        }
        match self.question_type.as_str() {
            "single-session-user"
            | "single-session-assistant"
            | "single-session-preference" => "information_extraction",
            "multi-session" => "multi_session",
            "temporal-reasoning" => "temporal_reasoning",
            "knowledge-update" => "knowledge_update",
            _ => "unknown",
        }
    }
}

/// A single conversational turn in a session.
#[derive(Debug, Deserialize)]
pub struct Turn {
    /// Either "user" or "assistant"
    pub role: String,
    /// The text content of this turn
    pub content: String,
    /// Whether this turn contains the answer to a benchmark question
    #[serde(default)]
    pub has_answer: bool,
}

/// Load LongMemEval dataset from a JSON file.
///
/// Expects a JSON array of LongMemEvalQuestion objects.
pub fn load_dataset(path: &std::path::Path) -> Result<Vec<LongMemEvalQuestion>, anyhow::Error> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let questions: Vec<LongMemEvalQuestion> = serde_json::from_reader(reader)?;
    Ok(questions)
}
