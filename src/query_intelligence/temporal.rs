/// Deterministic regex-based temporal hint parser
///
/// Parses natural language time expressions from query strings without any LLM call.
/// Used as a fallback when expansion is disabled, and as a pre-filter to reduce
/// the candidate set before vector similarity search.
///
/// All patterns are matched case-insensitively against the full query string.

use chrono::{DateTime, Datelike, Duration, TimeZone, Utc};
use regex::Regex;

use super::TimeRange;

/// Parse a temporal hint from the query string relative to `now`.
///
/// Returns `Some(TimeRange)` if a recognized time expression is found,
/// or `None` if no pattern matches. Only the first matching pattern is used.
pub fn parse_temporal_hint(query: &str, now: DateTime<Utc>) -> Option<TimeRange> {
    let q = query.to_lowercase();

    // --- relative named periods ---

    if q.contains("yesterday") {
        let start = Utc
            .with_ymd_and_hms(now.year(), now.month(), now.day(), 0, 0, 0)
            .single()?
            - Duration::days(1);
        let end = start + Duration::hours(23) + Duration::minutes(59) + Duration::seconds(59);
        return Some(TimeRange {
            after: Some(start),
            before: Some(end),
        });
    }

    if q.contains("today") {
        let start = Utc
            .with_ymd_and_hms(now.year(), now.month(), now.day(), 0, 0, 0)
            .single()?;
        return Some(TimeRange {
            after: Some(start),
            before: None,
        });
    }

    if q.contains("last week") || q.contains("past week") {
        return Some(TimeRange {
            after: Some(now - Duration::days(7)),
            before: None,
        });
    }

    if q.contains("last month") || q.contains("past month") {
        return Some(TimeRange {
            after: Some(now - Duration::days(30)),
            before: None,
        });
    }

    if q.contains("last year") || q.contains("past year") {
        return Some(TimeRange {
            after: Some(now - Duration::days(365)),
            before: None,
        });
    }

    // --- "a few X ago" patterns ---

    if q.contains("a few days ago") {
        return Some(TimeRange {
            after: Some(now - Duration::days(5)),
            before: Some(now - Duration::days(1)),
        });
    }

    if q.contains("a few weeks ago") {
        return Some(TimeRange {
            after: Some(now - Duration::days(28)),
            before: Some(now - Duration::days(7)),
        });
    }

    if q.contains("a few months ago") {
        return Some(TimeRange {
            after: Some(now - Duration::days(90)),
            before: Some(now - Duration::days(30)),
        });
    }

    // --- absolute date patterns ---

    // "after YYYY-MM-DD"
    let after_re = Regex::new(r"after\s+(\d{4}-\d{2}-\d{2})").ok()?;
    if let Some(cap) = after_re.captures(&q) {
        let date_str = format!("{}T00:00:00Z", &cap[1]);
        if let Ok(dt) = date_str.parse::<DateTime<Utc>>() {
            return Some(TimeRange {
                after: Some(dt),
                before: None,
            });
        }
    }

    // "before YYYY-MM-DD"
    let before_re = Regex::new(r"before\s+(\d{4}-\d{2}-\d{2})").ok()?;
    if let Some(cap) = before_re.captures(&q) {
        let date_str = format!("{}T23:59:59Z", &cap[1]);
        if let Ok(dt) = date_str.parse::<DateTime<Utc>>() {
            return Some(TimeRange {
                after: None,
                before: Some(dt),
            });
        }
    }

    // --- "between MONTH and MONTH" ---
    // e.g., "between January and March", "between march and june"
    let between_re =
        Regex::new(r"between\s+(\w+)\s+and\s+(\w+)").ok()?;
    if let Some(cap) = between_re.captures(&q) {
        let m1 = parse_month_name(&cap[1])?;
        let m2 = parse_month_name(&cap[2])?;
        let year = now.year();

        let start = Utc.with_ymd_and_hms(year, m1, 1, 0, 0, 0).single()?;
        // End: last day of m2 — use first day of m2+1 minus 1 second
        let (end_year, end_month) = if m2 == 12 {
            (year + 1, 1u32)
        } else {
            (year, m2 + 1)
        };
        let end = Utc
            .with_ymd_and_hms(end_year, end_month, 1, 0, 0, 0)
            .single()?
            - Duration::seconds(1);

        return Some(TimeRange {
            after: Some(start),
            before: Some(end),
        });
    }

    None
}

/// Convert a month name (English, case-insensitive) to its number (1–12).
fn parse_month_name(name: &str) -> Option<u32> {
    match name.to_lowercase().as_str() {
        "january" | "jan" => Some(1),
        "february" | "feb" => Some(2),
        "march" | "mar" => Some(3),
        "april" | "apr" => Some(4),
        "may" => Some(5),
        "june" | "jun" => Some(6),
        "july" | "jul" => Some(7),
        "august" | "aug" => Some(8),
        "september" | "sep" | "sept" => Some(9),
        "october" | "oct" => Some(10),
        "november" | "nov" => Some(11),
        "december" | "dec" => Some(12),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixed_now() -> DateTime<Utc> {
        // 2024-03-15 12:00:00 UTC
        Utc.with_ymd_and_hms(2024, 3, 15, 12, 0, 0).unwrap()
    }

    #[test]
    fn test_no_match_returns_none() {
        assert!(parse_temporal_hint("find my API keys", fixed_now()).is_none());
    }

    #[test]
    fn test_yesterday() {
        let now = fixed_now();
        let result = parse_temporal_hint("what did I do yesterday", now).unwrap();
        // yesterday = 2024-03-14
        assert_eq!(
            result.after.unwrap(),
            Utc.with_ymd_and_hms(2024, 3, 14, 0, 0, 0).unwrap()
        );
        assert_eq!(
            result.before.unwrap(),
            Utc.with_ymd_and_hms(2024, 3, 14, 23, 59, 59).unwrap()
        );
    }

    #[test]
    fn test_today() {
        let now = fixed_now();
        let result = parse_temporal_hint("notes from today", now).unwrap();
        assert_eq!(
            result.after.unwrap(),
            Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap()
        );
        assert!(result.before.is_none());
    }

    #[test]
    fn test_last_week() {
        let now = fixed_now();
        let result = parse_temporal_hint("what happened last week", now).unwrap();
        assert_eq!(result.after.unwrap(), now - Duration::days(7));
        assert!(result.before.is_none());
    }

    #[test]
    fn test_past_month() {
        let now = fixed_now();
        let result = parse_temporal_hint("memories from the past month", now).unwrap();
        assert_eq!(result.after.unwrap(), now - Duration::days(30));
    }

    #[test]
    fn test_last_year() {
        let now = fixed_now();
        let result = parse_temporal_hint("notes from last year", now).unwrap();
        assert_eq!(result.after.unwrap(), now - Duration::days(365));
    }

    #[test]
    fn test_a_few_days_ago() {
        let now = fixed_now();
        let result = parse_temporal_hint("I read that a few days ago", now).unwrap();
        assert_eq!(result.after.unwrap(), now - Duration::days(5));
        assert_eq!(result.before.unwrap(), now - Duration::days(1));
    }

    #[test]
    fn test_a_few_weeks_ago() {
        let now = fixed_now();
        let result = parse_temporal_hint("happened a few weeks ago", now).unwrap();
        assert_eq!(result.after.unwrap(), now - Duration::days(28));
        assert_eq!(result.before.unwrap(), now - Duration::days(7));
    }

    #[test]
    fn test_a_few_months_ago() {
        let now = fixed_now();
        let result = parse_temporal_hint("it was a few months ago", now).unwrap();
        assert_eq!(result.after.unwrap(), now - Duration::days(90));
        assert_eq!(result.before.unwrap(), now - Duration::days(30));
    }

    #[test]
    fn test_after_date() {
        let now = fixed_now();
        let result = parse_temporal_hint("entries after 2024-01-01", now).unwrap();
        assert_eq!(
            result.after.unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
        );
        assert!(result.before.is_none());
    }

    #[test]
    fn test_before_date() {
        let now = fixed_now();
        let result = parse_temporal_hint("notes before 2024-02-28", now).unwrap();
        assert!(result.after.is_none());
        assert_eq!(
            result.before.unwrap(),
            Utc.with_ymd_and_hms(2024, 2, 28, 23, 59, 59).unwrap()
        );
    }

    #[test]
    fn test_between_months() {
        let now = fixed_now();
        let result = parse_temporal_hint("between January and March", now).unwrap();
        assert_eq!(
            result.after.unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
        );
        // End of March = April 1 00:00:00 - 1 second = March 31 23:59:59
        assert_eq!(
            result.before.unwrap(),
            Utc.with_ymd_and_hms(2024, 3, 31, 23, 59, 59).unwrap()
        );
    }
}
