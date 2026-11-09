use serde::Serialize;

use crate::stats::median;
use crate::types::Session;

#[derive(Debug, Serialize)]
pub struct Summary {
    pub label: String,
    pub model: String,
    pub target: String,
    pub decode_tok_s: f64,
    pub prefill_tok_s: f64,
    pub ttft_ms: f64,
    pub peak_mb: Option<f64>,
    pub matches_reference: bool,
    pub token_match_rate: f64,
    pub sessions: usize,
}

#[derive(Debug, Serialize)]
pub struct DraftLogLine {
    pub date: String,
    pub milestone: String,
    pub title: String,
    pub commit: String,
    pub model: String,
    pub target: String,
    pub decode_tok_s: f64,
    pub prefill_tok_s: f64,
    pub ttft_ms: f64,
    pub peak_mb: Option<f64>,
    pub matches_reference: bool,
    pub token_match_rate: f64,
    pub sessions: usize,
    pub host: String,
    pub journal: String,
    pub flamegraph: String,
    pub headline: String,
}

pub fn summarize(sessions: &[Session]) -> Result<Summary, String> {
    let first = sessions
        .first()
        .ok_or_else(|| "cannot summarize zero sessions".to_string())?;

    let decode_tok_s = median(
        sessions
            .iter()
            .map(|session| session.session_medians.decode_tok_s)
            .collect(),
    )
    .unwrap_or(0.0);
    let prefill_tok_s = median(
        sessions
            .iter()
            .map(|session| session.session_medians.prefill_tok_s)
            .collect(),
    )
    .unwrap_or(0.0);
    let ttft_ms = median(
        sessions
            .iter()
            .map(|session| session.session_medians.ttft_ms)
            .collect(),
    )
    .unwrap_or(0.0);
    let peak_mb = median(
        sessions
            .iter()
            .filter_map(|session| session.peak_mb)
            .collect(),
    );

    let matches_reference = sessions.iter().all(|session| session.matches_reference);
    let matched_tokens: usize = sessions
        .iter()
        .flat_map(|session| &session.prompts)
        .map(|prompt| prompt.matched_tokens)
        .sum();
    let total_tokens: usize = sessions
        .iter()
        .flat_map(|session| &session.prompts)
        .map(|prompt| prompt.total_tokens)
        .sum();
    let token_match_rate = if total_tokens == 0 {
        if matches_reference {
            1.0
        } else {
            0.0
        }
    } else {
        matched_tokens as f64 / total_tokens as f64
    };

    Ok(Summary {
        label: first.label.clone(),
        model: first.model.clone(),
        target: "cpu-linux".to_string(),
        decode_tok_s,
        prefill_tok_s,
        ttft_ms,
        peak_mb,
        matches_reference,
        token_match_rate,
        sessions: sessions.len(),
    })
}

pub fn draft_log_line(summary: &Summary) -> DraftLogLine {
    DraftLogLine {
        date: "<date>".to_string(),
        milestone: summary.label.clone(),
        title: "<title>".to_string(),
        commit: "<commit>".to_string(),
        model: summary.model.clone(),
        target: summary.target.clone(),
        decode_tok_s: round3(summary.decode_tok_s),
        prefill_tok_s: round3(summary.prefill_tok_s),
        ttft_ms: round3(summary.ttft_ms),
        peak_mb: summary.peak_mb.map(round3),
        matches_reference: summary.matches_reference,
        token_match_rate: round6(summary.token_match_rate),
        sessions: summary.sessions,
        host: "<host>".to_string(),
        journal: "<journal>".to_string(),
        flamegraph: "<flamegraph>".to_string(),
        headline: "<headline>".to_string(),
    }
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::{draft_log_line, summarize};
    use crate::host::HostInfo;
    use crate::types::{Medians, PromptSession, Session};

    fn session(
        decode_tok_s: f64,
        prefill_tok_s: f64,
        ttft_ms: f64,
        peak_mb: Option<f64>,
        matched_tokens: usize,
        total_tokens: usize,
        matches_reference: bool,
    ) -> Session {
        Session {
            label: "N1".to_string(),
            model: "gpt2".to_string(),
            commit: "abc1234".to_string(),
            started_at: "2026-01-01T00:00:00Z".to_string(),
            host: HostInfo {
                nproc: 4,
                cpu: "test cpu".to_string(),
                memory_limit: "8589934592".to_string(),
                kernel: "test".to_string(),
            },
            worker_cmd: vec!["worker".to_string()],
            load_ms: 1.0,
            peak_mb,
            warmup: 1,
            runs: 2,
            prompts: vec![PromptSession {
                id: "p016".to_string(),
                n_prompt: 16,
                logit_max_abs_diff: 0.0,
                matched_tokens,
                total_tokens,
                runs: Vec::new(),
                medians: Medians {
                    decode_tok_s,
                    prefill_tok_s,
                    ttft_ms,
                },
            }],
            session_medians: Medians {
                decode_tok_s,
                prefill_tok_s,
                ttft_ms,
            },
            matches_reference,
        }
    }

    #[test]
    fn aggregates_medians_and_token_match_rate() {
        let sessions = vec![
            session(10.0, 100.0, 4.0, Some(100.0), 10, 10, true),
            session(20.0, 300.0, 8.0, Some(300.0), 7, 10, true),
        ];
        let summary = summarize(&sessions).unwrap();
        assert_eq!(summary.decode_tok_s, 15.0);
        assert_eq!(summary.prefill_tok_s, 200.0);
        assert_eq!(summary.ttft_ms, 6.0);
        assert_eq!(summary.peak_mb, Some(200.0));
        assert_eq!(summary.token_match_rate, 0.85);
        assert_eq!(summary.sessions, 2);
        assert!(summary.matches_reference);
        assert_eq!(summary.label, "N1");
        assert_eq!(summary.target, "cpu-linux");
    }

    #[test]
    fn ignores_null_peak_memory() {
        let sessions = vec![
            session(10.0, 100.0, 4.0, None, 10, 10, true),
            session(20.0, 300.0, 8.0, Some(300.0), 10, 10, true),
            session(30.0, 500.0, 12.0, None, 10, 10, true),
        ];
        let summary = summarize(&sessions).unwrap();
        assert_eq!(summary.peak_mb, Some(300.0));
        assert_eq!(summary.decode_tok_s, 20.0);
    }

    #[test]
    fn reports_a_mismatching_session() {
        let sessions = vec![
            session(10.0, 100.0, 4.0, Some(100.0), 10, 10, true),
            session(20.0, 300.0, 8.0, Some(300.0), 9, 10, false),
        ];
        let summary = summarize(&sessions).unwrap();
        assert!(!summary.matches_reference);
        assert_eq!(summary.token_match_rate, 0.95);
    }

    #[test]
    fn rejects_zero_sessions() {
        assert!(summarize(&[]).is_err());
    }

    #[test]
    fn draft_line_has_every_contract_field() {
        let summary = summarize(&[session(10.0, 100.0, 4.0, None, 10, 10, true)]).unwrap();
        let value = serde_json::to_value(draft_log_line(&summary)).unwrap();
        let keys = [
            "date",
            "milestone",
            "title",
            "commit",
            "model",
            "target",
            "decode_tok_s",
            "prefill_tok_s",
            "ttft_ms",
            "peak_mb",
            "matches_reference",
            "token_match_rate",
            "sessions",
            "host",
            "journal",
            "flamegraph",
            "headline",
        ];
        for key in keys {
            assert!(value.get(key).is_some(), "missing field {key}");
        }
        assert_eq!(value["date"], "<date>");
        assert_eq!(value["milestone"], "N1");
        assert_eq!(value["commit"], "<commit>");
        assert_eq!(value["host"], "<host>");
        assert_eq!(value["journal"], "<journal>");
        assert_eq!(value["flamegraph"], "<flamegraph>");
        assert_eq!(value["headline"], "<headline>");
        assert_eq!(value["peak_mb"], serde_json::Value::Null);
    }
}
