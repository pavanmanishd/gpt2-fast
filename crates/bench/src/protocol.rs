use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use serde_json::{json, Value};

#[derive(Debug, PartialEq)]
pub enum ResponseLine {
    Ready { load_ms: f64 },
    Logits(Vec<f32>),
    Token(u32),
    Tokens(Vec<u32>),
    Done,
}

pub fn parse_line(line: &str) -> Result<ResponseLine, String> {
    let value: Value = serde_json::from_str(line)
        .map_err(|error| format!("invalid JSON on worker stdout: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| format!("worker line is not a JSON object: {line}"))?;

    if let Some(ready) = object.get("ready") {
        if ready.as_bool() != Some(true) {
            return Err(format!("worker reported ready={ready}, expected true"));
        }
        let load_ms = object
            .get("load_ms")
            .and_then(Value::as_f64)
            .ok_or_else(|| "ready line has no numeric load_ms".to_string())?;
        return Ok(ResponseLine::Ready { load_ms });
    }

    if let Some(logits) = object.get("logits") {
        let values = logits
            .as_array()
            .ok_or_else(|| "logits is not an array".to_string())?;
        let mut floats = Vec::with_capacity(values.len());
        for value in values {
            let number = value
                .as_f64()
                .ok_or_else(|| format!("logit is not a number: {value}"))?;
            floats.push(number as f32);
        }
        return Ok(ResponseLine::Logits(floats));
    }

    if let Some(token) = object.get("token") {
        let id = token
            .as_u64()
            .ok_or_else(|| format!("token is not a non-negative integer: {token}"))?;
        let id = u32::try_from(id).map_err(|_| format!("token id out of range: {id}"))?;
        return Ok(ResponseLine::Token(id));
    }

    if let Some(tokens) = object.get("tokens") {
        let values = tokens
            .as_array()
            .ok_or_else(|| "tokens is not an array".to_string())?;
        let mut ids = Vec::with_capacity(values.len());
        for value in values {
            let id = value
                .as_u64()
                .ok_or_else(|| format!("batch token is not a non-negative integer: {value}"))?;
            ids.push(u32::try_from(id).map_err(|_| format!("batch token id out of range: {id}"))?);
        }
        return Ok(ResponseLine::Tokens(ids));
    }

    if let Some(done) = object.get("done") {
        if done.as_bool() != Some(true) {
            return Err(format!("worker reported done={done}, expected true"));
        }
        return Ok(ResponseLine::Done);
    }

    Err(format!("unrecognised worker line: {line}"))
}

pub struct Response {
    pub logits: Option<Vec<f32>>,
    pub tokens: Vec<u32>,
    pub request_at: Instant,
    pub first_token_at: Option<Instant>,
    pub last_token_at: Option<Instant>,
}

pub struct BatchResponse {
    pub steps: Vec<Vec<u32>>,
    pub request_at: Instant,
    pub first_step_at: Option<Instant>,
    pub last_step_at: Option<Instant>,
}

pub struct Worker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Worker {
    pub fn spawn(command: &[String]) -> Result<Worker, String> {
        let (program, args) = command
            .split_first()
            .ok_or_else(|| "empty worker command".to_string())?;
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("cannot start `{program}`: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "worker stdin was not piped".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "worker stdout was not piped".to_string())?;
        Ok(Worker {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    pub fn ready(&mut self) -> Result<f64, String> {
        let line = self.read_line()?;
        match parse_line(&line)? {
            ResponseLine::Ready { load_ms } => Ok(load_ms),
            _ => Err(format!("expected the ready line, got: {line}")),
        }
    }

    pub fn request(
        &mut self,
        ids: &[u32],
        max_new_tokens: usize,
        return_logits: bool,
    ) -> Result<Response, String> {
        let request = json!({
            "ids": ids,
            "max_new_tokens": max_new_tokens,
            "return_logits": return_logits,
        });
        self.write_request(&request)?;
        let request_at = Instant::now();

        let mut response = Response {
            logits: None,
            tokens: Vec::new(),
            request_at,
            first_token_at: None,
            last_token_at: None,
        };
        loop {
            let line = self.read_line()?;
            match parse_line(&line)? {
                ResponseLine::Logits(logits) => {
                    if response.logits.is_some() {
                        return Err("worker sent more than one logits line".to_string());
                    }
                    response.logits = Some(logits);
                }
                ResponseLine::Token(token) => {
                    let now = Instant::now();
                    if response.first_token_at.is_none() {
                        response.first_token_at = Some(now);
                    }
                    response.last_token_at = Some(now);
                    response.tokens.push(token);
                }
                ResponseLine::Done => break,
                ResponseLine::Tokens(_) => {
                    return Err("worker sent a batch line for a single request".to_string())
                }
                ResponseLine::Ready { .. } => {
                    return Err("worker sent a second ready line".to_string())
                }
            }
        }
        Ok(response)
    }

    pub fn request_batch(
        &mut self,
        prompts: &[&[u32]],
        max_new_tokens: usize,
    ) -> Result<BatchResponse, String> {
        let request = json!({
            "batch": prompts,
            "max_new_tokens": max_new_tokens,
        });
        self.write_request(&request)?;
        let request_at = Instant::now();

        let mut response = BatchResponse {
            steps: Vec::new(),
            request_at,
            first_step_at: None,
            last_step_at: None,
        };
        loop {
            let line = self.read_line()?;
            match parse_line(&line)? {
                ResponseLine::Tokens(tokens) => {
                    let now = Instant::now();
                    if response.first_step_at.is_none() {
                        response.first_step_at = Some(now);
                    }
                    response.last_step_at = Some(now);
                    response.steps.push(tokens);
                }
                ResponseLine::Done => break,
                ResponseLine::Logits(_) | ResponseLine::Token(_) | ResponseLine::Ready { .. } => {
                    return Err(format!("unexpected line in a batch response: {line}"))
                }
            }
        }
        Ok(response)
    }

    fn write_request(&mut self, request: &Value) -> Result<(), String> {
        let line = serde_json::to_string(request)
            .map_err(|error| format!("cannot encode request: {error}"))?;
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .map_err(|error| format!("cannot write to worker stdin: {error}"))
    }

    pub fn peak_mb(&self) -> Option<f64> {
        let status = fs::read_to_string(format!("/proc/{}/status", self.child.id())).ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kilobytes: f64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kilobytes / 1024.0);
            }
        }
        None
    }

    pub fn finish(mut self) -> Result<(), String> {
        drop(self.stdin);
        let status = self
            .child
            .wait()
            .map_err(|error| format!("waiting for worker: {error}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("worker exited with {status}"))
        }
    }

    fn read_line(&mut self) -> Result<String, String> {
        let mut line = String::new();
        let read = self
            .stdout
            .read_line(&mut line)
            .map_err(|error| format!("cannot read worker stdout: {error}"))?;
        if read == 0 {
            return Err("worker closed stdout before completing the request".to_string());
        }
        Ok(line.trim_end_matches(['\n', '\r']).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_line, ResponseLine};

    #[test]
    fn parses_ready_line() {
        assert_eq!(
            parse_line(r#"{"ready": true, "load_ms": 12.5}"#),
            Ok(ResponseLine::Ready { load_ms: 12.5 })
        );
    }

    #[test]
    fn parses_logits_line() {
        assert_eq!(
            parse_line(r#"{"logits": [1.5, -2.25, 0]}"#),
            Ok(ResponseLine::Logits(vec![1.5, -2.25, 0.0]))
        );
    }

    #[test]
    fn parses_token_line() {
        assert_eq!(
            parse_line(r#"{"token": 50256}"#),
            Ok(ResponseLine::Token(50256))
        );
    }

    #[test]
    fn parses_batch_token_line() {
        assert_eq!(
            parse_line(r#"{"tokens": [12, 0, 50256]}"#),
            Ok(ResponseLine::Tokens(vec![12, 0, 50256]))
        );
    }

    #[test]
    fn rejects_batch_tokens_that_are_not_integers() {
        assert!(parse_line(r#"{"tokens": [12, -1]}"#).is_err());
        assert!(parse_line(r#"{"tokens": 12}"#).is_err());
    }

    #[test]
    fn parses_done_line() {
        assert_eq!(parse_line(r#"{"done": true}"#), Ok(ResponseLine::Done));
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(parse_line("not json").is_err());
    }

    #[test]
    fn rejects_unknown_object() {
        assert!(parse_line(r#"{"failure": true}"#).is_err());
    }

    #[test]
    fn rejects_token_that_is_not_an_integer() {
        assert!(parse_line(r#"{"token": -1}"#).is_err());
    }

    #[test]
    fn rejects_ready_without_load_ms() {
        assert!(parse_line(r#"{"ready": true}"#).is_err());
    }
}
