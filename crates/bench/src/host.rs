use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct HostInfo {
    pub nproc: usize,
    pub cpu: String,
    pub memory_limit: String,
    pub kernel: String,
}

pub fn info() -> HostInfo {
    HostInfo {
        nproc: std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1),
        cpu: cpu_model(),
        memory_limit: memory_limit(),
        kernel: command_output("uname", &["-r"]).unwrap_or_else(|| "unknown".to_string()),
    }
}

pub fn timestamp_now() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format_timestamp(seconds)
}

fn cpu_model() -> String {
    if let Some(model) = command_output("sysctl", &["-n", "machdep.cpu.brand_string"]) {
        return model;
    }
    if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in contents.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                if let Some((_, value)) = rest.split_once(':') {
                    return value.trim().to_string();
                }
            }
        }
    }
    "unknown".to_string()
}

fn memory_limit() -> String {
    for path in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ] {
        if let Ok(value) = std::fs::read_to_string(path) {
            let value = value.trim();
            if !value.is_empty() {
                return value.to_string();
            }
        }
    }
    "unknown".to_string()
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn format_timestamp(seconds: u64) -> String {
    let days = (seconds / 86_400) as i64;
    let remainder = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        remainder / 3600,
        (remainder % 3600) / 60,
        remainder % 60
    )
}

// Howard Hinnant's civil_from_days, so no date library is needed.
fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = (z - era * 146_097) as u64;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era as i64 + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_prime + 2) / 5 + 1) as u32;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

#[cfg(test)]
mod tests {
    use super::format_timestamp;

    #[test]
    fn formats_unix_epoch() {
        assert_eq!(format_timestamp(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn formats_leap_day() {
        assert_eq!(format_timestamp(1_709_164_800), "2024-02-29T00:00:00Z");
    }
}
