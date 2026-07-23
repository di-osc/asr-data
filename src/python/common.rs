use std::sync::{Arc, RwLock};

use crate::audio::{
    AudioChannel as RustAudioChannel, AudioEncoding, AudioSource as RustAudioSource,
};
use crate::db::AudioDbError as RustAudioDbError;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use super::AsrDataError;

pub(super) type SharedAudio = Arc<RwLock<crate::doc::AudioDoc>>;

pub(super) fn py_error(error: impl std::fmt::Display) -> PyErr {
    AsrDataError::new_err(error.to_string())
}

pub(super) fn py_db_error(error: RustAudioDbError) -> PyErr {
    match error {
        RustAudioDbError::NotFound { audio_id } => PyKeyError::new_err(audio_id),
        error => py_error(error),
    }
}

pub(super) fn poisoned(label: &str) -> PyErr {
    PyRuntimeError::new_err(format!("{label} lock is poisoned"))
}

pub(super) fn audio_channel(value: &Bound<'_, PyAny>) -> PyResult<RustAudioChannel> {
    if let Ok(name) = value.extract::<String>() {
        return match name.to_ascii_lowercase().as_str() {
            "mono" => Ok(RustAudioChannel::Mono),
            "left" => Ok(RustAudioChannel::Left),
            "right" => Ok(RustAudioChannel::Right),
            _ => Err(PyValueError::new_err(format!(
                "unsupported audio channel {name:?}; expected mono, left, right, or an index"
            ))),
        };
    }
    let index = value.extract::<i64>().map_err(|_| {
        PyValueError::new_err("audio channel must be mono, left, right, or a non-negative index")
    })?;
    match index {
        ..0 => Err(PyValueError::new_err(
            "audio channel index must be non-negative",
        )),
        _ => u16::try_from(index)
            .map(RustAudioChannel::from_index)
            .map_err(|_| PyValueError::new_err("audio channel index exceeds u16")),
    }
}

pub(super) fn audio_channel_name(channel: RustAudioChannel) -> String {
    channel.name()
}

pub(super) fn encoding_name(encoding: &AudioEncoding) -> String {
    match encoding {
        AudioEncoding::Wav => "wav".to_string(),
        AudioEncoding::Flac => "flac".to_string(),
        AudioEncoding::Mp3 => "mp3".to_string(),
        AudioEncoding::Ogg => "ogg".to_string(),
        AudioEncoding::PcmS16Le => "pcm_s16le".to_string(),
        AudioEncoding::Other(value) => value.clone(),
        AudioEncoding::Unknown => "unknown".to_string(),
    }
}

pub(super) fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut output = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    output.push('…');
    output
}

/// Keep URLs useful in reprs without dumping long paths or signed query strings.
pub(super) fn summarize_url(value: &str, max_chars: usize) -> String {
    let (base, has_suffix) = match value.find(['?', '#']) {
        Some(index) => (&value[..index], true),
        None => (value, false),
    };
    let suffix = if has_suffix { "?…" } else { "" };
    let base_budget = max_chars.saturating_sub(suffix.chars().count());

    if base.chars().count() <= base_budget {
        return format!("{base}{suffix}");
    }
    if base_budget <= 1 {
        return truncate(base, base_budget);
    }

    let right_chars = (base_budget / 2).max(1);
    let left_chars = base_budget.saturating_sub(right_chars + 1);
    let left = base.chars().take(left_chars).collect::<String>();
    let right = base
        .chars()
        .rev()
        .take(right_chars)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("{left}…{right}{suffix}")
}

pub(super) fn format_duration_ms(duration_ms: f64) -> String {
    if duration_ms < 1_000.0 {
        return format!("{duration_ms:.0}ms");
    }
    let seconds = duration_ms / 1_000.0;
    if seconds < 60.0 {
        return format!("{seconds:.2}s");
    }
    let minutes = (seconds / 60.0).floor() as u64;
    let remaining_seconds = seconds - minutes as f64 * 60.0;
    if minutes < 60 {
        return format!("{minutes}m{remaining_seconds:04.1}s");
    }
    let hours = minutes / 60;
    let remaining_minutes = minutes % 60;
    format!("{hours}h{remaining_minutes:02}m{remaining_seconds:04.1}s")
}

pub(super) fn format_source_field(source: &RustAudioSource) -> String {
    match source {
        RustAudioSource::Path(path) => {
            format!("file={:?}", truncate(&path.display().to_string(), 72))
        }
        RustAudioSource::Url(url) => format!("url={:?}", summarize_url(url, 72)),
        RustAudioSource::Base64(data) => format!("base64_chars={}", data.len()),
        RustAudioSource::EncodedBytes(bytes) => format!("bytes={}", bytes.len()),
        RustAudioSource::PcmS16Le {
            bytes,
            sample_rate,
            channels,
        } => format!(
            "pcm_bytes={}, sample_rate={}, channels={}",
            bytes.len(),
            sample_rate,
            channels
        ),
    }
}

#[cfg(test)]
mod repr_tests {
    use super::summarize_url;

    #[test]
    fn url_summary_keeps_both_ends_and_hides_query_values() {
        let value = "https://audio.example.com/a/very/long/path/session_123456789.wav?token=secret";
        let summary = summarize_url(value, 48);

        assert_eq!(summary.chars().count(), 48);
        assert!(summary.starts_with("https://audio.example"));
        assert!(summary.ends_with("session_123456789.wav?…"));
        assert!(!summary.contains("secret"));
    }

    #[test]
    fn short_url_is_left_unchanged() {
        assert_eq!(
            summarize_url("https://example.com/audio.wav", 72),
            "https://example.com/audio.wav"
        );
    }
}
