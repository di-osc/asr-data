use std::sync::{Arc, RwLock};

use crate::audio::{
    AudioChannel as RustAudioChannel, AudioEncoding, AudioSource as RustAudioSource,
};
use crate::db::AudioDbError as RustAudioDbError;
use crate::timeline::{AnnotationSource, AnnotationStatus};
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

pub(super) fn annotation_status(value: &str) -> PyResult<AnnotationStatus> {
    match value.to_ascii_lowercase().as_str() {
        "partial" => Ok(AnnotationStatus::Partial),
        "final" => Ok(AnnotationStatus::Final),
        "revised" => Ok(AnnotationStatus::Revised),
        "deleted" => Ok(AnnotationStatus::Deleted),
        _ => Err(PyValueError::new_err(format!(
            "unsupported annotation status {value:?}"
        ))),
    }
}

pub(super) fn annotation_source(kind: &str, name: &str) -> PyResult<AnnotationSource> {
    match kind.to_ascii_lowercase().as_str() {
        "user" => Ok(AnnotationSource::User),
        "model" => Ok(AnnotationSource::Model(name.to_string())),
        "stage" => Ok(AnnotationSource::Stage(name.to_string())),
        "system" => Ok(AnnotationSource::System),
        _ => Err(PyValueError::new_err(format!(
            "unsupported annotation source kind {kind:?}; expected user, model, stage, or system"
        ))),
    }
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

pub(super) fn source_kind(source: &AnnotationSource) -> &'static str {
    match source {
        AnnotationSource::User => "user",
        AnnotationSource::Model(_) => "model",
        AnnotationSource::Stage(_) => "stage",
        AnnotationSource::System => "system",
    }
}

pub(super) fn source_name(source: &AnnotationSource) -> Option<&str> {
    match source {
        AnnotationSource::Model(name) | AnnotationSource::Stage(name) => Some(name),
        AnnotationSource::User | AnnotationSource::System => None,
    }
}

pub(super) fn status_name(status: &AnnotationStatus) -> &'static str {
    match status {
        AnnotationStatus::Partial => "partial",
        AnnotationStatus::Final => "final",
        AnnotationStatus::Revised => "revised",
        AnnotationStatus::Deleted => "deleted",
    }
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
        RustAudioSource::Url(url) => format!("url={:?}", truncate(url, 72)),
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
