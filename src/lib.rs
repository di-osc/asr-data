//! ASR data model types shared by offline and realtime inference.

pub mod audio;
mod db;
mod doc;
mod metrics;
#[cfg(feature = "python-bindings")]
mod python;
mod timeline;
mod utils;

pub use audio::{Audio, AudioChunk, AudioChunks, AudioError};
pub use audio::{AudioChannel, AudioEncoding, AudioFormat, AudioSource};
pub use db::{
    AudioDb, AudioDbError, AudioDbInfo, AudioDbMode, AudioQuery, DEFAULT_QUERY_LIMIT,
    MAX_QUERY_LIMIT, import_legacy_msgpack_to_db, read_audio_db_info,
};
pub use doc::{
    AudioChannelError, AudioDoc, AudioTimelineError, AudioValidationError, LegacyImportError,
    read_legacy_msgpack,
};
pub use metrics::{
    CerStats, TextNormalizationError, compute_cer, normalize_for_cer, normalize_zh_tn,
};
pub use timeline::{
    AcousticEvent, Annotation, AnnotationId, AnnotationPayload, AnnotationStatus, AudioId,
    LanguageTag, SpeakerId, SpeakerPayload, SpeechEvaluation, TextSpan, Timeline,
    TimelineEvalConfig, TimelineEvalError, TimelineEvaluation, TimelineId, Token, Transcript,
    Transcription, TranscriptionEvaluation, TranscriptionNormalization,
};
pub use utils::{DurationMs, SampleIndex, TimeRange};
