//! ASR data model types shared by offline and realtime inference.

pub mod audio;
mod db;
mod doc;
mod metrics;
#[cfg(feature = "python-bindings")]
mod python;
mod timeline;
mod utils;

pub use audio::SAMPLE_RATE_HZ;
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
pub use metrics::{CerStats, compute_cer, normalize_for_cer};
pub use timeline::{
    AcousticEvent, Annotation, AnnotationId, AnnotationPayload, AnnotationSource, AnnotationStatus,
    AudioId, Diagnostic, HotwordMatch, LanguageTag, SpeakerId, TextSpan, Timeline, TimelineId,
    Token, Transcript,
};
pub use utils::{DurationMs, SampleIndex, TimeRange};
