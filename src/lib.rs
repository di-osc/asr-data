//! ASR data model types shared by offline and realtime inference.

pub mod audio;
mod db;
mod doc;
mod evaluation;
mod metrics;
#[cfg(feature = "python-bindings")]
mod python;
mod timeline;
mod utils;

pub use audio::{AudioChannel, AudioEncoding, AudioFormat, AudioInfo, AudioSource};
pub use audio::{AudioChunk, AudioChunks, AudioError, Waveform};
pub use db::{
    AudioDb, AudioDbError, AudioDbInfo, AudioDbMode, AudioQuery, DEFAULT_QUERY_LIMIT,
    MAX_QUERY_LIMIT, read_audio_db_info,
};
pub use doc::{Audio, AudioChannelError, AudioTimelineError, AudioValidationError};
pub use evaluation::{
    DatasetActivityEvaluation, DatasetActivityEventEvaluation, DatasetEvalError, DatasetEvaluation,
    DatasetEvaluator, DatasetTranscriptionEvaluation, evaluate_dataset,
};
pub use metrics::{CerStats, TextNormalizationError, compute_cer, normalize_for_cer, normalize_zh};
pub use timeline::{
    ActivityEvaluation, ActivityEventEvaluation, Annotation, AudioActivity, AudioId, LanguageTag,
    Sentence, SpeakerId, SpeakerPayload, TimeSpan, TimeSpanConflictKind, TimeSpanId,
    TimeSpanOverlap, Timeline, TimelineEvalConfig, TimelineEvalError, TimelineEvaluation,
    TimelineId, TimelineSpanError, Token, Transcript, Transcription, TranscriptionEvaluation,
    TranscriptionNormalization,
};
pub use utils::{DurationMs, SampleIndex, TimeRange};
