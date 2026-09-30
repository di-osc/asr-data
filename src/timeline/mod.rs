//! 时间轴、标注和按来源评估转写 / 活动。

mod annotation;
mod data;
#[cfg(feature = "metrics")]
mod evaluation;
mod segment;

pub use annotation::{
    AudioEvent, AudioId, EventId, Gender, LanguageTag, SPEECH_EVENT_NAME, Speaker, SpeakerId,
    Speech, TimelineId, Token, Transcription,
};
pub use data::{EventConflictKind, EventOverlap, Timeline, TimelineEventError};
#[cfg(all(feature = "db", feature = "metrics"))]
pub(crate) use evaluation::normalize_transcription_text;
#[cfg(feature = "metrics")]
pub use evaluation::{
    ActivityEvaluation, ActivityEventEvaluation, TimelineEvalConfig, TimelineEvalError,
    TimelineEvaluation, TranscriptionEvaluation, TranscriptionNormalization,
};
pub use segment::{Sentence, Transcript};
