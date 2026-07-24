mod annotation;
mod data;
mod evaluation;
mod segment;

pub use annotation::{
    Annotation, AudioActivity, AudioId, LanguageTag, SpeakerId, SpeakerPayload, TimeSpan,
    TimeSpanId, TimelineId, Token, Transcription,
};
pub use data::{TimeSpanConflictKind, TimeSpanOverlap, Timeline, TimelineSpanError};
pub(crate) use evaluation::normalize_transcription_text;
pub use evaluation::{
    ActivityEvaluation, ActivityEventEvaluation, TimelineEvalConfig, TimelineEvalError,
    TimelineEvaluation, TranscriptionEvaluation, TranscriptionNormalization,
};
pub use segment::{Sentence, Transcript};
