//! ASR 数据模型：音频文档、时间轴标注、评估指标和 SQLite 存储。

pub mod audio;
mod dataset;
mod db;
mod doc;
mod evaluation;
mod metrics;
#[cfg(feature = "python-bindings")]
mod python;
mod timeline;
mod utils;

pub use audio::{AudioChannel, AudioEncoding, AudioFormat, AudioInfo, AudioSource};
pub use audio::{AudioChunk, AudioError, Waveform};
pub use dataset::{AudioDataset, AudioDatasetError};
pub use db::{
    AudioDb, AudioDbError, AudioDbInfo, AudioDbMode, AudioQuery, DEFAULT_QUERY_LIMIT,
    MAX_QUERY_LIMIT, read_audio_db_info,
};
pub use doc::{Audio, AudioChannelError, AudioStream, AudioTimelineError, AudioValidationError};
pub use evaluation::{
    DatasetActivityEvaluation, DatasetActivityEventEvaluation, DatasetEvalError, DatasetEvaluation,
    DatasetEvaluator, DatasetSpeakerEvaluation, DatasetTranscriptionEvaluation, evaluate_dataset,
};
pub use metrics::{
    CerStats, ChineseTextNormalizationOptions, TextNormalizationError, compute_cer,
    normalize_for_cer, normalize_zh, normalize_zh_with_options,
};
pub use timeline::{
    ActivityEvaluation, ActivityEventEvaluation, Annotation, AudioActivity, AudioId, LanguageTag,
    Sentence, Speaker, SpeakerId, TimeSpan, TimeSpanConflictKind, TimeSpanId, TimeSpanOverlap,
    Timeline, TimelineEvalConfig, TimelineEvalError, TimelineEvaluation, TimelineId,
    TimelineSpanError, Token, Transcript, Transcription, TranscriptionEvaluation,
    TranscriptionNormalization,
};
pub use utils::{DurationMs, SampleIndex, TimeRange};
