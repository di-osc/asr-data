//! ASR data model types shared by offline and realtime inference.

pub mod audio;
mod cer;
mod db;
mod doc;
mod extract_audio;
mod media;
#[cfg(feature = "python-bindings")]
mod python;
mod segment;
mod stream;
mod time;
mod timeline;
mod token;

pub use audio::waveform::{Waveform, WaveformError};
pub use audio::{AudioInput, AudioLoadOptions, AudioLoader, SAMPLE_RATE_HZ, normalize_audio_input};
pub use cer::{CerStats, compute_cer, normalize_for_cer};
pub use db::{
    AudioDb, AudioDbError, AudioDbInfo, AudioDbMode, AudioQuery, DEFAULT_QUERY_LIMIT,
    MAX_QUERY_LIMIT, import_legacy_msgpack_to_db, read_audio_db_info,
};
pub use doc::{
    AudioChannelError, AudioDoc, AudioValidationError, LegacyImportError, read_legacy_msgpack,
};
pub use extract_audio::{
    ExtractAudioSummary, extract_embedded_audio, extract_embedded_audio_from_db,
};
pub use media::{AudioChannel, AudioEncoding, AudioFormat, AudioSource};
pub use segment::{TextSpan, Transcript};
pub use stream::{AudioBytesStream, AudioChunk, AudioChunkList};
pub use time::{DurationMs, SampleIndex, TimeRange};
pub use timeline::{
    AcousticEvent, Annotation, AnnotationId, AnnotationPayload, AnnotationSource, AnnotationStatus,
    AudioId, Diagnostic, HotwordMatch, LanguageTag, SpeakerId, Timeline, TimelineId,
};
pub use token::Token;
