//! ASR data model types shared by offline and realtime inference.

pub mod audio;
mod cer;
mod db;
mod extract_audio;
mod fasr;
mod media;
#[cfg(feature = "python-bindings")]
mod python;
mod record;
mod segment;
mod stream;
mod time;
mod timeline;
mod token;
mod waveform;

pub use audio::{AudioInput, AudioLoadOptions, AudioLoader, SAMPLE_RATE_HZ, normalize_audio_input};
pub use cer::{CerStats, compute_cer, normalize_for_cer};
pub use db::{
    AudioDb, AudioDbError, AudioDbInfo, AudioDbMode, AudioQuery, DEFAULT_QUERY_LIMIT,
    MAX_QUERY_LIMIT, import_legacy_msgpack_to_db, read_audio_db_info,
};
pub use extract_audio::{
    ExtractAudioSummary, extract_embedded_audio, extract_embedded_audio_from_db,
};
pub use fasr::{FasrConvertSummary, convert_fasr_audiolist_to_db, read_fasr_audio_list};
pub use media::{AudioChannel, AudioEncoding, AudioFormat, AudioSource};
pub use record::{Audio, LegacyImportError, read_legacy_msgpack};
pub use segment::{TextSpan, Transcript};
pub use stream::{AudioBytesStream, AudioChunk, AudioChunkList};
pub use time::{DurationMs, SampleIndex, TimeRange};
pub use timeline::{
    AcousticEvent, Annotation, AnnotationId, AnnotationPayload, AnnotationSource, AnnotationStatus,
    AudioId, Diagnostic, HotwordMatch, LanguageTag, SpeakerId, Timeline, TimelineId,
};
pub use token::Token;
pub use waveform::{Waveform, WaveformError};
