use std::path::Path;

use asr_data::audio::{self, decode};
use asr_data::{
    Annotation, AnnotationPayload, Audio, AudioChannel, AudioChunk, AudioChunks, AudioDb,
    AudioDbError, AudioDbInfo, AudioDbMode, AudioDoc, AudioEncoding, AudioError, AudioFormat,
    AudioQuery, AudioSource, CerStats, DEFAULT_QUERY_LIMIT, DurationMs, MAX_QUERY_LIMIT,
    SampleIndex, SpeakerPayload, SpeechEvaluation, TextNormalizationError, TextSpan, TimeRange,
    Timeline, TimelineEvalConfig, TimelineEvalError, TimelineEvaluation, Token, Transcript,
    Transcription, TranscriptionEvaluation, TranscriptionNormalization, compute_cer,
    normalize_for_cer, normalize_zh, read_audio_db_info,
};

#[test]
fn stable_public_paths_compile() {
    let _: Option<Audio> = None;
    let _: Option<audio::Audio> = None;
    let _: Option<AudioDb> = None;
    let _: Option<AudioChunk> = None;
    let _: Option<AudioChunks> = None;
    let _: Option<AudioDoc> = None;
    let _: Option<Annotation> = None;
    let _: Option<AnnotationPayload> = None;
    let _: Option<AudioChannel> = None;
    let _: Option<AudioDbError> = None;
    let _: Option<AudioDbInfo> = None;
    let _: Option<AudioDbMode> = None;
    let _: Option<AudioEncoding> = None;
    let _: Option<AudioError> = None;
    let _: Option<AudioFormat> = None;
    let _: Option<AudioQuery> = None;
    let _: Option<AudioSource> = None;
    let _: Option<CerStats> = None;
    let _: Option<DurationMs> = None;
    let _: Option<SampleIndex> = None;
    let _: Option<SpeakerPayload> = None;
    let _: Option<TextSpan> = None;
    let _: Option<TimeRange> = None;
    let _: Option<Timeline> = None;
    let _: Option<Token> = None;
    let _: Option<Transcript> = None;
    let _: Option<Transcription> = None;
    let _: Option<TimelineEvalConfig> = None;
    let _: Option<TimelineEvalError> = None;
    let _: Option<TimelineEvaluation> = None;
    let _: Option<TranscriptionEvaluation> = None;
    let _: Option<SpeechEvaluation> = None;
    let _: Option<TranscriptionNormalization> = None;
    let _: Option<TextNormalizationError> = None;
    let _: fn(&str, &str) -> CerStats = compute_cer;
    let _: fn(&str, bool) -> String = normalize_for_cer;
    let _ = normalize_zh("2026");
    let timeline = Timeline::new("mono", DurationMs(1_000));
    let _ = timeline.eval(&TimelineEvalConfig::new().with_transcription("asr"));
    let _: fn(&Path) -> anyhow::Result<Audio> = decode::decode_path_audio;
    let _: usize = DEFAULT_QUERY_LIMIT;
    let _: usize = MAX_QUERY_LIMIT;
    let _ = || {
        let path = Path::new("unused");
        let _ = read_audio_db_info(path);
    };
}
