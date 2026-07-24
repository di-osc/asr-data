use std::path::Path;

use asr_data::audio::{self, decode};
use asr_data::{
    ActivityEvaluation, ActivityEventEvaluation, Annotation, Audio, AudioActivity, AudioChannel,
    AudioChunk, AudioChunks, AudioDb, AudioDbError, AudioDbInfo, AudioDbMode, AudioEncoding,
    AudioError, AudioFormat, AudioInfo, AudioQuery, AudioSource, AudioStream, CerStats,
    DEFAULT_QUERY_LIMIT, DatasetActivityEvaluation, DatasetActivityEventEvaluation,
    DatasetEvalError, DatasetEvaluation, DatasetEvaluator, DatasetTranscriptionEvaluation,
    DurationMs, MAX_QUERY_LIMIT, SampleIndex, Sentence, SpeakerPayload, TextNormalizationError,
    TimeRange, TimeSpan, Timeline, TimelineEvalConfig, TimelineEvalError, TimelineEvaluation,
    Token, Transcript, Transcription, TranscriptionEvaluation, TranscriptionNormalization,
    Waveform, compute_cer, evaluate_dataset, normalize_for_cer, normalize_zh, read_audio_db_info,
};

#[test]
fn stable_public_paths_compile() {
    let _: Option<Waveform> = None;
    let _: Option<audio::Waveform> = None;
    let _: Option<AudioDb> = None;
    let _: Option<AudioChunk> = None;
    let _: Option<AudioChunks> = None;
    let _: Option<Audio> = None;
    let _: Option<AudioStream> = None;
    let _: Option<TimeSpan> = None;
    let _: Option<Annotation> = None;
    let _: Option<AudioChannel> = None;
    let _: Option<AudioDbError> = None;
    let _: Option<AudioDbInfo> = None;
    let _: Option<AudioDbMode> = None;
    let _: Option<AudioEncoding> = None;
    let _: Option<AudioError> = None;
    let _: Option<AudioFormat> = None;
    let _: Option<AudioInfo> = None;
    let _: Option<AudioQuery> = None;
    let _: Option<AudioSource> = None;
    let _: Option<CerStats> = None;
    let _: Option<DurationMs> = None;
    let _: Option<SampleIndex> = None;
    let _: Option<SpeakerPayload> = None;
    let _: Option<Sentence> = None;
    let _: Option<TimeRange> = None;
    let _: Option<Timeline> = None;
    let _: Option<Token> = None;
    let _: Option<Transcript> = None;
    let _: Option<Transcription> = None;
    let _: Option<TimelineEvalConfig> = None;
    let _: Option<TimelineEvalError> = None;
    let _: Option<TimelineEvaluation> = None;
    let _: Option<TranscriptionEvaluation> = None;
    let _: Option<AudioActivity> = None;
    let _: Option<ActivityEvaluation> = None;
    let _: Option<ActivityEventEvaluation> = None;
    let _: Option<TranscriptionNormalization> = None;
    let _: Option<DatasetEvalError> = None;
    let _: Option<DatasetEvaluation> = None;
    let _: Option<DatasetEvaluator> = None;
    let _: Option<DatasetTranscriptionEvaluation> = None;
    let _: Option<DatasetActivityEvaluation> = None;
    let _: Option<DatasetActivityEventEvaluation> = None;
    let _: Option<TextNormalizationError> = None;
    let _: fn(&str, &str) -> CerStats = compute_cer;
    let _: fn(&str, bool) -> String = normalize_for_cer;
    let _ = normalize_zh("2026");
    let _ = evaluate_dataset([], &TimelineEvalConfig::new());
    let timeline = Timeline::new("mono", DurationMs(1_000));
    let _ = timeline.eval(&TimelineEvalConfig::new().with_transcription("asr"));
    let _: fn(&Path) -> anyhow::Result<Waveform> = decode::decode_path_audio;
    let _: usize = DEFAULT_QUERY_LIMIT;
    let _: usize = MAX_QUERY_LIMIT;
    let _ = || {
        let path = Path::new("unused");
        let _ = read_audio_db_info(path);
    };
}

#[test]
fn audio_stream_grows_timelines_and_converts_without_redecoding() {
    let source = AudioSource::from_pcm_s16le(vec![0; 2_000], 1_000, 1);
    let mut stream = source
        .stream_with_id("stream-1", 250)
        .expect("create stream");

    assert_eq!(stream.position_ms(), 0);
    assert_eq!(
        stream
            .timeline(AudioChannel::Mono)
            .expect("valid channel")
            .expect("mono timeline")
            .duration,
        DurationMs(0),
    );

    let first = stream.next().expect("first chunk").expect("decode chunk");
    assert_eq!(first.offset_ms, 0);
    assert_eq!(stream.position_ms(), 250);
    assert_eq!(
        stream
            .timeline(AudioChannel::Mono)
            .expect("valid channel")
            .expect("mono timeline")
            .duration,
        DurationMs(250),
    );

    let remaining = stream
        .by_ref()
        .collect::<Result<Vec<_>, _>>()
        .expect("remaining chunks");
    assert!(!remaining.is_empty());
    assert!(stream.is_complete());
    assert_eq!(stream.as_waveform().frame_count(), 1_000);

    let mut audio = stream.into_audio().expect("complete stream converts");
    assert_eq!(audio.id, "stream-1");
    assert_eq!(
        audio.as_waveform().expect("cached waveform").frame_count(),
        1_000
    );
}

#[test]
fn audio_and_stream_convenience_factories_are_public() {
    let pcm = vec![0; 2_000];
    let mut audio = Audio::from_pcm_s16le(pcm.clone(), 1_000, 1).expect("load audio");
    assert_eq!(audio.as_waveform().expect("waveform").frame_count(), 1_000);

    let mut stream = AudioStream::from_pcm_s16le(pcm, 1_000, 1, 250).expect("create audio stream");
    assert_eq!(stream.by_ref().count(), 4);
    assert!(stream.is_complete());
}

#[test]
fn timeline_uses_one_annotation_write_method() {
    let mut timeline = Timeline::new("audio", DurationMs(1_000));
    let reference = TimeSpan::new(
        TimeRange::new(DurationMs(0), DurationMs(1_000)),
        Annotation::Activity(AudioActivity::new().with_event("speech")),
        None,
    );
    timeline
        .annotate_span(true, reference)
        .expect("reference annotation");

    let prediction = TimeSpan::new(
        TimeRange::new(DurationMs(0), DurationMs(1_000)),
        Annotation::Transcription(Transcription::new("hello")),
        Some("asr".to_owned()),
    );
    timeline
        .annotate_span(false, prediction)
        .expect("prediction annotation");

    assert_eq!(timeline.reference.len(), 1);
    assert_eq!(timeline.prediction.len(), 1);
}
