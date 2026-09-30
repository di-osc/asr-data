use asr_data::{
    AudioEvent, Sentence, Speaker, Speech, TimeRange, Timeline, TimelineEvalConfig, Transcription,
    TranscriptionNormalization,
};

/// 给尚未写入时间轴的事件盖上区间和来源，供直接塞进向量的测试使用。
fn placed(start: usize, end: usize, source: Option<&str>, mut event: AudioEvent) -> AudioEvent {
    let source = source.map(str::to_owned);
    match &mut event {
        AudioEvent::Base {
            range,
            source: slot,
            ..
        } => {
            *range = TimeRange::new(start, end);
            *slot = source;
        }
        AudioEvent::Speech(speech) => {
            speech.range = TimeRange::new(start, end);
            speech.source = source;
        }
    }
    event
}

/// 构造活动事件。`None` 记为 `noise`，因为未命名活动已经取消。
fn activity(start: usize, end: usize, event: Option<&str>, source: Option<&str>) -> AudioEvent {
    let event = match event {
        Some("speech") => Speech::new().into(),
        Some(name) => AudioEvent::new(name),
        None => AudioEvent::new("noise"),
    };
    placed(start, end, source, event)
}

/// 构造覆盖整段区间的单句语音。
fn speech_text(text: &str, start: usize, end: usize) -> AudioEvent {
    Speech::new()
        .with_transcription(
            Transcription::new(text).with_sentences(vec![Sentence::new(text, start, end)]),
        )
        .into()
}

#[test]
fn evaluates_merged_activity_ranges() {
    let mut timeline = Timeline::new("audio", 1_000);
    timeline.reference.push(activity(100, 500, None, None));
    timeline.reference.push(activity(400, 600, None, None));
    timeline
        .prediction
        .push(activity(200, 700, None, Some("vad")));
    let result = timeline
        .evaluate(&TimelineEvalConfig::new().with_activity("vad"))
        .unwrap()
        .activity
        .remove("vad")
        .unwrap();
    assert_eq!(result.reference_ms, 500);
    assert_eq!(result.predicted_ms, 500);
    assert_eq!(result.true_positive_ms, 400);
    assert_eq!(result.false_positive_ms, 100);
    assert_eq!(result.false_negative_ms, 100);
    assert_eq!(result.true_negative_ms, 400);
    assert_eq!(result.precision(), 0.8);
    assert_eq!(result.recall(), 0.8);
    assert!((result.f1() - 0.8).abs() < f64::EPSILON);
    assert_eq!(result.iou(), 2.0 / 3.0);
    assert_eq!(result.events["noise"].true_positive_ms, 400);
    assert_eq!(result.events["noise"].false_positive_ms, 100);
    assert_eq!(result.events["noise"].false_negative_ms, 100);
}

#[test]
fn evaluates_activity_events_by_name() {
    let mut timeline = Timeline::new("audio", 1_000);
    timeline
        .reference
        .push(activity(100, 400, Some("speech"), None));
    timeline
        .reference
        .push(activity(500, 700, Some("music"), None));
    timeline
        .prediction
        .push(activity(100, 400, Some("cough"), Some("aed")));
    timeline
        .prediction
        .push(activity(500, 700, Some("speech"), Some("aed")));

    let result = timeline
        .evaluate(&TimelineEvalConfig::new().with_activity("aed"))
        .unwrap()
        .activity
        .remove("aed")
        .unwrap();

    assert_eq!(result.true_positive_ms, 500);
    assert_eq!(result.events["speech"].false_negative_ms, 300);
    assert_eq!(result.events["speech"].false_positive_ms, 200);
    assert_eq!(result.events["cough"].false_positive_ms, 300);
    assert_eq!(result.events["music"].false_negative_ms, 200);
}

#[test]
fn can_evaluate_transcription_without_normalization() {
    let mut timeline = Timeline::new("audio", 1_000);
    timeline
        .annotate(0, 1_000, speech_text("交易停滞", 0, 1_000))
        .unwrap();
    timeline
        .annotate_with(
            0,
            1_000,
            speech_text("交易停止", 0, 1_000),
            false,
            Some("asr"),
        )
        .unwrap();
    let config = TimelineEvalConfig::new()
        .with_transcription("asr")
        .with_transcription_normalization(TranscriptionNormalization::None);
    let result = timeline
        .evaluate(&config)
        .unwrap()
        .transcription
        .remove("asr")
        .unwrap();
    assert_eq!(result.stats.substitutions, 1);
    assert_eq!(result.matches(), 3);
    assert_eq!(result.stats.cer(), 0.25);
    assert!(!result.exact_match());
}

#[test]
fn automatically_evaluates_all_sources_with_references() {
    let mut timeline = Timeline::new("audio", 1_000);
    timeline
        .annotate(0, 1_000, speech_text("交易停滞", 0, 1_000))
        .unwrap();
    timeline
        .annotate_with(
            0,
            1_000,
            speech_text("交易停滞", 0, 1_000),
            false,
            Some("qwen"),
        )
        .unwrap();
    timeline
        .annotate_with(
            0,
            1_000,
            Speech::new()
                .with_speaker(Speaker::new("agent"))
                .with_transcription(
                    Transcription::new("交易停止").with_sentences(vec![Sentence::new(
                        "交易停止",
                        0,
                        1_000,
                    )]),
                ),
            false,
            Some("whisper"),
        )
        .unwrap();

    let result = timeline
        .evaluate(
            &TimelineEvalConfig::new()
                .with_transcription_normalization(TranscriptionNormalization::None),
        )
        .unwrap();
    assert_eq!(
        result.transcription.keys().cloned().collect::<Vec<_>>(),
        ["qwen", "whisper"]
    );
    assert_eq!(
        result.activity.keys().cloned().collect::<Vec<_>>(),
        ["qwen", "whisper"]
    );
    assert_eq!(result.transcription["qwen"].stats.cer(), 0.0);
    assert_eq!(result.transcription["whisper"].stats.cer(), 0.25);
}
