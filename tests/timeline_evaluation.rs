use asr_data::{
    Annotation, AudioActivity, DurationMs, Speaker, TimeRange, TimeSpan, Timeline,
    TimelineEvalConfig, Transcription, TranscriptionNormalization,
};

fn activity(start: u64, end: u64, event: Option<&str>, source: Option<&str>) -> TimeSpan {
    TimeSpan::new(
        TimeRange::new(DurationMs(start), DurationMs(end)),
        Annotation::Activity(AudioActivity {
            event: event.map(str::to_owned),
            confidence: None,
        }),
        source.map(str::to_owned),
    )
}

#[test]
fn evaluates_merged_activity_ranges() {
    let mut timeline = Timeline::new("audio", DurationMs(1_000));
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
    assert!(result.events.is_empty());
}

#[test]
fn evaluates_activity_events_and_masks_unknown_reference_ranges() {
    let mut timeline = Timeline::new("audio", DurationMs(1_000));
    timeline
        .reference
        .push(activity(100, 400, Some("speech"), None));
    timeline.reference.push(activity(500, 700, None, None));
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
    assert_eq!(result.events["speech"].false_positive_ms, 0);
    assert_eq!(result.events["cough"].false_positive_ms, 300);
}

#[test]
fn can_evaluate_transcription_without_normalization() {
    let mut timeline = Timeline::new("audio", DurationMs(1_000));
    timeline
        .annotate_span(0, 1_000, Transcription::new("交易停滞"))
        .unwrap();
    timeline
        .annotate_span_with(0, 1_000, Transcription::new("交易停止"), false, Some("asr"))
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
    let mut timeline = Timeline::new("audio", DurationMs(1_000));
    timeline
        .annotate_span(0, 1_000, Transcription::new("交易停滞"))
        .unwrap();
    timeline
        .annotate_span_with(
            0,
            1_000,
            Transcription::new("交易停滞"),
            false,
            Some("qwen"),
        )
        .unwrap();
    timeline
        .annotate_span_with(
            0,
            1_000,
            Speaker::new("agent").with_transcription(Transcription::new("交易停止")),
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
    assert!(result.activity.is_empty());
    assert_eq!(result.transcription["qwen"].stats.cer(), 0.0);
    assert_eq!(result.transcription["whisper"].stats.cer(), 0.25);
}
