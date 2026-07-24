use asr_data::{
    Annotation, AudioActivity, AudioDb, AudioFormat, AudioInfo, AudioQuery, AudioSource,
    DatasetEvaluator, DurationMs, TimeRange, TimeSpan, TimelineEvalConfig, Transcription,
    TranscriptionNormalization, evaluate_dataset,
};

fn doc(id: &str) -> asr_data::Audio {
    let source = AudioSource::PcmS16Le {
        bytes: vec![0; 32_000],
        sample_rate: 16_000,
        channels: 1,
    };
    let info = AudioInfo {
        sample_rate: 16_000,
        channels: 1,
        frame_count: 16_000,
        source_format: AudioFormat::pcm16_mono(16_000),
    };
    asr_data::Audio::with_id_from_info(id, source, &info)
}

fn transcription(text: &str, source: Option<&str>) -> TimeSpan {
    TimeSpan::new(
        TimeRange::new(DurationMs(0), DurationMs(1_000)),
        Annotation::Transcription(Transcription::new(text)),
        source.map(str::to_owned),
    )
}

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
fn aggregates_corpus_metrics_and_source_coverage() {
    let mut first = doc("first");
    let timeline = first.mono_timeline_mut().unwrap();
    timeline
        .annotate_span(true, transcription("aaaa", None))
        .unwrap();
    timeline
        .annotate_span(false, transcription("aaab", Some("qwen")))
        .unwrap();
    timeline
        .annotate_span(false, transcription("aaaa", Some("whisper")))
        .unwrap();
    timeline
        .annotate_span(true, activity(100, 500, Some("speech"), None))
        .unwrap();
    timeline
        .annotate_span(false, activity(200, 600, Some("speech"), Some("vad")))
        .unwrap();

    let mut second = doc("second");
    let timeline = second.mono_timeline_mut().unwrap();
    timeline
        .annotate_span(true, transcription("a", None))
        .unwrap();
    timeline
        .annotate_span(false, transcription("", Some("qwen")))
        .unwrap();
    timeline
        .annotate_span(true, activity(100, 500, Some("speech"), None))
        .unwrap();

    let third = doc("third");
    let docs = [first, second, third];
    let config = TimelineEvalConfig::new()
        .with_transcription_normalization(TranscriptionNormalization::None);
    let result = evaluate_dataset(&docs, &config).unwrap();

    assert_eq!(result.documents, 3);
    assert_eq!(result.timelines, 3);
    let qwen = &result.transcription["qwen"];
    assert_eq!(qwen.evaluated_timelines, 2);
    assert_eq!(qwen.unannotated_timelines, 1);
    assert_eq!(qwen.stats.substitutions, 1);
    assert_eq!(qwen.stats.deletions, 1);
    assert_eq!(qwen.stats.reference_chars, 5);
    assert!((qwen.cer() - 0.4).abs() < f64::EPSILON);
    assert_eq!(qwen.coverage(), 1.0);

    let whisper = &result.transcription["whisper"];
    assert_eq!(whisper.evaluated_timelines, 1);
    assert_eq!(whisper.missing_predictions, 1);
    assert_eq!(whisper.missing_prediction_ids, ["second:mono"]);
    assert_eq!(whisper.coverage(), 0.5);

    let vad = &result.activity["vad"];
    assert_eq!(vad.evaluated_timelines, 1);
    assert_eq!(vad.missing_predictions, 1);
    assert_eq!(vad.true_positive_ms, 300);
    assert_eq!(vad.false_positive_ms, 100);
    assert_eq!(vad.false_negative_ms, 100);
    assert_eq!(vad.true_negative_ms, 500);
    assert_eq!(vad.coverage(), 0.5);
    assert_eq!(vad.events["speech"].true_positive_ms, 300);
}

#[test]
fn audio_db_evaluation_pages_through_every_matching_document() {
    let path = std::env::temp_dir().join(format!(
        "asr-data-dataset-eval-{}.db",
        uuid::Uuid::new_v4().simple()
    ));
    let db = AudioDb::create(&path).unwrap();
    for index in 0..101 {
        let mut audio = doc(&format!("audio-{index:03}"));
        let timeline = audio.mono_timeline_mut().unwrap();
        timeline
            .annotate_span(true, transcription("a", None))
            .unwrap();
        timeline
            .annotate_span(false, transcription("a", Some("asr")))
            .unwrap();
        db.insert(&audio).unwrap();
    }

    let result = db
        .eval(
            &AudioQuery {
                limit: 17,
                ..AudioQuery::default()
            },
            &TimelineEvalConfig::new()
                .with_transcription("asr")
                .with_transcription_normalization(TranscriptionNormalization::None),
        )
        .unwrap();
    assert_eq!(result.documents, 101);
    assert_eq!(result.transcription["asr"].evaluated_timelines, 101);
    assert_eq!(result.transcription["asr"].cer(), 0.0);

    drop(db);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn streaming_evaluator_matches_the_convenience_function() {
    let mut audio = doc("one");
    let timeline = audio.mono_timeline_mut().unwrap();
    timeline
        .annotate_span(true, transcription("a", None))
        .unwrap();
    timeline
        .annotate_span(false, transcription("a", Some("asr")))
        .unwrap();
    let config = TimelineEvalConfig::new()
        .with_transcription("asr")
        .with_transcription_normalization(TranscriptionNormalization::None);

    let expected = evaluate_dataset([&audio], &config).unwrap();
    let mut evaluator = DatasetEvaluator::new(config);
    evaluator.push(&audio).unwrap();
    assert_eq!(evaluator.finish().unwrap(), expected);
}
