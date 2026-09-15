use asr_data::{
    AudioActivity, AudioDb, AudioFormat, AudioInfo, AudioQuery, AudioSource, DatasetEvaluator,
    SpeakerPayload, TimelineEvalConfig, Transcription, TranscriptionNormalization,
    evaluate_dataset,
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

fn activity(event: Option<&str>) -> AudioActivity {
    match event {
        Some(event) => AudioActivity::new().with_event(event),
        None => AudioActivity::new(),
    }
}

#[test]
fn aggregates_corpus_metrics_and_source_coverage() {
    let mut first = doc("first");
    let timeline = first.mono_timeline_mut().unwrap();
    timeline
        .annotate_span(0, 1_000, Transcription::new("aaaa"))
        .unwrap();
    timeline
        .annotate_span_with(0, 1_000, Transcription::new("aaab"), false, Some("qwen"))
        .unwrap();
    timeline
        .annotate_span_with(
            0,
            1_000,
            Transcription::new("aaaa"),
            false,
            Some("whisper"),
        )
        .unwrap();
    timeline
        .annotate_span(100, 500, activity(Some("speech")))
        .unwrap();
    timeline
        .annotate_span_with(200, 600, activity(Some("speech")), false, Some("vad"))
        .unwrap();

    let mut second = doc("second");
    let timeline = second.mono_timeline_mut().unwrap();
    timeline
        .annotate_span(0, 1_000, Transcription::new("a"))
        .unwrap();
    timeline
        .annotate_span_with(0, 1_000, Transcription::new(""), false, Some("qwen"))
        .unwrap();
    timeline
        .annotate_span(100, 500, activity(Some("speech")))
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
            .annotate_span(0, 1_000, Transcription::new("a"))
            .unwrap();
        timeline
            .annotate_span_with(0, 1_000, Transcription::new("a"), false, Some("asr"))
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
fn audio_db_speaker_evaluation_is_label_invariant() {
    let path = std::env::temp_dir().join(format!(
        "asr-data-speaker-eval-{}.db",
        uuid::Uuid::new_v4().simple()
    ));
    let db = AudioDb::create(&path).unwrap();
    let mut audio = doc("speakers");
    let timeline = audio.mono_timeline_mut().unwrap();
    timeline
        .annotate_span(0, 500, SpeakerPayload::new("alice"))
        .unwrap();
    timeline
        .annotate_span(500, 1_000, SpeakerPayload::new("bob"))
        .unwrap();
    timeline
        .annotate_span_with(
            0,
            500,
            SpeakerPayload::new("speaker_1"),
            false,
            Some("diarizer"),
        )
        .unwrap();
    timeline
        .annotate_span_with(
            500,
            1_000,
            SpeakerPayload::new("speaker_0"),
            false,
            Some("diarizer"),
        )
        .unwrap();
    db.insert(&audio).unwrap();

    let result = db
        .eval_speaker(&AudioQuery::default(), &["diarizer".to_owned()])
        .unwrap();
    let diarizer = &result["diarizer"];
    assert_eq!(diarizer.reference_speaker_ms, 1_000);
    assert_eq!(diarizer.correct_speaker_ms, 1_000);
    assert_eq!(diarizer.speaker_confusion_ms, 0);
    assert_eq!(diarizer.der(), 0.0);

    drop(db);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn streaming_evaluator_matches_the_convenience_function() {
    let mut audio = doc("one");
    let timeline = audio.mono_timeline_mut().unwrap();
    timeline
        .annotate_span(0, 1_000, Transcription::new("a"))
        .unwrap();
    timeline
        .annotate_span_with(0, 1_000, Transcription::new("a"), false, Some("asr"))
        .unwrap();
    let config = TimelineEvalConfig::new()
        .with_transcription("asr")
        .with_transcription_normalization(TranscriptionNormalization::None);

    let expected = evaluate_dataset([&audio], &config).unwrap();
    let mut evaluator = DatasetEvaluator::new(config);
    evaluator.push(&audio).unwrap();
    assert_eq!(evaluator.finish().unwrap(), expected);
}

#[test]
fn combined_sources_match_separate_source_evaluations_with_normalization() {
    let mut audio = doc("normalized");
    let timeline = audio.mono_timeline_mut().unwrap();
    timeline
        .annotate_span(0, 1_000, Transcription::new("今天是2024年1月2日"))
        .unwrap();
    timeline
        .annotate_span_with(
            0,
            1_000,
            Transcription::new("今天是二零二四年一月二日"),
            false,
            Some("qwen"),
        )
        .unwrap();
    timeline
        .annotate_span_with(
            0,
            1_000,
            Transcription::new("今天是2024年1月3日"),
            false,
            Some("whisper"),
        )
        .unwrap();

    let combined = evaluate_dataset(
        [&audio],
        &TimelineEvalConfig::new().with_all_transcriptions(),
    )
    .unwrap();
    for source in ["qwen", "whisper"] {
        let separate = evaluate_dataset(
            [&audio],
            &TimelineEvalConfig::new().with_transcription(source),
        )
        .unwrap();
        assert_eq!(
            combined.transcription[source],
            separate.transcription[source]
        );
    }
}
