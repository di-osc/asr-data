use std::collections::BTreeMap;

use asr_data::{
    Annotation, Audio, AudioChannel, AudioDb, AudioDbError, AudioDbMode, AudioEncoding, AudioError,
    AudioFormat, AudioQuery, AudioSource, DurationMs, MAX_QUERY_LIMIT, Sentence, SpeakerPayload,
    TimeRange, TimeSpan, Timeline, Token, Transcription, Waveform,
};

fn pcm_source(duration_ms: usize, channels: u16) -> AudioSource {
    AudioSource::from_pcm_s16le(
        vec![0; duration_ms * usize::from(channels) * 2],
        1_000,
        channels,
    )
}

#[test]
fn waveform_round_trips_pcm16_samples() {
    let waveform = Waveform::from_i16_pcm(&[0, 16_384, -16_384, 32_767], 16_000);

    assert_eq!(waveform.sample_rate, 16_000);
    assert_eq!(waveform.channels, 1);
    assert_eq!(waveform.duration_ms(), 0.25);
    assert_eq!(waveform.to_i16_pcm(), vec![0, 16_384, -16_384, 32_767]);
}

#[test]
fn stereo_waveform_uses_interleaved_frames_for_duration_and_slicing() {
    let waveform = Waveform::new_with_channels(
        vec![
            1.0, 10.0, // frame 0
            2.0, 20.0, // frame 1
            3.0, 30.0, // frame 2
            4.0, 40.0, // frame 3
        ],
        1_000,
        2,
    );

    assert_eq!(waveform.frame_count(), 4);
    assert_eq!(waveform.duration_ms(), 4.0);

    let slice = waveform.slice_ms(1, 3);
    assert_eq!(slice.channels, 2);
    assert_eq!(slice.sample_rate, 1_000);
    assert_eq!(slice.samples, vec![2.0, 20.0, 3.0, 30.0]);
}

#[test]
fn stereo_waveform_rejects_incomplete_interleaved_frames() {
    assert_eq!(
        Waveform::try_new_with_channels(vec![1.0, 2.0, 3.0], 16_000, 2),
        Err(AudioError::IncompleteFrame {
            samples: 3,
            channels: 2,
        })
    );
}

#[test]
fn stereo_pcm_bytes_reject_incomplete_interleaved_frames() {
    let mut bytes = Vec::new();
    for sample in [1_i16, 2_i16, 3_i16] {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }

    assert_eq!(
        Waveform::from_i16_pcm_bytes_with_channels(&bytes, 16_000, 2),
        Err(AudioError::IncompleteFrame {
            samples: 3,
            channels: 2,
        })
    );
}

#[test]
fn stereo_waveform_can_extract_channels_and_downmix_to_mono() -> Result<(), AudioError> {
    let waveform = Waveform::new_with_channels(
        vec![
            1.0, 3.0, // frame 0
            2.0, 4.0, // frame 1
        ],
        16_000,
        2,
    );

    let left = waveform.channel(0)?;
    assert_eq!(left.channels, 1);
    assert_eq!(left.samples, vec![1.0, 2.0]);

    let right = waveform.channel(1)?;
    assert_eq!(right.channels, 1);
    assert_eq!(right.samples, vec![3.0, 4.0]);

    let mono = waveform.to_mono()?;
    assert_eq!(mono.channels, 1);
    assert_eq!(mono.samples, vec![2.0, 3.0]);
    Ok(())
}

#[test]
fn append_rejects_waveforms_with_different_channel_counts() {
    let mut mono = Waveform::new(vec![0.0, 1.0], 16_000);
    let stereo = Waveform::new_with_channels(vec![0.0, 0.0, 1.0, 1.0], 16_000, 2);

    assert_eq!(mono.append(&stereo), Err(AudioError::InvalidChannelCount));
}

#[test]
fn time_range_reports_duration_and_overlap() {
    let range = TimeRange::new(DurationMs(100), DurationMs(240));

    assert_eq!(range.duration(), DurationMs(140));
    assert!(range.overlaps(&TimeRange::new(DurationMs(120), DurationMs(190))));
    assert!(!range.overlaps(&TimeRange::new(DurationMs(30), DurationMs(90))));
}

#[test]
fn annotation_model_is_status_free() {
    let mut timeline = Timeline::new("audio", DurationMs(100));
    let annotation = TimeSpan::new(
        TimeRange::new(DurationMs(0), DurationMs(100)),
        Annotation::Transcription(Transcription::new("hello")),
        None,
    );

    timeline.push_reference(annotation).unwrap();

    assert_eq!(timeline.reference_transcript().text, "hello");
}

fn activity_annotation(
    start: u64,
    end: u64,
    event: Option<&str>,
    source: Option<&str>,
) -> TimeSpan {
    TimeSpan::new(
        TimeRange::new(DurationMs(start), DurationMs(end)),
        Annotation::Activity(asr_data::AudioActivity {
            event: event.map(str::to_owned),
        }),
        source.map(str::to_owned),
    )
}

fn speaker_annotation(
    start: u64,
    end: u64,
    name: &str,
    text: Option<&str>,
    source: Option<&str>,
) -> TimeSpan {
    TimeSpan::new(
        TimeRange::new(DurationMs(start), DurationMs(end)),
        Annotation::Speaker(SpeakerPayload {
            name: name.to_owned(),
            transcription: text.map(Transcription::new),
        }),
        source.map(str::to_owned),
    )
}

fn transcription_annotation(start: u64, end: u64, text: &str, source: Option<&str>) -> TimeSpan {
    TimeSpan::new(
        TimeRange::new(DurationMs(start), DurationMs(end)),
        Annotation::Transcription(Transcription::new(text)),
        source.map(str::to_owned),
    )
}

#[test]
fn annotation_overlap_reference_activity_is_partitioned_by_event() {
    let mut timeline = Timeline::new("audio", DurationMs(300));
    let first = timeline
        .push_reference(activity_annotation(0, 100, None, None))
        .unwrap()
        .id
        .clone();

    assert_eq!(
        timeline
            .push_reference(activity_annotation(0, 100, None, None))
            .unwrap()
            .id,
        first
    );
    assert_eq!(timeline.reference.len(), 1);
    assert!(
        timeline
            .push_reference(activity_annotation(100, 200, None, None))
            .is_ok()
    );
    assert!(
        timeline
            .push_reference(activity_annotation(50, 150, None, None))
            .is_err()
    );
    assert!(
        timeline
            .push_reference(activity_annotation(50, 150, Some("speech"), None))
            .is_ok()
    );
    assert!(
        timeline
            .push_reference(activity_annotation(200, 250, Some("   "), None))
            .is_err()
    );
}

#[test]
fn annotation_overlap_reference_speakers_are_partitioned_by_name() {
    let mut timeline = Timeline::new("audio", DurationMs(300));
    timeline
        .push_reference(speaker_annotation(0, 100, "alice", None, None))
        .unwrap();

    assert!(
        timeline
            .push_reference(speaker_annotation(50, 150, "alice", None, None))
            .is_err()
    );
    assert!(
        timeline
            .push_reference(speaker_annotation(50, 150, "bob", None, None))
            .is_ok()
    );
}

#[test]
fn annotation_overlap_reference_text_uses_one_top_level_lane() {
    let mut timeline = Timeline::new("audio", DurationMs(300));
    timeline
        .push_reference(transcription_annotation(0, 100, "hello", None))
        .unwrap();

    assert!(
        timeline
            .push_reference(transcription_annotation(50, 150, "world", None))
            .is_err()
    );
    assert!(
        timeline
            .push_reference(speaker_annotation(50, 150, "alice", Some("world"), None))
            .is_err()
    );

    let mut speakers = Timeline::new("audio", DurationMs(300));
    speakers
        .push_reference(speaker_annotation(0, 100, "alice", Some("hello"), None))
        .unwrap();
    assert!(
        speakers
            .push_reference(speaker_annotation(50, 150, "bob", Some("world"), None))
            .is_ok()
    );
}

#[test]
fn annotation_overlap_prediction_is_partitioned_by_source() {
    let mut timeline = Timeline::new("audio", DurationMs(300));
    timeline
        .push_prediction(activity_annotation(0, 100, None, Some("vad-a")))
        .unwrap();

    assert!(
        timeline
            .push_prediction(activity_annotation(50, 150, None, Some("vad-a")))
            .is_err()
    );
    assert!(
        timeline
            .push_prediction(activity_annotation(50, 150, None, Some("vad-b")))
            .is_ok()
    );
    assert!(
        timeline
            .push_prediction(activity_annotation(150, 200, None, None))
            .is_err()
    );
    assert!(
        timeline
            .push_prediction(activity_annotation(150, 200, None, Some("   ")))
            .is_err()
    );
}

#[test]
fn annotation_overlap_prediction_speaker_and_text_rules_are_per_source() {
    let mut speakers = Timeline::new("audio", DurationMs(300));
    speakers
        .push_prediction(speaker_annotation(0, 100, "alice", None, Some("diarizer")))
        .unwrap();
    assert!(
        speakers
            .push_prediction(speaker_annotation(50, 150, "alice", None, Some("diarizer")))
            .is_err()
    );
    assert!(
        speakers
            .push_prediction(speaker_annotation(50, 150, "bob", None, Some("diarizer")))
            .is_ok()
    );
    assert!(
        speakers
            .push_prediction(speaker_annotation(50, 150, "alice", None, Some("other")))
            .is_ok()
    );

    let mut text = Timeline::new("audio", DurationMs(300));
    text.push_prediction(transcription_annotation(0, 100, "hello", Some("asr")))
        .unwrap();
    assert!(
        text.push_prediction(speaker_annotation(
            50,
            150,
            "alice",
            Some("world"),
            Some("asr")
        ))
        .is_err()
    );
    assert!(
        text.push_prediction(transcription_annotation(50, 150, "world", Some("other")))
            .is_ok()
    );
}

#[test]
fn annotation_overlap_prediction_relabel_is_atomic() {
    let mut timeline = Timeline::new("audio", DurationMs(300));
    timeline
        .push_prediction(activity_annotation(0, 100, None, Some("a")))
        .unwrap();
    timeline
        .push_prediction(activity_annotation(50, 150, None, Some("b")))
        .unwrap();
    let before = timeline.prediction.clone();

    assert!(timeline.relabel_prediction_source("b", "a").is_err());
    assert_eq!(timeline.prediction, before);
}

#[test]
fn prediction_sources_are_grouped_by_annotation_kind() {
    let mut timeline = Timeline::new("audio", DurationMs(300));
    timeline
        .push_prediction(activity_annotation(
            0,
            100,
            Some("speech"),
            Some("silero-vad"),
        ))
        .unwrap();
    timeline
        .push_prediction(TimeSpan::new(
            TimeRange::new(DurationMs(0), DurationMs(100)),
            Annotation::Transcription(Transcription::new("hello")),
            Some("qwen-asr".to_owned()),
        ))
        .unwrap();

    let sources = timeline.prediction_sources();
    assert_eq!(sources["activity"], vec!["silero-vad"]);
    assert_eq!(sources["transcription"], vec!["qwen-asr"]);
    assert!(sources["speaker"].is_empty());
    assert_eq!(sources.len(), 6);
}

#[test]
fn annotation_overlap_audio_validation_rejects_direct_vector_mutation() {
    let mut audio = Audio::with_id("audio", pcm_source(300, 1)).expect("audio document");
    let timeline = audio
        .ensure_timeline(AudioChannel::Mono, Some(DurationMs(300)))
        .unwrap();
    timeline
        .reference
        .push(activity_annotation(0, 100, None, None));
    timeline
        .reference
        .push(activity_annotation(50, 150, None, None));

    assert!(audio.validate().is_err());
}

#[test]
fn timeline_derives_transcript_from_all_text_annotations() {
    let mut timeline = Timeline::new("audio_1", DurationMs(100));
    timeline
        .push_reference(TimeSpan::new(
            TimeRange::new(DurationMs(0), DurationMs(40)),
            Annotation::Transcription(Transcription {
                text: "partial".to_string(),
                tokens: vec![],
                language: None,
                confidence: None,
            }),
            None,
        ))
        .unwrap();
    timeline
        .push_reference(TimeSpan::new(
            TimeRange::new(DurationMs(40), DurationMs(100)),
            Annotation::Transcription(Transcription {
                text: "hello".to_string(),
                tokens: vec![
                    Token::new("hello").with_range(TimeRange::new(DurationMs(40), DurationMs(100))),
                ],
                language: Some("English".to_string()),
                confidence: None,
            }),
            None,
        ))
        .unwrap();
    timeline
        .push_reference(TimeSpan::new(
            TimeRange::new(DurationMs(100), DurationMs(130)),
            Annotation::Sentence(Sentence {
                text: "world".to_string(),
                tokens: vec![],
                language: None,
            }),
            None,
        ))
        .unwrap();

    let transcript = timeline.reference_transcript();

    assert_eq!(transcript.text, "partial hello world");
    assert_eq!(transcript.language.as_deref(), Some("English"));
    assert_eq!(transcript.segments.len(), 3);
}

#[test]
fn audio_keeps_independent_channel_timelines() {
    let mut audio = Audio::with_id("call-1", pcm_source(100, 2)).expect("audio document");
    let transcription = |text: &str| {
        TimeSpan::new(
            TimeRange::new(DurationMs(0), DurationMs(100)),
            Annotation::Transcription(Transcription::new(text)),
            None,
        )
    };

    audio
        .ensure_timeline(AudioChannel::Left, Some(DurationMs(100)))
        .expect("left timeline")
        .push_reference(transcription("caller"))
        .unwrap();
    audio
        .ensure_timeline(AudioChannel::Right, None)
        .expect("right timeline")
        .push_reference(transcription("agent"))
        .unwrap();

    assert_eq!(
        audio
            .timeline(AudioChannel::Left)
            .expect("valid channel")
            .expect("left timeline")
            .reference_transcript()
            .text,
        "caller"
    );
    assert_eq!(
        audio
            .timeline(AudioChannel::Right)
            .expect("valid channel")
            .expect("right timeline")
            .reference_transcript()
            .text,
        "agent"
    );
    assert!(audio.mono_timeline().is_none());
    assert!(audio.timeline(AudioChannel::Channel(0)).is_err());
    assert!(audio.timeline(AudioChannel::Channel(1)).is_err());
}

#[test]
fn audio_timelines_own_a_shared_duration_and_channel_normalization() {
    let mut audio = Audio::with_id("call-1", pcm_source(500, 2)).expect("audio document");
    audio
        .ensure_timeline(AudioChannel::from_index(1), Some(DurationMs(500)))
        .expect("right timeline");

    assert_eq!(audio.id, "call-1");
    assert_eq!(audio.timeline_duration(), Some(DurationMs(500)));
    assert_eq!(AudioChannel::from_index(0), AudioChannel::Left);
    assert_eq!(AudioChannel::from_index(1), AudioChannel::Right);
    assert_eq!(AudioChannel::Right.index(), Some(1));
    assert_eq!(AudioChannel::Mono.index(), None);
    assert_eq!(AudioChannel::Right.name(), "right");
    assert!(
        audio
            .timelines()
            .values()
            .all(|timeline| timeline.audio_id == "call-1" && timeline.duration == DurationMs(500))
    );
    assert!(audio.validate().is_ok());
    assert!(
        audio
            .remove_timeline(AudioChannel::Right)
            .expect("canonical channel")
            .is_some()
    );
}

#[test]
fn waveform_splits_stereo_at_low_energy_without_changing_samples() -> Result<(), AudioError> {
    let mut samples = vec![1.0_f32; 62];
    for frame in 25..28 {
        samples[frame * 2] = 0.0;
        samples[frame * 2 + 1] = 0.0;
    }
    let waveform = Waveform::new_with_channels(samples.clone(), 10, 2);

    let chunks = waveform.split_at_low_energy(DurationMs(3_000))?;

    assert_eq!(chunks.len(), 2);
    assert!(chunks.iter().all(|chunk| chunk.frame_count() <= 30));
    assert!(chunks.iter().all(|chunk| chunk.channels == 2));
    assert_eq!(
        chunks
            .iter()
            .flat_map(|chunk| chunk.samples.iter().copied())
            .collect::<Vec<_>>(),
        samples
    );
    Ok(())
}

#[test]
fn waveform_low_energy_split_rejects_zero_duration() {
    let waveform = Waveform::new(vec![0.0], 16_000);
    assert_eq!(
        waveform.split_at_low_energy(DurationMs(0)),
        Err(AudioError::InvalidChunkSize)
    );
}

fn annotated_audio() -> Audio {
    let mut timeline = Timeline::new("audio_1", DurationMs(100));
    timeline
        .push_reference(TimeSpan::new(
            TimeRange::new(DurationMs(0), DurationMs(100)),
            Annotation::Transcription(Transcription::new("hello")),
            None,
        ))
        .unwrap();
    let audio = Audio::with_id("audio_1", pcm_source(100, 2))
        .expect("audio document")
        .with_timeline(timeline)
        .with_metadata_value("sha256", serde_json::json!("sha"));
    audio.with_metadata_value("model", serde_json::json!("qwen3-asr"))
}

#[test]
fn waveform_from_pcm_matches_source_load() {
    let bytes = vec![0u8, 0, 0xe8, 0x03, 0x18, 0xfc, 0xd0, 0x07];
    let mut via_source = AudioSource::from_pcm_s16le(bytes.clone(), 8_000, 2)
        .load()
        .expect("source load");
    let via_source = via_source.as_waveform().expect("loaded waveform");
    let via_waveform = Waveform::from_pcm_s16le(bytes, 8_000, 2).expect("waveform from pcm");
    assert_eq!(via_source, via_waveform);
}

#[test]
fn waveform_applies_optional_sample_rate_and_mono_explicitly() {
    let bytes = [0_i16, 1000, -1000, 2000]
        .into_iter()
        .flat_map(i16::to_le_bytes)
        .collect::<Vec<_>>();
    let source = AudioSource::from_pcm_s16le(bytes, 8_000, 2);

    let mut audio = source.load().expect("load source");
    let transformed = audio
        .as_waveform()
        .expect("waveform")
        .to_mono()
        .expect("mono")
        .resample(16_000)
        .expect("resample");

    assert_eq!(transformed.sample_rate, 16_000);
    assert_eq!(transformed.channels, 1);
    assert!(transformed.samples.iter().all(|sample| sample.is_finite()));
    assert!(
        transformed
            .samples
            .iter()
            .all(|sample| (-1.0..=1.0).contains(sample))
    );

    let preserved = audio.as_waveform().expect("preserve source format");
    assert_eq!(preserved.sample_rate, 8_000);
    assert_eq!(preserved.channels, 2);
}

#[test]
fn audio_source_loads_pcm_and_waveform_ops_preserve_original_format() {
    let bytes = [0_i16, 1000, -1000, 2000]
        .into_iter()
        .flat_map(i16::to_le_bytes)
        .collect::<Vec<_>>();
    let mut audio = Audio::with_id("pcm", AudioSource::from_pcm_s16le(bytes, 8_000, 2))
        .expect("audio document");

    let waveform = audio.as_waveform().expect("load PCM source");
    assert_eq!(waveform.sample_rate, 8_000);
    assert_eq!(waveform.channels, 2);
    assert_eq!(
        waveform.source_format,
        Some(AudioFormat {
            encoding: AudioEncoding::PcmS16Le,
            sample_rate: 8_000,
            channels: 2,
        })
    );

    let mono = waveform.to_mono().expect("downmix");
    let resampled = mono.resample(16_000).expect("resample");
    assert_eq!(resampled.sample_rate, 16_000);
    assert_eq!(resampled.channels, 1);
    assert_eq!(resampled.source_format, waveform.source_format);
}

#[test]
fn audio_source_probe_and_audio_doc_initialize_timelines_without_decoding() {
    let source = AudioSource::from_pcm_s16le(vec![0; 24], 1_000, 2);
    let info = source.probe().expect("probe PCM source");
    let doc = Audio::from_source(source).expect("document from stereo source");

    assert_eq!(info.sample_rate, 1_000);
    assert_eq!(info.channels, 2);
    assert_eq!(info.frame_count, 6);
    assert_eq!(info.duration_ms(), 6.0);
    assert_eq!(doc.timelines().len(), 2);
    assert_eq!(
        doc.timeline(AudioChannel::Left)
            .expect("left channel")
            .expect("left timeline")
            .duration,
        DurationMs(6)
    );
    assert!(
        doc.timeline(AudioChannel::Right)
            .expect("right channel")
            .is_some()
    );
    assert!(
        doc.timeline(AudioChannel::Mono)
            .expect("mono channel")
            .is_none()
    );
}

#[test]
fn audio_doc_uses_ceiling_duration_and_indexed_multichannel_timeline_names() {
    let mono_source = AudioSource::from_pcm_s16le(vec![0; 4], 3, 1);
    let mono_doc = Audio::from_source(mono_source).expect("mono document");
    assert_eq!(
        mono_doc.timelines().keys().copied().collect::<Vec<_>>(),
        vec![AudioChannel::Mono]
    );
    assert_eq!(mono_doc.timeline_duration(), Some(DurationMs(667)));

    let source = AudioSource::from_pcm_s16le(vec![0; 16], 1_000, 4);
    let multi_doc = Audio::from_source(source).expect("multichannel document");
    assert_eq!(
        multi_doc.timelines().keys().copied().collect::<Vec<_>>(),
        vec![
            AudioChannel::Left,
            AudioChannel::Right,
            AudioChannel::Channel(2),
            AudioChannel::Channel(3),
        ]
    );
}

#[test]
fn audio_source_probe_rejects_incomplete_pcm_frames() {
    let source = AudioSource::from_pcm_s16le(vec![0; 6], 16_000, 2);
    assert!(source.probe().is_err());
}

#[test]
fn audio_db_crud_and_difference_update() {
    let path = std::env::temp_dir().join(format!("asr-db-{}.vasr", uuid::Uuid::new_v4().simple()));
    let mut db = AudioDb::create(&path).expect("open AudioDb");
    let mut first = annotated_audio();
    first
        .ensure_timeline(AudioChannel::Left, None)
        .expect("left timeline")
        .push_reference(TimeSpan::new(
            TimeRange::new(DurationMs(0), DurationMs(100)),
            Annotation::Transcription(Transcription::new("caller")),
            None,
        ))
        .unwrap();
    first
        .ensure_timeline(AudioChannel::Right, None)
        .expect("right timeline")
        .push_reference(TimeSpan::new(
            TimeRange::new(DurationMs(0), DurationMs(100)),
            Annotation::Transcription(Transcription::new("agent")),
            None,
        ))
        .unwrap();
    let mut second = Audio::with_id("second", pcm_source(250, 1)).expect("second audio document");
    second
        .ensure_timeline(AudioChannel::Mono, Some(DurationMs(250)))
        .expect("mono timeline");
    db.insert(&first).expect("insert first");
    db.insert(&second).expect("insert second");
    db.set_metadata("dataset", &serde_json::json!("calls"))
        .expect("set database metadata");
    assert_eq!(
        db.metadata("dataset").expect("database metadata"),
        Some(serde_json::json!("calls"))
    );

    assert!(!db.update(&first).expect("unchanged update"));
    first = first.with_metadata_value("split", serde_json::json!("train"));
    assert!(db.update(&first).expect("changed update"));
    assert!(!db.update(&first).expect("second unchanged update"));
    let mut batch = vec![first.clone(), second.clone()];
    batch[0]
        .metadata
        .insert("batch".to_string(), serde_json::json!(1));
    batch[1]
        .metadata
        .insert("batch".to_string(), serde_json::json!(1));
    assert_eq!(db.update_many(&batch).expect("batch update"), 2);
    let first_page = db
        .query(&AudioQuery {
            limit: 1,
            ..AudioQuery::default()
        })
        .expect("first page");
    assert_eq!(first_page[0].audio_id(), "audio_1");
    let second_page = db
        .query(&AudioQuery {
            limit: 1,
            after: Some(first_page[0].audio_id()),
            ..AudioQuery::default()
        })
        .expect("second page");
    assert_eq!(second_page[0].audio_id(), "second");
    let filtered = db
        .query(&AudioQuery {
            min_duration: Some(DurationMs(50)),
            max_duration: Some(DurationMs(150)),
            metadata: BTreeMap::from([("split".to_string(), serde_json::json!("train"))]),
            ..AudioQuery::default()
        })
        .expect("filtered query");
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].metadata["split"], serde_json::json!("train"));
    assert!(matches!(
        db.query(&AudioQuery {
            limit: MAX_QUERY_LIMIT + 1,
            ..AudioQuery::default()
        }),
        Err(AudioDbError::QueryLimitExceeded { .. })
    ));
    let missing = Audio::with_id("missing", pcm_source(1, 1)).expect("missing audio document");
    assert!(matches!(
        db.update(&missing),
        Err(AudioDbError::NotFound { audio_id }) if audio_id == "missing"
    ));
    assert!(db.delete("second").expect("delete"));
    assert!(!db.delete("second").expect("delete missing"));
    drop(db);
    let db = AudioDb::open(&path, AudioDbMode::ReadOnly).expect("reopen");
    let remaining = db.query(&AudioQuery::default()).expect("query remaining");
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].audio_id(), "audio_1");
    assert_eq!(
        remaining[0]
            .timeline(AudioChannel::Left)
            .expect("valid channel")
            .expect("left timeline")
            .reference_transcript()
            .text,
        "caller"
    );
    assert_eq!(
        remaining[0]
            .timeline(AudioChannel::Right)
            .expect("valid channel")
            .expect("right timeline")
            .reference_transcript()
            .text,
        "agent"
    );
    std::fs::remove_file(path).ok();
}

#[test]
fn audio_db_rejects_every_schema_before_v10() {
    let path = std::env::temp_dir().join(format!(
        "asr-db-old-schema-{}.vasr",
        uuid::Uuid::new_v4().simple()
    ));
    assert_eq!(AudioDb::SCHEMA_VERSION, 10);

    for version in 1..10 {
        let connection = rusqlite::Connection::open(&path).expect("open v1 fixture");
        connection
            .pragma_update(None, "application_id", 0x5641_5352_i64)
            .expect("set application id");
        connection
            .pragma_update(None, "user_version", version)
            .expect("set schema version");
        drop(connection);

        assert!(matches!(
            AudioDb::open(&path, AudioDbMode::ReadOnly),
            Err(AudioDbError::UnsupportedSchema { found, expected: 10 }) if found == version
        ));
        std::fs::remove_file(&path).ok();
    }
}

#[test]
fn audio_db_create_and_open_are_explicit() {
    let path = std::env::temp_dir().join(format!(
        "asr-db-explicit-{}.db",
        uuid::Uuid::new_v4().simple()
    ));

    assert!(matches!(
        AudioDb::open(&path, AudioDbMode::ReadWrite),
        Err(AudioDbError::DatabaseNotFound { .. })
    ));
    let db = AudioDb::create(&path).expect("create database");
    drop(db);
    assert!(matches!(
        AudioDb::create(&path),
        Err(AudioDbError::AlreadyExists { .. })
    ));
    AudioDb::open(&path, AudioDbMode::ReadWrite).expect("open existing database");
    std::fs::remove_file(path).ok();
}
