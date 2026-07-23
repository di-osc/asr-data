use std::collections::BTreeMap;

use asr_data::{
    Annotation, AnnotationPayload, Audio, AudioChannel, AudioDb, AudioDbError, AudioDbMode,
    AudioDoc, AudioEncoding, AudioError, AudioFormat, AudioQuery, AudioSource, DurationMs,
    MAX_QUERY_LIMIT, TextSpan, TimeRange, Timeline, Token, Transcription,
    import_legacy_msgpack_to_db, read_legacy_msgpack,
};

#[test]
fn waveform_round_trips_pcm16_samples() {
    let waveform = Audio::from_i16_pcm(&[0, 16_384, -16_384, 32_767], 16_000);

    assert_eq!(waveform.sample_rate, 16_000);
    assert_eq!(waveform.channels, 1);
    assert_eq!(waveform.duration_ms(), 0.25);
    assert_eq!(waveform.to_i16_pcm(), vec![0, 16_384, -16_384, 32_767]);
}

#[test]
fn audio_ignores_legacy_normalization_state_and_no_longer_serializes_it() {
    let legacy = serde_json::json!({
        "samples": [0.25, -0.5],
        "sample_rate": 16_000,
        "channels": 1,
        "source_format": null,
        "is_normalized": true
    });
    let encoded = rmp_serde::to_vec_named(&legacy).expect("encode legacy audio");

    let audio: Audio = rmp_serde::from_slice(&encoded).expect("decode legacy audio");

    assert_eq!(audio.samples, vec![0.25, -0.5]);
    let current = rmp_serde::to_vec_named(&audio).expect("encode current audio");
    assert!(!String::from_utf8_lossy(&current).contains("is_normalized"));
}

#[test]
fn stereo_waveform_uses_interleaved_frames_for_duration_and_slicing() {
    let waveform = Audio::new_with_channels(
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
        Audio::try_new_with_channels(vec![1.0, 2.0, 3.0], 16_000, 2),
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
        Audio::from_i16_pcm_bytes_with_channels(&bytes, 16_000, 2),
        Err(AudioError::IncompleteFrame {
            samples: 3,
            channels: 2,
        })
    );
}

#[test]
fn stereo_waveform_can_extract_channels_and_downmix_to_mono() -> Result<(), AudioError> {
    let waveform = Audio::new_with_channels(
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
    let mut mono = Audio::new(vec![0.0, 1.0], 16_000);
    let stereo = Audio::new_with_channels(vec![0.0, 0.0, 1.0, 1.0], 16_000, 2);

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
    let annotation = Annotation::new(
        TimeRange::new(DurationMs(0), DurationMs(100)),
        AnnotationPayload::Transcription(Transcription::new("hello")),
        None,
    );

    timeline.push_reference(annotation).unwrap();

    assert_eq!(timeline.reference_transcript().text, "hello");
}

#[test]
fn timeline_derives_transcript_from_all_text_annotations() {
    let mut timeline = Timeline::new("audio_1", DurationMs(100));
    timeline
        .push_reference(Annotation::new(
            TimeRange::new(DurationMs(0), DurationMs(40)),
            AnnotationPayload::Transcription(Transcription {
                text: "partial".to_string(),
                tokens: vec![],
                language: None,
                confidence: None,
            }),
            None,
        ))
        .unwrap();
    timeline
        .push_reference(Annotation::new(
            TimeRange::new(DurationMs(40), DurationMs(100)),
            AnnotationPayload::Transcription(Transcription {
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
        .push_reference(Annotation::new(
            TimeRange::new(DurationMs(100), DurationMs(130)),
            AnnotationPayload::Sentence(TextSpan {
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
    let mut audio = AudioDoc::with_id("call-1", AudioSource::from_pcm_s16le(vec![0; 8], 8_000, 2));
    let transcription = |text: &str| {
        Annotation::new(
            TimeRange::new(DurationMs(0), DurationMs(100)),
            AnnotationPayload::Transcription(Transcription::new(text)),
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
    let mut audio = AudioDoc::with_id("call-1", AudioSource::from_encoded_bytes(vec![]));
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
    let waveform = Audio::new_with_channels(samples.clone(), 10, 2);

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
    let waveform = Audio::new(vec![0.0], 16_000);
    assert_eq!(
        waveform.split_at_low_energy(DurationMs(0)),
        Err(AudioError::InvalidChunkSize)
    );
}

fn annotated_audio() -> AudioDoc {
    let mut timeline = Timeline::new("audio_1", DurationMs(100));
    timeline
        .push_reference(Annotation::new(
            TimeRange::new(DurationMs(0), DurationMs(100)),
            AnnotationPayload::Transcription(Transcription::new("hello")),
            None,
        ))
        .unwrap();
    let audio = AudioDoc::with_id("audio_1", AudioSource::from_encoded_bytes(vec![1, 2, 3, 4]))
        .with_timeline(timeline)
        .with_metadata_value("sha256", serde_json::json!("sha"));
    audio.with_metadata_value("model", serde_json::json!("qwen3-asr"))
}

#[test]
fn waveform_from_pcm_matches_source_load() {
    let bytes = vec![0u8, 0, 0xe8, 0x03, 0x18, 0xfc, 0xd0, 0x07];
    let via_source = AudioSource::from_pcm_s16le(bytes.clone(), 8_000, 2)
        .load()
        .expect("source load");
    let via_waveform = Audio::from_pcm_s16le(bytes, 8_000, 2).expect("waveform from pcm");
    assert_eq!(via_source, via_waveform);
}

#[test]
fn audio_source_load_with_applies_optional_sample_rate_and_mono() {
    let bytes = [0_i16, 1000, -1000, 2000]
        .into_iter()
        .flat_map(i16::to_le_bytes)
        .collect::<Vec<_>>();
    let source = AudioSource::from_pcm_s16le(bytes, 8_000, 2);

    let transformed = source
        .load_with(Some(16_000), Some(true))
        .expect("transform source");

    assert_eq!(transformed.sample_rate, 16_000);
    assert_eq!(transformed.channels, 1);
    assert!(transformed.samples.iter().all(|sample| sample.is_finite()));
    assert!(
        transformed
            .samples
            .iter()
            .all(|sample| (-1.0..=1.0).contains(sample))
    );

    let preserved = source
        .load_with(None, Some(false))
        .expect("preserve source format");
    assert_eq!(preserved.sample_rate, 8_000);
    assert_eq!(preserved.channels, 2);
}

#[test]
fn audio_source_loads_pcm_and_waveform_ops_preserve_original_format() {
    let bytes = [0_i16, 1000, -1000, 2000]
        .into_iter()
        .flat_map(i16::to_le_bytes)
        .collect::<Vec<_>>();
    let audio = AudioDoc::with_id("pcm", AudioSource::from_pcm_s16le(bytes, 8_000, 2));

    let waveform = audio.source.load().expect("load PCM source");
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
fn audio_db_crud_and_difference_update() {
    let path = std::env::temp_dir().join(format!("asr-db-{}.vasr", uuid::Uuid::new_v4().simple()));
    let mut db = AudioDb::open(&path, AudioDbMode::ReadWrite).expect("open AudioDb");
    let mut first = annotated_audio();
    first
        .ensure_timeline(AudioChannel::Left, None)
        .expect("left timeline")
        .push_reference(Annotation::new(
            TimeRange::new(DurationMs(0), DurationMs(100)),
            AnnotationPayload::Transcription(Transcription::new("caller")),
            None,
        ))
        .unwrap();
    first
        .ensure_timeline(AudioChannel::Right, None)
        .expect("right timeline")
        .push_reference(Annotation::new(
            TimeRange::new(DurationMs(0), DurationMs(100)),
            AnnotationPayload::Transcription(Transcription::new("agent")),
            None,
        ))
        .unwrap();
    let mut second = AudioDoc::with_id("second", AudioSource::from_encoded_bytes(vec![5, 6, 7]));
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
    let missing = AudioDoc::with_id("missing", AudioSource::from_encoded_bytes(vec![]));
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
fn audio_db_v1_is_migrated_to_reference_prediction_v5() {
    let path = std::env::temp_dir().join(format!(
        "asr-db-v1-migration-{}.vasr",
        uuid::Uuid::new_v4().simple()
    ));
    let audio = annotated_audio();
    let source = rmp_serde::to_vec_named(&audio.source).expect("encode source");
    let timeline = rmp_serde::to_vec_named(audio.mono_timeline().expect("mono timeline"))
        .expect("encode timeline");
    let metadata = serde_json::to_string(&audio.metadata).expect("encode metadata");
    {
        let connection = rusqlite::Connection::open(&path).expect("open v1 fixture");
        connection
            .execute_batch(
                "PRAGMA application_id = 0x56415352;
                 PRAGMA user_version = 1;
                 CREATE TABLE metadata (
                     key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL
                 ) STRICT;
                 CREATE TABLE audios (
                     audio_id TEXT PRIMARY KEY NOT NULL,
                     source BLOB NOT NULL,
                     timeline BLOB NOT NULL,
                     metadata TEXT NOT NULL,
                     duration_ms INTEGER
                 ) STRICT;
                 CREATE INDEX audios_duration ON audios(duration_ms);",
            )
            .expect("create v1 schema");
        connection
            .execute(
                "INSERT INTO audios(audio_id, source, timeline, metadata, duration_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![audio.audio_id(), source, timeline, metadata, 100_i64],
            )
            .expect("insert v1 audio");
    }

    let db = AudioDb::open(&path, AudioDbMode::ReadWrite).expect("migrate v1 database");
    assert_eq!(AudioDb::SCHEMA_VERSION, 5);
    let migrated = db.query(&AudioQuery::default()).expect("query migrated");
    assert_eq!(
        migrated[0]
            .mono_timeline()
            .unwrap()
            .reference_transcript()
            .text,
        "hello"
    );
    drop(db);

    let connection = rusqlite::Connection::open(&path).expect("inspect v2 database");
    let version: i64 = connection
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .expect("version");
    assert_eq!(version, 5);
    assert_eq!(
        connection
            .query_row("SELECT COUNT(*) FROM audio_sources", [], |row| row
                .get::<_, i64>(0))
            .expect("source count"),
        1
    );
    assert_eq!(
        connection
            .query_row("SELECT COUNT(*) FROM timelines", [], |row| {
                row.get::<_, i64>(0)
            })
            .expect("timeline count"),
        1
    );
    std::fs::remove_file(path).ok();
}

#[test]
fn audio_db_v2_is_decoded_read_only_as_mono_timeline() {
    let path = std::env::temp_dir().join(format!(
        "asr-db-v2-read-only-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let audio = annotated_audio();
    let source = rmp_serde::to_vec_named(&audio.source).expect("encode source");
    let timeline = rmp_serde::to_vec_named(audio.mono_timeline().expect("mono timeline"))
        .expect("encode timeline");
    let metadata = serde_json::to_string(&audio.metadata).expect("encode metadata");
    {
        let connection = rusqlite::Connection::open(&path).expect("open v2 fixture");
        connection
            .execute_batch(
                "PRAGMA application_id = 0x56415352;
                 PRAGMA user_version = 2;
                 PRAGMA foreign_keys = ON;
                 CREATE TABLE metadata (
                     key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL
                 ) STRICT;
                 CREATE TABLE audios (
                     audio_id TEXT PRIMARY KEY NOT NULL,
                     metadata TEXT NOT NULL,
                     duration_ms INTEGER
                 ) STRICT;
                 CREATE TABLE audio_sources (
                     audio_id TEXT PRIMARY KEY NOT NULL REFERENCES audios(audio_id),
                     source BLOB NOT NULL
                 ) STRICT;
                 CREATE TABLE timelines (
                     audio_id TEXT PRIMARY KEY NOT NULL REFERENCES audios(audio_id),
                     timeline BLOB NOT NULL
                 ) STRICT;",
            )
            .expect("create v2 schema");
        connection
            .execute(
                "INSERT INTO audios(audio_id, metadata, duration_ms) VALUES (?1, ?2, ?3)",
                rusqlite::params![audio.audio_id(), metadata, 100_i64],
            )
            .expect("insert audio");
        connection
            .execute(
                "INSERT INTO audio_sources(audio_id, source) VALUES (?1, ?2)",
                rusqlite::params![audio.audio_id(), source],
            )
            .expect("insert source");
        connection
            .execute(
                "INSERT INTO timelines(audio_id, timeline) VALUES (?1, ?2)",
                rusqlite::params![audio.audio_id(), timeline],
            )
            .expect("insert timeline");
    }

    let db = AudioDb::open(&path, AudioDbMode::ReadOnly).expect("open v2 read-only");
    let loaded = db.query(&AudioQuery::default()).expect("query v2");
    assert_eq!(
        loaded[0]
            .mono_timeline()
            .unwrap()
            .reference_transcript()
            .text,
        "hello"
    );
    assert_eq!(loaded[0].timelines().len(), 1);
    drop(db);
    std::fs::remove_file(path).ok();
}

#[test]
fn audio_db_v3_is_decoded_read_only_with_top_level_identity_and_duration() {
    let path = std::env::temp_dir().join(format!(
        "asr-db-v3-read-only-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let audio = annotated_audio();
    let source = rmp_serde::to_vec_named(&audio.source).expect("encode source");
    let timelines = rmp_serde::to_vec_named(audio.timelines()).expect("encode timelines");
    let metadata = serde_json::to_string(&audio.metadata).expect("encode metadata");
    {
        let connection = rusqlite::Connection::open(&path).expect("open v3 fixture");
        connection
            .execute_batch(
                "PRAGMA application_id = 0x56415352;
                 PRAGMA user_version = 3;
                 CREATE TABLE metadata (
                     key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL
                 ) STRICT;
                 CREATE TABLE audios (
                     audio_id TEXT PRIMARY KEY NOT NULL,
                     metadata TEXT NOT NULL,
                     duration_ms INTEGER
                 ) STRICT;
                 CREATE TABLE audio_sources (
                     audio_id TEXT PRIMARY KEY NOT NULL REFERENCES audios(audio_id),
                     source BLOB NOT NULL
                 ) STRICT;
                 CREATE TABLE timelines (
                     audio_id TEXT PRIMARY KEY NOT NULL REFERENCES audios(audio_id),
                     timeline BLOB NOT NULL
                 ) STRICT;",
            )
            .expect("create v3 schema");
        connection
            .execute(
                "INSERT INTO audios(audio_id, metadata, duration_ms) VALUES (?1, ?2, ?3)",
                rusqlite::params![audio.audio_id(), metadata, 100_i64],
            )
            .expect("insert audio");
        connection
            .execute(
                "INSERT INTO audio_sources(audio_id, source) VALUES (?1, ?2)",
                rusqlite::params![audio.audio_id(), source],
            )
            .expect("insert source");
        connection
            .execute(
                "INSERT INTO timelines(audio_id, timeline) VALUES (?1, ?2)",
                rusqlite::params![audio.audio_id(), timelines],
            )
            .expect("insert timelines");
    }

    let db = AudioDb::open(&path, AudioDbMode::ReadOnly).expect("open v3 read-only");
    let loaded = db.query(&AudioQuery::default()).expect("query v3");
    assert_eq!(loaded[0].id, "audio_1");
    assert_eq!(loaded[0].timeline_duration(), Some(DurationMs(100)));
    assert_eq!(
        loaded[0]
            .mono_timeline()
            .unwrap()
            .reference_transcript()
            .text,
        "hello"
    );
    loaded[0].validate().expect("valid migrated audio");
    drop(db);
    std::fs::remove_file(path).ok();
}

#[test]
fn legacy_v1_record_list_is_migrated_when_read() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/lbg_call-100.vasr.msgpack");
    if !path.exists() {
        return;
    }
    let audios = read_legacy_msgpack(&path).expect("read legacy v1 fixture");
    assert!(!audios.is_empty());
    assert!(audios.iter().all(|audio| {
        !audio.mono_timeline().unwrap().audio_id.is_empty()
            && audio.mono_timeline().unwrap().annotation_count() != 0
    }));
    let db_path =
        std::env::temp_dir().join(format!("legacy-{}.vasr", uuid::Uuid::new_v4().simple()));
    let imported = import_legacy_msgpack_to_db(path, &db_path).expect("import legacy file");
    assert_eq!(imported, audios.len());
    let db = AudioDb::open(&db_path, AudioDbMode::ReadOnly).expect("open imported db");
    assert!(
        !db.query(&AudioQuery::default())
            .expect("query imported")
            .is_empty()
    );
    drop(db);
    std::fs::remove_file(db_path).ok();
}
