use std::time::{Duration, SystemTime};

use asr_data::{Audio, AudioDb, AudioDbError, AudioQuery, AudioSource};

#[test]
fn filters_by_automatic_creation_and_update_times() {
    assert_eq!(AudioDb::SCHEMA_VERSION, 11);
    let path = std::env::temp_dir().join(format!(
        "asr-db-timestamps-{}.db",
        uuid::Uuid::new_v4().simple()
    ));
    let db = AudioDb::create(&path).expect("open AudioDb");
    let mut audio = Audio::with_id(
        "timestamped",
        AudioSource::from_pcm_s16le(vec![0, 0], 1_000, 1),
    )
    .expect("create timestamped audio");
    let before_insert = SystemTime::now() - Duration::from_secs(1);
    db.insert(&audio).expect("insert timestamped audio");
    let after_insert = SystemTime::now() + Duration::from_secs(1);

    assert_eq!(
        db.query(&AudioQuery {
            created_from: Some(before_insert),
            created_until: Some(after_insert),
            ..AudioQuery::default()
        })
        .expect("query creation range")
        .len(),
        1
    );
    assert!(
        db.query(&AudioQuery {
            created_until: Some(before_insert),
            ..AudioQuery::default()
        })
        .expect("query before creation")
        .is_empty()
    );

    std::thread::sleep(Duration::from_millis(2));
    let update_boundary = SystemTime::now();
    assert!(!db.update(&audio).expect("no-op update"));
    assert!(
        db.query(&AudioQuery {
            updated_from: Some(update_boundary),
            ..AudioQuery::default()
        })
        .expect("query unchanged update")
        .is_empty()
    );

    std::thread::sleep(Duration::from_millis(2));
    audio
        .metadata
        .insert("changed".to_owned(), serde_json::json!(true));
    assert!(db.update(&audio).expect("changed update"));
    assert_eq!(
        db.query(&AudioQuery {
            updated_from: Some(update_boundary),
            ..AudioQuery::default()
        })
        .expect("query changed update")
        .len(),
        1
    );
    assert!(
        db.query(&AudioQuery {
            created_from: Some(update_boundary),
            ..AudioQuery::default()
        })
        .expect("creation time remains unchanged")
        .is_empty()
    );

    assert!(matches!(
        db.query(&AudioQuery {
            created_from: Some(after_insert),
            created_until: Some(before_insert),
            ..AudioQuery::default()
        }),
        Err(AudioDbError::InvalidCreatedTimeRange)
    ));
    assert!(matches!(
        db.query(&AudioQuery {
            updated_from: Some(after_insert),
            updated_until: Some(before_insert),
            ..AudioQuery::default()
        }),
        Err(AudioDbError::InvalidUpdatedTimeRange)
    ));

    drop(db);
    std::fs::remove_file(path).ok();
}
