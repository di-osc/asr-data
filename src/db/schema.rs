use std::collections::BTreeMap;

use rusqlite::{Connection, params};

use crate::audio::AudioChannel;
use crate::timeline::Timeline;

use super::query::{decode, encode};
use super::{
    APPLICATION_ID, AudioDbError, CHANNEL_TIMELINE_SCHEMA_VERSION, LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION, SPLIT_TABLE_SCHEMA_VERSION,
};

pub(super) fn initialize(connection: &Connection) -> Result<(), AudioDbError> {
    let current: i64 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if current != 0 {
        migrate(connection)?;
        validate(connection)?;
        return Ok(());
    }
    connection.execute_batch(&format!(
        "PRAGMA application_id = {APPLICATION_ID};
         PRAGMA user_version = {SCHEMA_VERSION};
         PRAGMA foreign_keys = ON;
         CREATE TABLE metadata (
             key   TEXT PRIMARY KEY NOT NULL,
             value TEXT NOT NULL
         ) STRICT;
         CREATE TABLE audios (
             audio_id   TEXT PRIMARY KEY NOT NULL,
             metadata   TEXT NOT NULL,
             duration_ms INTEGER
         ) STRICT;
         CREATE TABLE audio_sources (
             audio_id TEXT PRIMARY KEY NOT NULL
                 REFERENCES audios(audio_id) ON DELETE CASCADE,
             source BLOB NOT NULL
         ) STRICT;
         CREATE TABLE timelines (
             audio_id TEXT PRIMARY KEY NOT NULL
                 REFERENCES audios(audio_id) ON DELETE CASCADE,
             timeline BLOB NOT NULL
         ) STRICT;
         CREATE INDEX audios_duration ON audios(duration_ms);"
    ))?;
    Ok(())
}

pub(super) fn configure(connection: &Connection) -> Result<(), AudioDbError> {
    connection.pragma_update(None, "foreign_keys", true)?;
    Ok(())
}

fn migrate(connection: &Connection) -> Result<(), AudioDbError> {
    let mut version: i64 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if version == LEGACY_SCHEMA_VERSION {
        migrate_v1_to_v2(connection)?;
        version = SPLIT_TABLE_SCHEMA_VERSION;
    }
    if version == SPLIT_TABLE_SCHEMA_VERSION {
        migrate_v2_to_v3(connection)?;
        version = CHANNEL_TIMELINE_SCHEMA_VERSION;
    }
    if version == CHANNEL_TIMELINE_SCHEMA_VERSION {
        connection.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    }
    Ok(())
}

fn migrate_v1_to_v2(connection: &Connection) -> Result<(), AudioDbError> {
    let application_id: i64 =
        connection.pragma_query_value(None, "application_id", |row| row.get(0))?;
    if application_id != APPLICATION_ID {
        return Err(AudioDbError::InvalidApplicationId {
            found: application_id,
        });
    }
    connection.pragma_update(None, "foreign_keys", false)?;
    let result = connection.execute_batch(&format!(
        "BEGIN IMMEDIATE;
         ALTER TABLE audios RENAME TO audios_v1;
         DROP INDEX IF EXISTS audios_duration;
         CREATE TABLE audios (
             audio_id TEXT PRIMARY KEY NOT NULL,
             metadata TEXT NOT NULL,
             duration_ms INTEGER
         ) STRICT;
         CREATE TABLE audio_sources (
             audio_id TEXT PRIMARY KEY NOT NULL
                 REFERENCES audios(audio_id) ON DELETE CASCADE,
             source BLOB NOT NULL
         ) STRICT;
         CREATE TABLE timelines (
             audio_id TEXT PRIMARY KEY NOT NULL
                 REFERENCES audios(audio_id) ON DELETE CASCADE,
             timeline BLOB NOT NULL
         ) STRICT;
         INSERT INTO audios(audio_id, metadata, duration_ms)
             SELECT audio_id, metadata, duration_ms FROM audios_v1;
         INSERT INTO audio_sources(audio_id, source)
             SELECT audio_id, source FROM audios_v1;
         INSERT INTO timelines(audio_id, timeline)
             SELECT audio_id, timeline FROM audios_v1;
         DROP TABLE audios_v1;
         CREATE INDEX audios_duration ON audios(duration_ms);
         PRAGMA user_version = {SPLIT_TABLE_SCHEMA_VERSION};
         COMMIT;"
    ));
    if result.is_err() {
        let _ = connection.execute_batch("ROLLBACK;");
    }
    connection.pragma_update(None, "foreign_keys", true)?;
    result.map_err(AudioDbError::from)
}

fn migrate_v2_to_v3(connection: &Connection) -> Result<(), AudioDbError> {
    let encoded_timelines = {
        let mut statement = connection.prepare("SELECT audio_id, timeline FROM timelines")?;
        let rows = statement.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        let mut encoded = Vec::new();
        for row in rows {
            let (audio_id, bytes) = row?;
            let timeline: Timeline = decode(&bytes)?;
            let timelines = BTreeMap::from([(AudioChannel::Mono, timeline)]);
            encoded.push((audio_id, encode(&timelines)?));
        }
        encoded
    };

    connection.execute_batch("BEGIN IMMEDIATE;")?;
    let result = (|| {
        for (audio_id, timelines) in encoded_timelines {
            connection.execute(
                "UPDATE timelines SET timeline = ?1 WHERE audio_id = ?2",
                params![timelines, audio_id],
            )?;
        }
        connection.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        connection.execute_batch("COMMIT;")?;
        Ok::<(), AudioDbError>(())
    })();
    if result.is_err() {
        let _ = connection.execute_batch("ROLLBACK;");
    }
    result
}

pub(super) fn validate(connection: &Connection) -> Result<i64, AudioDbError> {
    let application_id: i64 =
        connection.pragma_query_value(None, "application_id", |row| row.get(0))?;
    if application_id != APPLICATION_ID {
        return Err(AudioDbError::InvalidApplicationId {
            found: application_id,
        });
    }
    let version: i64 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if version != SCHEMA_VERSION
        && version != SPLIT_TABLE_SCHEMA_VERSION
        && version != CHANNEL_TIMELINE_SCHEMA_VERSION
        && version != LEGACY_SCHEMA_VERSION
    {
        return Err(AudioDbError::UnsupportedSchema {
            found: version,
            expected: SCHEMA_VERSION,
        });
    }
    Ok(version)
}
