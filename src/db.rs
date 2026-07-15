use std::collections::BTreeMap;
use std::path::Path;

use rusqlite::{
    Connection, OpenFlags, OptionalExtension, Transaction, params, params_from_iter,
    types::Value as SqlValue,
};
use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::{Audio, DurationMs};

const SCHEMA_VERSION: i64 = 4;
const CHANNEL_TIMELINE_SCHEMA_VERSION: i64 = 3;
const SPLIT_TABLE_SCHEMA_VERSION: i64 = 2;
const LEGACY_SCHEMA_VERSION: i64 = 1;
const APPLICATION_ID: i64 = 0x5641_5352; // "VASR"
pub const DEFAULT_QUERY_LIMIT: usize = 100;
pub const MAX_QUERY_LIMIT: usize = 10_000;

#[derive(Debug, Error)]
pub enum AudioDbError {
    #[error("audio database error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("failed to encode audio database value: {0}")]
    Encode(#[from] rmp_serde::encode::Error),
    #[error("failed to decode audio database value: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
    #[error("failed to encode audio database metadata: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid audio: {0}")]
    Validation(#[from] crate::AudioValidationError),
    #[error("unsupported audio database schema version {found}; expected {expected}")]
    UnsupportedSchema { found: i64, expected: i64 },
    #[error("file is not an ASR AudioDb (application id {found:#x})")]
    InvalidApplicationId { found: i64 },
    #[error("audio {audio_id:?} does not exist")]
    NotFound { audio_id: String },
    #[error("audio query limit {limit} exceeds the maximum of {max}")]
    QueryLimitExceeded { limit: usize, max: usize },
    #[error("audio query minimum duration exceeds its maximum duration")]
    InvalidDurationRange,
    #[error("failed to import legacy MessagePack audio data: {0}")]
    LegacyImport(#[from] crate::LegacyImportError),
}

pub struct AudioDb {
    connection: Connection,
    schema_version: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioQuery {
    pub limit: usize,
    pub after: Option<String>,
    pub min_duration: Option<DurationMs>,
    pub max_duration: Option<DurationMs>,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl Default for AudioQuery {
    fn default() -> Self {
        Self {
            limit: DEFAULT_QUERY_LIMIT,
            after: None,
            min_duration: None,
            max_duration: None,
            metadata: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioDbMode {
    ReadWrite,
    ReadOnly,
}

pub(crate) struct AudioDbTransaction<'db> {
    transaction: Transaction<'db>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioDbInfo {
    pub schema_version: i64,
    pub audios: usize,
    pub total_duration: DurationMs,
}

impl AudioDb {
    pub const SCHEMA_VERSION: i64 = SCHEMA_VERSION;

    pub fn open(path: impl AsRef<Path>, mode: AudioDbMode) -> Result<Self, AudioDbError> {
        match mode {
            AudioDbMode::ReadWrite => {
                let connection = Connection::open_with_flags(
                    path,
                    OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
                )?;
                initialize(&connection)?;
                configure(&connection)?;
                Ok(Self {
                    connection,
                    schema_version: SCHEMA_VERSION,
                })
            }
            AudioDbMode::ReadOnly => {
                let connection =
                    Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
                let schema_version = validate(&connection)?;
                configure(&connection)?;
                Ok(Self {
                    connection,
                    schema_version,
                })
            }
        }
    }

    pub fn insert(&self, audio: &Audio) -> Result<(), AudioDbError> {
        insert_with(&self.connection, audio)
    }

    /// Updates only the parts of an existing audio that differ from its stored value.
    /// Returns `true` when at least one part changed and `false` for a no-op update.
    pub fn update(&self, audio: &Audio) -> Result<bool, AudioDbError> {
        update_with(&self.connection, audio)
    }

    pub fn query(&self, query: &AudioQuery) -> Result<Vec<Audio>, AudioDbError> {
        query_with(&self.connection, query, self.schema_version)
    }

    pub fn get(&self, audio_id: &str) -> Result<Option<Audio>, AudioDbError> {
        get_with(&self.connection, audio_id, self.schema_version)
    }

    pub fn contains(&self, audio_id: &str) -> Result<bool, AudioDbError> {
        Ok(self
            .connection
            .query_row(
                "SELECT 1 FROM audios WHERE audio_id = ?1",
                [audio_id],
                |_| Ok(()),
            )
            .optional()?
            .is_some())
    }

    pub fn delete(&self, audio_id: &str) -> Result<bool, AudioDbError> {
        Ok(self
            .connection
            .execute("DELETE FROM audios WHERE audio_id = ?1", [audio_id])?
            != 0)
    }

    pub fn update_many(&mut self, audios: &[Audio]) -> Result<usize, AudioDbError> {
        let transaction = self.connection.transaction()?;
        let mut updated = 0;
        for audio in audios {
            updated += usize::from(update_with(&transaction, audio)?);
        }
        transaction.commit()?;
        Ok(updated)
    }

    pub(crate) fn len(&self) -> Result<usize, AudioDbError> {
        let count: i64 = self
            .connection
            .query_row("SELECT COUNT(*) FROM audios", [], |row| row.get(0))?;
        Ok(usize::try_from(count).unwrap_or(usize::MAX))
    }

    pub(crate) fn total_duration(&self) -> Result<DurationMs, AudioDbError> {
        let duration: i64 = self.connection.query_row(
            "SELECT COALESCE(SUM(duration_ms), 0) FROM audios",
            [],
            |row| row.get(0),
        )?;
        Ok(DurationMs(u64::try_from(duration).unwrap_or_default()))
    }

    pub(crate) fn load_all(&self) -> Result<Vec<Audio>, AudioDbError> {
        let mut audios = Vec::new();
        let mut after = None;
        loop {
            let page = self.query(&AudioQuery {
                limit: MAX_QUERY_LIMIT,
                after,
                ..AudioQuery::default()
            })?;
            if page.is_empty() {
                break;
            }
            after = page.last().map(Audio::audio_id);
            audios.extend(page);
        }
        Ok(audios)
    }

    pub fn set_metadata(&self, key: &str, value: &serde_json::Value) -> Result<(), AudioDbError> {
        let value = serde_json::to_string(value)?;
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?1, ?2)\
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn metadata(&self, key: &str) -> Result<Option<serde_json::Value>, AudioDbError> {
        self.connection
            .query_row("SELECT value FROM metadata WHERE key = ?1", [key], |row| {
                row.get::<_, String>(0)
            })
            .optional()?
            .map(|value| serde_json::from_str(&value).map_err(AudioDbError::from))
            .transpose()
    }

    pub fn all_metadata(&self) -> Result<BTreeMap<String, serde_json::Value>, AudioDbError> {
        let mut statement = self
            .connection
            .prepare("SELECT key, value FROM metadata ORDER BY key")?;
        let rows = statement.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut metadata = BTreeMap::new();
        for row in rows {
            let (key, value) = row?;
            metadata.insert(key, serde_json::from_str(&value)?);
        }
        Ok(metadata)
    }

    pub fn delete_metadata(&self, key: &str) -> Result<bool, AudioDbError> {
        Ok(self
            .connection
            .execute("DELETE FROM metadata WHERE key = ?1", [key])?
            != 0)
    }

    pub(crate) fn transaction(&mut self) -> Result<AudioDbTransaction<'_>, AudioDbError> {
        Ok(AudioDbTransaction {
            transaction: self.connection.transaction()?,
        })
    }
}

impl AudioDbTransaction<'_> {
    pub(crate) fn insert(&self, audio: &Audio) -> Result<(), AudioDbError> {
        insert_with(&self.transaction, audio)
    }

    pub(crate) fn commit(self) -> Result<(), AudioDbError> {
        self.transaction.commit()?;
        Ok(())
    }
}

pub fn import_legacy_msgpack_to_db(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> Result<usize, AudioDbError> {
    let audios = crate::read_legacy_msgpack(input)?;
    let count = audios.len();
    let mut db = AudioDb::open(output, AudioDbMode::ReadWrite)?;
    let transaction = db.transaction()?;
    for audio in &audios {
        transaction.insert(audio)?;
    }
    transaction.commit()?;
    Ok(count)
}

pub fn read_audio_db_info(path: impl AsRef<Path>) -> Result<AudioDbInfo, AudioDbError> {
    let db = AudioDb::open(path, AudioDbMode::ReadOnly)?;
    Ok(AudioDbInfo {
        schema_version: db.schema_version,
        audios: db.len()?,
        total_duration: db.total_duration()?,
    })
}

fn initialize(connection: &Connection) -> Result<(), AudioDbError> {
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

fn configure(connection: &Connection) -> Result<(), AudioDbError> {
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
            let timeline: crate::Timeline = decode(&bytes)?;
            let timelines = BTreeMap::from([(crate::AudioChannel::Mono, timeline)]);
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

fn validate(connection: &Connection) -> Result<i64, AudioDbError> {
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

fn insert_with(connection: &Connection, audio: &Audio) -> Result<(), AudioDbError> {
    audio.validate()?;
    let source = encode(&audio.source)?;
    let timeline = encode(audio.timelines())?;
    let metadata = serde_json::to_string(&audio.metadata)?;
    let duration = audio
        .duration
        .map(|duration| i64::try_from(duration.0).unwrap_or(i64::MAX));
    connection.execute_batch("SAVEPOINT asr_write")?;
    let result = (|| {
        connection.execute(
            "INSERT INTO audios(audio_id, metadata, duration_ms) VALUES (?1, ?2, ?3)",
            params![audio.id, metadata, duration],
        )?;
        connection.execute(
            "INSERT INTO audio_sources(audio_id, source) VALUES (?1, ?2)",
            params![audio.id, source],
        )?;
        connection.execute(
            "INSERT INTO timelines(audio_id, timeline) VALUES (?1, ?2)",
            params![audio.id, timeline],
        )?;
        Ok::<(), AudioDbError>(())
    })();
    match result {
        Ok(()) => {
            connection.execute_batch("RELEASE asr_write")?;
            Ok(())
        }
        Err(error) => {
            let _ = connection.execute_batch("ROLLBACK TO asr_write; RELEASE asr_write;");
            Err(error)
        }
    }
}

fn update_with(connection: &Connection, audio: &Audio) -> Result<bool, AudioDbError> {
    audio.validate()?;
    let audio_id = &audio.id;
    let source = encode(&audio.source)?;
    let timeline = encode(audio.timelines())?;
    let metadata = serde_json::to_string(&audio.metadata)?;
    let duration = audio
        .duration
        .map(|duration| i64::try_from(duration.0).unwrap_or(i64::MAX));

    connection.execute_batch("SAVEPOINT asr_update")?;
    let result = (|| {
        let exists = connection
            .query_row(
                "SELECT 1 FROM audios WHERE audio_id = ?1",
                [audio_id],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if !exists {
            return Err(AudioDbError::NotFound {
                audio_id: audio_id.clone(),
            });
        }

        let audio_changed = connection.execute(
            "UPDATE audios
             SET metadata = ?1, duration_ms = ?2
             WHERE audio_id = ?3
               AND (metadata IS NOT ?1 OR duration_ms IS NOT ?2)",
            params![metadata, duration, audio_id],
        )? != 0;
        let source_changed = connection.execute(
            "UPDATE audio_sources
             SET source = ?1
             WHERE audio_id = ?2 AND source IS NOT ?1",
            params![source, audio_id],
        )? != 0;
        let timeline_changed = connection.execute(
            "UPDATE timelines
             SET timeline = ?1
             WHERE audio_id = ?2 AND timeline IS NOT ?1",
            params![timeline, audio_id],
        )? != 0;

        Ok::<bool, AudioDbError>(audio_changed || source_changed || timeline_changed)
    })();
    match result {
        Ok(changed) => {
            connection.execute_batch("RELEASE asr_update")?;
            Ok(changed)
        }
        Err(error) => {
            let _ = connection.execute_batch("ROLLBACK TO asr_update; RELEASE asr_update;");
            Err(error)
        }
    }
}

fn query_with(
    connection: &Connection,
    query: &AudioQuery,
    schema_version: i64,
) -> Result<Vec<Audio>, AudioDbError> {
    if query.limit > MAX_QUERY_LIMIT {
        return Err(AudioDbError::QueryLimitExceeded {
            limit: query.limit,
            max: MAX_QUERY_LIMIT,
        });
    }
    if query.limit == 0 {
        return Ok(Vec::new());
    }
    if query
        .min_duration
        .zip(query.max_duration)
        .is_some_and(|(minimum, maximum)| minimum > maximum)
    {
        return Err(AudioDbError::InvalidDurationRange);
    }

    let mut sql = if schema_version == LEGACY_SCHEMA_VERSION {
        String::from(
            "SELECT audios.source, audios.timeline, audios.metadata,
                    audios.audio_id, audios.duration_ms FROM audios",
        )
    } else {
        String::from(
            "SELECT audio_sources.source, timelines.timeline, audios.metadata,
                    audios.audio_id, audios.duration_ms
             FROM audios
             JOIN audio_sources USING (audio_id)
             JOIN timelines USING (audio_id)",
        )
    };
    let mut predicates = Vec::new();
    let mut parameters = Vec::<SqlValue>::new();

    if let Some(after) = &query.after {
        let parameter = push_sql_parameter(&mut parameters, SqlValue::Text(after.clone()));
        predicates.push(format!("audios.audio_id > {parameter}"));
    }
    if let Some(minimum) = query.min_duration {
        let value = i64::try_from(minimum.0).unwrap_or(i64::MAX);
        let parameter = push_sql_parameter(&mut parameters, SqlValue::Integer(value));
        predicates.push(format!("audios.duration_ms >= {parameter}"));
    }
    if let Some(maximum) = query.max_duration {
        let value = i64::try_from(maximum.0).unwrap_or(i64::MAX);
        let parameter = push_sql_parameter(&mut parameters, SqlValue::Integer(value));
        predicates.push(format!("audios.duration_ms <= {parameter}"));
    }
    for (key, value) in &query.metadata {
        let key_parameter = push_sql_parameter(&mut parameters, SqlValue::Text(key.clone()));
        let value_parameter = push_sql_parameter(
            &mut parameters,
            SqlValue::Text(serde_json::to_string(value)?),
        );
        predicates.push(format!(
            "EXISTS (
                 SELECT 1 FROM json_each(audios.metadata) AS metadata_entry
                 WHERE metadata_entry.key = {key_parameter}
                   AND metadata_entry.type = json_type({value_parameter}, '$')
                   AND metadata_entry.value IS json_extract({value_parameter}, '$')
             )"
        ));
    }

    if !predicates.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&predicates.join(" AND "));
    }
    let limit = i64::try_from(query.limit).unwrap_or(i64::MAX);
    let limit_parameter = push_sql_parameter(&mut parameters, SqlValue::Integer(limit));
    sql.push_str(&format!(
        " ORDER BY audios.audio_id LIMIT {limit_parameter}"
    ));

    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(parameters.iter()), |row| {
        decode_audio_row(row, schema_version)
    })?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(AudioDbError::from)
}

fn push_sql_parameter(parameters: &mut Vec<SqlValue>, value: SqlValue) -> String {
    parameters.push(value);
    format!("?{}", parameters.len())
}

fn get_with(
    connection: &Connection,
    audio_id: &str,
    schema_version: i64,
) -> Result<Option<Audio>, AudioDbError> {
    let sql = if schema_version == LEGACY_SCHEMA_VERSION {
        "SELECT source, timeline, metadata, audio_id, duration_ms
         FROM audios WHERE audio_id = ?1"
    } else {
        "SELECT audio_sources.source, timelines.timeline, audios.metadata,
                audios.audio_id, audios.duration_ms
         FROM audios
         JOIN audio_sources USING (audio_id)
         JOIN timelines USING (audio_id)
         WHERE audios.audio_id = ?1"
    };
    connection
        .query_row(sql, [audio_id], |row| decode_audio_row(row, schema_version))
        .optional()
        .map_err(AudioDbError::from)
}

fn decode_audio_row(row: &rusqlite::Row<'_>, schema_version: i64) -> rusqlite::Result<Audio> {
    let source: Vec<u8> = row.get(0)?;
    let timeline: Vec<u8> = row.get(1)?;
    let metadata: String = row.get(2)?;
    let audio_id: String = row.get(3)?;
    let duration = row
        .get::<_, Option<i64>>(4)?
        .map(|value| DurationMs(u64::try_from(value).unwrap_or_default()));
    let source = decode(&source).map_err(sql_conversion_error)?;
    let timelines = if schema_version >= CHANNEL_TIMELINE_SCHEMA_VERSION {
        decode(&timeline).map_err(sql_conversion_error)?
    } else {
        let timeline = decode(&timeline).map_err(sql_conversion_error)?;
        BTreeMap::from([(crate::AudioChannel::Mono, timeline)])
    };
    let metadata = serde_json::from_str(&metadata).map_err(sql_conversion_error)?;
    let mut timelines: BTreeMap<crate::AudioChannel, crate::Timeline> = timelines;
    for timeline in timelines.values_mut() {
        timeline.audio_id.clone_from(&audio_id);
        timeline.duration = duration;
    }
    Ok(Audio {
        id: audio_id,
        duration,
        source,
        timelines,
        metadata,
    })
}

fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec_named(value)
}

fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, rmp_serde::decode::Error> {
    rmp_serde::from_slice(bytes)
}

fn sql_conversion_error(error: impl std::error::Error + Send + Sync + 'static) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Blob, Box::new(error))
}
