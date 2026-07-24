use std::collections::BTreeMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::audio::AudioChannel;
use crate::doc::Audio;
use crate::timeline::Timeline;
use crate::utils::DurationMs;
use rusqlite::{
    Connection, OpenFlags, OptionalExtension, params, params_from_iter, types::Value as SqlValue,
};
use serde::{Serialize, de::DeserializeOwned};

use super::schema::{configure, initialize, validate};
use super::{
    AudioDb, AudioDbError, AudioDbInfo, AudioDbMode, AudioQuery, MAX_QUERY_LIMIT, SCHEMA_VERSION,
};

impl AudioDb {
    pub const SCHEMA_VERSION: i64 = SCHEMA_VERSION;

    pub fn create(path: impl AsRef<Path>) -> Result<Self, AudioDbError> {
        let path = path.as_ref();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::AlreadyExists {
                    AudioDbError::AlreadyExists {
                        path: path.to_path_buf(),
                    }
                } else {
                    AudioDbError::Io(error)
                }
            })?;
        drop(file);
        let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
        initialize(&connection)?;
        configure(&connection)?;
        Ok(Self { connection })
    }

    pub fn open(path: impl AsRef<Path>, mode: AudioDbMode) -> Result<Self, AudioDbError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(AudioDbError::DatabaseNotFound {
                path: path.to_path_buf(),
            });
        }
        let flags = match mode {
            AudioDbMode::ReadWrite => OpenFlags::SQLITE_OPEN_READ_WRITE,
            AudioDbMode::ReadOnly => OpenFlags::SQLITE_OPEN_READ_ONLY,
        };
        let connection = Connection::open_with_flags(path, flags)?;
        validate(&connection)?;
        configure(&connection)?;
        Ok(Self { connection })
    }

    pub fn insert(&self, audio: &Audio) -> Result<(), AudioDbError> {
        insert_with(&self.connection, audio, now_unix_millis())
    }

    /// Updates only the parts of an existing audio that differ from its stored value.
    /// Returns `true` when at least one part changed and `false` for a no-op update.
    pub fn update(&self, audio: &Audio) -> Result<bool, AudioDbError> {
        update_with(&self.connection, audio, now_unix_millis())
    }

    pub fn query(&self, query: &AudioQuery) -> Result<Vec<Audio>, AudioDbError> {
        query_with(&self.connection, query)
    }

    pub fn get(&self, audio_id: &str) -> Result<Option<Audio>, AudioDbError> {
        get_with(&self.connection, audio_id)
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
        let updated_at_ms = now_unix_millis();
        let mut updated = 0;
        for audio in audios {
            updated += usize::from(update_with(&transaction, audio, updated_at_ms)?);
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
}

pub fn read_audio_db_info(path: impl AsRef<Path>) -> Result<AudioDbInfo, AudioDbError> {
    let db = AudioDb::open(path, AudioDbMode::ReadOnly)?;
    Ok(AudioDbInfo {
        schema_version: SCHEMA_VERSION,
        audios: db.len()?,
        total_duration: db.total_duration()?,
    })
}

fn insert_with(
    connection: &Connection,
    audio: &Audio,
    timestamp_ms: i64,
) -> Result<(), AudioDbError> {
    audio.validate()?;
    let source = encode(&audio.source)?;
    let info = encode(&audio.info)?;
    let timeline = encode(audio.timelines())?;
    let metadata = serde_json::to_string(&audio.metadata)?;
    let duration = audio
        .timeline_duration()
        .map(|duration| i64::try_from(duration.0).unwrap_or(i64::MAX));
    connection.execute_batch("SAVEPOINT asr_write")?;
    let result = (|| {
        connection.execute(
            "INSERT INTO audios(
                 audio_id, metadata, duration_ms, created_at_ms, updated_at_ms
             ) VALUES (?1, ?2, ?3, ?4, ?4)",
            params![audio.id, metadata, duration, timestamp_ms],
        )?;
        connection.execute(
            "INSERT INTO audio_sources(audio_id, source, info) VALUES (?1, ?2, ?3)",
            params![audio.id, source, info],
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

fn update_with(
    connection: &Connection,
    audio: &Audio,
    updated_at_ms: i64,
) -> Result<bool, AudioDbError> {
    audio.validate()?;
    let audio_id = &audio.id;
    let source = encode(&audio.source)?;
    let info = encode(&audio.info)?;
    let timeline = encode(audio.timelines())?;
    let metadata = serde_json::to_string(&audio.metadata)?;
    let duration = audio
        .timeline_duration()
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
             SET source = ?1, info = ?2
             WHERE audio_id = ?3
               AND (source IS NOT ?1 OR info IS NOT ?2)",
            params![source, info, audio_id],
        )? != 0;
        let timeline_changed = connection.execute(
            "UPDATE timelines
             SET timeline = ?1
             WHERE audio_id = ?2 AND timeline IS NOT ?1",
            params![timeline, audio_id],
        )? != 0;

        let changed = audio_changed || source_changed || timeline_changed;
        if changed {
            connection.execute(
                "UPDATE audios
                 SET updated_at_ms = MAX(updated_at_ms, ?1)
                 WHERE audio_id = ?2",
                params![updated_at_ms, audio_id],
            )?;
        }
        Ok::<bool, AudioDbError>(changed)
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

fn query_with(connection: &Connection, query: &AudioQuery) -> Result<Vec<Audio>, AudioDbError> {
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
    if query
        .created_from
        .zip(query.created_until)
        .is_some_and(|(start, end)| start > end)
    {
        return Err(AudioDbError::InvalidCreatedTimeRange);
    }
    if query
        .updated_from
        .zip(query.updated_until)
        .is_some_and(|(start, end)| start > end)
    {
        return Err(AudioDbError::InvalidUpdatedTimeRange);
    }

    let mut sql = String::from(
        "SELECT audio_sources.source, audio_sources.info, timelines.timeline,
                audios.metadata, audios.audio_id
         FROM audios
         JOIN audio_sources USING (audio_id)
         JOIN timelines USING (audio_id)",
    );
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
    if let Some(start) = query.created_from {
        let parameter = push_sql_parameter(
            &mut parameters,
            SqlValue::Integer(system_time_to_query_boundary_millis(start)),
        );
        predicates.push(format!("audios.created_at_ms >= {parameter}"));
    }
    if let Some(end) = query.created_until {
        let parameter = push_sql_parameter(
            &mut parameters,
            SqlValue::Integer(system_time_to_query_boundary_millis(end)),
        );
        predicates.push(format!("audios.created_at_ms < {parameter}"));
    }
    if let Some(start) = query.updated_from {
        let parameter = push_sql_parameter(
            &mut parameters,
            SqlValue::Integer(system_time_to_query_boundary_millis(start)),
        );
        predicates.push(format!("audios.updated_at_ms >= {parameter}"));
    }
    if let Some(end) = query.updated_until {
        let parameter = push_sql_parameter(
            &mut parameters,
            SqlValue::Integer(system_time_to_query_boundary_millis(end)),
        );
        predicates.push(format!("audios.updated_at_ms < {parameter}"));
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
        decode_audio_row(row)
    })?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(AudioDbError::from)
}

fn push_sql_parameter(parameters: &mut Vec<SqlValue>, value: SqlValue) -> String {
    parameters.push(value);
    format!("?{}", parameters.len())
}

fn now_unix_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or_default()
}

fn system_time_to_query_boundary_millis(time: SystemTime) -> i64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => {
            let millis =
                duration.as_millis() + u128::from(duration.subsec_nanos() % 1_000_000 != 0);
            i64::try_from(millis).unwrap_or(i64::MAX)
        }
        Err(error) => -i64::try_from(error.duration().as_millis()).unwrap_or(i64::MAX),
    }
}

fn get_with(connection: &Connection, audio_id: &str) -> Result<Option<Audio>, AudioDbError> {
    let sql = "SELECT audio_sources.source, audio_sources.info, timelines.timeline,
                      audios.metadata, audios.audio_id
               FROM audios
               JOIN audio_sources USING (audio_id)
               JOIN timelines USING (audio_id)
               WHERE audios.audio_id = ?1";
    connection
        .query_row(sql, [audio_id], decode_audio_row)
        .optional()
        .map_err(AudioDbError::from)
}

fn decode_audio_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Audio> {
    let source: Vec<u8> = row.get(0)?;
    let info: Vec<u8> = row.get(1)?;
    let timeline: Vec<u8> = row.get(2)?;
    let metadata: String = row.get(3)?;
    let audio_id: String = row.get(4)?;
    let source = decode(&source).map_err(sql_conversion_error)?;
    let info = decode(&info).map_err(sql_conversion_error)?;
    let timelines: BTreeMap<AudioChannel, Timeline> =
        decode(&timeline).map_err(sql_conversion_error)?;
    let metadata = serde_json::from_str(&metadata).map_err(sql_conversion_error)?;
    let audio = Audio {
        id: audio_id,
        source,
        info,
        timelines,
        metadata,
        waveform: None,
    };
    audio.validate().map_err(sql_conversion_error)?;
    Ok(audio)
}

pub(super) fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec_named(value)
}

pub(super) fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, rmp_serde::decode::Error> {
    rmp_serde::from_slice(bytes)
}

fn sql_conversion_error(error: impl std::error::Error + Send + Sync + 'static) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Blob, Box::new(error))
}
