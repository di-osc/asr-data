use rusqlite::Connection;

use super::{APPLICATION_ID, AudioDbError, SCHEMA_VERSION};

/// 空库则创建 schema；已有库则校验 application_id 和版本。
///
/// # Errors
///
/// schema 不匹配或 SQLite 执行失败时返回错误。
pub(super) fn initialize(connection: &Connection) -> Result<(), AudioDbError> {
    let current: i64 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if current != 0 {
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
             audio_id      TEXT PRIMARY KEY NOT NULL,
             metadata      TEXT NOT NULL,
             duration_ms   INTEGER,
             created_at_ms INTEGER NOT NULL,
             updated_at_ms INTEGER NOT NULL
         ) STRICT;
         CREATE TABLE audio_sources (
             audio_id TEXT PRIMARY KEY NOT NULL
                 REFERENCES audios(audio_id) ON DELETE CASCADE,
             source     BLOB NOT NULL,
             info BLOB NOT NULL
         ) STRICT;
         CREATE TABLE timelines (
             audio_id TEXT PRIMARY KEY NOT NULL
                 REFERENCES audios(audio_id) ON DELETE CASCADE,
             timeline BLOB NOT NULL
         ) STRICT;
         CREATE INDEX audios_duration ON audios(duration_ms);
         CREATE INDEX audios_created_at ON audios(created_at_ms);
         CREATE INDEX audios_updated_at ON audios(updated_at_ms);"
    ))?;
    Ok(())
}

/// 打开已有连接后打开外键约束。
pub(super) fn configure(connection: &Connection) -> Result<(), AudioDbError> {
    connection.pragma_update(None, "foreign_keys", true)?;
    Ok(())
}

/// 确认文件是本库 schema，且版本恰好等于 [`SCHEMA_VERSION`](super::SCHEMA_VERSION)。
///
/// # Errors
///
/// application_id 或 user_version 不匹配时返回错误。
pub(super) fn validate(connection: &Connection) -> Result<(), AudioDbError> {
    let application_id: i64 =
        connection.pragma_query_value(None, "application_id", |row| row.get(0))?;
    if application_id != APPLICATION_ID {
        return Err(AudioDbError::InvalidApplicationId {
            found: application_id,
        });
    }
    let version: i64 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if version != SCHEMA_VERSION {
        return Err(AudioDbError::UnsupportedSchema {
            found: version,
            expected: SCHEMA_VERSION,
        });
    }
    Ok(())
}
