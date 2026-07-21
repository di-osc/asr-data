use std::collections::BTreeMap;

use rusqlite::{Connection, Transaction};
use thiserror::Error;

use crate::doc::{AudioValidationError, LegacyImportError};
use crate::utils::DurationMs;

mod query;
mod schema;

pub use query::{import_legacy_msgpack_to_db, read_audio_db_info};

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
    Validation(#[from] AudioValidationError),
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
    LegacyImport(#[from] LegacyImportError),
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
