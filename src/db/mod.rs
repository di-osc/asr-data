use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::SystemTime;

use rusqlite::Connection;
use thiserror::Error;

use crate::doc::AudioValidationError;
use crate::utils::DurationMs;

mod query;
mod schema;

pub use query::read_audio_db_info;

const SCHEMA_VERSION: i64 = 8;
const APPLICATION_ID: i64 = 0x5641_5352; // "VASR"
pub const DEFAULT_QUERY_LIMIT: usize = 100;
pub const MAX_QUERY_LIMIT: usize = 10_000;

#[derive(Debug, Error)]
pub enum AudioDbError {
    #[error("audio database already exists at {path:?}")]
    AlreadyExists { path: PathBuf },
    #[error("audio database does not exist at {path:?}")]
    DatabaseNotFound { path: PathBuf },
    #[error("audio database filesystem error: {0}")]
    Io(#[from] std::io::Error),
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
    #[error("audio query created_from exceeds created_until")]
    InvalidCreatedTimeRange,
    #[error("audio query updated_from exceeds updated_until")]
    InvalidUpdatedTimeRange,
}

pub struct AudioDb {
    connection: Connection,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioQuery {
    pub limit: usize,
    pub after: Option<String>,
    pub min_duration: Option<DurationMs>,
    pub max_duration: Option<DurationMs>,
    pub created_from: Option<SystemTime>,
    pub created_until: Option<SystemTime>,
    pub updated_from: Option<SystemTime>,
    pub updated_until: Option<SystemTime>,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl Default for AudioQuery {
    fn default() -> Self {
        Self {
            limit: DEFAULT_QUERY_LIMIT,
            after: None,
            min_duration: None,
            max_duration: None,
            created_from: None,
            created_until: None,
            updated_from: None,
            updated_until: None,
            metadata: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioDbMode {
    ReadWrite,
    ReadOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioDbInfo {
    pub schema_version: i64,
    pub audios: usize,
    pub total_duration: DurationMs,
}
