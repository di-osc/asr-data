//! SQLite 音频文档库：schema、查询和 MessagePack 编解码。

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

const SCHEMA_VERSION: i64 = 11;
const APPLICATION_ID: i64 = 0x5641_5352; // "VASR"
/// 查询默认返回条数。
pub const DEFAULT_QUERY_LIMIT: usize = 100;
/// 单次查询允许的最大条数。
pub const MAX_QUERY_LIMIT: usize = 10_000;

/// SQLite 音频库操作失败的原因。
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

/// SQLite 持久化的音频文档集合。
pub struct AudioDb {
    connection: Connection,
}

/// 按时长、创建/更新时间和 metadata 过滤的查询条件。
#[derive(Debug, Clone, PartialEq)]
pub struct AudioQuery {
    /// 返回条数上限，受 [`MAX_QUERY_LIMIT`] 约束。
    pub limit: usize,
    /// 从该 `audio_id` 之后继续翻页（不含自身）。
    pub after: Option<String>,
    /// 最短时长（含）。
    pub min_duration: Option<DurationMs>,
    /// 最长时长（含）。
    pub max_duration: Option<DurationMs>,
    /// 创建时间下界（含）。
    pub created_from: Option<SystemTime>,
    /// 创建时间上界（含）。
    pub created_until: Option<SystemTime>,
    /// 更新时间下界（含）。
    pub updated_from: Option<SystemTime>,
    /// 更新时间上界（含）。
    pub updated_until: Option<SystemTime>,
    /// 要求文档 metadata 包含这些键值。
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

/// 打开数据库时的读写模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioDbMode {
    /// 允许插入、更新和删除。
    ReadWrite,
    /// 只读打开已有数据库。
    ReadOnly,
}

/// 数据库概要：schema 版本、文档数和总时长。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioDbInfo {
    /// 当前 schema 版本号。
    pub schema_version: i64,
    /// 已存储的音频文档数。
    pub audios: usize,
    /// 所有文档时长之和。
    pub total_duration: DurationMs,
}
