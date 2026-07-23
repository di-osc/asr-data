use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::db::{AudioDb as RustAudioDb, AudioDbMode, AudioQuery};
use crate::doc::AudioDoc as RustAudioDoc;
use crate::utils::DurationMs;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDateTime, PyDict};

use super::common::{format_duration_ms, poisoned, py_db_error, py_error, truncate};
use super::doc::PyAudioDoc;
use super::evaluation::{PyDatasetEvaluation, eval_config};

/// 持久化 AudioDoc 的 SQLite 数据库。
///
/// 使用 AudioDB.create 创建新数据库，使用 AudioDB.open 打开已有数据库。
#[pyclass(name = "AudioDB")]
struct PyAudioDb {
    inner: Arc<Mutex<RustAudioDb>>,
    path: String,
    read_only: bool,
}

#[pyclass(name = "AudioDBIterator")]
struct PyAudioDbIterator {
    inner: Arc<Mutex<RustAudioDb>>,
    audios: std::vec::IntoIter<RustAudioDoc>,
    after: Option<String>,
    exhausted: bool,
}

#[pymethods]
impl PyAudioDbIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyAudioDoc>> {
        loop {
            if let Some(audio) = self.audios.next() {
                return PyAudioDoc::from_rust(py, audio).map(Some);
            }
            if self.exhausted {
                return Ok(None);
            }

            let page = self
                .inner
                .lock()
                .map_err(|_| poisoned("AudioDB"))?
                .query(&AudioQuery {
                    after: self.after.clone(),
                    ..AudioQuery::default()
                })
                .map_err(py_error)?;
            if page.is_empty() {
                self.exhausted = true;
                return Ok(None);
            }
            self.after = page.last().map(RustAudioDoc::audio_id);
            self.audios = page.into_iter();
        }
    }
}

#[pymethods]
impl PyAudioDb {
    /// 创建新的数据库。
    ///
    /// Args:
    ///     path: 新数据库文件路径。
    ///
    /// Returns:
    ///     可读写的 AudioDB。
    ///
    /// Raises:
    ///     FileExistsError: 目标路径已经存在。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB
    ///     >>> with TemporaryDirectory() as directory:
    ///     ...     db = AudioDB.create(f"{directory}/dataset.db")
    #[staticmethod]
    fn create(path: String) -> PyResult<Self> {
        let db = RustAudioDb::create(&path);
        Ok(Self {
            inner: Arc::new(Mutex::new(db.map_err(py_db_error)?)),
            path,
            read_only: false,
        })
    }

    /// 打开并校验已有数据库。
    ///
    /// Args:
    ///     path: 已有数据库文件路径。
    ///     read_only: 是否以只读模式打开。
    ///
    /// Returns:
    ///     已打开的 AudioDB。
    ///
    /// Raises:
    ///     FileNotFoundError: 数据库不存在。
    ///     AsrDataError: 文件不是受支持的 asr-data 数据库。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB
    ///     >>> with TemporaryDirectory() as directory:
    ///     ...     path = f"{directory}/dataset.db"
    ///     ...     _ = AudioDB.create(path)
    ///     ...     db = AudioDB.open(path)
    #[staticmethod]
    #[pyo3(signature = (path, read_only=false))]
    fn open(path: String, read_only: bool) -> PyResult<Self> {
        let mode = if read_only {
            AudioDbMode::ReadOnly
        } else {
            AudioDbMode::ReadWrite
        };
        let db = RustAudioDb::open(&path, mode);
        Ok(Self {
            inner: Arc::new(Mutex::new(db.map_err(py_db_error)?)),
            path,
            read_only,
        })
    }

    /// 插入一条新 AudioDoc。
    ///
    /// Args:
    ///     audio: 要插入的完整 AudioDoc。
    ///
    /// Returns:
    ///     None。
    ///
    /// Raises:
    ///     AsrDataError: ID 已存在或文档校验失败。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB, AudioDoc, AudioSource
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> doc = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000), id="one")
    ///     >>> db.insert(doc)
    fn insert(&self, py: Python<'_>, audio: PyRef<'_, PyAudioDoc>) -> PyResult<()> {
        let audio = audio.cloned_inner(py)?;
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .insert(&audio)
            .map_err(py_error)
    }

    /// 更新已有 AudioDoc，仅在内容变化时写入。
    ///
    /// Args:
    ///     audio: 包含新内容的完整 AudioDoc。
    ///
    /// Returns:
    ///     实际发生更新时为 True，否则为 False。
    ///
    /// Raises:
    ///     KeyError: 文档 ID 不存在。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB, AudioDoc, AudioSource
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> doc = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000), id="one")
    ///     >>> db.insert(doc)
    ///     >>> doc.metadata["checked"] = True
    ///     >>> changed = db.update(doc)
    fn update(&self, py: Python<'_>, audio: PyRef<'_, PyAudioDoc>) -> PyResult<bool> {
        let audio = audio.cloned_inner(py)?;
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .update(&audio)
            .map_err(py_db_error)
    }

    /// 按游标、时长、时间和 metadata 查询文档。
    ///
    /// Args:
    ///     limit: 最大返回数量，默认为 100。
    ///     after: 上一页最后一个 AudioDoc ID。
    ///     min_duration_ms: 可选最短时长。
    ///     max_duration_ms: 可选最长时长。
    ///     created_from: 带时区的创建时间下界。
    ///     created_until: 带时区的创建时间上界，不包含。
    ///     updated_from: 带时区的修改时间下界。
    ///     updated_until: 带时区的修改时间上界，不包含。
    ///     metadata: 要精确匹配的 JSON metadata。
    ///
    /// Returns:
    ///     按 AudioDoc ID 排序的文档列表。
    ///
    /// Raises:
    ///     ValueError: 范围反向、datetime 无时区或 limit 无效。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB, AudioDoc, AudioSource
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> doc = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000), id="one")
    ///     >>> doc.metadata["split"] = "test"
    ///     >>> db.insert(doc)
    ///     >>> page = db.query(limit=10, metadata={"split": "test"})
    #[pyo3(signature = (
        limit=100,
        *,
        after=None,
        min_duration_ms=None,
        max_duration_ms=None,
        created_from=None,
        created_until=None,
        updated_from=None,
        updated_until=None,
        metadata=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn query(
        &self,
        py: Python<'_>,
        limit: usize,
        after: Option<String>,
        min_duration_ms: Option<u64>,
        max_duration_ms: Option<u64>,
        created_from: Option<&Bound<'_, PyAny>>,
        created_until: Option<&Bound<'_, PyAny>>,
        updated_from: Option<&Bound<'_, PyAny>>,
        updated_until: Option<&Bound<'_, PyAny>>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<PyAudioDoc>> {
        let created_from = datetime_to_system_time(created_from, "created_from")?;
        let created_until = datetime_to_system_time(created_until, "created_until")?;
        let updated_from = datetime_to_system_time(updated_from, "updated_from")?;
        let updated_until = datetime_to_system_time(updated_until, "updated_until")?;
        validate_time_range(
            created_from,
            created_until,
            "created_from must not exceed created_until",
        )?;
        validate_time_range(
            updated_from,
            updated_until,
            "updated_from must not exceed updated_until",
        )?;
        let metadata = metadata
            .map(|metadata| {
                pythonize::depythonize::<BTreeMap<String, serde_json::Value>>(metadata.as_any())
                    .map_err(py_error)
            })
            .transpose()?
            .unwrap_or_default();
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .query(&AudioQuery {
                limit,
                after,
                min_duration: min_duration_ms.map(DurationMs),
                max_duration: max_duration_ms.map(DurationMs),
                created_from,
                created_until,
                updated_from,
                updated_until,
                metadata,
            })
            .map_err(py_error)?
            .into_iter()
            .map(|audio| PyAudioDoc::from_rust(py, audio))
            .collect()
    }

    /// 评测数据库中全部匹配文档。
    ///
    /// 不传 source 时自动发现全部可评测来源。内部使用 ID 游标分页，
    /// batch_size 只控制单批读取量，不限制最终数据集大小。
    ///
    /// Args:
    ///     transcription: 转写来源或来源列表。
    ///     speech: Speech 来源或来源列表。
    ///     normalize: 是否执行中文文本标准化。
    ///     batch_size: 每批读取的文档数。
    ///     after: 可选起始 AudioDoc ID 游标。
    ///     min_duration_ms: 可选最短时长。
    ///     max_duration_ms: 可选最长时长。
    ///     created_from: 创建时间下界。
    ///     created_until: 创建时间上界，不包含。
    ///     updated_from: 修改时间下界。
    ///     updated_until: 修改时间上界，不包含。
    ///     metadata: 要精确匹配的 JSON metadata。
    ///
    /// Returns:
    ///     按任务和 source 分组的数据集级评测结果。
    ///
    /// Raises:
    ///     ValueError: batch_size 为零或筛选范围无效。
    ///     AsrDataError: 没有可评测内容或显式 source 不存在。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB, AudioDoc, AudioSource
    ///     >>> from asr_data.annotation import Transcription
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> doc = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000), id="one")
    ///     >>> timeline = doc.timeline("mono")
    ///     >>> _ = timeline.reference.add_transcription(
    ///     ...     0, timeline.duration_ms, Transcription("你好")
    ///     ... )
    ///     >>> _ = timeline.prediction.add_transcription(
    ///     ...     0, timeline.duration_ms, Transcription("你好"), source="qwen-asr"
    ///     ... )
    ///     >>> doc.metadata["split"] = "test"
    ///     >>> db.insert(doc)
    ///     >>> result = db.eval(
    ///     ...     transcription="qwen-asr",
    ///     ...     metadata={"split": "test"},
    ///     ... )
    #[pyo3(signature = (
        *,
        transcription=None,
        speech=None,
        normalize=true,
        batch_size=100,
        after=None,
        min_duration_ms=None,
        max_duration_ms=None,
        created_from=None,
        created_until=None,
        updated_from=None,
        updated_until=None,
        metadata=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn eval(
        &self,
        transcription: Option<&Bound<'_, PyAny>>,
        speech: Option<&Bound<'_, PyAny>>,
        normalize: bool,
        batch_size: usize,
        after: Option<String>,
        min_duration_ms: Option<u64>,
        max_duration_ms: Option<u64>,
        created_from: Option<&Bound<'_, PyAny>>,
        created_until: Option<&Bound<'_, PyAny>>,
        updated_from: Option<&Bound<'_, PyAny>>,
        updated_until: Option<&Bound<'_, PyAny>>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDatasetEvaluation> {
        if batch_size == 0 {
            return Err(PyValueError::new_err(
                "batch_size must be greater than zero",
            ));
        }
        let config = eval_config(transcription, speech, normalize)?;
        let created_from = datetime_to_system_time(created_from, "created_from")?;
        let created_until = datetime_to_system_time(created_until, "created_until")?;
        let updated_from = datetime_to_system_time(updated_from, "updated_from")?;
        let updated_until = datetime_to_system_time(updated_until, "updated_until")?;
        validate_time_range(
            created_from,
            created_until,
            "created_from must not exceed created_until",
        )?;
        validate_time_range(
            updated_from,
            updated_until,
            "updated_from must not exceed updated_until",
        )?;
        let metadata = metadata
            .map(|metadata| {
                pythonize::depythonize::<BTreeMap<String, serde_json::Value>>(metadata.as_any())
                    .map_err(py_error)
            })
            .transpose()?
            .unwrap_or_default();
        let inner = self
            .inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .eval(
                &AudioQuery {
                    limit: batch_size,
                    after,
                    min_duration: min_duration_ms.map(DurationMs),
                    max_duration: max_duration_ms.map(DurationMs),
                    created_from,
                    created_until,
                    updated_from,
                    updated_until,
                    metadata,
                },
                &config,
            )
            .map_err(py_error)?;
        Ok(PyDatasetEvaluation { inner })
    }

    fn __getitem__(&self, py: Python<'_>, audio_id: &str) -> PyResult<PyAudioDoc> {
        let audio = self
            .inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .get(audio_id)
            .map_err(py_error)?
            .ok_or_else(|| PyKeyError::new_err(audio_id.to_string()))?;
        PyAudioDoc::from_rust(py, audio)
    }

    fn __contains__(&self, audio_id: &str) -> PyResult<bool> {
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .contains(audio_id)
            .map_err(py_error)
    }

    /// 删除指定 ID 的文档并返回是否确实删除。
    ///
    /// Args:
    ///     audio_id: 文档 ID。
    ///
    /// Returns:
    ///     文档存在并被删除时为 True。
    ///
    /// Raises:
    ///     AsrDataError: 数据库写入失败。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB, AudioDoc, AudioSource
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> db.insert(AudioDoc(
    ///     ...     AudioSource.from_pcm(b"\0\0" * 10, 16000), id="sample-1"
    ///     ... ))
    ///     >>> deleted = db.delete("sample-1")
    fn delete(&self, audio_id: &str) -> PyResult<bool> {
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .delete(audio_id)
            .map_err(py_error)
    }

    /// 在单个事务中批量更新文档。
    ///
    /// Args:
    ///     audios: 要更新的完整 AudioDoc 列表。
    ///
    /// Returns:
    ///     实际发生变化的文档数量。
    ///
    /// Raises:
    ///     KeyError: 任一文档 ID 不存在。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB, AudioDoc, AudioSource
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> docs = [
    ///     ...     AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000), id=name)
    ///     ...     for name in ("first", "second")
    ///     ... ]
    ///     >>> for doc in docs:
    ///     ...     db.insert(doc)
    ///     ...     doc.metadata["checked"] = True
    ///     >>> changed = db.update_many(docs)
    fn update_many(&self, py: Python<'_>, audios: Vec<Py<PyAudioDoc>>) -> PyResult<usize> {
        let audios = audios
            .iter()
            .map(|audio| audio.bind(py).borrow().cloned_inner(py))
            .collect::<PyResult<Vec<_>>>()?;
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .update_many(&audios)
            .map_err(py_db_error)
    }

    /// 设置数据库级 JSON metadata。
    ///
    /// Args:
    ///     key: metadata 键。
    ///     value: 可序列化为 JSON 的值。
    ///
    /// Returns:
    ///     None。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> db.set_metadata("version", "2026-07")
    fn set_metadata(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value: serde_json::Value = pythonize::depythonize(value).map_err(py_error)?;
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .set_metadata(key, &value)
            .map_err(py_error)
    }

    /// 读取一个数据库级 metadata 值。
    ///
    /// Args:
    ///     key: metadata 键。
    ///
    /// Returns:
    ///     解码后的值；不存在时为 None。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> db.set_metadata("version", "2026-07")
    ///     >>> value = db.metadata_value("version")
    fn metadata_value<'py>(
        &self,
        py: Python<'py>,
        key: &str,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .metadata(key)
            .map_err(py_error)?
            .map(|value| pythonize::pythonize(py, &value).map_err(py_error))
            .transpose()
    }

    /// 全部数据库级 metadata 的副本。
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let metadata = self
            .inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .all_metadata()
            .map_err(py_error)?;
        pythonize::pythonize(py, &metadata).map_err(py_error)
    }

    /// 删除数据库级 metadata 并返回键是否存在。
    ///
    /// Args:
    ///     key: metadata 键。
    ///
    /// Returns:
    ///     键存在并被删除时为 True。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> db.set_metadata("version", "2026-07")
    ///     >>> deleted = db.delete_metadata("version")
    fn delete_metadata(&self, key: &str) -> PyResult<bool> {
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .delete_metadata(key)
            .map_err(py_error)
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .len()
            .map_err(py_error)
    }

    fn __iter__(&self) -> PyResult<PyAudioDbIterator> {
        Ok(PyAudioDbIterator {
            inner: Arc::clone(&self.inner),
            audios: Vec::new().into_iter(),
            after: None,
            exhausted: false,
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        let db = self.inner.lock().map_err(|_| poisoned("AudioDB"))?;
        let len = db.len().map_err(py_error)?;
        let duration = db.total_duration().map_err(py_error)?;
        let mode = if self.read_only {
            "read-only"
        } else {
            "read-write"
        };
        Ok(format!(
            "AudioDB(path={:?}, mode={:?}, audios={}, duration={:?})",
            truncate(&self.path, 72),
            mode,
            len,
            format_duration_ms(duration.0 as f64)
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        let db = self.inner.lock().map_err(|_| poisoned("AudioDB"))?;
        let len = db.len().map_err(py_error)?;
        Ok(format!(
            "AudioDB({:?}, {} audios)",
            truncate(&self.path, 72),
            len
        ))
    }
}

fn datetime_to_system_time(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<SystemTime>> {
    let Some(value) = value else {
        return Ok(None);
    };
    value.cast::<PyDateTime>().map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a datetime.datetime instance"))
    })?;
    if value.call_method0("utcoffset")?.is_none() {
        return Err(PyValueError::new_err(format!(
            "{name} must be timezone-aware"
        )));
    }
    let seconds = value.call_method0("timestamp")?.extract::<f64>()?;
    let milliseconds = seconds * 1_000.0;
    if !milliseconds.is_finite() || milliseconds < i64::MIN as f64 || milliseconds > i64::MAX as f64
    {
        return Err(PyValueError::new_err(format!(
            "{name} is outside the supported datetime range"
        )));
    }
    let milliseconds = milliseconds.ceil() as i64;
    let duration = Duration::from_millis(milliseconds.unsigned_abs());
    let time = if milliseconds >= 0 {
        UNIX_EPOCH.checked_add(duration)
    } else {
        UNIX_EPOCH.checked_sub(duration)
    }
    .ok_or_else(|| {
        PyValueError::new_err(format!("{name} is outside the supported datetime range"))
    })?;
    Ok(Some(time))
}

fn validate_time_range(
    start: Option<SystemTime>,
    end: Option<SystemTime>,
    message: &'static str,
) -> PyResult<()> {
    if start.zip(end).is_some_and(|(start, end)| start > end) {
        return Err(PyValueError::new_err(message));
    }
    Ok(())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAudioDb>()?;
    module.add_class::<PyAudioDbIterator>()?;
    Ok(())
}
