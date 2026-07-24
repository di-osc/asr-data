use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::dataset::{AudioDataset as RustAudioDataset, AudioDatasetError};
use crate::db::{AudioDb as RustAudioDb, AudioDbMode, AudioQuery};
use crate::doc::Audio as RustAudio;
use crate::utils::DurationMs;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDateTime, PyDict};

use super::audio::async_runtime;
use super::common::{format_duration_ms, poisoned, py_db_error, py_error, truncate};
use super::doc::PyAudio;
use super::evaluation::{PyDatasetEvaluation, eval_config};

/// 持久化 Audio 的 SQLite 数据库。
///
/// 使用 AudioDB.create 创建新数据库，使用 AudioDB.open 打开已有数据库。
#[pyclass(name = "AudioDB")]
#[derive(Clone)]
struct PyAudioDb {
    inner: Arc<Mutex<RustAudioDb>>,
    path: String,
    read_only: bool,
}

/// 具名、带版本，并由可选 train、val、test AudioDB 支撑的数据集。
///
/// Args:
///     train: 可选训练集 AudioDB。
///     val: 可选验证集 AudioDB。
///     test: 可选测试集 AudioDB。
///
/// Examples:
///     >>> from tempfile import TemporaryDirectory
///     >>> from asr_data import AudioDB, AudioDataset
///     >>> directory = TemporaryDirectory()
///     >>> train = AudioDB.create(f"{directory.name}/train.db")
///     >>> dataset = AudioDataset(train=train)
///     >>> len(dataset.train)
///     0
///     >>> dataset.val is None
///     True
#[pyclass(name = "AudioDataset", frozen)]
struct PyAudioDataset {
    name: String,
    version: String,
    license: String,
    train: Option<PyAudioDb>,
    val: Option<PyAudioDb>,
    test: Option<PyAudioDb>,
}

#[pymethods]
impl PyAudioDataset {
    #[new]
    #[pyo3(signature = (train=None, val=None, test=None))]
    fn new(
        train: Option<PyRef<'_, PyAudioDb>>,
        val: Option<PyRef<'_, PyAudioDb>>,
        test: Option<PyRef<'_, PyAudioDb>>,
    ) -> Self {
        Self {
            name: String::new(),
            version: String::new(),
            license: String::new(),
            train: train.map(|db| db.clone()),
            val: val.map(|db| db.clone()),
            test: test.map(|db| db.clone()),
        }
    }

    /// 通过 modelhub 下载完整 ModelScope 数据集仓库。
    ///
    /// 下载和缓存完全由 modelhub 管理。未传 cache_dir 时使用
    /// modelhub/ModelScope 的默认缓存目录，未传 revision 时使用 master。
    /// name 使用 repo_id，version 使用实际 revision，license 从同一
    /// revision 的 README.md front matter 读取。仓库中不存在的切分数据库
    /// 返回 None，存在的数据库只读打开。
    ///
    /// Args:
    ///     repo_id: ModelScope 数据集仓库 ID。
    ///     revision: 可选仓库 revision，默认 master。
    ///     cache_dir: 可选 ModelScope 缓存根目录。
    ///
    /// Returns:
    ///     train、val、test 为只读 AudioDB 或 None 的 AudioDataset。
    ///
    /// Raises:
    ///     ValueError: repo_id、revision 或 README.md license 无效。
    ///     AsrDataError: 整仓下载失败，或已有切分数据库不是受支持的 AudioDB。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDataset
    ///     >>> dataset = AudioDataset.from_modelscope("di-osc/aishell-1")
    #[staticmethod]
    #[pyo3(signature = (
        repo_id,
        *,
        revision=None,
        cache_dir=None
    ))]
    fn from_modelscope(
        py: Python<'_>,
        repo_id: String,
        revision: Option<String>,
        cache_dir: Option<PathBuf>,
    ) -> PyResult<Self> {
        let dataset = py
            .detach(move || {
                async_runtime().block_on(RustAudioDataset::from_modelscope(
                    &repo_id,
                    revision.as_deref(),
                    cache_dir.as_deref(),
                ))
            })
            .map_err(py_dataset_error)?;
        let name = dataset.name().to_owned();
        let version = dataset.version().to_owned();
        let license = dataset.license().to_owned();
        let train_path = dataset
            .train_database_path()
            .map(|path| path.display().to_string());
        let val_path = dataset
            .val_database_path()
            .map(|path| path.display().to_string());
        let test_path = dataset
            .test_database_path()
            .map(|path| path.display().to_string());
        let (train, val, test) = dataset.into_databases();
        Ok(Self {
            name,
            version,
            license,
            train: wrap_optional_dataset_db(train_path, train),
            val: wrap_optional_dataset_db(val_path, val),
            test: wrap_optional_dataset_db(test_path, test),
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn version(&self) -> &str {
        &self.version
    }

    #[getter]
    fn license(&self) -> &str {
        &self.license
    }

    /// 返回训练集的只读 AudioDB。
    #[getter]
    fn train(&self) -> Option<PyAudioDb> {
        self.train.clone()
    }

    /// 返回验证集的只读 AudioDB。
    #[getter]
    fn val(&self) -> Option<PyAudioDb> {
        self.val.clone()
    }

    /// 返回测试集的只读 AudioDB。
    #[getter]
    fn test(&self) -> Option<PyAudioDb> {
        self.test.clone()
    }

    fn __repr__(&self) -> PyResult<String> {
        let train_len = optional_dataset_db_len(&self.train)?;
        let val_len = optional_dataset_db_len(&self.val)?;
        let test_len = optional_dataset_db_len(&self.test)?;
        Ok(format!(
            "AudioDataset(name={:?}, version={:?}, license={:?}, train={}, val={}, test={})",
            truncate(&self.name, 48),
            truncate(&self.version, 32),
            truncate(&self.license, 32),
            format_optional_count(train_len),
            format_optional_count(val_len),
            format_optional_count(test_len),
        ))
    }

    fn __str__(&self) -> String {
        if self.version.is_empty() {
            self.name.clone()
        } else {
            format!("{}@{}", self.name, self.version)
        }
    }
}

fn wrap_optional_dataset_db(path: Option<String>, db: Option<RustAudioDb>) -> Option<PyAudioDb> {
    match (path, db) {
        (Some(path), Some(db)) => Some(PyAudioDb {
            inner: Arc::new(Mutex::new(db)),
            path,
            read_only: true,
        }),
        (None, None) => None,
        _ => unreachable!("AudioDataset database and path must both be present or absent"),
    }
}

fn optional_dataset_db_len(db: &Option<PyAudioDb>) -> PyResult<Option<usize>> {
    db.as_ref()
        .map(|db| {
            db.inner
                .lock()
                .map_err(|_| poisoned("AudioDB"))?
                .len()
                .map_err(py_error)
        })
        .transpose()
}

fn format_optional_count(count: Option<usize>) -> String {
    count.map_or_else(|| "None".to_owned(), |count| count.to_string())
}

#[pyclass(name = "AudioDBIterator")]
struct PyAudioDbIterator {
    inner: Arc<Mutex<RustAudioDb>>,
    audios: std::vec::IntoIter<RustAudio>,
    after: Option<String>,
    exhausted: bool,
}

#[pymethods]
impl PyAudioDbIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyAudio>> {
        loop {
            if let Some(audio) = self.audios.next() {
                return PyAudio::from_rust(py, audio).map(Some);
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
            self.after = page.last().map(RustAudio::audio_id);
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

    /// 插入一条新 Audio。
    ///
    /// Args:
    ///     audio: 要插入的完整 Audio。
    ///
    /// Returns:
    ///     None。
    ///
    /// Raises:
    ///     AsrDataError: ID 已存在或文档校验失败。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB, Audio, AudioSource
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000), id="one")
    ///     >>> db.insert(doc)
    fn insert(&self, py: Python<'_>, audio: PyRef<'_, PyAudio>) -> PyResult<()> {
        let audio = audio.cloned_inner(py)?;
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .insert(&audio)
            .map_err(py_error)
    }

    /// 更新已有 Audio，仅在内容变化时写入。
    ///
    /// Args:
    ///     audio: 包含新内容的完整 Audio。
    ///
    /// Returns:
    ///     实际发生更新时为 True，否则为 False。
    ///
    /// Raises:
    ///     KeyError: 文档 ID 不存在。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB, Audio, AudioSource
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000), id="one")
    ///     >>> db.insert(doc)
    ///     >>> doc.metadata["checked"] = True
    ///     >>> changed = db.update(doc)
    fn update(&self, py: Python<'_>, audio: PyRef<'_, PyAudio>) -> PyResult<bool> {
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
    ///     after: 上一页最后一个 Audio ID。
    ///     min_duration_ms: 可选最短时长。
    ///     max_duration_ms: 可选最长时长。
    ///     created_from: 带时区的创建时间下界。
    ///     created_until: 带时区的创建时间上界，不包含。
    ///     updated_from: 带时区的修改时间下界。
    ///     updated_until: 带时区的修改时间上界，不包含。
    ///     metadata: 要精确匹配的 JSON metadata。
    ///
    /// Returns:
    ///     按 Audio ID 排序的文档列表。
    ///
    /// Raises:
    ///     ValueError: 范围反向、datetime 无时区或 limit 无效。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB, Audio, AudioSource
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000), id="one")
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
    ) -> PyResult<Vec<PyAudio>> {
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
            .map(|audio| PyAudio::from_rust(py, audio))
            .collect()
    }

    /// 评测数据库中全部匹配文档。
    ///
    /// 不传 source 时自动发现全部可评测来源。内部使用 ID 游标分页，
    /// batch_size 只控制单批读取量，不限制最终数据集大小。
    ///
    /// Args:
    ///     transcription: 转写来源或来源列表。
    ///     activity: Activity 来源或来源列表。
    ///     normalize: 是否执行中文文本标准化。
    ///     batch_size: 每批读取的文档数。
    ///     after: 可选起始 Audio ID 游标。
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
    ///     >>> from asr_data import AudioDB, Audio, AudioSource
    ///     >>> from asr_data.annotation import Transcription
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000), id="one")
    ///     >>> timeline = doc.timeline("mono")
    ///     >>> _ = timeline.annotate_span(
    ///     ...     0, timeline.duration_ms, Transcription("你好"), is_reference=True
    ///     ... )
    ///     >>> _ = timeline.annotate_span(
    ///     ...     0, timeline.duration_ms, Transcription("你好"),
    ///     ...     is_reference=False, source="qwen-asr"
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
        activity=None,
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
        py: Python<'_>,
        transcription: Option<&Bound<'_, PyAny>>,
        activity: Option<&Bound<'_, PyAny>>,
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
        let config = eval_config(transcription, activity, normalize)?;
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
        let query = AudioQuery {
            limit: batch_size,
            after,
            min_duration: min_duration_ms.map(DurationMs),
            max_duration: max_duration_ms.map(DurationMs),
            created_from,
            created_until,
            updated_from,
            updated_until,
            metadata,
        };
        let inner = py.detach(|| {
            let db = self.inner.lock().map_err(|_| poisoned("AudioDB"))?;
            db.eval(&query, &config).map_err(py_error)
        })?;
        Ok(PyDatasetEvaluation { inner })
    }

    fn __getitem__(&self, py: Python<'_>, audio_id: &str) -> PyResult<PyAudio> {
        let audio = self
            .inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .get(audio_id)
            .map_err(py_error)?
            .ok_or_else(|| PyKeyError::new_err(audio_id.to_string()))?;
        PyAudio::from_rust(py, audio)
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
    ///     >>> from asr_data import AudioDB, Audio, AudioSource
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> db.insert(Audio(
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
    ///     audios: 要更新的完整 Audio 列表。
    ///
    /// Returns:
    ///     实际发生变化的文档数量。
    ///
    /// Raises:
    ///     KeyError: 任一文档 ID 不存在。
    ///
    /// Examples:
    ///     >>> from tempfile import TemporaryDirectory
    ///     >>> from asr_data import AudioDB, Audio, AudioSource
    ///     >>> directory = TemporaryDirectory()
    ///     >>> db = AudioDB.create(f"{directory.name}/dataset.db")
    ///     >>> docs = [
    ///     ...     Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000), id=name)
    ///     ...     for name in ("first", "second")
    ///     ... ]
    ///     >>> for doc in docs:
    ///     ...     db.insert(doc)
    ///     ...     doc.metadata["checked"] = True
    ///     >>> changed = db.update_many(docs)
    fn update_many(&self, py: Python<'_>, audios: Vec<Py<PyAudio>>) -> PyResult<usize> {
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

fn py_dataset_error(error: AudioDatasetError) -> PyErr {
    match error {
        error @ (AudioDatasetError::EmptyRepositoryId
        | AudioDatasetError::EmptyRevision
        | AudioDatasetError::InvalidLicense { .. }) => PyValueError::new_err(error.to_string()),
        AudioDatasetError::Database { source, .. } => py_db_error(source),
        AudioDatasetError::ModelScopeDownload { .. } | AudioDatasetError::ReadLicense { .. } => {
            py_error(error)
        }
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAudioDataset>()?;
    module.add_class::<PyAudioDb>()?;
    module.add_class::<PyAudioDbIterator>()?;
    Ok(())
}
