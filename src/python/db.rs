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
    #[staticmethod]
    fn create(path: String) -> PyResult<Self> {
        let db = RustAudioDb::create(&path);
        Ok(Self {
            inner: Arc::new(Mutex::new(db.map_err(py_db_error)?)),
            path,
            read_only: false,
        })
    }

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

    fn insert(&self, py: Python<'_>, audio: PyRef<'_, PyAudioDoc>) -> PyResult<()> {
        let audio = audio.cloned_inner(py)?;
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .insert(&audio)
            .map_err(py_error)
    }

    fn update(&self, py: Python<'_>, audio: PyRef<'_, PyAudioDoc>) -> PyResult<bool> {
        let audio = audio.cloned_inner(py)?;
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .update(&audio)
            .map_err(py_db_error)
    }

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

    fn delete(&self, audio_id: &str) -> PyResult<bool> {
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .delete(audio_id)
            .map_err(py_error)
    }

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

    fn set_metadata(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value: serde_json::Value = pythonize::depythonize(value).map_err(py_error)?;
        self.inner
            .lock()
            .map_err(|_| poisoned("AudioDB"))?
            .set_metadata(key, &value)
            .map_err(py_error)
    }

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
