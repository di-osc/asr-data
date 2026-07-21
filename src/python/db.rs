use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::db::{AudioDb as RustAudioDb, AudioDbMode, AudioQuery};
use crate::doc::AudioDoc as RustAudioDoc;
use crate::utils::DurationMs;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

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
    #[new]
    #[pyo3(signature = (path, read_only=false))]
    fn new(path: String, read_only: bool) -> PyResult<Self> {
        let mode = if read_only {
            AudioDbMode::ReadOnly
        } else {
            AudioDbMode::ReadWrite
        };
        let db = RustAudioDb::open(&path, mode);
        Ok(Self {
            inner: Arc::new(Mutex::new(db.map_err(py_error)?)),
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

    #[pyo3(signature = (limit=100, *, after=None, min_duration_ms=None, max_duration_ms=None, metadata=None))]
    fn query(
        &self,
        py: Python<'_>,
        limit: usize,
        after: Option<String>,
        min_duration_ms: Option<u64>,
        max_duration_ms: Option<u64>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<PyAudioDoc>> {
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

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAudioDb>()?;
    module.add_class::<PyAudioDbIterator>()?;
    Ok(())
}
