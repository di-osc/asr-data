use std::sync::{Arc, RwLock};

use crate::audio::AudioSource as RustAudioSource;
use crate::doc::AudioDoc as RustAudioDoc;
use crate::utils::DurationMs;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::audio::{py_source_from_rust, rust_source_from_py};
use super::common::{
    SharedAudio, audio_channel, audio_channel_name, format_duration_ms, format_source_field,
    poisoned, py_error, truncate,
};
use super::timeline::PyTimeline;

#[pyclass(name = "AudioDoc")]
pub(super) struct PyAudioDoc {
    inner: SharedAudio,
    metadata: Py<PyDict>,
}

impl PyAudioDoc {
    pub(super) fn from_rust(py: Python<'_>, audio: RustAudioDoc) -> PyResult<Self> {
        let metadata = PyDict::new(py);
        for (key, value) in &audio.metadata {
            metadata.set_item(key, pythonize::pythonize(py, value).map_err(py_error)?)?;
        }
        Ok(Self {
            inner: Arc::new(RwLock::new(audio)),
            metadata: metadata.unbind(),
        })
    }

    fn build(py: Python<'_>, source: RustAudioSource, id: Option<String>) -> PyResult<Self> {
        let audio = match id {
            Some(id) => RustAudioDoc::with_id(id, source),
            None => RustAudioDoc::new(source),
        };
        Self::from_rust(py, audio)
    }

    pub(super) fn cloned_inner(&self, py: Python<'_>) -> PyResult<RustAudioDoc> {
        let mut audio = self.inner.read().map_err(|_| poisoned("audio"))?.clone();
        audio.metadata =
            pythonize::depythonize(self.metadata.bind(py).as_any()).map_err(py_error)?;
        Ok(audio)
    }
}

#[pymethods]
impl PyAudioDoc {
    #[new]
    #[pyo3(signature = (source, id=None))]
    fn new(py: Python<'_>, source: &Bound<'_, PyAny>, id: Option<String>) -> PyResult<Self> {
        let source = rust_source_from_py(source)?;
        Self::build(py, source, id)
    }

    #[getter]
    fn source(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let audio = self.inner.read().map_err(|_| poisoned("audio"))?;
        py_source_from_rust(py, &audio.source)
    }

    #[getter]
    fn id(&self) -> PyResult<String> {
        Ok(self.inner.read().map_err(|_| poisoned("audio"))?.id.clone())
    }

    fn timeline(&self, channel: &Bound<'_, PyAny>) -> PyResult<Option<PyTimeline>> {
        let channel = audio_channel(channel)?;
        let exists = self
            .inner
            .read()
            .map_err(|_| poisoned("audio"))?
            .timeline(channel)
            .map_err(py_error)?
            .is_some();
        Ok(exists.then(|| PyTimeline {
            audio: Arc::clone(&self.inner),
            channel,
        }))
    }

    #[pyo3(signature = (channel, duration_ms=None))]
    fn ensure_timeline(
        &self,
        channel: &Bound<'_, PyAny>,
        duration_ms: Option<u64>,
    ) -> PyResult<PyTimeline> {
        let channel = audio_channel(channel)?;
        self.inner
            .write()
            .map_err(|_| poisoned("audio"))?
            .ensure_timeline(channel, duration_ms.map(DurationMs))
            .map_err(py_error)?;
        Ok(PyTimeline {
            audio: Arc::clone(&self.inner),
            channel,
        })
    }

    fn remove_timeline(&self, channel: &Bound<'_, PyAny>) -> PyResult<bool> {
        let channel = audio_channel(channel)?;
        Ok(self
            .inner
            .write()
            .map_err(|_| poisoned("audio"))?
            .remove_timeline(channel)
            .map_err(py_error)?
            .is_some())
    }

    fn validate(&self) -> PyResult<()> {
        self.inner
            .read()
            .map_err(|_| poisoned("audio"))?
            .validate()
            .map_err(py_error)
    }

    #[getter]
    fn timelines<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let channels = self
            .inner
            .read()
            .map_err(|_| poisoned("audio"))?
            .timelines()
            .keys()
            .copied()
            .collect::<Vec<_>>();
        let timelines = PyDict::new(py);
        for channel in channels {
            timelines.set_item(
                audio_channel_name(channel),
                Py::new(
                    py,
                    PyTimeline {
                        audio: Arc::clone(&self.inner),
                        channel,
                    },
                )?,
            )?;
        }
        Ok(timelines)
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        self.metadata.bind(py).clone()
    }

    fn __repr__(&self) -> PyResult<String> {
        let audio = self.inner.read().map_err(|_| poisoned("audio"))?;
        let mut fields = vec![
            format!("id={:?}", truncate(&audio.id, 40)),
            format_source_field(&audio.source),
        ];
        if let Some(duration) = audio.timeline_duration() {
            fields.push(format!(
                "duration={:?}",
                format_duration_ms(duration.0 as f64)
            ));
        }
        let annotation_count = audio
            .timelines()
            .values()
            .map(|timeline| timeline.annotations.len())
            .sum::<usize>();
        if annotation_count != 0 {
            fields.push(format!("annotations={annotation_count}"));
        }
        Ok(format!("AudioDoc({})", fields.join(", ")))
    }

    fn __str__(&self) -> PyResult<String> {
        let audio = self.inner.read().map_err(|_| poisoned("audio"))?;
        let id = truncate(&audio.id, 40);
        Ok(match audio.timeline_duration() {
            Some(duration) => {
                format!(
                    "AudioDoc {:?} ({})",
                    id,
                    format_duration_ms(duration.0 as f64)
                )
            }
            None => format!("AudioDoc {id:?}"),
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAudioDoc>()?;
    Ok(())
}
