use std::sync::{Arc, Mutex, RwLock};

use crate::audio::AudioSource as RustAudioSource;
use crate::doc::AudioDoc as RustAudioDoc;
use crate::utils::DurationMs;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::audio::{
    PyAudioInfo, async_runtime, py_audio_info_from_rust, py_source_from_rust, rust_source_from_py,
};
use super::common::{
    SharedAudio, audio_channel, audio_channel_name, format_duration_ms, format_source_field,
    poisoned, py_error, truncate,
};
use super::timeline::PyTimeline;

/// 音频来源、元信息、时间轴、标注和业务 metadata 的集合。
///
/// 构造时会探测音频信息并根据声道自动创建 timeline，但不会解码浮点波形。
///
/// Args:
///     source: AudioSource、路径或 URL。
///     id: 可选稳定文档 ID；省略时自动生成。
///
/// Raises:
///     AsrDataError: 来源无法探测。
///
/// Examples:
///     >>> from asr_data import AudioDoc, AudioSource
///     >>> source = AudioSource.from_pcm(b"\0\0" * 16000, 16000)
///     >>> doc = AudioDoc(source, id="sample-1")
///     >>> doc.timeline("mono").duration_ms
///     1000
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
        let audio = py
            .detach(move || match id {
                Some(id) => RustAudioDoc::with_id_from_source(id, source),
                None => RustAudioDoc::from_source(source),
            })
            .map_err(py_error)?;
        Self::from_rust(py, audio)
    }

    pub(super) fn cloned_inner(&self, py: Python<'_>) -> PyResult<RustAudioDoc> {
        let mut audio = self.inner.read().map_err(|_| poisoned("audio"))?.clone();
        audio.metadata =
            pythonize::depythonize(self.metadata.bind(py).as_any()).map_err(py_error)?;
        Ok(audio)
    }
}

type AsyncDocResult = Arc<Mutex<Option<Result<RustAudioDoc, String>>>>;

#[pyclass(name = "_AudioDocTask")]
struct PyAudioDocTask {
    result: AsyncDocResult,
}

#[pymethods]
impl PyAudioDocTask {
    fn done(&self) -> PyResult<bool> {
        Ok(self
            .result
            .lock()
            .map_err(|_| poisoned("audio document task"))?
            .is_some())
    }

    fn result(&self, py: Python<'_>) -> PyResult<PyAudioDoc> {
        let result = self
            .result
            .lock()
            .map_err(|_| poisoned("audio document task"))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("audio document has not completed"))?;
        PyAudioDoc::from_rust(py, result.map_err(super::AsrDataError::new_err)?)
    }
}

fn spawn_doc_from_source(source: RustAudioSource, id: Option<String>) -> PyAudioDocTask {
    let result: AsyncDocResult = Arc::new(Mutex::new(None));
    let task_result = Arc::clone(&result);
    async_runtime().spawn(async move {
        let doc = match id {
            Some(id) => RustAudioDoc::with_id_afrom_source(id, source).await,
            None => RustAudioDoc::afrom_source(source).await,
        }
        .map_err(|error| format!("{error:#}"));
        if let Ok(mut slot) = task_result.lock() {
            *slot = Some(doc);
        }
    });
    PyAudioDocTask { result }
}

#[pymethods]
impl PyAudioDoc {
    #[new]
    #[pyo3(signature = (source, id=None))]
    fn new(py: Python<'_>, source: &Bound<'_, PyAny>, id: Option<String>) -> PyResult<Self> {
        let source = rust_source_from_py(source)?;
        Self::build(py, source, id)
    }

    #[staticmethod]
    #[pyo3(signature = (source, id=None))]
    fn _start_afrom_source(
        source: &Bound<'_, PyAny>,
        id: Option<String>,
    ) -> PyResult<PyAudioDocTask> {
        Ok(spawn_doc_from_source(rust_source_from_py(source)?, id))
    }

    /// 创建文档时保存的 AudioSource。
    #[getter]
    fn source(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let audio = self.inner.read().map_err(|_| poisoned("audio"))?;
        py_source_from_rust(py, &audio.source)
    }

    /// 不含解码样本的 AudioInfo。
    #[getter]
    fn audio_info(&self) -> PyResult<PyAudioInfo> {
        let audio = self.inner.read().map_err(|_| poisoned("audio"))?;
        Ok(py_audio_info_from_rust(&audio.audio_info))
    }

    /// 文档唯一 ID。
    #[getter]
    fn id(&self) -> PyResult<String> {
        Ok(self.inner.read().map_err(|_| poisoned("audio"))?.id.clone())
    }

    /// 查询指定声道的 timeline，不存在时返回 None。
    ///
    /// Args:
    ///     channel: ``"mono"``、``"left"``、``"right"`` 或声道索引。
    ///
    /// Returns:
    ///     对应 Timeline；不存在时为 None。
    ///
    /// Raises:
    ///     ValueError: 声道名称或索引无效。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> doc = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000))
    ///     >>> doc.timeline("mono").duration_ms
    ///     1
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
    /// 取得或创建指定声道的 timeline。
    ///
    /// Args:
    ///     channel: 声道名称或索引。
    ///     duration_ms: 可选时长；必须与文档音频时长一致。
    ///
    /// Returns:
    ///     已有或新建的 Timeline。
    ///
    /// Raises:
    ///     ValueError: 时长无效或与文档不一致。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> doc = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000))
    ///     >>> doc.ensure_timeline("mono") is not None
    ///     True
    fn ensure_timeline(
        &self,
        channel: &Bound<'_, PyAny>,
        duration_ms: Option<f64>,
    ) -> PyResult<PyTimeline> {
        let channel = audio_channel(channel)?;
        let duration_ms = duration_ms
            .map(|value| {
                if !value.is_finite() || value < 0.0 || value.ceil() > u64::MAX as f64 {
                    return Err(PyValueError::new_err(
                        "duration_ms must be a finite non-negative number",
                    ));
                }
                Ok(DurationMs(value.ceil() as u64))
            })
            .transpose()?;
        self.inner
            .write()
            .map_err(|_| poisoned("audio"))?
            .ensure_timeline(channel, duration_ms)
            .map_err(py_error)?;
        Ok(PyTimeline {
            audio: Arc::clone(&self.inner),
            channel,
        })
    }

    /// 删除指定声道的 timeline 并返回是否存在。
    ///
    /// Args:
    ///     channel: 声道名称或索引。
    ///
    /// Returns:
    ///     确实删除时为 True。
    ///
    /// Raises:
    ///     ValueError: 声道无效。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> doc = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000))
    ///     >>> doc.remove_timeline("mono")
    ///     True
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

    /// 校验声道、时长、annotation 范围、source 和重叠约束。
    ///
    /// Raises:
    ///     AsrDataError: 文档包含无效数据。
    ///
    /// Returns:
    ///     None。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> doc = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000))
    ///     >>> doc.validate() is None
    ///     True
    fn validate(&self) -> PyResult<()> {
        self.inner
            .read()
            .map_err(|_| poisoned("audio"))?
            .validate()
            .map_err(py_error)
    }

    /// 以声道名称为键的全部 timeline。
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

    /// 可原地修改的文档级 JSON metadata。
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
            .map(|timeline| timeline.annotation_count())
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
    module.add_class::<PyAudioDocTask>()?;
    Ok(())
}
