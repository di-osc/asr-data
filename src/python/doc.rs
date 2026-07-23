use std::sync::{Arc, Mutex, RwLock};

use crate::audio::AudioSource as RustAudioSource;
use crate::doc::Audio as RustAudio;
use crate::utils::DurationMs;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::audio::{
    PyAudioInfo, PyAudioIterator, PyWaveform, async_runtime, py_audio_info_from_rust,
    py_source_from_rust, rust_source_from_py, stream_audio,
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
///     >>> from asr_data import Audio, AudioSource
///     >>> source = AudioSource.from_pcm(b"\0\0" * 16000, 16000)
///     >>> audio = Audio(source, id="sample-1")
///     >>> audio.timeline("mono").duration_ms
///     1000
#[pyclass(name = "Audio")]
pub(super) struct PyAudio {
    inner: SharedAudio,
    metadata: Py<PyDict>,
}

impl PyAudio {
    pub(super) fn from_rust(py: Python<'_>, audio: RustAudio) -> PyResult<Self> {
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
                Some(id) => RustAudio::with_id_from_source(id, source),
                None => RustAudio::from_source(source),
            })
            .map_err(py_error)?;
        Self::from_rust(py, audio)
    }

    pub(super) fn cloned_inner(&self, py: Python<'_>) -> PyResult<RustAudio> {
        let mut audio = self.inner.read().map_err(|_| poisoned("audio"))?.clone();
        audio.metadata =
            pythonize::depythonize(self.metadata.bind(py).as_any()).map_err(py_error)?;
        Ok(audio)
    }
}

type AsyncAudioResult = Arc<Mutex<Option<Result<RustAudio, String>>>>;

#[pyclass(name = "_AudioTask")]
struct PyAudioTask {
    result: AsyncAudioResult,
}

#[pymethods]
impl PyAudioTask {
    fn done(&self) -> PyResult<bool> {
        Ok(self
            .result
            .lock()
            .map_err(|_| poisoned("audio document task"))?
            .is_some())
    }

    fn result(&self, py: Python<'_>) -> PyResult<PyAudio> {
        let result = self
            .result
            .lock()
            .map_err(|_| poisoned("audio document task"))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("audio document has not completed"))?;
        PyAudio::from_rust(py, result.map_err(super::AsrDataError::new_err)?)
    }
}

fn spawn_audio_from_source(source: RustAudioSource, id: Option<String>) -> PyAudioTask {
    let result: AsyncAudioResult = Arc::new(Mutex::new(None));
    let task_result = Arc::clone(&result);
    async_runtime().spawn(async move {
        let audio = match id {
            Some(id) => RustAudio::with_id_afrom_source(id, source).await,
            None => RustAudio::afrom_source(source).await,
        }
        .map_err(|error| format!("{error:#}"));
        if let Ok(mut slot) = task_result.lock() {
            *slot = Some(audio);
        }
    });
    PyAudioTask { result }
}

#[pymethods]
impl PyAudio {
    #[new]
    #[pyo3(signature = (source, id=None))]
    fn new(py: Python<'_>, source: &Bound<'_, PyAny>, id: Option<String>) -> PyResult<Self> {
        let source = rust_source_from_py(source)?;
        Self::build(py, source, id)
    }

    #[staticmethod]
    #[pyo3(signature = (source, id=None))]
    fn _start_afrom_source(source: &Bound<'_, PyAny>, id: Option<String>) -> PyResult<PyAudioTask> {
        Ok(spawn_audio_from_source(rust_source_from_py(source)?, id))
    }

    /// 创建文档时保存的 AudioSource。
    #[getter]
    fn source(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let audio = self.inner.read().map_err(|_| poisoned("audio"))?;
        py_source_from_rust(py, &audio.source)
    }

    /// 不含解码样本的 AudioInfo。
    #[getter]
    fn info(&self) -> PyResult<PyAudioInfo> {
        let audio = self.inner.read().map_err(|_| poisoned("audio"))?;
        Ok(py_audio_info_from_rust(&audio.info))
    }

    /// 文档唯一 ID。
    #[getter]
    fn id(&self) -> PyResult<String> {
        Ok(self.inner.read().map_err(|_| poisoned("audio"))?.id.clone())
    }

    /// 解码后的波形是否已经保留在内存中。
    #[getter]
    fn is_loaded(&self) -> PyResult<bool> {
        Ok(self
            .inner
            .read()
            .map_err(|_| poisoned("audio"))?
            .is_loaded())
    }

    /// 返回完整波形；尚未加载时会按需解码。
    ///
    /// Returns:
    ///     当前 Audio 的完整 Waveform。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> audio = AudioSource.from_pcm(b"\0\0" * 10, 16000).open()
    ///     >>> audio.as_waveform().frame_count
    ///     10
    fn as_waveform(&self) -> PyResult<PyWaveform> {
        let waveform = self
            .inner
            .write()
            .map_err(|_| poisoned("audio"))?
            .as_waveform()
            .map_err(py_error)?;
        Ok(PyWaveform::from_rust(waveform))
    }

    /// 释放运行时波形缓存，保留 info、timeline 和标注。
    ///
    /// Returns:
    ///     None。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> audio = AudioSource.from_pcm(b"\0\0", 16000).load()
    ///     >>> audio.unload()
    fn unload(&self) -> PyResult<()> {
        self.inner.write().map_err(|_| poisoned("audio"))?.unload();
        Ok(())
    }

    /// 按固定时长顺序读取 AudioChunk。
    ///
    /// Args:
    ///     chunk_size_ms: 每个 chunk 的目标时长，单位为毫秒。
    ///
    /// Returns:
    ///     可迭代的 AudioIterator。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> audio = AudioSource.from_pcm(b"\0\0" * 16000, 16000).open()
    ///     >>> len(list(audio.stream(500)))
    ///     2
    #[pyo3(signature = (chunk_size_ms=100))]
    fn stream(&self, py: Python<'_>, chunk_size_ms: u64) -> PyResult<PyAudioIterator> {
        if chunk_size_ms == 0 {
            return Err(PyValueError::new_err(
                "chunk_size_ms must be greater than zero",
            ));
        }
        stream_audio(py, self.inner.clone(), chunk_size_ms)
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
    ///     >>> from asr_data import Audio, AudioSource
    ///     >>> audio = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
    ///     >>> audio.timeline("mono").duration_ms
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
    ///     >>> from asr_data import Audio, AudioSource
    ///     >>> audio = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
    ///     >>> audio.ensure_timeline("mono") is not None
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
    ///     >>> from asr_data import Audio, AudioSource
    ///     >>> audio = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
    ///     >>> audio.remove_timeline("mono")
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
    ///     >>> from asr_data import Audio, AudioSource
    ///     >>> audio = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
    ///     >>> audio.validate() is None
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
        let span_count = audio
            .timelines()
            .values()
            .map(|timeline| timeline.span_count())
            .sum::<usize>();
        if span_count != 0 {
            fields.push(format!("annotations={span_count}"));
        }
        Ok(format!("Audio({})", fields.join(", ")))
    }

    fn __str__(&self) -> PyResult<String> {
        let audio = self.inner.read().map_err(|_| poisoned("audio"))?;
        let id = truncate(&audio.id, 40);
        Ok(match audio.timeline_duration() {
            Some(duration) => {
                format!("Audio {:?} ({})", id, format_duration_ms(duration.0 as f64))
            }
            None => format!("Audio {id:?}"),
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAudio>()?;
    module.add_class::<PyAudioTask>()?;
    Ok(())
}
