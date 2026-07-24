use std::sync::{Arc, Mutex, RwLock};

use crate::audio::AudioSource as RustAudioSource;
use crate::doc::Audio as RustAudio;
use crate::utils::DurationMs;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};

use super::audio::{
    PyAudioInfo, PyWaveform, async_runtime, display_rust_waveform, py_audio_info_from_rust,
    py_source_from_rust, rust_source_from_py,
};
use super::common::{
    SharedAudio, audio_channel, audio_channel_name, format_duration_ms, format_source_field,
    poisoned, py_error, truncate,
};
use super::timeline::PyTimeline;

/// 音频来源、元信息、时间轴、标注和业务 metadata 的集合。
///
/// 构造时会完整解码音频并根据声道自动创建最终时长的 timeline。
///
/// Args:
///     source: AudioSource、路径或 URL。
///     id: 可选的文档 ID；省略时自动生成。
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
                Some(id) => source.load_with_id(id),
                None => source.load(),
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
        let audio = tokio::task::spawn_blocking(move || match id {
            Some(id) => source.load_with_id(id),
            None => source.load(),
        })
        .await
        .map_err(|error| anyhow::anyhow!("audio loader worker failed: {error}"))
        .and_then(|result| result)
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

    /// 从本地文件完整加载 Audio。
    ///
    /// Args:
    ///     path: 音频文件路径。
    ///     id: 可选的文档 ID。
    ///
    /// Returns:
    ///     已完整解码的 Audio。
    ///
    /// Raises:
    ///     AsrDataError: 文件无法读取或解码。
    ///
    /// Examples:
    ///     >>> audio = Audio.from_path("audio.wav", id="sample")
    #[staticmethod]
    #[pyo3(signature = (path, *, id=None))]
    fn from_path(py: Python<'_>, path: String, id: Option<String>) -> PyResult<Self> {
        Self::build(py, RustAudioSource::from_path(path), id)
    }

    /// 从 URL 完整加载 Audio。
    ///
    /// Args:
    ///     url: HTTP、HTTPS 或 file URL。
    ///     id: 可选的文档 ID。
    ///
    /// Returns:
    ///     已完整解码的 Audio。
    ///
    /// Raises:
    ///     AsrDataError: URL 无法读取或音频无法解码。
    ///
    /// Examples:
    ///     >>> audio = Audio.from_url("https://example.com/audio.wav")
    #[staticmethod]
    #[pyo3(signature = (url, *, id=None))]
    fn from_url(py: Python<'_>, url: String, id: Option<String>) -> PyResult<Self> {
        Self::build(py, RustAudioSource::from_url(url), id)
    }

    /// 从带容器或编码信息的音频字节完整加载 Audio。
    ///
    /// Args:
    ///     data: WAV、MP3 等编码音频字节。
    ///     id: 可选的文档 ID。
    ///
    /// Returns:
    ///     已完整解码的 Audio。
    ///
    /// Raises:
    ///     AsrDataError: 字节不是受支持的音频。
    ///
    /// Examples:
    ///     >>> audio = Audio.from_bytes(encoded_audio)
    #[staticmethod]
    #[pyo3(signature = (data, *, id=None))]
    fn from_bytes(py: Python<'_>, data: &Bound<'_, PyBytes>, id: Option<String>) -> PyResult<Self> {
        Self::build(
            py,
            RustAudioSource::from_encoded_bytes(data.as_bytes().to_vec()),
            id,
        )
    }

    /// 从 base64 编码音频完整加载 Audio。
    ///
    /// Args:
    ///     data: 编码音频的 base64 字符串。
    ///     id: 可选的文档 ID。
    ///
    /// Returns:
    ///     已完整解码的 Audio。
    ///
    /// Raises:
    ///     AsrDataError: base64 或其中的音频无效。
    ///
    /// Examples:
    ///     >>> audio = Audio.from_base64(encoded)
    #[staticmethod]
    #[pyo3(signature = (data, *, id=None))]
    fn from_base64(py: Python<'_>, data: String, id: Option<String>) -> PyResult<Self> {
        Self::build(py, RustAudioSource::from_base64(data), id)
    }

    /// 从 PCM S16LE 字节完整加载 Audio。
    ///
    /// Args:
    ///     data: 交错排列的 PCM S16LE 字节。
    ///     sample_rate: 每秒每声道采样帧数。
    ///     channels: 声道数，默认为 1。
    ///     id: 可选的文档 ID。
    ///
    /// Returns:
    ///     已完整解码的 Audio。
    ///
    /// Raises:
    ///     AsrDataError: PCM 参数或字节长度无效。
    ///
    /// Examples:
    ///     >>> audio = Audio.from_pcm(b"\0\0" * 16000, 16000)
    #[staticmethod]
    #[pyo3(signature = (data, sample_rate, channels=1, *, id=None))]
    fn from_pcm(
        py: Python<'_>,
        data: &Bound<'_, PyBytes>,
        sample_rate: u32,
        channels: u16,
        id: Option<String>,
    ) -> PyResult<Self> {
        Self::build(
            py,
            RustAudioSource::from_pcm_s16le(data.as_bytes().to_vec(), sample_rate, channels),
            id,
        )
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

    /// 返回完整波形。
    ///
    /// Returns:
    ///     当前 Audio 的完整 Waveform。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> audio = AudioSource.from_pcm(b"\0\0" * 10, 16000).load()
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

    /// 在 Jupyter 中显示完整音频播放器。
    ///
    /// Args:
    ///     start_ms: 可选播放起始时间。
    ///     end_ms: 可选播放结束时间。
    ///     autoplay: 是否自动播放。
    ///
    /// Returns:
    ///     None；播放器直接发送到当前 Jupyter 输出。
    ///
    /// Raises:
    ///     ValueError: 结束时间早于起始时间。
    ///     AsrDataError: IPython 不可用。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> audio = AudioSource.from_pcm(b"\0\0" * 100, 1000).load()
    ///     >>> audio.display(end_ms=50)
    #[pyo3(signature = (start_ms=None, end_ms=None, autoplay=false))]
    fn display(
        &self,
        py: Python<'_>,
        start_ms: Option<u64>,
        end_ms: Option<u64>,
        autoplay: bool,
    ) -> PyResult<()> {
        let waveform = self
            .inner
            .write()
            .map_err(|_| poisoned("audio"))?
            .as_waveform()
            .map_err(py_error)?;
        display_rust_waveform(py, waveform, start_ms, end_ms, autoplay)
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
