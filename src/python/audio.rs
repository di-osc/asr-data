use std::sync::mpsc::{Receiver, TryRecvError, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};

use crate::audio::{
    AudioChunk as RustAudioChunk, AudioFormat as RustAudioFormat, AudioInfo as RustAudioInfo,
    AudioSource as RustAudioSource, Waveform as RustWaveform,
};
use crate::utils::DurationMs;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, ndarray::ArrayView1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};

use super::AsrDataError;
use super::common::{
    SharedAudio, audio_channel, encoding_name, format_duration_ms, poisoned, py_error,
    summarize_url, truncate,
};
use super::doc::PyAudio;
use super::timeline::PyTimeline;

/// 音频编码、采样率和声道数组成的格式信息。
#[pyclass(name = "AudioFormat", frozen)]
#[derive(Clone)]
pub(super) struct PyAudioFormat {
    inner: RustAudioFormat,
}

#[pymethods]
impl PyAudioFormat {
    /// 编码名称，例如 ``"wav"`` 或 ``"pcm_s16le"``。
    #[getter]
    fn encoding(&self) -> String {
        encoding_name(&self.inner.encoding)
    }

    /// 每秒每个声道的采样帧数。
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    /// 声道数。
    #[getter]
    fn channels(&self) -> u16 {
        self.inner.channels
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioFormat(encoding={:?}, sample_rate={}, channels={})",
            self.encoding(),
            self.sample_rate(),
            self.channels()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "{}/{}Hz/{}ch",
            self.encoding(),
            self.sample_rate(),
            self.channels()
        )
    }
}

/// 不包含解码采样的音频元信息。
#[pyclass(name = "AudioInfo", frozen)]
#[derive(Clone)]
pub(super) struct PyAudioInfo {
    inner: RustAudioInfo,
}

pub(super) fn py_audio_info_from_rust(info: &RustAudioInfo) -> PyAudioInfo {
    PyAudioInfo {
        inner: info.clone(),
    }
}

#[pymethods]
impl PyAudioInfo {
    /// 原始采样率。
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    /// 原始声道数。
    #[getter]
    fn channels(&self) -> u16 {
        self.inner.channels
    }

    /// 每个声道的有效采样帧数。
    #[getter]
    fn frame_count(&self) -> u64 {
        self.inner.frame_count
    }

    /// 根据帧数和采样率计算的时长，单位为毫秒。
    #[getter]
    fn duration_ms(&self) -> f64 {
        self.inner.duration_ms()
    }

    /// 原始音频格式。
    #[getter]
    fn source_format(&self) -> PyAudioFormat {
        PyAudioFormat {
            inner: self.inner.source_format.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioInfo(frames={}, duration={}, sample_rate={}, channels={}, source_format={:?})",
            self.frame_count(),
            format_duration_ms(self.duration_ms()),
            self.sample_rate(),
            self.channels(),
            self.source_format().__str__(),
        )
    }
}

/// 已解码到内存中的音频波形。
///
/// Args:
///     samples: 一维 float32 兼容数组；多声道样本按帧交错排列。
///     sample_rate: 每秒每个声道的采样帧数。
///     channels: 声道数，默认为 1。
///
/// Raises:
///     ValueError: 采样率或声道数无效，或者样本数不能整除声道数。
///
/// Examples:
///     >>> import numpy as np
///     >>> from asr_data import Waveform
///     >>> audio = Waveform(np.zeros(16000, dtype=np.float32), 16000)
///     >>> audio.duration_ms
///     1000.0
#[pyclass(name = "Waveform")]
pub(super) struct PyWaveform {
    inner: RustWaveform,
    numpy_owner: Option<Py<PyArray1<f32>>>,
}

impl PyWaveform {
    pub(super) fn from_rust(inner: RustWaveform) -> Self {
        Self {
            inner,
            numpy_owner: None,
        }
    }

    fn materialize(&self, py: Python<'_>) -> PyResult<RustWaveform> {
        let Some(owner) = &self.numpy_owner else {
            return Ok(self.inner.clone());
        };
        let samples = owner.bind(py).readonly().as_slice()?.to_vec();
        let mut audio = RustWaveform::try_new_with_channels(
            samples,
            self.inner.sample_rate,
            self.inner.channels,
        )
        .map_err(py_error)?;
        audio.source_format = self.inner.source_format.clone();
        Ok(audio)
    }
}

#[pymethods]
impl PyWaveform {
    #[new]
    #[pyo3(signature = (samples, sample_rate, channels=1))]
    fn new(samples: &Bound<'_, PyAny>, sample_rate: u32, channels: u16) -> PyResult<Self> {
        let py = samples.py();
        let numpy = py.import("numpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", numpy.getattr("float32")?)?;
        kwargs.set_item("order", "C")?;
        let samples = numpy
            .getattr("asarray")?
            .call((samples,), Some(&kwargs))?
            .cast_into::<PyArray1<f32>>()?
            .unbind();
        {
            let array = samples.bind(py).readonly();
            let len = array
                .as_slice()
                .map_err(|_| {
                    PyValueError::new_err(
                        "samples must be a one-dimensional C-contiguous float32 array",
                    )
                })?
                .len();
            if channels == 0 || !len.is_multiple_of(usize::from(channels)) {
                return Err(PyValueError::new_err(
                    "samples must contain complete audio frames",
                ));
            }
            Ok(Self {
                inner: RustWaveform::new_with_channels(Vec::new(), sample_rate, channels),
                numpy_owner: Some(samples),
            })
        }
    }

    /// 从本地文件加载并解码音频。
    ///
    /// Args:
    ///     path: 本地音频文件路径。
    ///
    /// Returns:
    ///     解码后的完整 Waveform。
    ///
    /// Raises:
    ///     AsrDataError: 文件无法读取或音频无法解码。
    ///
    /// Examples:
    ///     >>> from tempfile import NamedTemporaryFile
    ///     >>> from urllib.request import urlretrieve
    ///     >>> from asr_data import Waveform
    ///     >>> url = "https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav"
    ///     >>> with NamedTemporaryFile(suffix=".wav") as file:
    ///     ...     _ = urlretrieve(url, file.name)
    ///     ...     audio = Waveform.from_path(file.name)
    #[staticmethod]
    fn from_path(py: Python<'_>, path: String) -> PyResult<Self> {
        py.detach(move || RustWaveform::from_path(path))
            .map(Self::from_rust)
            .map_err(py_error)
    }

    /// 从 HTTP 或 HTTPS URL 下载并解码音频。
    ///
    /// Args:
    ///     url: 音频 URL。
    ///
    /// Returns:
    ///     解码后的完整 Waveform。
    ///
    /// Raises:
    ///     AsrDataError: 请求失败或音频无法解码。
    ///
    /// Examples:
    ///     >>> from asr_data import Waveform
    ///     >>> audio = Waveform.from_url(
    ///     ...     "https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav"
    ///     ... )
    #[staticmethod]
    fn from_url(py: Python<'_>, url: String) -> PyResult<Self> {
        py.detach(move || RustWaveform::from_url(url))
            .map(Self::from_rust)
            .map_err(py_error)
    }

    /// 从 WAV、MP3 等编码字节解码音频。
    ///
    /// Args:
    ///     data: 包含音频容器或编码信息的字节。
    ///
    /// Returns:
    ///     解码后的完整 Waveform。
    ///
    /// Raises:
    ///     AsrDataError: 字节不是受支持的音频。
    ///
    /// Examples:
    ///     >>> from urllib.request import urlopen
    ///     >>> from asr_data import Waveform
    ///     >>> url = "https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav"
    ///     >>> audio = Waveform.from_bytes(urlopen(url).read())
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = data.as_bytes().to_vec();
        py.detach(move || RustWaveform::from_encoded_bytes(bytes))
            .map(Self::from_rust)
            .map_err(py_error)
    }

    /// 从 base64 编码的音频字符串解码音频。
    ///
    /// Args:
    ///     data: base64 字符串或 data URL。
    ///
    /// Returns:
    ///     解码后的完整 Waveform。
    ///
    /// Raises:
    ///     AsrDataError: base64 或音频编码无效。
    ///
    /// Examples:
    ///     >>> import base64
    ///     >>> from urllib.request import urlopen
    ///     >>> from asr_data import Waveform
    ///     >>> url = "https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav"
    ///     >>> data = base64.b64encode(urlopen(url).read()).decode()
    ///     >>> audio = Waveform.from_base64(data)
    #[staticmethod]
    fn from_base64(py: Python<'_>, data: String) -> PyResult<Self> {
        py.detach(move || RustWaveform::from_base64(data))
            .map(Self::from_rust)
            .map_err(py_error)
    }

    /// 从 PCM S16LE 原始字节创建音频。
    ///
    /// Args:
    ///     data: 按帧交错的有符号 16 位小端 PCM 字节。
    ///     sample_rate: 采样率。
    ///     channels: 声道数，默认为 1。
    ///
    /// Returns:
    ///     转换为 float32 样本的 Waveform。
    ///
    /// Raises:
    ///     ValueError: PCM 参数或帧长度无效。
    ///
    /// Examples:
    ///     >>> from asr_data import Waveform
    ///     >>> Waveform.from_pcm(b"\0\0" * 16000, 16000).duration_ms
    ///     1000.0
    #[staticmethod]
    #[pyo3(signature = (data, sample_rate, channels=1))]
    fn from_pcm(
        py: Python<'_>,
        data: &Bound<'_, PyBytes>,
        sample_rate: u32,
        channels: u16,
    ) -> PyResult<Self> {
        let bytes = data.as_bytes().to_vec();
        py.detach(move || RustWaveform::from_pcm_s16le(bytes, sample_rate, channels))
            .map(Self::from_rust)
            .map_err(py_error)
    }

    /// 加载任意 AudioSource 并解码完整音频。
    ///
    /// Args:
    ///     source: 要加载的 AudioSource。
    ///
    /// Returns:
    ///     解码后的完整 Waveform。
    ///
    /// Raises:
    ///     AsrDataError: 来源无法读取或解码。
    ///
    /// Examples:
    ///     >>> from asr_data import Waveform, AudioSource
    ///     >>> source = AudioSource.from_pcm(b"\0\0" * 10, 16000)
    ///     >>> Waveform.from_source(source).frame_count
    ///     10
    #[staticmethod]
    fn from_source(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Self> {
        let source = rust_source_from_py(source)?;
        py.detach(move || RustWaveform::from_source(&source))
            .map(Self::from_rust)
            .map_err(py_error)
    }

    #[staticmethod]
    fn _start_aload_from_path(path: String) -> PyResult<PyAudioLoadTask> {
        spawn_source_aload(RustAudioSource::from_path(path), None, None)
    }

    #[staticmethod]
    fn _start_aload_from_source(source: &Bound<'_, PyAny>) -> PyResult<PyAudioLoadTask> {
        spawn_source_aload(rust_source_from_py(source)?, None, None)
    }

    /// 当前采样率。
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    /// 当前声道数。
    #[getter]
    fn channels(&self) -> u16 {
        self.inner.channels
    }

    /// 每个声道的采样帧数。
    #[getter]
    fn frame_count(&self, py: Python<'_>) -> usize {
        self.numpy_owner.as_ref().map_or_else(
            || self.inner.frame_count(),
            |array| array.bind(py).len().unwrap_or(0) / usize::from(self.inner.channels),
        )
    }

    /// 音频时长，单位为毫秒。
    #[getter]
    fn duration_ms(&self, py: Python<'_>) -> f64 {
        self.frame_count(py) as f64 * 1000.0 / f64::from(self.inner.sample_rate)
    }

    /// 解码前检测到的可选原始格式。
    #[getter]
    fn source_format(&self) -> Option<PyAudioFormat> {
        self.inner
            .source_format
            .clone()
            .map(|inner| PyAudioFormat { inner })
    }

    /// 一维只读 NumPy float32 样本视图。
    #[getter]
    #[allow(unsafe_code)]
    fn samples<'py>(this: Bound<'py, Self>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let audio = this.borrow();
        if let Some(owner) = &audio.numpy_owner {
            let view = owner
                .bind(this.py())
                .call_method0("view")?
                .cast_into::<PyArray1<f32>>()?;
            view.call_method1("setflags", (false,))?;
            return Ok(view);
        }
        let view = ArrayView1::from(&audio.inner.samples);
        // SAFETY: the returned ndarray keeps `this` as its base owner. PyWaveform is
        // frozen from Python and none of its Rust methods mutate/reallocate samples.
        let array = unsafe { PyArray1::borrow_from_array(&view, this.into_any()) };
        array.call_method1("setflags", (false,))?;
        Ok(array)
    }

    /// 在 Jupyter 中显示音频播放器。
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
    ///     >>> import numpy as np
    ///     >>> from asr_data import Waveform
    ///     >>> audio = Waveform(np.zeros(16000), 16000)
    ///     >>> audio.display(start_ms=0, end_ms=500)
    #[pyo3(signature = (start_ms=None, end_ms=None, autoplay=false))]
    fn display(
        &self,
        py: Python<'_>,
        start_ms: Option<u64>,
        end_ms: Option<u64>,
        autoplay: bool,
    ) -> PyResult<()> {
        if let (Some(start), Some(end)) = (start_ms, end_ms)
            && end < start
        {
            return Err(PyValueError::new_err(
                "end_ms must be greater than or equal to start_ms",
            ));
        }
        let ipython = py.import("IPython.display").map_err(|_| {
            AsrDataError::new_err(
                "Waveform.display() requires IPython; install it with `pip install ipython`",
            )
        })?;
        let materialized = self.materialize(py)?;
        let audio = match (start_ms, end_ms) {
            (None, None) => materialized,
            (start, end) => materialized.slice_ms(
                start.unwrap_or(0),
                end.unwrap_or_else(|| materialized.duration_ms().ceil() as u64),
            ),
        };
        let samples = audio.samples.clone().into_pyarray(py);
        let data: Bound<'_, PyAny> = if audio.channels == 1 {
            samples.into_any()
        } else {
            // Rust stores interleaved frames; IPython expects [channels, frames].
            samples
                .call_method1(
                    "reshape",
                    (audio.frame_count(), usize::from(audio.channels)),
                )?
                .getattr("T")?
        };
        let kwargs = PyDict::new(py);
        kwargs.set_item("data", data)?;
        kwargs.set_item("rate", audio.sample_rate)?;
        kwargs.set_item("autoplay", autoplay)?;
        let player = ipython.getattr("Audio")?.call((), Some(&kwargs))?;
        ipython.getattr("display")?.call1((player,))?;
        Ok(())
    }

    /// 混合所有声道并返回新的单声道 Waveform。
    ///
    /// Returns:
    ///     不修改原对象的新 Waveform。
    ///
    /// Examples:
    ///     >>> import numpy as np
    ///     >>> from asr_data import Waveform
    ///     >>> Waveform(np.zeros(20), 16000, 2).to_mono().channels
    ///     1
    fn to_mono(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self::from_rust(
            self.materialize(py)?.to_mono().map_err(py_error)?,
        ))
    }

    /// 提取指定声道并返回新的单声道 Waveform。
    ///
    /// Args:
    ///     index: 从 0 开始的声道索引。
    ///
    /// Returns:
    ///     提取出的单声道 Waveform。
    ///
    /// Raises:
    ///     AsrDataError: 索引超出范围。
    ///
    /// Examples:
    ///     >>> import numpy as np
    ///     >>> from asr_data import Waveform
    ///     >>> Waveform(np.zeros(20), 16000, 2).channel(0).channels
    ///     1
    fn channel(&self, py: Python<'_>, index: u16) -> PyResult<Self> {
        Ok(Self::from_rust(
            self.materialize(py)?.channel(index).map_err(py_error)?,
        ))
    }

    /// 重采样并返回新的 Waveform。
    ///
    /// Args:
    ///     sample_rate: 目标采样率。
    ///
    /// Returns:
    ///     不修改原对象的新 Waveform。
    ///
    /// Raises:
    ///     ValueError: 目标采样率为零。
    ///
    /// Examples:
    ///     >>> import numpy as np
    ///     >>> from asr_data import Waveform
    ///     >>> Waveform(np.zeros(160), 16000).resample(8000).sample_rate
    ///     8000
    fn resample(&self, py: Python<'_>, sample_rate: u32) -> PyResult<Self> {
        let waveform = self.materialize(py)?;
        py.detach(move || waveform.resample(sample_rate))
            .map(Self::from_rust)
            .map_err(py_error)
    }

    /// 按半开毫秒范围截取并返回新的 Waveform。
    ///
    /// Args:
    ///     start_ms: 起始时间，包含。
    ///     end_ms: 结束时间，不包含。
    ///
    /// Returns:
    ///     截取后的新 Waveform。
    ///
    /// Examples:
    ///     >>> import numpy as np
    ///     >>> from asr_data import Waveform
    ///     >>> Waveform(np.zeros(16000), 16000).slice_ms(0, 500).duration_ms
    ///     500.0
    fn slice_ms(&self, py: Python<'_>, start_ms: u64, end_ms: u64) -> PyResult<Self> {
        Ok(Self::from_rust(
            self.materialize(py)?.slice_ms(start_ms, end_ms),
        ))
    }

    /// 在低能量位置拆分为不超过目标时长的片段。
    ///
    /// Args:
    ///     max_duration_ms: 每段的最大目标时长。
    ///
    /// Returns:
    ///     保持原顺序的 Waveform 列表。
    ///
    /// Raises:
    ///     ValueError: max_duration_ms 为零。
    ///
    /// Examples:
    ///     >>> import numpy as np
    ///     >>> from asr_data import Waveform
    ///     >>> len(Waveform(np.zeros(32000), 16000).split_at_low_energy(1000))
    ///     3
    fn split_at_low_energy(&self, py: Python<'_>, max_duration_ms: u64) -> PyResult<Vec<Self>> {
        self.materialize(py)?
            .split_at_low_energy(DurationMs(max_duration_ms))
            .map(|chunks| chunks.into_iter().map(Self::from_rust).collect())
            .map_err(py_error)
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.numpy_owner
            .as_ref()
            .map_or(self.inner.samples.len(), |array| {
                array.bind(py).len().unwrap_or(0)
            })
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let source_format = self.source_format().map_or_else(
            || "None".to_string(),
            |format| format!("{:?}", format.__str__()),
        );
        format!(
            "Waveform(frames={}, duration={}, sample_rate={}, channels={}, source_format={})",
            self.frame_count(py),
            format_duration_ms(self.duration_ms(py)),
            self.sample_rate(),
            self.channels(),
            source_format,
        )
    }

    fn __str__(&self, py: Python<'_>) -> String {
        format!(
            "Waveform({}, {}Hz, {}ch)",
            format_duration_ms(self.duration_ms(py)),
            self.sample_rate(),
            self.channels()
        )
    }
}

type AsyncLoadResult = Arc<Mutex<Option<Result<RustWaveform, String>>>>;
type AsyncProbeResult = Arc<Mutex<Option<Result<RustAudioInfo, String>>>>;

pub(super) fn async_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|error| panic!("failed to create audio async runtime: {error}"))
    })
}

#[pyclass(name = "_AudioLoadTask")]
pub(super) struct PyAudioLoadTask {
    result: AsyncLoadResult,
}

#[pymethods]
impl PyAudioLoadTask {
    fn done(&self) -> PyResult<bool> {
        Ok(self
            .result
            .lock()
            .map_err(|_| poisoned("audio load task"))?
            .is_some())
    }

    fn result(&self) -> PyResult<PyWaveform> {
        let result = self
            .result
            .lock()
            .map_err(|_| poisoned("audio load task"))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("audio load has not completed"))?;
        result
            .map(PyWaveform::from_rust)
            .map_err(AsrDataError::new_err)
    }
}

#[pyclass(name = "_AudioProbeTask")]
struct PyAudioProbeTask {
    result: AsyncProbeResult,
}

#[pymethods]
impl PyAudioProbeTask {
    fn done(&self) -> PyResult<bool> {
        Ok(self
            .result
            .lock()
            .map_err(|_| poisoned("audio probe task"))?
            .is_some())
    }

    fn result(&self) -> PyResult<PyAudioInfo> {
        let result = self
            .result
            .lock()
            .map_err(|_| poisoned("audio probe task"))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("audio probe has not completed"))?;
        result
            .map(|inner| PyAudioInfo { inner })
            .map_err(AsrDataError::new_err)
    }
}

fn spawn_source_aprobe(source: RustAudioSource) -> PyAudioProbeTask {
    let result: AsyncProbeResult = Arc::new(Mutex::new(None));
    let task_result = Arc::clone(&result);
    async_runtime().spawn(async move {
        let info = source.aprobe().await.map_err(|error| format!("{error:#}"));
        if let Ok(mut slot) = task_result.lock() {
            *slot = Some(info);
        }
    });
    PyAudioProbeTask { result }
}

fn spawn_source_aload(
    source: RustAudioSource,
    sample_rate: Option<u32>,
    mono: Option<bool>,
) -> PyResult<PyAudioLoadTask> {
    let result: AsyncLoadResult = Arc::new(Mutex::new(None));
    let task_result = Arc::clone(&result);
    async_runtime().spawn(async move {
        let loaded = tokio::task::spawn_blocking(move || {
            crate::audio::transform_loaded_audio(source.decode_waveform()?, sample_rate, mono)
        })
        .await
        .map_err(|error| format!("waveform loader worker failed: {error}"))
        .and_then(|result| result.map_err(|error| format!("{error:#}")));
        if let Ok(mut slot) = task_result.lock() {
            *slot = Some(loaded);
        }
    });
    Ok(PyAudioLoadTask { result })
}

struct AsyncStreamState {
    receiver: Receiver<Result<RustAudioChunk, String>>,
    pending: Option<Result<RustAudioChunk, String>>,
    finished: bool,
}

impl AsyncStreamState {
    fn poll(&mut self) {
        if self.pending.is_some() || self.finished {
            return;
        }
        match self.receiver.try_recv() {
            Ok(item) => self.pending = Some(item),
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => self.finished = true,
        }
    }
}

#[pyclass(name = "_AudioStreamTask")]
struct PyAudioStreamTask {
    state: Mutex<AsyncStreamState>,
}

#[pymethods]
impl PyAudioStreamTask {
    fn next_result(&self) -> PyResult<Option<PyAudioChunk>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| poisoned("audio stream task"))?;
        state.poll();
        state
            .pending
            .take()
            .transpose()
            .map(|chunk| chunk.map(|inner| PyAudioChunk { inner, audio: None }))
            .map_err(AsrDataError::new_err)
    }

    fn done(&self) -> PyResult<bool> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| poisoned("audio stream task"))?;
        state.poll();
        Ok(state.finished && state.pending.is_none())
    }
}

#[allow(dead_code)]
fn spawn_source_astream(
    source: RustAudioSource,
    chunk_size_ms: u64,
    sample_rate: Option<u32>,
    mono: Option<bool>,
) -> PyResult<PyAudioStreamTask> {
    if chunk_size_ms == 0 {
        return Err(py_error("chunk size must be greater than zero"));
    }
    if sample_rate == Some(0) {
        return Err(py_error("sample rate must be greater than zero"));
    }
    let (sender, receiver) = sync_channel(2);
    async_runtime().spawn_blocking(move || {
        let stream =
            crate::audio::stream::SourceAudioStream::new(source, chunk_size_ms, sample_rate, mono);
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                let _ = sender.send(Err(format!("{error:#}")));
                return;
            }
        };
        for chunk in &mut stream {
            if sender
                .send(chunk.map_err(|error| format!("{error:#}")))
                .is_err()
            {
                break;
            }
        }
    });
    Ok(PyAudioStreamTask {
        state: Mutex::new(AsyncStreamState {
            receiver,
            pending: None,
            finished: false,
        }),
    })
}

/// 同步产生 AudioChunk 的音频流迭代器。
#[pyclass(name = "AudioIterator")]
pub(super) struct PyAudioIterator {
    chunks: Mutex<AudioIteratorChunks>,
    audio: Option<SharedAudio>,
    accumulated: Vec<f32>,
    position_ms: u64,
    finished: bool,
}

enum AudioIteratorChunks {
    Source(Box<crate::audio::stream::SourceAudioStream>),
    Loaded(crate::audio::AudioChunks),
}

impl AudioIteratorChunks {
    fn next_chunk(&mut self) -> Option<anyhow::Result<RustAudioChunk>> {
        match self {
            Self::Source(chunks) => chunks.next(),
            Self::Loaded(chunks) => chunks.next().map(Ok),
        }
    }
}

/// 流式音频中的一个连续解码片段。
#[pyclass(name = "AudioChunk")]
#[derive(Clone)]
struct PyAudioChunk {
    inner: RustAudioChunk,
    audio: Option<SharedAudio>,
}

#[pymethods]
impl PyAudioChunk {
    /// 当前采样率。
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    /// 当前声道数。
    #[getter]
    fn channels(&self) -> u16 {
        self.inner.channels
    }

    /// 片段相对来源起点的偏移，单位为毫秒。
    #[getter]
    fn offset_ms(&self) -> u64 {
        self.inner.offset_ms
    }

    /// 是否为流中的最后一个片段。
    #[getter]
    fn is_final(&self) -> bool {
        self.inner.is_final
    }

    /// 每个声道的采样帧数。
    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }

    /// 片段时长，单位为毫秒。
    #[getter]
    fn duration_ms(&self) -> f64 {
        self.inner.duration_ms()
    }

    /// 片段结束位置，使用完整 Audio 的全局毫秒坐标。
    #[getter]
    fn end_ms(&self) -> u64 {
        self.inner
            .offset_ms
            .saturating_add(self.inner.duration_ms().ceil() as u64)
    }

    /// 将片段局部时间范围转换为完整 Audio 的全局时间范围。
    ///
    /// Args:
    ///     start_ms: chunk 内的局部起始时间。
    ///     end_ms: chunk 内的局部结束时间。
    ///
    /// Returns:
    ///     ``(start_ms, end_ms)`` 全局时间范围。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> audio = AudioSource.from_pcm(b"\0\0" * 1600, 16000).open()
    ///     >>> chunk = next(audio.stream(50))
    ///     >>> chunk.to_global_span(0, 10)
    ///     (0, 10)
    fn to_global_span(&self, start_ms: u64, end_ms: u64) -> PyResult<(u64, u64)> {
        if end_ms < start_ms || end_ms as f64 > self.inner.duration_ms().ceil() {
            return Err(PyValueError::new_err(
                "local span must be ordered and contained in the chunk",
            ));
        }
        Ok((
            self.inner.offset_ms.saturating_add(start_ms),
            self.inner.offset_ms.saturating_add(end_ms),
        ))
    }

    /// 返回该片段的 Waveform，可选提取指定声道。
    ///
    /// Args:
    ///     channel: 可选的 mono、left、right 或声道索引。
    ///
    /// Returns:
    ///     当前 chunk 的 Waveform。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> audio = AudioSource.from_pcm(b"\0\0" * 1600, 16000).open()
    ///     >>> chunk = next(audio.stream(50))
    ///     >>> chunk.as_waveform().duration_ms
    ///     50.0
    #[pyo3(signature = (channel=None))]
    fn as_waveform(&self, channel: Option<&Bound<'_, PyAny>>) -> PyResult<PyWaveform> {
        let waveform = RustWaveform::new_with_channels(
            self.inner.samples.clone(),
            self.inner.sample_rate,
            self.inner.channels,
        );
        let waveform = match channel {
            None => waveform,
            Some(value) => match audio_channel(value)? {
                crate::audio::AudioChannel::Mono => waveform.to_mono().map_err(py_error)?,
                crate::audio::AudioChannel::Left => waveform.channel(0).map_err(py_error)?,
                crate::audio::AudioChannel::Right => waveform.channel(1).map_err(py_error)?,
                crate::audio::AudioChannel::Channel(index) => {
                    waveform.channel(index).map_err(py_error)?
                }
            },
        };
        Ok(PyWaveform::from_rust(waveform))
    }

    /// 返回父 Audio 的完整 Timeline，而不是 chunk 局部时间轴。
    ///
    /// Args:
    ///     channel: mono、left、right 或声道索引。
    ///
    /// Returns:
    ///     父 Audio 的完整 Timeline。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> audio = AudioSource.from_pcm(b"\0\0" * 1600, 16000).open()
    ///     >>> chunk = next(audio.stream(50))
    ///     >>> chunk.timeline("mono").duration_ms
    ///     100
    fn timeline(&self, channel: &Bound<'_, PyAny>) -> PyResult<PyTimeline> {
        let audio = self
            .audio
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("this chunk is not bound to an Audio"))?;
        let channel = audio_channel(channel)?;
        let exists = audio
            .read()
            .map_err(|_| poisoned("audio"))?
            .timeline(channel)
            .map_err(py_error)?
            .is_some();
        if !exists {
            return Err(PyValueError::new_err("selected timeline does not exist"));
        }
        Ok(PyTimeline {
            audio: audio.clone(),
            channel,
        })
    }

    /// 解码前检测到的可选原始格式。
    #[getter]
    fn source_format(&self) -> Option<PyAudioFormat> {
        self.inner
            .source_format
            .clone()
            .map(|inner| PyAudioFormat { inner })
    }

    /// 一维只读 NumPy float32 样本视图。
    #[getter]
    #[allow(unsafe_code)]
    fn samples<'py>(this: Bound<'py, Self>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let chunk = this.borrow();
        let view = ArrayView1::from(&chunk.inner.samples);
        // SAFETY: the ndarray owns a reference to `this`, whose samples are never
        // mutated or reallocated after the view is created.
        let array = unsafe { PyArray1::borrow_from_array(&view, this.into_any()) };
        array.call_method1("setflags", (false,))?;
        Ok(array)
    }

    /// 混合所有声道并返回新的单声道片段。
    ///
    /// Returns:
    ///     不修改原片段的新 AudioChunk。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> chunk = next(AudioSource.from_pcm(b"\0\0" * 10, 16000).open().stream())
    ///     >>> chunk.to_mono().channels
    ///     1
    fn to_mono(&self) -> PyResult<Self> {
        self.inner
            .to_mono()
            .map(|inner| Self {
                inner,
                audio: self.audio.clone(),
            })
            .map_err(py_error)
    }

    /// 提取指定声道并返回新的单声道片段。
    ///
    /// Args:
    ///     index: 从 0 开始的声道索引。
    ///
    /// Returns:
    ///     提取出的 AudioChunk。
    ///
    /// Raises:
    ///     AsrDataError: 索引超出范围。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> source = AudioSource.from_pcm(b"\0\0\0\0" * 10, 16000, 2)
    ///     >>> next(source.open().stream()).channel(0).channels
    ///     1
    fn channel(&self, index: u16) -> PyResult<Self> {
        self.inner
            .channel(index)
            .map(|inner| Self {
                inner,
                audio: self.audio.clone(),
            })
            .map_err(py_error)
    }

    /// 重采样并返回新的片段。
    ///
    /// Args:
    ///     sample_rate: 目标采样率。
    ///
    /// Returns:
    ///     重采样后的 AudioChunk。
    ///
    /// Raises:
    ///     ValueError: 目标采样率为零。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> chunk = next(AudioSource.from_pcm(b"\0\0" * 100, 16000).open().stream())
    ///     >>> chunk.resample(8000).sample_rate
    ///     8000
    fn resample(&self, py: Python<'_>, sample_rate: u32) -> PyResult<Self> {
        let chunk = self.inner.clone();
        py.detach(move || chunk.resample(sample_rate))
            .map(|inner| Self {
                inner,
                audio: self.audio.clone(),
            })
            .map_err(py_error)
    }

    /// 按相对片段起点的半开毫秒范围截取片段。
    ///
    /// Args:
    ///     start_ms: 相对起始时间。
    ///     end_ms: 相对结束时间。
    ///
    /// Returns:
    ///     截取后的 AudioChunk。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> chunk = next(AudioSource.from_pcm(b"\0\0" * 16000, 16000).open().stream())
    ///     >>> chunk.slice_ms(0, 50).duration_ms
    ///     50.0
    fn slice_ms(&self, start_ms: u64, end_ms: u64) -> Self {
        Self {
            inner: self.inner.slice_ms(start_ms, end_ms),
            audio: self.audio.clone(),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.samples.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioChunk(frames={}, offset_ms={}, duration={}, sample_rate={}, channels={}, is_final={})",
            self.frame_count(),
            self.offset_ms(),
            format_duration_ms(self.duration_ms()),
            self.sample_rate(),
            self.channels(),
            self.is_final(),
        )
    }
}

#[pymethods]
impl PyAudioIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyAudioChunk>> {
        if self.finished {
            return Ok(None);
        }
        let next = self
            .chunks
            .get_mut()
            .map_err(|_| poisoned("audio iterator"))?
            .next_chunk()
            .transpose()
            .map_err(py_error)?;
        let Some(inner) = next else {
            self.finish(false)?;
            return Ok(None);
        };
        self.position_ms = inner
            .offset_ms
            .saturating_add(inner.duration_ms().ceil() as u64);
        if self.audio.is_some() {
            self.accumulated.extend_from_slice(&inner.samples);
        }
        if inner.is_final {
            self.finish(true)?;
        }
        Ok(Some(PyAudioChunk {
            inner,
            audio: self.audio.clone(),
        }))
    }

    #[getter]
    fn position_ms(&self) -> u64 {
        self.position_ms
    }

    #[getter]
    fn is_finished(&self) -> bool {
        self.finished
    }

    fn close(&mut self) -> PyResult<()> {
        self.finish(false)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.close()
    }
}

impl PyAudioIterator {
    fn finish(&mut self, complete: bool) -> PyResult<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        if let Some(audio) = &self.audio {
            let mut audio = audio.write().map_err(|_| poisoned("audio"))?;
            if complete && !audio.is_loaded() {
                let waveform = RustWaveform::try_new_with_channels(
                    std::mem::take(&mut self.accumulated),
                    audio.info.sample_rate,
                    audio.info.channels,
                )
                .map_err(py_error)?
                .with_source_format(audio.info.source_format.clone());
                audio.waveform = Some(waveform);
            }
            audio.end_stream();
        }
        Ok(())
    }
}

impl Drop for PyAudioIterator {
    fn drop(&mut self) {
        let _ = self.finish(false);
    }
}

#[allow(dead_code)]
fn stream_source(
    py: Python<'_>,
    source: RustAudioSource,
    chunk_size_ms: u64,
    sample_rate: Option<u32>,
    mono: Option<bool>,
) -> PyResult<PyAudioIterator> {
    let chunks = py
        .detach(move || {
            crate::audio::stream::SourceAudioStream::new(source, chunk_size_ms, sample_rate, mono)
        })
        .map_err(py_error)?;
    Ok(PyAudioIterator {
        chunks: Mutex::new(AudioIteratorChunks::Source(Box::new(chunks))),
        audio: None,
        accumulated: Vec::new(),
        position_ms: 0,
        finished: false,
    })
}

pub(super) fn stream_audio(
    py: Python<'_>,
    audio: SharedAudio,
    chunk_size_ms: u64,
) -> PyResult<PyAudioIterator> {
    let chunks_result = {
        let mut value = audio.write().map_err(|_| poisoned("audio"))?;
        value.begin_stream().map_err(py_error)?;
        if let Some(waveform) = value.waveform.clone() {
            waveform
                .into_chunks_ms(chunk_size_ms)
                .map(AudioIteratorChunks::Loaded)
                .map_err(py_error)
        } else {
            let source = value.source.clone();
            drop(value);
            py.detach(move || {
                crate::audio::stream::SourceAudioStream::new(source, chunk_size_ms, None, None)
            })
            .map(|stream| AudioIteratorChunks::Source(Box::new(stream)))
            .map_err(py_error)
        }
    };
    let chunks = match chunks_result {
        Ok(chunks) => chunks,
        Err(error) => {
            if let Ok(mut value) = audio.write() {
                value.end_stream();
            }
            return Err(error);
        }
    };
    Ok(PyAudioIterator {
        chunks: Mutex::new(chunks),
        audio: Some(audio),
        accumulated: Vec::new(),
        position_ms: 0,
        finished: false,
    })
}

#[allow(dead_code)]
fn load_source(
    py: Python<'_>,
    source: RustAudioSource,
    sample_rate: Option<u32>,
    mono: Option<bool>,
) -> PyResult<PyWaveform> {
    py.detach(move || {
        crate::audio::transform_loaded_audio(source.decode_waveform()?, sample_rate, mono)
    })
    .map(PyWaveform::from_rust)
    .map_err(py_error)
}

/// 尚未解码的音频来源描述。
///
/// AudioSource 保留路径、URL、编码字节、base64 或 PCM 参数；真正的 I/O
/// 和解码发生在 probe、load 或 stream。
///
/// Examples:
///     >>> from asr_data import AudioSource
///     >>> source = AudioSource.from_pcm(b"\0\0" * 16000, sample_rate=16000)
///     >>> source.kind
///     'pcm'
#[pyclass(name = "AudioSource", frozen)]
#[derive(Clone)]
struct PyAudioSource {
    inner: RustAudioSource,
}

#[pymethods]
impl PyAudioSource {
    /// 从本地文件路径创建来源。
    ///
    /// Args:
    ///     path: 相对或绝对文件路径。
    ///
    /// Returns:
    ///     尚未加载的 AudioSource。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> AudioSource.from_path("audio.wav").path
    ///     'audio.wav'
    #[staticmethod]
    fn from_path(path: String) -> Self {
        Self {
            inner: RustAudioSource::from_path(path),
        }
    }

    /// 从 HTTP 或 HTTPS URL 创建来源。
    ///
    /// Args:
    ///     url: 音频地址。
    ///
    /// Returns:
    ///     尚未发起请求的 AudioSource。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> AudioSource.from_url("https://example.com/a.wav").kind
    ///     'url'
    #[staticmethod]
    fn from_url(url: String) -> Self {
        Self {
            inner: RustAudioSource::from_url(url),
        }
    }

    /// 从 WAV、MP3 等编码音频字节创建来源。
    ///
    /// Args:
    ///     data: 带容器或编码信息的音频字节。
    ///
    /// Returns:
    ///     保存编码字节的 AudioSource。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> AudioSource.from_bytes(b"RIFF").kind
    ///     'bytes'
    #[staticmethod]
    fn from_bytes(data: &Bound<'_, PyBytes>) -> Self {
        Self {
            inner: RustAudioSource::from_encoded_bytes(data.as_bytes().to_vec()),
        }
    }

    /// 从 base64 字符串或 data URL 创建来源。
    ///
    /// Args:
    ///     data: base64 内容。
    ///
    /// Returns:
    ///     保存原字符串的 AudioSource。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> AudioSource.from_base64("UklGRg==").kind
    ///     'base64'
    #[staticmethod]
    fn from_base64(data: String) -> Self {
        Self {
            inner: RustAudioSource::from_base64(data),
        }
    }

    /// 从 PCM S16LE 原始字节创建来源。
    ///
    /// Args:
    ///     data: 按帧交错的有符号 16 位小端字节。
    ///     sample_rate: 采样率。
    ///     channels: 声道数，默认为 1。
    ///
    /// Returns:
    ///     保存 PCM 数据和格式参数的 AudioSource。
    ///
    /// Raises:
    ///     ValueError: 采样率、声道数或 PCM 帧长度无效。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> AudioSource.from_pcm(b"\0\0" * 10, 16000).channels
    ///     1
    #[staticmethod]
    #[pyo3(signature = (data, sample_rate, channels=1))]
    fn from_pcm(data: &Bound<'_, PyBytes>, sample_rate: u32, channels: u16) -> Self {
        Self {
            inner: RustAudioSource::from_pcm_s16le(data.as_bytes().to_vec(), sample_rate, channels),
        }
    }

    /// 来源类型：path、url、bytes、base64 或 pcm。
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            RustAudioSource::Path(_) => "path",
            RustAudioSource::Url(_) => "url",
            RustAudioSource::EncodedBytes(_) => "bytes",
            RustAudioSource::Base64(_) => "base64",
            RustAudioSource::PcmS16Le { .. } => "pcm",
        }
    }

    /// Path 来源的路径，否则为 None。
    #[getter]
    fn path(&self) -> Option<String> {
        match &self.inner {
            RustAudioSource::Path(path) => Some(path.display().to_string()),
            _ => None,
        }
    }

    /// URL 来源的地址，否则为 None。
    #[getter]
    fn url(&self) -> Option<String> {
        match &self.inner {
            RustAudioSource::Url(url) => Some(url.clone()),
            _ => None,
        }
    }

    /// 编码字节来源的内容，否则为 None。
    #[getter]
    fn bytes(&self, py: Python<'_>) -> Option<Py<PyBytes>> {
        match &self.inner {
            RustAudioSource::EncodedBytes(bytes) => Some(PyBytes::new(py, bytes).unbind()),
            _ => None,
        }
    }

    /// base64 来源的内容，否则为 None。
    #[getter]
    fn base64(&self) -> Option<String> {
        match &self.inner {
            RustAudioSource::Base64(data) => Some(data.clone()),
            _ => None,
        }
    }

    /// PCM 来源的原始字节，否则为 None。
    #[getter]
    fn pcm(&self, py: Python<'_>) -> Option<Py<PyBytes>> {
        match &self.inner {
            RustAudioSource::PcmS16Le { bytes, .. } => Some(PyBytes::new(py, bytes).unbind()),
            _ => None,
        }
    }

    /// PCM 来源的采样率，否则为 None。
    #[getter]
    fn sample_rate(&self) -> Option<u32> {
        match &self.inner {
            RustAudioSource::PcmS16Le { sample_rate, .. } => Some(*sample_rate),
            _ => None,
        }
    }

    /// PCM 来源的声道数，否则为 None。
    #[getter]
    fn channels(&self) -> Option<u16> {
        match &self.inner {
            RustAudioSource::PcmS16Le { channels, .. } => Some(*channels),
            _ => None,
        }
    }

    #[pyo3(signature = (*, id=None))]
    /// 探测来源并创建尚未加载波形的 Audio。
    ///
    /// Args:
    ///     id: 可选稳定 Audio ID。
    ///
    /// Returns:
    ///     尚未加载波形但已经包含 info 和 timelines 的 Audio。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> _ = AudioSource.from_pcm(b"\0\0", 16000).open(id="sample")
    fn open(&self, py: Python<'_>, id: Option<String>) -> PyResult<PyAudio> {
        let source = self.inner.clone();
        let audio = py
            .detach(move || match id {
                Some(id) => source.open_with_id(id),
                None => source.open(),
            })
            .map_err(py_error)?;
        PyAudio::from_rust(py, audio)
    }

    #[pyo3(signature = (*, id=None))]
    /// 同步解码来源并返回已加载的 Audio。
    ///
    /// Args:
    ///     id: 可选稳定 Audio ID。
    ///
    /// Returns:
    ///     已携带完整波形和 timeline 的 Audio。
    ///
    /// Raises:
    ///     AsrDataError: 来源无法读取、解码或转换。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> source = AudioSource.from_pcm(b"\0\0" * 16000, 16000)
    ///     >>> source.load().as_waveform().duration_ms
    ///     1000.0
    fn load(&self, py: Python<'_>, id: Option<String>) -> PyResult<PyAudio> {
        let source = self.inner.clone();
        let audio = py
            .detach(move || match id {
                Some(id) => source.load_with_id(id),
                None => source.load(),
            })
            .map_err(py_error)?;
        PyAudio::from_rust(py, audio)
    }

    /// 读取格式和时长信息，但不解码为浮点采样。
    ///
    /// Returns:
    ///     不包含采样数据的 AudioInfo。
    ///
    /// Raises:
    ///     AsrDataError: 来源无法读取或探测。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> AudioSource.from_pcm(b"\0\0" * 16000, 16000).probe().duration_ms
    ///     1000.0
    fn probe(&self, py: Python<'_>) -> PyResult<PyAudioInfo> {
        let source = self.inner.clone();
        py.detach(move || source.probe())
            .map(|inner| PyAudioInfo { inner })
            .map_err(py_error)
    }

    fn _start_aprobe(&self) -> PyAudioProbeTask {
        spawn_source_aprobe(self.inner.clone())
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            RustAudioSource::Path(path) => {
                format!(
                    "AudioSource(path={:?})",
                    truncate(&path.display().to_string(), 72)
                )
            }
            RustAudioSource::Url(url) => {
                format!("AudioSource(url={:?})", summarize_url(url, 72))
            }
            RustAudioSource::EncodedBytes(bytes) => {
                format!("AudioSource(bytes={})", bytes.len())
            }
            RustAudioSource::Base64(data) => {
                format!("AudioSource(base64_chars={})", data.len())
            }
            RustAudioSource::PcmS16Le {
                bytes,
                sample_rate,
                channels,
            } => format!(
                "AudioSource(pcm_bytes={}, sample_rate={}, channels={})",
                bytes.len(),
                sample_rate,
                channels
            ),
        }
    }
}

pub(super) fn rust_source_from_py(obj: &Bound<'_, PyAny>) -> PyResult<RustAudioSource> {
    obj.extract::<PyRef<'_, PyAudioSource>>()
        .map(|source| source.inner.clone())
        .map_err(|_| PyValueError::new_err("expected AudioSource"))
}

pub(super) fn py_source_from_rust(py: Python<'_>, source: &RustAudioSource) -> PyResult<Py<PyAny>> {
    Ok(Py::new(
        py,
        PyAudioSource {
            inner: source.clone(),
        },
    )?
    .into_any())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAudioFormat>()?;
    module.add_class::<PyAudioInfo>()?;
    module.add_class::<PyWaveform>()?;
    module.add_class::<PyAudioChunk>()?;
    module.add_class::<PyAudioSource>()?;
    module.add_class::<PyAudioLoadTask>()?;
    module.add_class::<PyAudioProbeTask>()?;
    module.add_class::<PyAudioStreamTask>()?;
    module.add_class::<PyAudioIterator>()?;
    Ok(())
}
