use std::sync::mpsc::{Receiver, TryRecvError, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};

use crate::audio::{
    Audio as RustAudio, AudioChunk as RustAudioChunk, AudioFormat as RustAudioFormat,
    AudioInfo as RustAudioInfo, AudioSource as RustAudioSource,
};
use crate::utils::DurationMs;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, ndarray::ArrayView1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};

use super::AsrDataError;
use super::common::{
    encoding_name, format_duration_ms, poisoned, py_error, summarize_url, truncate,
};

#[pyclass(name = "AudioFormat", frozen)]
#[derive(Clone)]
pub(super) struct PyAudioFormat {
    inner: RustAudioFormat,
}

#[pymethods]
impl PyAudioFormat {
    #[getter]
    fn encoding(&self) -> String {
        encoding_name(&self.inner.encoding)
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

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
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    #[getter]
    fn channels(&self) -> u16 {
        self.inner.channels
    }

    #[getter]
    fn frame_count(&self) -> u64 {
        self.inner.frame_count
    }

    #[getter]
    fn duration_ms(&self) -> f64 {
        self.inner.duration_ms()
    }

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

#[pyclass(name = "Audio")]
pub(super) struct PyAudio {
    inner: RustAudio,
    numpy_owner: Option<Py<PyArray1<f32>>>,
}

impl PyAudio {
    fn from_rust(inner: RustAudio) -> Self {
        Self {
            inner,
            numpy_owner: None,
        }
    }

    fn materialize(&self, py: Python<'_>) -> PyResult<RustAudio> {
        let Some(owner) = &self.numpy_owner else {
            return Ok(self.inner.clone());
        };
        let samples = owner.bind(py).readonly().as_slice()?.to_vec();
        let mut audio =
            RustAudio::try_new_with_channels(samples, self.inner.sample_rate, self.inner.channels)
                .map_err(py_error)?;
        audio.source_format = self.inner.source_format.clone();
        Ok(audio)
    }
}

#[pymethods]
impl PyAudio {
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
                inner: RustAudio::new_with_channels(Vec::new(), sample_rate, channels),
                numpy_owner: Some(samples),
            })
        }
    }

    #[staticmethod]
    fn from_path(py: Python<'_>, path: String) -> PyResult<Self> {
        py.detach(move || RustAudio::from_path(path))
            .map(Self::from_rust)
            .map_err(py_error)
    }

    #[staticmethod]
    fn from_url(py: Python<'_>, url: String) -> PyResult<Self> {
        py.detach(move || RustAudio::from_url(url))
            .map(Self::from_rust)
            .map_err(py_error)
    }

    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = data.as_bytes().to_vec();
        py.detach(move || RustAudio::from_encoded_bytes(bytes))
            .map(Self::from_rust)
            .map_err(py_error)
    }

    #[staticmethod]
    fn from_base64(py: Python<'_>, data: String) -> PyResult<Self> {
        py.detach(move || RustAudio::from_base64(data))
            .map(Self::from_rust)
            .map_err(py_error)
    }

    #[staticmethod]
    #[pyo3(signature = (data, sample_rate, channels=1))]
    fn from_pcm(
        py: Python<'_>,
        data: &Bound<'_, PyBytes>,
        sample_rate: u32,
        channels: u16,
    ) -> PyResult<Self> {
        let bytes = data.as_bytes().to_vec();
        py.detach(move || RustAudio::from_pcm_s16le(bytes, sample_rate, channels))
            .map(Self::from_rust)
            .map_err(py_error)
    }

    #[staticmethod]
    fn from_source(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Self> {
        let source = rust_source_from_py(source)?;
        py.detach(move || RustAudio::from_source(&source))
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

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    #[getter]
    fn channels(&self) -> u16 {
        self.inner.channels
    }

    #[getter]
    fn frame_count(&self, py: Python<'_>) -> usize {
        self.numpy_owner.as_ref().map_or_else(
            || self.inner.frame_count(),
            |array| array.bind(py).len().unwrap_or(0) / usize::from(self.inner.channels),
        )
    }

    #[getter]
    fn duration_ms(&self, py: Python<'_>) -> f64 {
        self.frame_count(py) as f64 * 1000.0 / f64::from(self.inner.sample_rate)
    }

    #[getter]
    fn source_format(&self) -> Option<PyAudioFormat> {
        self.inner
            .source_format
            .clone()
            .map(|inner| PyAudioFormat { inner })
    }

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
        // SAFETY: the returned ndarray keeps `this` as its base owner. PyAudio is
        // frozen from Python and none of its Rust methods mutate/reallocate samples.
        let array = unsafe { PyArray1::borrow_from_array(&view, this.into_any()) };
        array.call_method1("setflags", (false,))?;
        Ok(array)
    }

    /// Displays an IPython audio player for an optional millisecond range.
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
                "Audio.display() requires IPython; install it with `pip install ipython`",
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

    fn to_mono(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self::from_rust(
            self.materialize(py)?.to_mono().map_err(py_error)?,
        ))
    }

    fn channel(&self, py: Python<'_>, index: u16) -> PyResult<Self> {
        Ok(Self::from_rust(
            self.materialize(py)?.channel(index).map_err(py_error)?,
        ))
    }

    fn resample(&self, py: Python<'_>, sample_rate: u32) -> PyResult<Self> {
        let waveform = self.materialize(py)?;
        py.detach(move || waveform.resample(sample_rate))
            .map(Self::from_rust)
            .map_err(py_error)
    }

    fn slice_ms(&self, py: Python<'_>, start_ms: u64, end_ms: u64) -> PyResult<Self> {
        Ok(Self::from_rust(
            self.materialize(py)?.slice_ms(start_ms, end_ms),
        ))
    }

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
            "Audio(frames={}, duration={}, sample_rate={}, channels={}, source_format={})",
            self.frame_count(py),
            format_duration_ms(self.duration_ms(py)),
            self.sample_rate(),
            self.channels(),
            source_format,
        )
    }

    fn __str__(&self, py: Python<'_>) -> String {
        format!(
            "Audio({}, {}Hz, {}ch)",
            format_duration_ms(self.duration_ms(py)),
            self.sample_rate(),
            self.channels()
        )
    }
}

type AsyncLoadResult = Arc<Mutex<Option<Result<RustAudio, String>>>>;
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

    fn result(&self) -> PyResult<PyAudio> {
        let result = self
            .result
            .lock()
            .map_err(|_| poisoned("audio load task"))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("audio load has not completed"))?;
        result
            .map(PyAudio::from_rust)
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
        let loaded = source
            .aload_with(sample_rate, mono)
            .await
            .map_err(|error| format!("{error:#}"));
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
            .map(|chunk| chunk.map(|inner| PyAudioChunk { inner }))
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

#[pyclass(name = "AudioIterator", unsendable)]
struct PyAudioIterator {
    chunks: crate::audio::stream::SourceAudioStream,
}

#[pyclass(name = "AudioChunk")]
#[derive(Clone)]
struct PyAudioChunk {
    inner: RustAudioChunk,
}

#[pymethods]
impl PyAudioChunk {
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    #[getter]
    fn channels(&self) -> u16 {
        self.inner.channels
    }

    #[getter]
    fn offset_ms(&self) -> u64 {
        self.inner.offset_ms
    }

    #[getter]
    fn is_final(&self) -> bool {
        self.inner.is_final
    }

    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }

    #[getter]
    fn duration_ms(&self) -> f64 {
        self.inner.duration_ms()
    }

    #[getter]
    fn source_format(&self) -> Option<PyAudioFormat> {
        self.inner
            .source_format
            .clone()
            .map(|inner| PyAudioFormat { inner })
    }

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

    fn to_mono(&self) -> PyResult<Self> {
        self.inner
            .to_mono()
            .map(|inner| Self { inner })
            .map_err(py_error)
    }

    fn channel(&self, index: u16) -> PyResult<Self> {
        self.inner
            .channel(index)
            .map(|inner| Self { inner })
            .map_err(py_error)
    }

    fn resample(&self, py: Python<'_>, sample_rate: u32) -> PyResult<Self> {
        let chunk = self.inner.clone();
        py.detach(move || chunk.resample(sample_rate))
            .map(|inner| Self { inner })
            .map_err(py_error)
    }

    fn slice_ms(&self, start_ms: u64, end_ms: u64) -> Self {
        Self {
            inner: self.inner.slice_ms(start_ms, end_ms),
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
        self.chunks
            .next()
            .transpose()
            .map(|chunk| chunk.map(|inner| PyAudioChunk { inner }))
            .map_err(py_error)
    }
}

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
    Ok(PyAudioIterator { chunks })
}

fn load_source(
    py: Python<'_>,
    source: RustAudioSource,
    sample_rate: Option<u32>,
    mono: Option<bool>,
) -> PyResult<PyAudio> {
    py.detach(move || source.load_with(sample_rate, mono))
        .map(PyAudio::from_rust)
        .map_err(py_error)
}

#[pyclass(name = "AudioSource", frozen)]
#[derive(Clone)]
struct PyAudioSource {
    inner: RustAudioSource,
}

#[pymethods]
impl PyAudioSource {
    #[staticmethod]
    fn from_path(path: String) -> Self {
        Self {
            inner: RustAudioSource::from_path(path),
        }
    }

    #[staticmethod]
    fn from_url(url: String) -> Self {
        Self {
            inner: RustAudioSource::from_url(url),
        }
    }

    #[staticmethod]
    fn from_bytes(data: &Bound<'_, PyBytes>) -> Self {
        Self {
            inner: RustAudioSource::from_encoded_bytes(data.as_bytes().to_vec()),
        }
    }

    #[staticmethod]
    fn from_base64(data: String) -> Self {
        Self {
            inner: RustAudioSource::from_base64(data),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (data, sample_rate, channels=1))]
    fn from_pcm(data: &Bound<'_, PyBytes>, sample_rate: u32, channels: u16) -> Self {
        Self {
            inner: RustAudioSource::from_pcm_s16le(data.as_bytes().to_vec(), sample_rate, channels),
        }
    }

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

    #[getter]
    fn path(&self) -> Option<String> {
        match &self.inner {
            RustAudioSource::Path(path) => Some(path.display().to_string()),
            _ => None,
        }
    }

    #[getter]
    fn url(&self) -> Option<String> {
        match &self.inner {
            RustAudioSource::Url(url) => Some(url.clone()),
            _ => None,
        }
    }

    #[getter]
    fn bytes(&self, py: Python<'_>) -> Option<Py<PyBytes>> {
        match &self.inner {
            RustAudioSource::EncodedBytes(bytes) => Some(PyBytes::new(py, bytes).unbind()),
            _ => None,
        }
    }

    #[getter]
    fn base64(&self) -> Option<String> {
        match &self.inner {
            RustAudioSource::Base64(data) => Some(data.clone()),
            _ => None,
        }
    }

    #[getter]
    fn pcm(&self, py: Python<'_>) -> Option<Py<PyBytes>> {
        match &self.inner {
            RustAudioSource::PcmS16Le { bytes, .. } => Some(PyBytes::new(py, bytes).unbind()),
            _ => None,
        }
    }

    #[getter]
    fn sample_rate(&self) -> Option<u32> {
        match &self.inner {
            RustAudioSource::PcmS16Le { sample_rate, .. } => Some(*sample_rate),
            _ => None,
        }
    }

    #[getter]
    fn channels(&self) -> Option<u16> {
        match &self.inner {
            RustAudioSource::PcmS16Le { channels, .. } => Some(*channels),
            _ => None,
        }
    }

    #[pyo3(signature = (*, sample_rate=None, mono=None))]
    fn load(
        &self,
        py: Python<'_>,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudio> {
        load_source(py, self.inner.clone(), sample_rate, mono)
    }

    fn probe(&self, py: Python<'_>) -> PyResult<PyAudioInfo> {
        let source = self.inner.clone();
        py.detach(move || source.probe())
            .map(|inner| PyAudioInfo { inner })
            .map_err(py_error)
    }

    fn _start_aprobe(&self) -> PyAudioProbeTask {
        spawn_source_aprobe(self.inner.clone())
    }

    #[pyo3(signature = (chunk_size_ms=100, *, sample_rate=None, mono=None))]
    fn stream(
        &self,
        py: Python<'_>,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioIterator> {
        stream_source(py, self.inner.clone(), chunk_size_ms, sample_rate, mono)
    }

    #[pyo3(signature = (*, sample_rate=None, mono=None))]
    fn _start_aload(
        &self,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioLoadTask> {
        spawn_source_aload(self.inner.clone(), sample_rate, mono)
    }

    #[pyo3(signature = (chunk_size_ms=100, *, sample_rate=None, mono=None))]
    fn _start_astream(
        &self,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioStreamTask> {
        spawn_source_astream(self.inner.clone(), chunk_size_ms, sample_rate, mono)
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
    module.add_class::<PyAudio>()?;
    module.add_class::<PyAudioChunk>()?;
    module.add_class::<PyAudioSource>()?;
    module.add_class::<PyAudioLoadTask>()?;
    module.add_class::<PyAudioProbeTask>()?;
    module.add_class::<PyAudioStreamTask>()?;
    module.add_class::<PyAudioIterator>()?;
    Ok(())
}
