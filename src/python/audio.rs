use std::sync::mpsc::{Receiver, TryRecvError, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};

use crate::audio::{
    Audio as RustAudio, AudioChunk as RustAudioChunk, AudioFormat as RustAudioFormat,
    AudioSource as RustAudioSource,
};
use crate::utils::DurationMs;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, ndarray::ArrayView1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};

use super::AsrDataError;
use super::common::{encoding_name, format_duration_ms, poisoned, py_error, truncate};

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

#[pyclass(name = "AudioPath", frozen)]
#[derive(Clone)]
struct PyAudioPath {
    path: String,
}

#[pymethods]
impl PyAudioPath {
    #[new]
    fn new(path: String) -> Self {
        Self { path }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        "path"
    }

    #[getter]
    fn path(&self) -> String {
        self.path.clone()
    }

    #[pyo3(signature = (*, sample_rate=None, mono=None))]
    fn load(
        &self,
        py: Python<'_>,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudio> {
        load_source(
            py,
            RustAudioSource::from_path(self.path.clone()),
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (chunk_size_ms=100, *, sample_rate=None, mono=None))]
    fn stream(
        &self,
        py: Python<'_>,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioIterator> {
        stream_source(
            py,
            RustAudioSource::from_path(self.path.clone()),
            chunk_size_ms,
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (*, sample_rate=None, mono=None))]
    fn _start_aload(
        &self,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioLoadTask> {
        spawn_source_aload(
            RustAudioSource::from_path(self.path.clone()),
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (chunk_size_ms=100, *, sample_rate=None, mono=None))]
    fn _start_astream(
        &self,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioStreamTask> {
        spawn_source_astream(
            RustAudioSource::from_path(self.path.clone()),
            chunk_size_ms,
            sample_rate,
            mono,
        )
    }

    fn __repr__(&self) -> String {
        format!("AudioPath({:?})", truncate(&self.path, 72))
    }
}

#[pyclass(name = "AudioUrl", frozen)]
#[derive(Clone)]
struct PyAudioUrl {
    url: String,
}

#[pymethods]
impl PyAudioUrl {
    #[new]
    fn new(url: String) -> Self {
        Self { url }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        "url"
    }

    #[getter]
    fn url(&self) -> String {
        self.url.clone()
    }

    #[pyo3(signature = (*, sample_rate=None, mono=None))]
    fn load(
        &self,
        py: Python<'_>,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudio> {
        load_source(
            py,
            RustAudioSource::from_url(self.url.clone()),
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (chunk_size_ms=100, *, sample_rate=None, mono=None))]
    fn stream(
        &self,
        py: Python<'_>,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioIterator> {
        stream_source(
            py,
            RustAudioSource::from_url(self.url.clone()),
            chunk_size_ms,
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (*, sample_rate=None, mono=None))]
    fn _start_aload(
        &self,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioLoadTask> {
        spawn_source_aload(
            RustAudioSource::from_url(self.url.clone()),
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (chunk_size_ms=100, *, sample_rate=None, mono=None))]
    fn _start_astream(
        &self,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioStreamTask> {
        spawn_source_astream(
            RustAudioSource::from_url(self.url.clone()),
            chunk_size_ms,
            sample_rate,
            mono,
        )
    }

    fn __repr__(&self) -> String {
        format!("AudioUrl({:?})", truncate(&self.url, 72))
    }
}

#[pyclass(name = "AudioBytes", frozen)]
#[derive(Clone)]
struct PyAudioBytes {
    bytes: Vec<u8>,
}

#[pymethods]
impl PyAudioBytes {
    #[new]
    fn new(data: &Bound<'_, PyBytes>) -> Self {
        Self {
            bytes: data.as_bytes().to_vec(),
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        "bytes"
    }

    #[getter]
    fn byte_size(&self) -> usize {
        self.bytes.len()
    }

    #[pyo3(signature = (*, sample_rate=None, mono=None))]
    fn load(
        &self,
        py: Python<'_>,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudio> {
        load_source(
            py,
            RustAudioSource::from_encoded_bytes(self.bytes.clone()),
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (chunk_size_ms=100, *, sample_rate=None, mono=None))]
    fn stream(
        &self,
        py: Python<'_>,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioIterator> {
        stream_source(
            py,
            RustAudioSource::from_encoded_bytes(self.bytes.clone()),
            chunk_size_ms,
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (*, sample_rate=None, mono=None))]
    fn _start_aload(
        &self,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioLoadTask> {
        spawn_source_aload(
            RustAudioSource::from_encoded_bytes(self.bytes.clone()),
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (chunk_size_ms=100, *, sample_rate=None, mono=None))]
    fn _start_astream(
        &self,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioStreamTask> {
        spawn_source_astream(
            RustAudioSource::from_encoded_bytes(self.bytes.clone()),
            chunk_size_ms,
            sample_rate,
            mono,
        )
    }

    fn __repr__(&self) -> String {
        format!("AudioBytes(byte_size={})", self.bytes.len())
    }
}

#[pyclass(name = "AudioBase64", frozen)]
#[derive(Clone)]
struct PyAudioBase64 {
    data: String,
}

#[pymethods]
impl PyAudioBase64 {
    #[new]
    fn new(data: String) -> Self {
        Self { data }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        "base64"
    }

    #[getter]
    fn data(&self) -> String {
        self.data.clone()
    }

    #[pyo3(signature = (*, sample_rate=None, mono=None))]
    fn load(
        &self,
        py: Python<'_>,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudio> {
        load_source(
            py,
            RustAudioSource::from_base64(self.data.clone()),
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (chunk_size_ms=100, *, sample_rate=None, mono=None))]
    fn stream(
        &self,
        py: Python<'_>,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioIterator> {
        stream_source(
            py,
            RustAudioSource::from_base64(self.data.clone()),
            chunk_size_ms,
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (*, sample_rate=None, mono=None))]
    fn _start_aload(
        &self,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioLoadTask> {
        spawn_source_aload(
            RustAudioSource::from_base64(self.data.clone()),
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (chunk_size_ms=100, *, sample_rate=None, mono=None))]
    fn _start_astream(
        &self,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioStreamTask> {
        spawn_source_astream(
            RustAudioSource::from_base64(self.data.clone()),
            chunk_size_ms,
            sample_rate,
            mono,
        )
    }

    fn __repr__(&self) -> String {
        format!("AudioBase64(chars={})", self.data.len())
    }
}

#[pyclass(name = "AudioPcm", frozen)]
#[derive(Clone)]
struct PyAudioPcm {
    bytes: Vec<u8>,
    sample_rate: u32,
    channels: u16,
}

#[pymethods]
impl PyAudioPcm {
    #[new]
    #[pyo3(signature = (data, sample_rate, channels=1))]
    fn new(data: &Bound<'_, PyBytes>, sample_rate: u32, channels: u16) -> Self {
        Self {
            bytes: data.as_bytes().to_vec(),
            sample_rate,
            channels,
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        "pcm"
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[getter]
    fn channels(&self) -> u16 {
        self.channels
    }

    #[getter]
    fn byte_size(&self) -> usize {
        self.bytes.len()
    }

    #[pyo3(signature = (*, sample_rate=None, mono=None))]
    fn load(
        &self,
        py: Python<'_>,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudio> {
        load_source(
            py,
            RustAudioSource::from_pcm_s16le(self.bytes.clone(), self.sample_rate, self.channels),
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (chunk_size_ms=100, *, sample_rate=None, mono=None))]
    fn stream(
        &self,
        py: Python<'_>,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioIterator> {
        stream_source(
            py,
            RustAudioSource::from_pcm_s16le(self.bytes.clone(), self.sample_rate, self.channels),
            chunk_size_ms,
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (*, sample_rate=None, mono=None))]
    fn _start_aload(
        &self,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioLoadTask> {
        spawn_source_aload(
            RustAudioSource::from_pcm_s16le(self.bytes.clone(), self.sample_rate, self.channels),
            sample_rate,
            mono,
        )
    }

    #[pyo3(signature = (chunk_size_ms=100, *, sample_rate=None, mono=None))]
    fn _start_astream(
        &self,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> PyResult<PyAudioStreamTask> {
        spawn_source_astream(
            RustAudioSource::from_pcm_s16le(self.bytes.clone(), self.sample_rate, self.channels),
            chunk_size_ms,
            sample_rate,
            mono,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioPcm(byte_size={}, sample_rate={}, channels={})",
            self.bytes.len(),
            self.sample_rate,
            self.channels
        )
    }
}

pub(super) fn rust_source_from_py(obj: &Bound<'_, PyAny>) -> PyResult<RustAudioSource> {
    if let Ok(v) = obj.extract::<PyRef<'_, PyAudioPath>>() {
        return Ok(RustAudioSource::from_path(v.path.clone()));
    }
    if let Ok(v) = obj.extract::<PyRef<'_, PyAudioUrl>>() {
        return Ok(RustAudioSource::from_url(v.url.clone()));
    }
    if let Ok(v) = obj.extract::<PyRef<'_, PyAudioBytes>>() {
        return Ok(RustAudioSource::from_encoded_bytes(v.bytes.clone()));
    }
    if let Ok(v) = obj.extract::<PyRef<'_, PyAudioBase64>>() {
        return Ok(RustAudioSource::from_base64(v.data.clone()));
    }
    if let Ok(v) = obj.extract::<PyRef<'_, PyAudioPcm>>() {
        return Ok(RustAudioSource::from_pcm_s16le(
            v.bytes.clone(),
            v.sample_rate,
            v.channels,
        ));
    }
    Err(PyValueError::new_err(
        "expected AudioPath, AudioUrl, AudioBytes, AudioBase64, or AudioPcm",
    ))
}

pub(super) fn py_source_from_rust(py: Python<'_>, source: &RustAudioSource) -> PyResult<Py<PyAny>> {
    match source {
        RustAudioSource::Path(path) => Ok(Py::new(
            py,
            PyAudioPath {
                path: path.display().to_string(),
            },
        )?
        .into_any()),
        RustAudioSource::Url(url) => Ok(Py::new(py, PyAudioUrl { url: url.clone() })?.into_any()),
        RustAudioSource::EncodedBytes(bytes) => Ok(Py::new(
            py,
            PyAudioBytes {
                bytes: bytes.clone(),
            },
        )?
        .into_any()),
        RustAudioSource::Base64(data) => {
            Ok(Py::new(py, PyAudioBase64 { data: data.clone() })?.into_any())
        }
        RustAudioSource::PcmS16Le {
            bytes,
            sample_rate,
            channels,
        } => Ok(Py::new(
            py,
            PyAudioPcm {
                bytes: bytes.clone(),
                sample_rate: *sample_rate,
                channels: *channels,
            },
        )?
        .into_any()),
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAudioFormat>()?;
    module.add_class::<PyAudio>()?;
    module.add_class::<PyAudioChunk>()?;
    module.add_class::<PyAudioPath>()?;
    module.add_class::<PyAudioUrl>()?;
    module.add_class::<PyAudioBytes>()?;
    module.add_class::<PyAudioBase64>()?;
    module.add_class::<PyAudioPcm>()?;
    module.add_class::<PyAudioLoadTask>()?;
    module.add_class::<PyAudioStreamTask>()?;
    module.add_class::<PyAudioIterator>()?;
    Ok(())
}
