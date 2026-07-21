use std::collections::BTreeMap;
use std::sync::mpsc::{Receiver, TryRecvError, sync_channel};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::{
    Annotation as RustAnnotation, AnnotationPayload, AnnotationSource, AnnotationStatus,
    Audio as RustAudio, AudioChannel as RustAudioChannel, AudioChunk as RustAudioChunk,
    AudioDb as RustAudioDb, AudioDbError as RustAudioDbError, AudioDbMode,
    AudioDoc as RustAudioDoc, AudioEncoding, AudioFormat as RustAudioFormat,
    AudioLoadOptions as RustAudioLoadOptions, AudioQuery, AudioSource as RustAudioSource,
    DurationMs, TextSpan, TimeRange, Timeline as RustTimeline, Transcript as RustTranscript,
};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, ndarray::ArrayView1};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};

create_exception!(_native, AsrDataError, PyException);

fn py_error(error: impl std::fmt::Display) -> PyErr {
    AsrDataError::new_err(error.to_string())
}

fn py_db_error(error: RustAudioDbError) -> PyErr {
    match error {
        RustAudioDbError::NotFound { audio_id } => PyKeyError::new_err(audio_id),
        error => py_error(error),
    }
}

fn poisoned(label: &str) -> PyErr {
    PyRuntimeError::new_err(format!("{label} lock is poisoned"))
}

fn annotation_status(value: &str) -> PyResult<AnnotationStatus> {
    match value.to_ascii_lowercase().as_str() {
        "partial" => Ok(AnnotationStatus::Partial),
        "final" => Ok(AnnotationStatus::Final),
        "revised" => Ok(AnnotationStatus::Revised),
        "deleted" => Ok(AnnotationStatus::Deleted),
        _ => Err(PyValueError::new_err(format!(
            "unsupported annotation status {value:?}"
        ))),
    }
}

fn annotation_source(kind: &str, name: &str) -> PyResult<AnnotationSource> {
    match kind.to_ascii_lowercase().as_str() {
        "user" => Ok(AnnotationSource::User),
        "model" => Ok(AnnotationSource::Model(name.to_string())),
        "stage" => Ok(AnnotationSource::Stage(name.to_string())),
        "system" => Ok(AnnotationSource::System),
        _ => Err(PyValueError::new_err(format!(
            "unsupported annotation source kind {kind:?}; expected user, model, stage, or system"
        ))),
    }
}

fn audio_channel(value: &Bound<'_, PyAny>) -> PyResult<RustAudioChannel> {
    if let Ok(name) = value.extract::<String>() {
        return match name.to_ascii_lowercase().as_str() {
            "mono" => Ok(RustAudioChannel::Mono),
            "left" => Ok(RustAudioChannel::Left),
            "right" => Ok(RustAudioChannel::Right),
            _ => Err(PyValueError::new_err(format!(
                "unsupported audio channel {name:?}; expected mono, left, right, or an index"
            ))),
        };
    }
    let index = value.extract::<i64>().map_err(|_| {
        PyValueError::new_err("audio channel must be mono, left, right, or a non-negative index")
    })?;
    match index {
        ..0 => Err(PyValueError::new_err(
            "audio channel index must be non-negative",
        )),
        _ => u16::try_from(index)
            .map(RustAudioChannel::from_index)
            .map_err(|_| PyValueError::new_err("audio channel index exceeds u16")),
    }
}

fn audio_channel_name(channel: RustAudioChannel) -> String {
    channel.name()
}

fn source_kind(source: &AnnotationSource) -> &'static str {
    match source {
        AnnotationSource::User => "user",
        AnnotationSource::Model(_) => "model",
        AnnotationSource::Stage(_) => "stage",
        AnnotationSource::System => "system",
    }
}

fn source_name(source: &AnnotationSource) -> Option<&str> {
    match source {
        AnnotationSource::Model(name) | AnnotationSource::Stage(name) => Some(name),
        AnnotationSource::User | AnnotationSource::System => None,
    }
}

fn status_name(status: &AnnotationStatus) -> &'static str {
    match status {
        AnnotationStatus::Partial => "partial",
        AnnotationStatus::Final => "final",
        AnnotationStatus::Revised => "revised",
        AnnotationStatus::Deleted => "deleted",
    }
}

fn encoding_name(encoding: &AudioEncoding) -> String {
    match encoding {
        AudioEncoding::Wav => "wav".to_string(),
        AudioEncoding::Flac => "flac".to_string(),
        AudioEncoding::Mp3 => "mp3".to_string(),
        AudioEncoding::Ogg => "ogg".to_string(),
        AudioEncoding::PcmS16Le => "pcm_s16le".to_string(),
        AudioEncoding::Other(value) => value.clone(),
        AudioEncoding::Unknown => "unknown".to_string(),
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut output = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    output.push('…');
    output
}

fn format_duration_ms(duration_ms: f64) -> String {
    if duration_ms < 1_000.0 {
        return format!("{duration_ms:.0}ms");
    }
    let seconds = duration_ms / 1_000.0;
    if seconds < 60.0 {
        return format!("{seconds:.2}s");
    }
    let minutes = (seconds / 60.0).floor() as u64;
    let remaining_seconds = seconds - minutes as f64 * 60.0;
    if minutes < 60 {
        return format!("{minutes}m{remaining_seconds:04.1}s");
    }
    let hours = minutes / 60;
    let remaining_minutes = minutes % 60;
    format!("{hours}h{remaining_minutes:02}m{remaining_seconds:04.1}s")
}

fn format_source_field(source: &RustAudioSource) -> String {
    match source {
        RustAudioSource::Path(path) => {
            format!("file={:?}", truncate(&path.display().to_string(), 72))
        }
        RustAudioSource::Url(url) => format!("url={:?}", truncate(url, 72)),
        RustAudioSource::Base64(data) => format!("base64_chars={}", data.len()),
        RustAudioSource::EncodedBytes(bytes) => format!("bytes={}", bytes.len()),
        RustAudioSource::PcmS16Le {
            bytes,
            sample_rate,
            channels,
        } => format!(
            "pcm_bytes={}, sample_rate={}, channels={}",
            bytes.len(),
            sample_rate,
            channels
        ),
    }
}

#[pyclass(name = "AudioFormat", frozen)]
#[derive(Clone)]
struct PyAudioFormat {
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
struct PyAudio {
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
        spawn_source_aload(
            RustAudioSource::from_path(path),
            RustAudioLoadOptions::default(),
        )
    }

    #[staticmethod]
    fn _start_aload_from_source(source: &Bound<'_, PyAny>) -> PyResult<PyAudioLoadTask> {
        spawn_source_aload(
            rust_source_from_py(source)?,
            RustAudioLoadOptions::default(),
        )
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

#[pyclass(name = "Annotation", frozen)]
#[derive(Clone)]
struct PyAnnotation {
    inner: RustAnnotation,
}

#[pymethods]
impl PyAnnotation {
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    fn start_ms(&self) -> u64 {
        self.inner.range.start.0
    }

    #[getter]
    fn end_ms(&self) -> u64 {
        self.inner.range.end.0
    }

    #[getter]
    fn status(&self) -> &'static str {
        status_name(&self.inner.status)
    }

    #[getter]
    fn confidence(&self) -> Option<f32> {
        self.inner.confidence
    }

    #[getter]
    fn source_kind(&self) -> &'static str {
        source_kind(&self.inner.source)
    }

    #[getter]
    fn source(&self) -> Option<String> {
        source_name(&self.inner.source).map(str::to_string)
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner.payload {
            AnnotationPayload::Speech => "speech",
            AnnotationPayload::Silence => "silence",
            AnnotationPayload::Token(_) => "token",
            AnnotationPayload::Transcription(_) => "transcription",
            AnnotationPayload::Sentence(_) => "sentence",
            AnnotationPayload::Speaker(_) => "speaker",
            AnnotationPayload::Language(_) => "language",
            AnnotationPayload::Hotword(_) => "hotword",
            AnnotationPayload::AcousticEvent(_) => "acoustic_event",
            AnnotationPayload::Diagnostic(_) => "diagnostic",
        }
    }

    #[getter]
    fn text(&self) -> Option<String> {
        match &self.inner.payload {
            AnnotationPayload::Transcription(span) | AnnotationPayload::Sentence(span) => {
                Some(span.text.clone())
            }
            AnnotationPayload::Token(token) => Some(token.text.clone()),
            _ => None,
        }
    }

    #[getter]
    fn speaker(&self) -> Option<String> {
        match &self.inner.payload {
            AnnotationPayload::Speaker(speaker) => Some(speaker.clone()),
            _ => None,
        }
    }

    #[getter]
    fn language(&self) -> Option<String> {
        match &self.inner.payload {
            AnnotationPayload::Transcription(span) | AnnotationPayload::Sentence(span) => {
                span.language.clone()
            }
            AnnotationPayload::Language(language) => Some(language.clone()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        let text = self
            .text()
            .map(|text| format!(", text={:?}", truncate(&text, 60)))
            .unwrap_or_default();
        let confidence = self
            .confidence()
            .map(|value| format!(", confidence={value:.3}"))
            .unwrap_or_default();
        format!(
            "Annotation(id={:?}, kind={:?}, range={}..{}ms, status={:?}{text}{confidence})",
            truncate(&self.id(), 20),
            self.kind(),
            self.start_ms(),
            self.end_ms(),
            self.status(),
        )
    }

    fn __str__(&self) -> String {
        let text = self
            .text()
            .map(|text| format!(": {:?}", truncate(&text, 60)))
            .unwrap_or_default();
        format!(
            "{} [{}..{}ms]{text}",
            self.kind(),
            self.start_ms(),
            self.end_ms()
        )
    }
}

#[pyclass(name = "Transcript", frozen)]
#[derive(Clone)]
struct PyTranscript {
    inner: RustTranscript,
}

#[pymethods]
impl PyTranscript {
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    #[getter]
    fn language(&self) -> Option<String> {
        self.inner.language.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Transcript(text={:?}, language={:?})",
            truncate(&self.text(), 100),
            self.language()
        )
    }

    fn __str__(&self) -> String {
        self.text()
    }
}

type SharedAudio = Arc<RwLock<RustAudioDoc>>;
type AsyncLoadResult = Arc<Mutex<Option<Result<RustAudio, String>>>>;

fn async_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|error| panic!("failed to create audio async runtime: {error}"))
    })
}

#[pyclass(name = "Timeline")]
#[derive(Clone)]
struct PyTimeline {
    audio: SharedAudio,
    channel: RustAudioChannel,
}

#[pymethods]
impl PyTimeline {
    #[getter]
    fn id(&self) -> PyResult<String> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self.selected(&audio)?.id.clone())
    }

    #[getter]
    fn audio_id(&self) -> PyResult<String> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self.selected(&audio)?.audio_id.clone())
    }

    #[setter]
    fn set_audio_id(&self, value: String) -> PyResult<()> {
        let mut audio = self.audio.write().map_err(|_| poisoned("audio"))?;
        audio.set_audio_id(value);
        Ok(())
    }

    #[getter]
    fn duration_ms(&self) -> PyResult<u64> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self.selected(&audio)?.duration.0)
    }

    #[getter]
    fn annotations(&self) -> PyResult<Vec<PyAnnotation>> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self
            .selected(&audio)?
            .annotations
            .iter()
            .cloned()
            .map(|inner| PyAnnotation { inner })
            .collect())
    }

    #[pyo3(signature = (start_ms, end_ms, source="vad", confidence=None, source_kind="stage"))]
    fn add_speech(
        &self,
        start_ms: u64,
        end_ms: u64,
        source: &str,
        confidence: Option<f32>,
        source_kind: &str,
    ) -> PyResult<PyAnnotation> {
        self.add_payload(
            start_ms,
            end_ms,
            AnnotationPayload::Speech,
            source,
            source_kind,
            confidence,
            AnnotationStatus::Final,
        )
    }

    #[pyo3(signature = (start_ms, end_ms, source="vad", confidence=None, source_kind="stage"))]
    fn add_silence(
        &self,
        start_ms: u64,
        end_ms: u64,
        source: &str,
        confidence: Option<f32>,
        source_kind: &str,
    ) -> PyResult<PyAnnotation> {
        self.add_payload(
            start_ms,
            end_ms,
            AnnotationPayload::Silence,
            source,
            source_kind,
            confidence,
            AnnotationStatus::Final,
        )
    }

    #[pyo3(signature = (start_ms, end_ms, text, source="asr", language=None, confidence=None, status="final", source_kind="stage"))]
    #[allow(clippy::too_many_arguments)]
    fn add_transcription(
        &self,
        start_ms: u64,
        end_ms: u64,
        text: String,
        source: &str,
        language: Option<String>,
        confidence: Option<f32>,
        status: &str,
        source_kind: &str,
    ) -> PyResult<PyAnnotation> {
        self.add_payload(
            start_ms,
            end_ms,
            AnnotationPayload::Transcription(TextSpan {
                text,
                tokens: Vec::new(),
                language,
            }),
            source,
            source_kind,
            confidence,
            annotation_status(status)?,
        )
    }

    #[pyo3(signature = (start_ms, end_ms, speaker, source="diarization", confidence=None, status="final", source_kind="stage"))]
    #[allow(clippy::too_many_arguments)]
    fn add_speaker(
        &self,
        start_ms: u64,
        end_ms: u64,
        speaker: String,
        source: &str,
        confidence: Option<f32>,
        status: &str,
        source_kind: &str,
    ) -> PyResult<PyAnnotation> {
        self.add_payload(
            start_ms,
            end_ms,
            AnnotationPayload::Speaker(speaker),
            source,
            source_kind,
            confidence,
            annotation_status(status)?,
        )
    }

    #[pyo3(signature = (source, source_kind="model"))]
    fn annotations_by_source(
        &self,
        source: &str,
        source_kind: &str,
    ) -> PyResult<Vec<PyAnnotation>> {
        let expected = annotation_source(source_kind, source)?;
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self
            .selected(&audio)?
            .annotations
            .iter()
            .filter(|annotation| annotation.source == expected)
            .cloned()
            .map(|inner| PyAnnotation { inner })
            .collect())
    }

    #[pyo3(signature = (source, source_kind="model"))]
    fn transcript_by_source(&self, source: &str, source_kind: &str) -> PyResult<PyTranscript> {
        let expected = annotation_source(source_kind, source)?;
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(PyTranscript {
            inner: self.selected(&audio)?.transcript_by_source(&expected),
        })
    }

    #[pyo3(signature = (source, source_kind="model"))]
    fn remove_annotations_by_source(&self, source: &str, source_kind: &str) -> PyResult<usize> {
        let expected = annotation_source(source_kind, source)?;
        let mut audio = self.audio.write().map_err(|_| poisoned("audio"))?;
        let timeline = self.selected_mut(&mut audio)?;
        Ok(timeline.remove_annotations_by_source(&expected))
    }

    #[pyo3(signature = (from_source, to_source, from_source_kind="stage", to_source_kind="model"))]
    fn relabel_annotations_source(
        &self,
        from_source: &str,
        to_source: &str,
        from_source_kind: &str,
        to_source_kind: &str,
    ) -> PyResult<usize> {
        let from = annotation_source(from_source_kind, from_source)?;
        let to = annotation_source(to_source_kind, to_source)?;
        let mut audio = self.audio.write().map_err(|_| poisoned("audio"))?;
        Ok(self
            .selected_mut(&mut audio)?
            .relabel_annotations_source(&from, to))
    }

    fn transcript(&self) -> PyResult<PyTranscript> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(PyTranscript {
            inner: self.selected(&audio)?.transcript(),
        })
    }

    fn __len__(&self) -> PyResult<usize> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self.selected(&audio)?.annotations.len())
    }

    fn __repr__(&self) -> PyResult<String> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        let timeline = self.selected(&audio)?;
        let duration = format!("{:?}", format_duration_ms(timeline.duration.0 as f64));
        Ok(format!(
            "Timeline(id={:?}, audio_id={:?}, duration={}, annotations={})",
            truncate(&timeline.id, 24),
            truncate(&timeline.audio_id, 40),
            duration,
            timeline.annotations.len()
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        let timeline = self.selected(&audio)?;
        let duration = format_duration_ms(timeline.duration.0 as f64);
        Ok(format!(
            "Timeline({}, {} annotations)",
            duration,
            timeline.annotations.len()
        ))
    }
}

impl PyTimeline {
    fn selected<'a>(&self, audio: &'a RustAudioDoc) -> PyResult<&'a RustTimeline> {
        audio
            .timeline(self.channel)
            .map_err(py_error)?
            .ok_or_else(|| PyRuntimeError::new_err("selected timeline does not exist"))
    }

    fn selected_mut<'a>(&self, audio: &'a mut RustAudioDoc) -> PyResult<&'a mut RustTimeline> {
        audio
            .timeline_mut(self.channel)
            .map_err(py_error)?
            .ok_or_else(|| PyRuntimeError::new_err("selected timeline does not exist"))
    }

    #[allow(clippy::too_many_arguments)]
    fn add_payload(
        &self,
        start_ms: u64,
        end_ms: u64,
        payload: AnnotationPayload,
        source: &str,
        source_kind: &str,
        confidence: Option<f32>,
        status: AnnotationStatus,
    ) -> PyResult<PyAnnotation> {
        if end_ms < start_ms {
            return Err(PyValueError::new_err("end_ms must be >= start_ms"));
        }
        let mut annotation = RustAnnotation::new(
            TimeRange::new(DurationMs(start_ms), DurationMs(end_ms)),
            payload,
            annotation_source(source_kind, source)?,
            status,
        );
        annotation.confidence = confidence;
        let mut audio = self.audio.write().map_err(|_| poisoned("audio"))?;
        self.selected_mut(&mut audio)?.push(annotation.clone());
        Ok(PyAnnotation { inner: annotation })
    }
}

#[pyclass(name = "AudioDoc")]
struct PyAudioDoc {
    inner: SharedAudio,
    metadata: Py<PyDict>,
}

#[pyclass(name = "_AudioLoadTask")]
struct PyAudioLoadTask {
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
    options: RustAudioLoadOptions,
) -> PyResult<PyAudioLoadTask> {
    let result: AsyncLoadResult = Arc::new(Mutex::new(None));
    let task_result = Arc::clone(&result);
    async_runtime().spawn(async move {
        let loaded = source
            .aload_with_options(&options)
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
    options: RustAudioLoadOptions,
) -> PyResult<PyAudioStreamTask> {
    if chunk_size_ms == 0 {
        return Err(py_error("chunk size must be greater than zero"));
    }
    if options.sample_rate == Some(0) {
        return Err(py_error("sample rate must be greater than zero"));
    }
    let (sender, receiver) = sync_channel(2);
    async_runtime().spawn_blocking(move || {
        let stream = crate::audio::stream::SourceAudioStream::new(source, chunk_size_ms, options);
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
            crate::audio::stream::SourceAudioStream::new(
                source,
                chunk_size_ms,
                RustAudioLoadOptions { sample_rate, mono },
            )
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
    let options = RustAudioLoadOptions { sample_rate, mono };
    py.detach(move || crate::AudioLoader.load(&source, &options))
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
            RustAudioLoadOptions { sample_rate, mono },
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
            RustAudioLoadOptions { sample_rate, mono },
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
            RustAudioLoadOptions { sample_rate, mono },
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
            RustAudioLoadOptions { sample_rate, mono },
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
            RustAudioLoadOptions { sample_rate, mono },
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
            RustAudioLoadOptions { sample_rate, mono },
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
            RustAudioLoadOptions { sample_rate, mono },
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
            RustAudioLoadOptions { sample_rate, mono },
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
            RustAudioLoadOptions { sample_rate, mono },
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
            RustAudioLoadOptions { sample_rate, mono },
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

fn rust_source_from_py(obj: &Bound<'_, PyAny>) -> PyResult<RustAudioSource> {
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

fn py_source_from_rust(py: Python<'_>, source: &RustAudioSource) -> PyResult<Py<PyAny>> {
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

impl PyAudioDoc {
    fn from_rust(py: Python<'_>, audio: RustAudioDoc) -> PyResult<Self> {
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

    fn cloned_inner(&self, py: Python<'_>) -> PyResult<RustAudioDoc> {
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

#[pymodule]
fn _native(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = async_runtime();
    module.add("AsrDataError", py.get_type::<AsrDataError>())?;
    module.add_class::<PyAudioFormat>()?;
    module.add_class::<PyAudio>()?;
    module.add_class::<PyAudioChunk>()?;
    module.add_class::<PyAnnotation>()?;
    module.add_class::<PyTranscript>()?;
    module.add_class::<PyTimeline>()?;
    module.add_class::<PyAudioPath>()?;
    module.add_class::<PyAudioUrl>()?;
    module.add_class::<PyAudioBytes>()?;
    module.add_class::<PyAudioBase64>()?;
    module.add_class::<PyAudioPcm>()?;
    module.add_class::<PyAudioDoc>()?;
    module.add_class::<PyAudioLoadTask>()?;
    module.add_class::<PyAudioStreamTask>()?;
    module.add_class::<PyAudioIterator>()?;
    module.add_class::<PyAudioDb>()?;
    module.add_class::<PyAudioDbIterator>()?;
    Ok(())
}
