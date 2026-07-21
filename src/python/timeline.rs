use crate::audio::AudioChannel as RustAudioChannel;
use crate::doc::AudioDoc as RustAudioDoc;
use crate::timeline::{
    Annotation as RustAnnotation, AnnotationPayload, AnnotationStatus, TextSpan,
    Timeline as RustTimeline, Transcript as RustTranscript,
};
use crate::utils::{DurationMs, TimeRange};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use super::common::{
    SharedAudio, annotation_source, annotation_status, format_duration_ms, poisoned, py_error,
    source_kind, source_name, status_name, truncate,
};

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

#[pyclass(name = "Timeline")]
#[derive(Clone)]
pub(super) struct PyTimeline {
    pub(super) audio: SharedAudio,
    pub(super) channel: RustAudioChannel,
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

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAnnotation>()?;
    module.add_class::<PyTranscript>()?;
    module.add_class::<PyTimeline>()?;
    Ok(())
}
