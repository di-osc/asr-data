use crate::audio::AudioChannel as RustAudioChannel;
use crate::doc::AudioDoc as RustAudioDoc;
use crate::timeline::{
    Annotation as RustAnnotation, AnnotationPayload, AnnotationStatus, SpeakerPayload,
    Timeline as RustTimeline, Token as RustToken, Transcript as RustTranscript,
    Transcription as RustTranscription,
};
use crate::utils::{DurationMs, TimeRange};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use super::common::{
    SharedAudio, annotation_status, format_duration_ms, poisoned, py_error, status_name, truncate,
};

#[pyclass(name = "Token", frozen)]
#[derive(Clone)]
struct PyToken {
    inner: RustToken,
}

#[pymethods]
impl PyToken {
    #[new]
    #[pyo3(signature = (text, *, start_ms=None, end_ms=None, confidence=None))]
    fn new(
        text: String,
        start_ms: Option<u64>,
        end_ms: Option<u64>,
        confidence: Option<f32>,
    ) -> PyResult<Self> {
        let range = match (start_ms, end_ms) {
            (None, None) => None,
            (Some(start), Some(end)) if end >= start => {
                Some(TimeRange::new(DurationMs(start), DurationMs(end)))
            }
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err("end_ms must be >= start_ms"));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "start_ms and end_ms must be provided together",
                ));
            }
        };
        Ok(Self {
            inner: RustToken {
                text,
                range,
                confidence,
            },
        })
    }

    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    #[getter]
    fn start_ms(&self) -> Option<u64> {
        self.inner.range.map(|range| range.start.0)
    }

    #[getter]
    fn end_ms(&self) -> Option<u64> {
        self.inner.range.map(|range| range.end.0)
    }

    #[getter]
    fn confidence(&self) -> Option<f32> {
        self.inner.confidence
    }

    fn __repr__(&self) -> String {
        let range = match self.inner.range {
            Some(range) => format!(", range={}..{}ms", range.start.0, range.end.0),
            None => String::new(),
        };
        let confidence = self
            .inner
            .confidence
            .map(|value| format!(", confidence={value:.3}"))
            .unwrap_or_default();
        format!(
            "Token(text={:?}{range}{confidence})",
            truncate(&self.inner.text, 40)
        )
    }
}

#[pyclass(name = "Transcription", frozen)]
#[derive(Clone)]
struct PyTranscription {
    inner: RustTranscription,
}

#[pymethods]
impl PyTranscription {
    #[new]
    #[pyo3(signature = (text, *, tokens=None, language=None, confidence=None))]
    fn new(
        text: String,
        tokens: Option<Vec<PyRef<'_, PyToken>>>,
        language: Option<String>,
        confidence: Option<f32>,
    ) -> Self {
        Self {
            inner: RustTranscription {
                text,
                tokens: tokens
                    .unwrap_or_default()
                    .into_iter()
                    .map(|token| token.inner.clone())
                    .collect(),
                language,
                confidence,
            },
        }
    }

    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    #[getter]
    fn tokens(&self) -> Vec<PyToken> {
        self.inner
            .tokens
            .iter()
            .cloned()
            .map(|inner| PyToken { inner })
            .collect()
    }

    #[getter]
    fn language(&self) -> Option<String> {
        self.inner.language.clone()
    }

    #[getter]
    fn confidence(&self) -> Option<f32> {
        self.inner.confidence
    }

    fn __repr__(&self) -> String {
        let mut fields = vec![format!("text={:?}", truncate(&self.inner.text, 60))];
        if let Some(language) = &self.inner.language {
            fields.push(format!("language={language:?}"));
        }
        if !self.inner.tokens.is_empty() {
            fields.push(format!("tokens={}", self.inner.tokens.len()));
        }
        if let Some(confidence) = self.inner.confidence {
            fields.push(format!("confidence={confidence:.3}"));
        }
        format!("Transcription({})", fields.join(", "))
    }
}

#[pyclass(name = "Speaker", frozen)]
#[derive(Clone)]
struct PySpeaker {
    inner: SpeakerPayload,
}

#[pymethods]
impl PySpeaker {
    #[new]
    #[pyo3(signature = (name, *, transcription=None))]
    fn new(name: String, transcription: Option<PyRef<'_, PyTranscription>>) -> Self {
        Self {
            inner: SpeakerPayload {
                name,
                transcription: transcription.map(|value| value.inner.clone()),
            },
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn transcription(&self) -> Option<PyTranscription> {
        self.inner
            .transcription
            .clone()
            .map(|inner| PyTranscription { inner })
    }

    fn __repr__(&self) -> String {
        let transcription = self
            .inner
            .transcription
            .as_ref()
            .map(|value| {
                format!(
                    ", transcription=Transcription(text={:?}, tokens={})",
                    truncate(&value.text, 40),
                    value.tokens.len()
                )
            })
            .unwrap_or_default();
        format!("Speaker(name={:?}{transcription})", self.inner.name)
    }
}

#[pyclass(name = "Annotation")]
#[derive(Clone)]
struct PyAnnotation {
    audio: SharedAudio,
    channel: RustAudioChannel,
    group: AnnotationGroup,
    annotation_id: String,
}

#[derive(Clone, Copy)]
enum AnnotationGroup {
    Reference,
    Prediction,
}

#[pymethods]
impl PyAnnotation {
    #[getter]
    fn id(&self) -> String {
        self.annotation_id.clone()
    }

    #[getter]
    fn start_ms(&self) -> PyResult<u64> {
        Ok(self.snapshot()?.range.start.0)
    }

    #[getter]
    fn end_ms(&self) -> PyResult<u64> {
        Ok(self.snapshot()?.range.end.0)
    }

    #[getter]
    fn status(&self) -> PyResult<&'static str> {
        Ok(status_name(&self.snapshot()?.status))
    }

    #[getter]
    fn confidence(&self) -> PyResult<Option<f32>> {
        Ok(self.snapshot()?.confidence)
    }

    #[getter]
    fn source(&self) -> PyResult<Option<String>> {
        Ok(self.snapshot()?.source)
    }

    #[getter]
    fn kind(&self) -> PyResult<&'static str> {
        Ok(annotation_kind(&self.snapshot()?.payload))
    }

    #[getter]
    fn text(&self) -> PyResult<Option<String>> {
        Ok(match &self.snapshot()?.payload {
            AnnotationPayload::Transcription(transcription) => Some(transcription.text.clone()),
            AnnotationPayload::Sentence(span) => Some(span.text.clone()),
            AnnotationPayload::Token(token) => Some(token.text.clone()),
            _ => None,
        })
    }

    #[getter]
    fn name(&self) -> PyResult<Option<String>> {
        Ok(match &self.snapshot()?.payload {
            AnnotationPayload::Speaker(speaker) => Some(speaker.name.clone()),
            _ => None,
        })
    }

    #[getter]
    fn transcription(&self) -> PyResult<Option<PyTranscription>> {
        Ok(match &self.snapshot()?.payload {
            AnnotationPayload::Transcription(transcription) => Some(PyTranscription {
                inner: transcription.clone(),
            }),
            AnnotationPayload::Speaker(speaker) => speaker
                .transcription
                .clone()
                .map(|inner| PyTranscription { inner }),
            _ => None,
        })
    }

    #[setter]
    fn set_transcription(&self, transcription: Option<PyRef<'_, PyTranscription>>) -> PyResult<()> {
        let transcription = transcription.map(|value| value.inner.clone());
        let mut audio = self.audio.write().map_err(|_| poisoned("audio"))?;
        let timeline = audio
            .timeline_mut(self.channel)
            .map_err(py_error)?
            .ok_or_else(|| PyRuntimeError::new_err("selected timeline does not exist"))?;
        let annotations = annotations_mut(timeline, self.group);
        let index = annotations
            .iter()
            .position(|annotation| annotation.id == self.annotation_id)
            .ok_or_else(|| PyRuntimeError::new_err("annotation no longer exists"))?;
        let annotation_range = annotations[index].range;
        match &mut annotations[index].payload {
            AnnotationPayload::Speaker(speaker) => {
                validate_speaker_transcription(annotation_range, &transcription)?;
                speaker.transcription = transcription;
            }
            AnnotationPayload::Transcription(current) => {
                let transcription = transcription.ok_or_else(|| {
                    PyValueError::new_err(
                        "a transcription annotation cannot have an empty transcription",
                    )
                })?;
                validate_transcription_range(
                    annotation_range,
                    &transcription,
                    "transcription annotation",
                )?;
                *current = transcription;
            }
            _ => {
                return Err(PyValueError::new_err(
                    "transcription can only be set on a speaker or transcription annotation",
                ));
            }
        }

        let updated = annotations[index].clone();
        annotations
            .retain(|annotation| annotation.id == updated.id || !annotation.content_eq(&updated));
        Ok(())
    }

    #[getter]
    fn language(&self) -> PyResult<Option<String>> {
        Ok(match &self.snapshot()?.payload {
            AnnotationPayload::Transcription(transcription) => transcription.language.clone(),
            AnnotationPayload::Sentence(span) => span.language.clone(),
            AnnotationPayload::Language(language) => Some(language.clone()),
            _ => None,
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        let annotation = self.snapshot()?;
        let text = match &annotation.payload {
            AnnotationPayload::Transcription(transcription) => {
                format!(", text={:?}", truncate(&transcription.text, 60))
            }
            AnnotationPayload::Sentence(span) => format!(", text={:?}", truncate(&span.text, 60)),
            AnnotationPayload::Token(token) => {
                format!(", text={:?}", truncate(&token.text, 60))
            }
            _ => String::new(),
        };
        let confidence = annotation
            .confidence
            .map(|value| format!(", confidence={value:.3}"))
            .unwrap_or_default();
        let speaker = match &annotation.payload {
            AnnotationPayload::Speaker(speaker) => {
                let transcription = speaker
                    .transcription
                    .as_ref()
                    .map(|value| {
                        format!(
                            ", transcription=Transcription(text={:?}, tokens={})",
                            truncate(&value.text, 40),
                            value.tokens.len()
                        )
                    })
                    .unwrap_or_default();
                format!(", name={:?}{transcription}", speaker.name)
            }
            _ => String::new(),
        };
        Ok(format!(
            "Annotation(id={:?}, kind={:?}, range={}..{}ms, status={:?}{speaker}{text}{confidence})",
            truncate(&annotation.id, 20),
            annotation_kind(&annotation.payload),
            annotation.range.start.0,
            annotation.range.end.0,
            status_name(&annotation.status),
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        let annotation = self.snapshot()?;
        let text = match &annotation.payload {
            AnnotationPayload::Transcription(transcription) => {
                format!(": {:?}", truncate(&transcription.text, 60))
            }
            AnnotationPayload::Sentence(span) => format!(": {:?}", truncate(&span.text, 60)),
            AnnotationPayload::Token(token) => {
                format!(": {:?}", truncate(&token.text, 60))
            }
            _ => String::new(),
        };
        Ok(format!(
            "{} [{}..{}ms]{text}",
            annotation_kind(&annotation.payload),
            annotation.range.start.0,
            annotation.range.end.0
        ))
    }
}

fn annotations(timeline: &RustTimeline, group: AnnotationGroup) -> &Vec<RustAnnotation> {
    match group {
        AnnotationGroup::Reference => &timeline.reference,
        AnnotationGroup::Prediction => &timeline.prediction,
    }
}

fn annotations_mut(
    timeline: &mut RustTimeline,
    group: AnnotationGroup,
) -> &mut Vec<RustAnnotation> {
    match group {
        AnnotationGroup::Reference => &mut timeline.reference,
        AnnotationGroup::Prediction => &mut timeline.prediction,
    }
}

impl PyAnnotation {
    fn snapshot(&self) -> PyResult<RustAnnotation> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        let timeline = audio
            .timeline(self.channel)
            .map_err(py_error)?
            .ok_or_else(|| PyRuntimeError::new_err("selected timeline does not exist"))?;
        annotations(timeline, self.group)
            .iter()
            .find(|annotation| annotation.id == self.annotation_id)
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err("annotation no longer exists"))
    }
}

fn annotation_kind(payload: &AnnotationPayload) -> &'static str {
    match payload {
        AnnotationPayload::Speech => "speech",
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

fn validate_speaker_transcription(
    speaker_range: TimeRange,
    transcription: &Option<RustTranscription>,
) -> PyResult<()> {
    if let Some(transcription) = transcription {
        validate_transcription_range(speaker_range, transcription, "speaker annotation")?;
    }
    Ok(())
}

fn validate_transcription_range(
    annotation_range: TimeRange,
    transcription: &RustTranscription,
    annotation_kind: &str,
) -> PyResult<()> {
    for token in &transcription.tokens {
        if let Some(range) = token.range
            && (range.start < annotation_range.start || range.end > annotation_range.end)
        {
            return Err(PyValueError::new_err(format!(
                "token range must be within the {annotation_kind} range"
            )));
        }
    }
    Ok(())
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
    fn reference(&self) -> PyReferenceAnnotations {
        PyReferenceAnnotations {
            core: AnnotationCollectionCore::new(self, AnnotationGroup::Reference),
        }
    }

    #[getter]
    fn prediction(&self) -> PyPredictionAnnotations {
        PyPredictionAnnotations {
            core: AnnotationCollectionCore::new(self, AnnotationGroup::Prediction),
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        let timeline = self.selected(&audio)?;
        let duration = format!("{:?}", format_duration_ms(timeline.duration.0 as f64));
        Ok(format!(
            "Timeline(id={:?}, audio_id={:?}, duration={}, reference={}, prediction={})",
            truncate(&timeline.id, 24),
            truncate(&timeline.audio_id, 40),
            duration,
            timeline.reference.len(),
            timeline.prediction.len()
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        let timeline = self.selected(&audio)?;
        let duration = format_duration_ms(timeline.duration.0 as f64);
        Ok(format!(
            "Timeline({}, {} reference, {} prediction)",
            duration,
            timeline.reference.len(),
            timeline.prediction.len()
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
}

#[derive(Clone)]
struct AnnotationCollectionCore {
    audio: SharedAudio,
    channel: RustAudioChannel,
    group: AnnotationGroup,
}

impl AnnotationCollectionCore {
    fn new(timeline: &PyTimeline, group: AnnotationGroup) -> Self {
        Self {
            audio: timeline.audio.clone(),
            channel: timeline.channel,
            group,
        }
    }

    fn annotation_handle(&self, annotation_id: String) -> PyAnnotation {
        PyAnnotation {
            audio: self.audio.clone(),
            channel: self.channel,
            group: self.group,
            annotation_id,
        }
    }

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

    fn all(&self) -> PyResult<Vec<PyAnnotation>> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(annotations(self.selected(&audio)?, self.group)
            .iter()
            .map(|annotation| self.annotation_handle(annotation.id.clone()))
            .collect())
    }

    fn len(&self) -> PyResult<usize> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(annotations(self.selected(&audio)?, self.group).len())
    }

    #[allow(clippy::too_many_arguments)]
    fn add_payload(
        &self,
        start_ms: u64,
        end_ms: u64,
        payload: AnnotationPayload,
        source: Option<&str>,
        confidence: Option<f32>,
        status: AnnotationStatus,
    ) -> PyResult<PyAnnotation> {
        if end_ms < start_ms {
            return Err(PyValueError::new_err("end_ms must be >= start_ms"));
        }
        let mut annotation = RustAnnotation::new(
            TimeRange::new(DurationMs(start_ms), DurationMs(end_ms)),
            payload,
            source.map(str::to_string),
            status,
        );
        annotation.confidence = confidence;
        let mut audio = self.audio.write().map_err(|_| poisoned("audio"))?;
        let timeline = self.selected_mut(&mut audio)?;
        if end_ms > timeline.duration.0 {
            return Err(PyValueError::new_err(format!(
                "annotation end_ms ({end_ms}) must not exceed timeline duration_ms ({})",
                timeline.duration.0
            )));
        }
        let annotation_id = match self.group {
            AnnotationGroup::Reference => timeline.push_reference_unique(annotation),
            AnnotationGroup::Prediction => timeline.push_prediction_unique(annotation),
        }
        .id
        .clone();
        Ok(self.annotation_handle(annotation_id))
    }
}

#[pyclass(name = "ReferenceAnnotations")]
#[derive(Clone)]
struct PyReferenceAnnotations {
    core: AnnotationCollectionCore,
}

#[pymethods]
impl PyReferenceAnnotations {
    #[getter]
    fn annotations(&self) -> PyResult<Vec<PyAnnotation>> {
        self.core.all()
    }

    #[pyo3(signature = (start_ms, end_ms, confidence=None))]
    fn add_speech(
        &self,
        start_ms: u64,
        end_ms: u64,
        confidence: Option<f32>,
    ) -> PyResult<PyAnnotation> {
        self.core.add_payload(
            start_ms,
            end_ms,
            AnnotationPayload::Speech,
            None,
            confidence,
            AnnotationStatus::Final,
        )
    }

    #[pyo3(signature = (start_ms, end_ms, transcription, confidence=None, status="final"))]
    fn add_transcription(
        &self,
        start_ms: u64,
        end_ms: u64,
        transcription: PyRef<'_, PyTranscription>,
        confidence: Option<f32>,
        status: &str,
    ) -> PyResult<PyAnnotation> {
        validate_transcription_range(
            TimeRange::new(DurationMs(start_ms), DurationMs(end_ms)),
            &transcription.inner,
            "transcription annotation",
        )?;
        self.core.add_payload(
            start_ms,
            end_ms,
            AnnotationPayload::Transcription(transcription.inner.clone()),
            None,
            confidence,
            annotation_status(status)?,
        )
    }

    #[pyo3(signature = (start_ms, end_ms, speaker, confidence=None, status="final"))]
    fn add_speaker(
        &self,
        start_ms: u64,
        end_ms: u64,
        speaker: PyRef<'_, PySpeaker>,
        confidence: Option<f32>,
        status: &str,
    ) -> PyResult<PyAnnotation> {
        add_speaker(
            &self.core,
            start_ms,
            end_ms,
            &speaker.inner,
            None,
            confidence,
            status,
        )
    }

    fn transcript(&self) -> PyResult<PyTranscript> {
        let audio = self.core.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(PyTranscript {
            inner: self.core.selected(&audio)?.reference_transcript(),
        })
    }

    fn __len__(&self) -> PyResult<usize> {
        self.core.len()
    }
}

#[pyclass(name = "PredictionAnnotations")]
#[derive(Clone)]
struct PyPredictionAnnotations {
    core: AnnotationCollectionCore,
}

#[pymethods]
impl PyPredictionAnnotations {
    #[getter]
    fn annotations(&self) -> PyResult<Vec<PyAnnotation>> {
        self.core.all()
    }

    #[getter]
    fn sources(&self) -> PyResult<Vec<String>> {
        let audio = self.core.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self
            .core
            .selected(&audio)?
            .prediction_sources()
            .into_iter()
            .map(str::to_string)
            .collect())
    }

    #[pyo3(signature = (start_ms, end_ms, *, source, confidence=None))]
    fn add_speech(
        &self,
        start_ms: u64,
        end_ms: u64,
        source: &str,
        confidence: Option<f32>,
    ) -> PyResult<PyAnnotation> {
        self.add_simple(
            start_ms,
            end_ms,
            AnnotationPayload::Speech,
            source,
            confidence,
        )
    }

    #[pyo3(signature = (start_ms, end_ms, transcription, *, source, confidence=None, status="final"))]
    #[allow(clippy::too_many_arguments)]
    fn add_transcription(
        &self,
        start_ms: u64,
        end_ms: u64,
        transcription: PyRef<'_, PyTranscription>,
        source: &str,
        confidence: Option<f32>,
        status: &str,
    ) -> PyResult<PyAnnotation> {
        validate_source(source)?;
        validate_transcription_range(
            TimeRange::new(DurationMs(start_ms), DurationMs(end_ms)),
            &transcription.inner,
            "transcription annotation",
        )?;
        self.core.add_payload(
            start_ms,
            end_ms,
            AnnotationPayload::Transcription(transcription.inner.clone()),
            Some(source),
            confidence,
            annotation_status(status)?,
        )
    }

    #[pyo3(signature = (start_ms, end_ms, speaker, *, source, confidence=None, status="final"))]
    #[allow(clippy::too_many_arguments)]
    fn add_speaker(
        &self,
        start_ms: u64,
        end_ms: u64,
        speaker: PyRef<'_, PySpeaker>,
        source: &str,
        confidence: Option<f32>,
        status: &str,
    ) -> PyResult<PyAnnotation> {
        validate_source(source)?;
        add_speaker(
            &self.core,
            start_ms,
            end_ms,
            &speaker.inner,
            Some(source),
            confidence,
            status,
        )
    }

    fn by_source(&self, source: &str) -> PyResult<Vec<PyAnnotation>> {
        let audio = self.core.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self
            .core
            .selected(&audio)?
            .predictions_by_source(source)
            .map(|annotation| self.core.annotation_handle(annotation.id.clone()))
            .collect())
    }

    fn transcript(&self, source: &str) -> PyResult<PyTranscript> {
        let audio = self.core.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(PyTranscript {
            inner: self.core.selected(&audio)?.prediction_transcript(source),
        })
    }

    fn remove_by_source(&self, source: &str) -> PyResult<usize> {
        let mut audio = self.core.audio.write().map_err(|_| poisoned("audio"))?;
        Ok(self
            .core
            .selected_mut(&mut audio)?
            .remove_predictions_by_source(source))
    }

    fn relabel_source(&self, from_source: &str, to_source: &str) -> PyResult<usize> {
        validate_source(to_source)?;
        let mut audio = self.core.audio.write().map_err(|_| poisoned("audio"))?;
        Ok(self
            .core
            .selected_mut(&mut audio)?
            .relabel_prediction_source(from_source, to_source))
    }

    fn __len__(&self) -> PyResult<usize> {
        self.core.len()
    }
}

impl PyPredictionAnnotations {
    fn add_simple(
        &self,
        start_ms: u64,
        end_ms: u64,
        payload: AnnotationPayload,
        source: &str,
        confidence: Option<f32>,
    ) -> PyResult<PyAnnotation> {
        validate_source(source)?;
        self.core.add_payload(
            start_ms,
            end_ms,
            payload,
            Some(source),
            confidence,
            AnnotationStatus::Final,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn add_speaker(
    core: &AnnotationCollectionCore,
    start_ms: u64,
    end_ms: u64,
    speaker: &SpeakerPayload,
    source: Option<&str>,
    confidence: Option<f32>,
    status: &str,
) -> PyResult<PyAnnotation> {
    validate_speaker_transcription(
        TimeRange::new(DurationMs(start_ms), DurationMs(end_ms)),
        &speaker.transcription,
    )?;
    core.add_payload(
        start_ms,
        end_ms,
        AnnotationPayload::Speaker(speaker.clone()),
        source,
        confidence,
        annotation_status(status)?,
    )
}

fn validate_source(source: &str) -> PyResult<()> {
    if source.trim().is_empty() {
        return Err(PyValueError::new_err(
            "prediction source must be a non-empty string",
        ));
    }
    Ok(())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyToken>()?;
    module.add_class::<PyTranscription>()?;
    module.add_class::<PySpeaker>()?;
    module.add_class::<PyAnnotation>()?;
    module.add_class::<PyTranscript>()?;
    module.add_class::<PyReferenceAnnotations>()?;
    module.add_class::<PyPredictionAnnotations>()?;
    module.add_class::<PyTimeline>()?;
    Ok(())
}
