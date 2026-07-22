use crate::timeline::{SpeakerPayload, Token as RustToken, Transcription as RustTranscription};
use crate::utils::{DurationMs, TimeRange};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::common::truncate;

#[pyclass(name = "Token", module = "asr_data.annotation", frozen)]
#[derive(Clone)]
pub(super) struct PyToken {
    pub(super) inner: RustToken,
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

#[pyclass(name = "Transcription", module = "asr_data.annotation", frozen)]
#[derive(Clone)]
pub(super) struct PyTranscription {
    pub(super) inner: RustTranscription,
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

#[pyclass(name = "Speaker", module = "asr_data.annotation", frozen)]
#[derive(Clone)]
pub(super) struct PySpeaker {
    pub(super) inner: SpeakerPayload,
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

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyToken>()?;
    module.add_class::<PyTranscription>()?;
    module.add_class::<PySpeaker>()?;
    Ok(())
}
