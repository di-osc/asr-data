use crate::timeline::{
    AudioActivity as RustAudioActivity, SpeakerPayload, Token as RustToken,
    Transcription as RustTranscription,
};
use crate::utils::{DurationMs, TimeRange};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::common::truncate;

/// 音频中的一个活动事件 payload。
///
/// Args:
///     event: 可选事件名称；省略时只表示存在活动。
///
/// Raises:
///     ValueError: event 仅包含空白字符。
///
/// Examples:
///     >>> from asr_data.annotation import AudioActivity
///     >>> AudioActivity(event="speech").event
///     'speech'
///     >>> AudioActivity().event is None
///     True
#[pyclass(name = "AudioActivity", module = "asr_data.annotation", frozen)]
#[derive(Clone)]
pub(super) struct PyAudioActivity {
    pub(super) inner: RustAudioActivity,
}

#[pymethods]
impl PyAudioActivity {
    #[new]
    #[pyo3(signature = (*, event=None))]
    fn new(event: Option<String>) -> PyResult<Self> {
        if event
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(PyValueError::new_err(
                "event must contain at least one non-whitespace character",
            ));
        }
        Ok(Self {
            inner: RustAudioActivity { event },
        })
    }

    /// 可选事件名称。
    #[getter]
    fn event(&self) -> Option<String> {
        self.inner.event.clone()
    }

    fn __repr__(&self) -> String {
        match &self.inner.event {
            Some(event) => format!("AudioActivity(event={event:?})"),
            None => "AudioActivity()".to_owned(),
        }
    }
}

/// 转写中的细粒度文本单元。
///
/// Args:
///     text: Token 文本。
///     start_ms: 可选起始时间，单位为毫秒。
///     end_ms: 可选结束时间，单位为毫秒。
///     confidence: 可选置信度。
///
/// Raises:
///     ValueError: 时间参数没有成对提供，或者结束时间早于起始时间。
///
/// Examples:
///     >>> from asr_data.annotation import Token
///     >>> token = Token("你好", start_ms=0, end_ms=300, confidence=0.9)
///     >>> token.text
///     '你好'
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

    /// Token 文本。
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    /// 可选起始时间，单位为毫秒。
    #[getter]
    fn start_ms(&self) -> Option<u64> {
        self.inner.range.map(|range| range.start.0)
    }

    /// 可选结束时间，单位为毫秒。
    #[getter]
    fn end_ms(&self) -> Option<u64> {
        self.inner.range.map(|range| range.end.0)
    }

    /// 可选 token 级置信度。
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

/// 完整转写文本及其 token、语言和置信度。
///
/// Args:
///     text: 完整转写文本。
///     tokens: 可选 Token 列表。
///     language: 可选语言标签。
///     confidence: 可选转写级置信度。
///
/// Examples:
///     >>> from asr_data.annotation import Token, Transcription
///     >>> value = Transcription("你好", tokens=[Token("你好")], language="zh")
///     >>> value.language
///     'zh'
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

    /// 完整转写文本。
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    /// Token 列表的副本。
    #[getter]
    fn tokens(&self) -> Vec<PyToken> {
        self.inner
            .tokens
            .iter()
            .cloned()
            .map(|inner| PyToken { inner })
            .collect()
    }

    /// 可选语言标签。
    #[getter]
    fn language(&self) -> Option<String> {
        self.inner.language.clone()
    }

    /// 可选转写级置信度。
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

/// 一次说话人发话的 payload。
///
/// Args:
///     name: 说话人名称或稳定业务标识。
///     transcription: 该次发话携带的可选完整转写。
///
/// Examples:
///     >>> from asr_data.annotation import Speaker, Transcription
///     >>> speaker = Speaker("agent", transcription=Transcription("你好"))
///     >>> speaker.name
///     'agent'
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

    /// 说话人名称或稳定业务标识。
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// 该次发话携带的可选完整转写。
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
    module.add_class::<PyAudioActivity>()?;
    module.add_class::<PyToken>()?;
    module.add_class::<PyTranscription>()?;
    module.add_class::<PySpeaker>()?;
    Ok(())
}
