use crate::timeline::{
    AudioEvent as RustAudioEvent, Gender, Sentence as RustSentence, Speaker, Speech as RustSpeech,
    Token as RustToken, Transcription as RustTranscription,
};
use crate::utils::TimeRange;
use pyo3::PyClassInitializer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use super::common::truncate;

/// 词级转写 token。时间是时间轴上的绝对毫秒。
///
/// Args:
///     text: Token 文本。
///     start_ms: 可选起始时间，单位为毫秒。
///     end_ms: 可选结束时间，单位为毫秒。
///     confidence: 可选置信度。
///
/// Raises:
///     ValueError: 时间参数没有成对提供，或者结束时间不晚于起始时间。
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
        Ok(Self {
            inner: RustToken {
                text,
                range: optional_range(start_ms, end_ms)?,
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
    fn start_ms(&self) -> Option<usize> {
        self.inner.range.map(|range| range.start_ms)
    }

    /// 可选结束时间，单位为毫秒。
    #[getter]
    fn end_ms(&self) -> Option<usize> {
        self.inner.range.map(|range| range.end_ms)
    }

    /// 可选 token 级置信度。
    #[getter]
    fn confidence(&self) -> Option<f32> {
        self.inner.confidence
    }

    fn __repr__(&self) -> String {
        let range = match self.inner.range {
            Some(range) => format!(", range={}..{}ms", range.start_ms, range.end_ms),
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

/// 一句转写。时间是时间轴上的绝对毫秒，并且必须落在所属语音内。
///
/// Args:
///     text: 整句文本。
///     start_ms: 起始时间，单位为毫秒。
///     end_ms: 结束时间，单位为毫秒。
///     tokens: 可选 Token 列表。
///
/// Raises:
///     ValueError: 结束时间不晚于起始时间。
///
/// Examples:
///     >>> from asr_data.annotation import Sentence
///     >>> Sentence("你好", 0, 300).text
///     '你好'
#[pyclass(name = "Sentence", module = "asr_data.annotation", frozen)]
#[derive(Clone)]
pub(super) struct PySentence {
    pub(super) inner: RustSentence,
}

#[pymethods]
impl PySentence {
    #[new]
    #[pyo3(signature = (text, start_ms, end_ms, *, tokens=None))]
    fn new(
        text: String,
        start_ms: u64,
        end_ms: u64,
        tokens: Option<Vec<PyRef<'_, PyToken>>>,
    ) -> PyResult<Self> {
        let Some(range) = optional_range(Some(start_ms), Some(end_ms))? else {
            return Err(PyValueError::new_err("end_ms must be > start_ms"));
        };
        Ok(Self {
            inner: RustSentence {
                text,
                range,
                tokens: tokens
                    .unwrap_or_default()
                    .into_iter()
                    .map(|token| token.inner.clone())
                    .collect(),
            },
        })
    }

    /// 整句文本。
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    /// 起始时间，单位为毫秒。
    #[getter]
    fn start_ms(&self) -> usize {
        self.inner.range.start_ms
    }

    /// 结束时间，单位为毫秒。
    #[getter]
    fn end_ms(&self) -> usize {
        self.inner.range.end_ms
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

    fn __repr__(&self) -> String {
        format!(
            "Sentence(text={:?}, range={}..{}ms, tokens={})",
            truncate(&self.inner.text, 40),
            self.inner.range.start_ms,
            self.inner.range.end_ms,
            self.inner.tokens.len()
        )
    }
}

/// 一段语音里的转写。至少要有全文，句子和 token 都可以空着。
///
/// Args:
///     text: 整段转写文本。
///     sentences: 可选句子列表。省略或空列表都合法。
///     confidence: 可选整段置信度。
///
/// Examples:
///     >>> from asr_data.annotation import Sentence, Transcription
///     >>> Transcription("你好").text
///     '你好'
///     >>> Transcription("你好", sentences=[Sentence("你好", 0, 300)]).sentences[0].tokens
///     []
#[pyclass(name = "Transcription", module = "asr_data.annotation", frozen)]
#[derive(Clone)]
pub(super) struct PyTranscription {
    pub(super) inner: RustTranscription,
}

#[pymethods]
impl PyTranscription {
    #[new]
    #[pyo3(signature = (text, *, sentences=None, confidence=None))]
    fn new(
        text: String,
        sentences: Option<Vec<PyRef<'_, PySentence>>>,
        confidence: Option<f32>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RustTranscription {
                text,
                sentences: sentences
                    .unwrap_or_default()
                    .into_iter()
                    .map(|sentence| sentence.inner.clone())
                    .collect(),
                confidence,
            },
        })
    }

    /// 整段转写文本。
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    /// 句子列表的副本。
    #[getter]
    fn sentences(&self) -> Vec<PySentence> {
        self.inner
            .sentences
            .iter()
            .cloned()
            .map(|inner| PySentence { inner })
            .collect()
    }

    /// 可选整段置信度。
    #[getter]
    fn confidence(&self) -> Option<f32> {
        self.inner.confidence
    }

    fn __repr__(&self) -> String {
        let mut fields = vec![
            format!("text={:?}", truncate(&self.inner.text, 60)),
            format!("sentences={}", self.inner.sentences.len()),
        ];
        if let Some(confidence) = self.inner.confidence {
            fields.push(format!("confidence={confidence:.3}"));
        }
        format!("Transcription({})", fields.join(", "))
    }
}

/// 一段语音上的说话人。一段语音最多一个说话人。
///
/// Args:
///     name: 说话人名称或业务标识。
///     gender: 可选性别，取 ``male``、``female`` 或 ``unknown``。省略表示还没标。
///
/// Raises:
///     ValueError: 名字是空白，或性别不是这三个值。
///
/// Examples:
///     >>> from asr_data.annotation import Speaker
///     >>> Speaker("agent", gender="female").gender
///     'female'
#[pyclass(name = "Speaker", module = "asr_data.annotation", frozen)]
#[derive(Clone)]
pub(super) struct PySpeaker {
    pub(super) inner: Speaker,
}

#[pymethods]
impl PySpeaker {
    #[new]
    #[pyo3(signature = (name, *, gender=None))]
    fn new(name: String, gender: Option<String>) -> PyResult<Self> {
        if name.trim().is_empty() {
            return Err(PyValueError::new_err(
                "speaker name must contain at least one non-whitespace character",
            ));
        }
        let gender = match gender.as_deref() {
            None => None,
            Some(value) => Some(Gender::parse(value).ok_or_else(|| {
                PyValueError::new_err("gender must be 'male', 'female', or 'unknown'")
            })?),
        };
        Ok(Self {
            inner: Speaker { name, gender },
        })
    }

    /// 说话人名称或业务标识。
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// 可选性别：``male``、``female`` 或 ``unknown``。
    #[getter]
    fn gender(&self) -> Option<&'static str> {
        self.inner.gender.map(Gender::as_str)
    }

    fn __repr__(&self) -> String {
        let gender = self
            .inner
            .gender
            .map(|value| format!(", gender={:?}", value.as_str()))
            .unwrap_or_default();
        format!("Speaker(name={:?}{gender})", self.inner.name)
    }
}

/// 音频事件基类。音乐、噪声、静音以及其他非语音事件都用它。
///
/// `Speech` 从本类派生。直接构造时事件名不能是 ``speech``。
///
/// Args:
///     name: 事件名，例如 ``music``、``noise``、``silence``。
///     confidence: 可选检测置信度。
///
/// Raises:
///     ValueError: 事件名为空白，或者是 ``speech``。
///
/// Examples:
///     >>> from asr_data.annotation import AudioEvent
///     >>> AudioEvent("music").name
///     'music'
#[pyclass(name = "AudioEvent", module = "asr_data.annotation", subclass)]
#[derive(Clone)]
pub(super) struct PyAudioEvent {
    pub(super) inner: RustAudioEvent,
}

#[pymethods]
impl PyAudioEvent {
    #[new]
    #[pyo3(signature = (name, *, confidence=None))]
    fn new(name: String, confidence: Option<f32>) -> PyResult<Self> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(PyValueError::new_err(
                "event name must contain at least one non-whitespace character",
            ));
        }
        if trimmed == "speech" {
            return Err(PyValueError::new_err(
                "speech events must be constructed with Speech",
            ));
        }
        let mut inner = RustAudioEvent::new(name);
        if let Some(confidence) = confidence {
            inner = inner.with_confidence(confidence);
        }
        Ok(Self { inner })
    }

    /// 事件 ID。尚未写入时间轴时为空字符串。
    #[getter]
    fn id(&self) -> String {
        self.inner.id().to_owned()
    }

    /// 事件名。语音事件固定为 ``speech``。
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_owned()
    }

    /// 起始时间，单位为毫秒。尚未写入时间轴时为 0。
    #[getter]
    fn start_ms(&self) -> usize {
        self.inner.range().start_ms
    }

    /// 结束时间，单位为毫秒。尚未写入时间轴时为 0。
    #[getter]
    fn end_ms(&self) -> usize {
        self.inner.range().end_ms
    }

    /// 预测来源。参考事件为 ``None``。
    #[getter]
    fn source(&self) -> Option<String> {
        self.inner.source().map(str::to_owned)
    }

    /// 可选检测置信度。
    #[getter]
    fn confidence(&self) -> Option<f32> {
        self.inner.confidence()
    }

    fn __repr__(&self) -> String {
        let confidence = self
            .inner
            .confidence()
            .map(|value| format!(", confidence={value:.3}"))
            .unwrap_or_default();
        format!("AudioEvent(name={:?}{confidence})", self.inner.name())
    }
}

/// 从 [`AudioEvent`] 派生的语音事件。语种、说话人和转写都可以缺。
///
/// Args:
///     language: 可选 BCP-47 语种标签。
///     speaker: 可选说话人。一段语音只有一个。
///     transcription: 可选转写。
///     confidence: 可选活动检测置信度。
///
/// Examples:
///     >>> from asr_data.annotation import Speech
///     >>> isinstance(Speech(), AudioEvent)
///     True
#[pyclass(name = "Speech", module = "asr_data.annotation", extends = PyAudioEvent)]
#[derive(Clone)]
pub(super) struct PySpeech;

#[pymethods]
impl PySpeech {
    #[new]
    #[pyo3(signature = (*, language=None, speaker=None, transcription=None, confidence=None))]
    fn new(
        language: Option<String>,
        speaker: Option<PyRef<'_, PySpeaker>>,
        transcription: Option<PyRef<'_, PyTranscription>>,
        confidence: Option<f32>,
    ) -> (Self, PyAudioEvent) {
        let mut speech = RustSpeech::new();
        if let Some(language) = language {
            speech = speech.with_language(language);
        }
        if let Some(speaker) = speaker {
            speech = speech.with_speaker(speaker.inner.clone());
        }
        if let Some(transcription) = transcription {
            speech = speech.with_transcription(transcription.inner.clone());
        }
        if let Some(confidence) = confidence {
            speech = speech.with_confidence(confidence);
        }
        (
            Self,
            PyAudioEvent {
                inner: speech.into(),
            },
        )
    }

    /// 可选语种标签。
    #[getter]
    fn language(self_: PyRef<'_, Self>) -> Option<String> {
        self_.as_super().inner.language().map(str::to_owned)
    }

    /// 可选说话人。
    #[getter]
    fn speaker(self_: PyRef<'_, Self>) -> Option<PySpeaker> {
        self_
            .as_super()
            .inner
            .speaker()
            .cloned()
            .map(|inner| PySpeaker { inner })
    }

    /// 可选转写。
    #[getter]
    fn transcription(self_: PyRef<'_, Self>) -> Option<PyTranscription> {
        self_
            .as_super()
            .inner
            .transcription()
            .cloned()
            .map(|inner| PyTranscription { inner })
    }

    fn __repr__(self_: PyRef<'_, Self>) -> String {
        let event = &self_.as_super().inner;
        let mut fields = Vec::new();
        if let Some(language) = event.language() {
            fields.push(format!("language={language:?}"));
        }
        if let Some(speaker) = event.speaker() {
            fields.push(format!("speaker={:?}", speaker.name));
        }
        if let Some(transcription) = event.transcription() {
            fields.push(format!(
                "transcription={:?}",
                truncate(&transcription.text, 40)
            ));
        }
        if let Some(confidence) = event.confidence() {
            fields.push(format!("confidence={confidence:.3}"));
        }
        format!("Speech({})", fields.join(", "))
    }
}

/// 从 Python 对象取出事件。`Speech` 必须先于基类匹配，因为它也是 `AudioEvent`。
pub(super) fn event_from_py(value: &Bound<'_, PyAny>) -> PyResult<RustAudioEvent> {
    if let Ok(speech) = value.extract::<PyRef<'_, PySpeech>>() {
        return Ok(speech.as_super().inner.clone());
    }
    if let Ok(event) = value.extract::<PyRef<'_, PyAudioEvent>>() {
        return Ok(event.inner.clone());
    }
    Err(PyTypeError::new_err("event must be AudioEvent or Speech"))
}

/// 把事件交回 Python。语音事件返回 `Speech` 实例。
pub(super) fn event_to_py(py: Python<'_>, event: RustAudioEvent) -> PyResult<Py<PyAny>> {
    if event.is_speech() {
        let initializer =
            PyClassInitializer::from(PyAudioEvent { inner: event }).add_subclass(PySpeech);
        return Ok(Py::new(py, initializer)?.into_any());
    }
    Ok(Py::new(py, PyAudioEvent { inner: event })?.into_any())
}

/// 成对的起止时间。两个都省略时返回 `None`。
fn optional_range(start_ms: Option<u64>, end_ms: Option<u64>) -> PyResult<Option<TimeRange>> {
    match (start_ms, end_ms) {
        (None, None) => Ok(None),
        (Some(start), Some(end)) if end > start => {
            Ok(Some(TimeRange::new(start as usize, end as usize)))
        }
        (Some(_), Some(_)) => Err(PyValueError::new_err("end_ms must be > start_ms")),
        _ => Err(PyValueError::new_err(
            "start_ms and end_ms must be provided together",
        )),
    }
}

/// 把本模块的 Python 类型注册进 `_native`。
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyToken>()?;
    module.add_class::<PySentence>()?;
    module.add_class::<PyTranscription>()?;
    module.add_class::<PySpeaker>()?;
    module.add_class::<PyAudioEvent>()?;
    module.add_class::<PySpeech>()?;
    Ok(())
}
