use crate::audio::AudioChannel as RustAudioChannel;
use crate::doc::AudioDoc as RustAudioDoc;
use crate::timeline::{
    Annotation as RustAnnotation, AnnotationPayload, SpeakerPayload,
    SpeechEvaluation as RustSpeechEvaluation, Timeline as RustTimeline, TimelineEvalConfig,
    TimelineEvaluation as RustTimelineEvaluation, Transcript as RustTranscript,
    Transcription as RustTranscription, TranscriptionEvaluation as RustTranscriptionEvaluation,
    TranscriptionNormalization,
};
use crate::utils::{DurationMs, TimeRange};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use super::annotation::{PySpeaker, PyToken, PyTranscription};
use super::common::{SharedAudio, format_duration_ms, poisoned, py_error, truncate};

/// Timeline 上一条带时间范围的标注记录。
///
/// payload 可整体替换；替换操作会重新校验类型、token 范围和重叠规则。
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
    /// 自动生成且稳定的 annotation ID。
    #[getter]
    fn id(&self) -> String {
        self.annotation_id.clone()
    }

    /// 起始时间，单位为毫秒，包含该位置。
    #[getter]
    fn start_ms(&self) -> PyResult<u64> {
        Ok(self.snapshot()?.range.start.0)
    }

    /// 结束时间，单位为毫秒，不包含该位置。
    #[getter]
    fn end_ms(&self) -> PyResult<u64> {
        Ok(self.snapshot()?.range.end.0)
    }

    /// 可选 annotation 级置信度。
    #[getter]
    fn confidence(&self) -> PyResult<Option<f32>> {
        Ok(self.snapshot()?.confidence)
    }

    /// Prediction 来源；reference 始终为 None。
    #[getter]
    fn source(&self) -> PyResult<Option<String>> {
        Ok(self.snapshot()?.source)
    }

    /// AnnotationKind 字符串。
    #[getter]
    fn kind(&self) -> PyResult<&'static str> {
        Ok(annotation_kind(&self.snapshot()?.payload))
    }

    /// 当前 payload；Speech 返回 None。
    #[getter]
    fn payload(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self.snapshot()?.payload {
            AnnotationPayload::Speech => py.None(),
            AnnotationPayload::Token(inner) => Py::new(py, PyToken { inner })?.into_any(),
            AnnotationPayload::Transcription(inner) => {
                Py::new(py, PyTranscription { inner })?.into_any()
            }
            AnnotationPayload::Speaker(inner) => Py::new(py, PySpeaker { inner })?.into_any(),
            payload @ (AnnotationPayload::Sentence(_)
            | AnnotationPayload::Language(_)
            | AnnotationPayload::AcousticEvent(_)) => pythonize::pythonize(py, &payload)
                .map_err(py_error)?
                .unbind(),
        })
    }

    /// 整体替换 payload，并原子执行完整校验。
    #[setter]
    fn set_payload(&self, payload: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut audio = self.audio.write().map_err(|_| poisoned("audio"))?;
        let timeline = audio
            .timeline_mut(self.channel)
            .map_err(py_error)?
            .ok_or_else(|| PyRuntimeError::new_err("selected timeline does not exist"))?;
        let mut candidate = timeline.clone();
        let annotations = annotations_mut(&mut candidate, self.group);
        let index = annotations
            .iter()
            .position(|annotation| annotation.id == self.annotation_id)
            .ok_or_else(|| PyRuntimeError::new_err("annotation no longer exists"))?;
        let annotation_range = annotations[index].range;
        match &mut annotations[index].payload {
            AnnotationPayload::Speech => {
                if !payload.is_none() {
                    return Err(PyValueError::new_err(
                        "a speech annotation payload must be None",
                    ));
                }
            }
            AnnotationPayload::Token(current) => {
                let token = payload.extract::<PyRef<'_, PyToken>>().map_err(|_| {
                    PyValueError::new_err("a token annotation payload must be Token")
                })?;
                if let Some(range) = token.inner.range
                    && (range.start < annotation_range.start || range.end > annotation_range.end)
                {
                    return Err(PyValueError::new_err(
                        "token range must be within the token annotation range",
                    ));
                }
                *current = token.inner.clone();
            }
            AnnotationPayload::Transcription(current) => {
                let transcription =
                    payload
                        .extract::<PyRef<'_, PyTranscription>>()
                        .map_err(|_| {
                            PyValueError::new_err(
                                "a transcription annotation payload must be Transcription",
                            )
                        })?;
                validate_transcription_range(
                    annotation_range,
                    &transcription.inner,
                    "transcription annotation",
                )?;
                *current = transcription.inner.clone();
            }
            AnnotationPayload::Speaker(current) => {
                let speaker = payload.extract::<PyRef<'_, PySpeaker>>().map_err(|_| {
                    PyValueError::new_err("a speaker annotation payload must be Speaker")
                })?;
                validate_speaker_transcription(annotation_range, &speaker.inner.transcription)?;
                *current = speaker.inner.clone();
            }
            AnnotationPayload::Sentence(_)
            | AnnotationPayload::Language(_)
            | AnnotationPayload::AcousticEvent(_) => {
                return Err(PyValueError::new_err(
                    "this annotation payload type cannot be replaced from Python",
                ));
            }
        }

        let updated = annotations[index].clone();
        annotations
            .retain(|annotation| annotation.id == updated.id || !annotation.content_eq(&updated));
        candidate.validate_annotations().map_err(py_error)?;
        *timeline = candidate;
        Ok(())
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
            "Annotation(id={:?}, kind={:?}, range={}..{}ms{speaker}{text}{confidence})",
            truncate(&annotation.id, 20),
            annotation_kind(&annotation.payload),
            annotation.range.start.0,
            annotation.range.end.0,
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
    payload.kind()
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

/// 按时间顺序组合得到的转写视图。
#[pyclass(name = "Transcript", frozen)]
#[derive(Clone)]
struct PyTranscript {
    inner: RustTranscript,
}

#[pymethods]
impl PyTranscript {
    /// 组合后的完整文本。
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    /// 首个可用语言标签。
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

/// 单个 prediction source 的 timeline 转写评测结果。
#[pyclass(name = "TranscriptionEvaluation", frozen)]
#[derive(Clone)]
struct PyTranscriptionEvaluation {
    inner: RustTranscriptionEvaluation,
}

#[pymethods]
impl PyTranscriptionEvaluation {
    /// Prediction source。
    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    /// 原始参考文本。
    #[getter]
    fn reference(&self) -> String {
        self.inner.reference.clone()
    }

    /// 原始预测文本。
    #[getter]
    fn hypothesis(&self) -> String {
        self.inner.hypothesis.clone()
    }

    /// 标准化后的参考文本。
    #[getter]
    fn normalized_reference(&self) -> String {
        self.inner.normalized_reference.clone()
    }

    /// 标准化后的预测文本。
    #[getter]
    fn normalized_hypothesis(&self) -> String {
        self.inner.normalized_hypothesis.clone()
    }

    /// 标准化模式：``"zh_tn"`` 或 ``"none"``。
    #[getter]
    fn normalization(&self) -> &'static str {
        match self.inner.normalization {
            TranscriptionNormalization::None => "none",
            TranscriptionNormalization::ChineseTn => "zh_tn",
        }
    }

    /// 匹配字符数。
    #[getter]
    fn matches(&self) -> usize {
        self.inner.matches()
    }

    /// 替换字符数。
    #[getter]
    fn substitutions(&self) -> usize {
        self.inner.stats.substitutions
    }

    /// 删除字符数。
    #[getter]
    fn deletions(&self) -> usize {
        self.inner.stats.deletions
    }

    /// 插入字符数。
    #[getter]
    fn insertions(&self) -> usize {
        self.inner.stats.insertions
    }

    /// 参考文本字符数。
    #[getter]
    fn reference_chars(&self) -> usize {
        self.inner.stats.reference_chars
    }

    /// 预测文本字符数。
    #[getter]
    fn hypothesis_chars(&self) -> usize {
        self.inner.hypothesis_chars
    }

    /// 字符错误率。
    #[getter]
    fn cer(&self) -> f64 {
        self.inner.stats.cer()
    }

    /// 字符级 precision。
    #[getter]
    fn precision(&self) -> f64 {
        self.inner.precision()
    }

    /// 字符级 recall。
    #[getter]
    fn recall(&self) -> f64 {
        self.inner.recall()
    }

    /// 字符级 F1。
    #[getter]
    fn f1(&self) -> f64 {
        self.inner.f1()
    }

    /// 标准化后的文本是否完全一致。
    #[getter]
    fn exact_match(&self) -> bool {
        self.inner.exact_match()
    }

    fn __repr__(&self) -> String {
        format!(
            "TranscriptionEvaluation(source={:?}, cer={:.4}, matches={}, substitutions={}, deletions={}, insertions={})",
            self.inner.source,
            self.inner.stats.cer(),
            self.inner.matches(),
            self.inner.stats.substitutions,
            self.inner.stats.deletions,
            self.inner.stats.insertions,
        )
    }
}

/// 单个 prediction source 的 timeline Speech 评测结果。
#[pyclass(name = "SpeechEvaluation", frozen)]
#[derive(Clone)]
struct PySpeechEvaluation {
    inner: RustSpeechEvaluation,
}

#[pymethods]
impl PySpeechEvaluation {
    /// Prediction source。
    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    /// Reference 人声总时长，单位为毫秒。
    #[getter]
    fn reference_ms(&self) -> u64 {
        self.inner.reference_ms
    }

    /// Prediction 人声总时长，单位为毫秒。
    #[getter]
    fn predicted_ms(&self) -> u64 {
        self.inner.predicted_ms
    }

    /// 正确预测为人声的时长。
    #[getter]
    fn true_positive_ms(&self) -> u64 {
        self.inner.true_positive_ms
    }

    /// 正确预测为静音的时长。
    #[getter]
    fn true_negative_ms(&self) -> u64 {
        self.inner.true_negative_ms
    }

    /// 误报人声的时长。
    #[getter]
    fn false_positive_ms(&self) -> u64 {
        self.inner.false_positive_ms
    }

    /// 漏报人声的时长。
    #[getter]
    fn false_negative_ms(&self) -> u64 {
        self.inner.false_negative_ms
    }

    /// Speech precision。
    #[getter]
    fn precision(&self) -> f64 {
        self.inner.precision()
    }

    /// Speech recall。
    #[getter]
    fn recall(&self) -> f64 {
        self.inner.recall()
    }

    /// Speech F1。
    #[getter]
    fn f1(&self) -> f64 {
        self.inner.f1()
    }

    /// Reference 与 prediction 的区间 IoU。
    #[getter]
    fn iou(&self) -> f64 {
        self.inner.iou()
    }

    fn __repr__(&self) -> String {
        format!(
            "SpeechEvaluation(source={:?}, precision={:.4}, recall={:.4}, f1={:.4}, iou={:.4})",
            self.inner.source,
            self.inner.precision(),
            self.inner.recall(),
            self.inner.f1(),
            self.inner.iou(),
        )
    }
}

/// Timeline 评测的组合结果。
#[pyclass(name = "TimelineEvaluation", frozen)]
#[derive(Clone)]
struct PyTimelineEvaluation {
    inner: RustTimelineEvaluation,
}

#[pymethods]
impl PyTimelineEvaluation {
    /// 按 prediction source 分组的转写结果。
    #[getter]
    fn transcription(&self) -> std::collections::BTreeMap<String, PyTranscriptionEvaluation> {
        self.inner
            .transcription
            .iter()
            .map(|(source, inner)| {
                (
                    source.clone(),
                    PyTranscriptionEvaluation {
                        inner: inner.clone(),
                    },
                )
            })
            .collect()
    }

    /// 按 prediction source 分组的 Speech 结果。
    #[getter]
    fn speech(&self) -> std::collections::BTreeMap<String, PySpeechEvaluation> {
        self.inner
            .speech
            .iter()
            .map(|(source, inner)| {
                (
                    source.clone(),
                    PySpeechEvaluation {
                        inner: inner.clone(),
                    },
                )
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TimelineEvaluation(transcription={}, speech={})",
            self.inner.transcription.len(),
            self.inner.speech.len(),
        )
    }
}

/// 一个声道上的参考真值和模型预测时间轴。
#[pyclass(name = "Timeline")]
#[derive(Clone)]
pub(super) struct PyTimeline {
    pub(super) audio: SharedAudio,
    pub(super) channel: RustAudioChannel,
}

#[pymethods]
impl PyTimeline {
    /// Timeline 唯一 ID。
    #[getter]
    fn id(&self) -> PyResult<String> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self.selected(&audio)?.id.clone())
    }

    /// 所属 AudioDoc ID。
    #[getter]
    fn audio_id(&self) -> PyResult<String> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self.selected(&audio)?.audio_id.clone())
    }

    /// 修改所属 AudioDoc ID。
    #[setter]
    fn set_audio_id(&self, value: String) -> PyResult<()> {
        let mut audio = self.audio.write().map_err(|_| poisoned("audio"))?;
        audio.set_audio_id(value);
        Ok(())
    }

    /// Timeline 总时长，单位为毫秒。
    #[getter]
    fn duration_ms(&self) -> PyResult<u64> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self.selected(&audio)?.duration.0)
    }

    /// 不带 source 的 reference 标注集合。
    #[getter]
    fn reference(&self) -> PyReferenceAnnotations {
        PyReferenceAnnotations {
            core: AnnotationCollectionCore::new(self, AnnotationGroup::Reference),
        }
    }

    /// 必须带 source 的 prediction 标注集合。
    #[getter]
    fn prediction(&self) -> PyPredictionAnnotations {
        PyPredictionAnnotations {
            core: AnnotationCollectionCore::new(self, AnnotationGroup::Prediction),
        }
    }

    /// 评测一个或多个 prediction source。
    ///
    /// 不传 source 时自动发现所有具有对应 reference 的来源。只传一个任务
    /// 参数时只评测该任务。
    ///
    /// Args:
    ///     transcription: 转写来源或来源名称列表。
    ///     speech: Speech 来源或来源名称列表。
    ///     normalize: 是否在计算 CER 前执行中文文本标准化。
    ///
    /// Returns:
    ///     按任务和 source 分组的 TimelineEvaluation。
    ///
    /// Raises:
    ///     AsrDataError: reference 缺失、显式 source 不存在或没有可评测内容。
    ///     TypeError: source 参数不是字符串或字符串序列。
    ///     ValueError: source 是空字符串。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> from asr_data.annotation import Transcription
    ///     >>> timeline = AudioDoc(
    ///     ...     AudioSource.from_pcm(b"\0\0" * 10, 16000)
    ///     ... ).timeline("mono")
    ///     >>> _ = timeline.reference.add_transcription(
    ///     ...     0, timeline.duration_ms, Transcription("你好")
    ///     ... )
    ///     >>> _ = timeline.prediction.add_transcription(
    ///     ...     0, timeline.duration_ms, Transcription("你好"), source="qwen-asr"
    ///     ... )
    ///     >>> result = timeline.eval()
    ///     >>> result.transcription["qwen-asr"].cer
    ///     0.0
    #[pyo3(signature = (*, transcription=None, speech=None, normalize=true))]
    fn eval(
        &self,
        transcription: Option<&Bound<'_, PyAny>>,
        speech: Option<&Bound<'_, PyAny>>,
        normalize: bool,
    ) -> PyResult<PyTimelineEvaluation> {
        let normalization = if normalize {
            TranscriptionNormalization::ChineseTn
        } else {
            TranscriptionNormalization::None
        };
        let config = TimelineEvalConfig {
            transcription_sources: extract_eval_sources(transcription, "transcription")?,
            speech_sources: extract_eval_sources(speech, "speech")?,
            transcription_normalization: normalization,
        };
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        let inner = self.selected(&audio)?.eval(&config).map_err(py_error)?;
        Ok(PyTimelineEvaluation { inner })
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

pub(super) fn extract_eval_sources(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<Vec<String>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if let Ok(source) = value.extract::<String>() {
        validate_source(&source)?;
        return Ok(Some(vec![source]));
    }
    let sources = value.extract::<Vec<String>>().map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a string or a sequence of strings"))
    })?;
    for source in &sources {
        validate_source(source)?;
    }
    Ok(Some(sources))
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

    fn add_payload(
        &self,
        start_ms: u64,
        end_ms: u64,
        payload: AnnotationPayload,
        source: Option<&str>,
        confidence: Option<f32>,
    ) -> PyResult<PyAnnotation> {
        if end_ms < start_ms {
            return Err(PyValueError::new_err("end_ms must be >= start_ms"));
        }
        let mut annotation = RustAnnotation::new(
            TimeRange::new(DurationMs(start_ms), DurationMs(end_ms)),
            payload,
            source.map(str::to_string),
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
            AnnotationGroup::Reference => timeline.push_reference(annotation),
            AnnotationGroup::Prediction => timeline.push_prediction(annotation),
        }
        .map_err(py_error)?
        .id
        .clone();
        Ok(self.annotation_handle(annotation_id))
    }
}

/// Timeline 的参考真值标注集合。
#[pyclass(name = "ReferenceAnnotations")]
#[derive(Clone)]
struct PyReferenceAnnotations {
    core: AnnotationCollectionCore,
}

#[pymethods]
impl PyReferenceAnnotations {
    /// 全部 reference annotation 句柄。
    #[getter]
    fn annotations(&self) -> PyResult<Vec<PyAnnotation>> {
        self.core.all()
    }

    #[pyo3(signature = (start_ms, end_ms, confidence=None))]
    /// 添加 Speech 区间。
    ///
    /// Args:
    ///     start_ms: 起始时间，包含。
    ///     end_ms: 结束时间，不包含。
    ///     confidence: 可选置信度。
    ///
    /// Returns:
    ///     新建或已有的 Annotation。
    ///
    /// Raises:
    ///     ValueError: 时间范围无效。
    ///     AsrDataError: 与已有 reference Speech 重叠。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> timeline = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
    ///     >>> annotation = timeline.reference.add_speech(0, timeline.duration_ms)
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
        )
    }

    #[pyo3(signature = (start_ms, end_ms, transcription, confidence=None))]
    /// 添加完整转写 reference。
    ///
    /// Args:
    ///     start_ms: 起始时间。
    ///     end_ms: 结束时间。
    ///     transcription: Transcription payload。
    ///     confidence: 可选 annotation 级置信度。
    ///
    /// Raises:
    ///     ValueError: 时间范围或 token 范围无效。
    ///     AsrDataError: 与已有标注冲突。
    ///
    /// Returns:
    ///     新建或已有的 Annotation。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> from asr_data.annotation import Transcription
    ///     >>> timeline = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
    ///     >>> item = timeline.reference.add_transcription(
    ///     ...     0, timeline.duration_ms, Transcription("你好")
    ///     ... )
    fn add_transcription(
        &self,
        start_ms: u64,
        end_ms: u64,
        transcription: PyRef<'_, PyTranscription>,
        confidence: Option<f32>,
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
        )
    }

    #[pyo3(signature = (start_ms, end_ms, speaker, confidence=None))]
    /// 添加一次说话人发话 reference。
    ///
    /// Args:
    ///     start_ms: 起始时间。
    ///     end_ms: 结束时间。
    ///     speaker: Speaker payload。
    ///     confidence: 可选 annotation 级置信度。
    ///
    /// Returns:
    ///     新建或已有的 Annotation。
    ///
    /// Raises:
    ///     ValueError: 时间或内嵌 token 范围无效。
    ///     AsrDataError: 与同名说话人已有发话冲突。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> from asr_data.annotation import Speaker
    ///     >>> timeline = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
    ///     >>> item = timeline.reference.add_speaker(
    ///     ...     0, timeline.duration_ms, Speaker("agent")
    ///     ... )
    fn add_speaker(
        &self,
        start_ms: u64,
        end_ms: u64,
        speaker: PyRef<'_, PySpeaker>,
        confidence: Option<f32>,
    ) -> PyResult<PyAnnotation> {
        add_speaker(
            &self.core,
            start_ms,
            end_ms,
            &speaker.inner,
            None,
            confidence,
        )
    }

    /// 按时间顺序组合全部 reference 文本。
    ///
    /// Returns:
    ///     组合后的 Transcript。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> from asr_data.annotation import Transcription
    ///     >>> timeline = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
    ///     >>> _ = timeline.reference.add_transcription(
    ///     ...     0, timeline.duration_ms, Transcription("你好")
    ///     ... )
    ///     >>> timeline.reference.transcript().text
    ///     '你好'
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

/// Timeline 的模型 prediction 标注集合。
#[pyclass(name = "PredictionAnnotations")]
#[derive(Clone)]
struct PyPredictionAnnotations {
    core: AnnotationCollectionCore,
}

#[pymethods]
impl PyPredictionAnnotations {
    /// 全部 prediction annotation 句柄。
    #[getter]
    fn annotations(&self) -> PyResult<Vec<PyAnnotation>> {
        self.core.all()
    }

    /// 按 AnnotationKind 分组、排序并去重的 source 字典。
    #[getter]
    fn sources(&self) -> PyResult<std::collections::BTreeMap<&'static str, Vec<String>>> {
        let audio = self.core.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self
            .core
            .selected(&audio)?
            .prediction_sources()
            .into_iter()
            .map(|(kind, sources)| {
                (
                    kind,
                    sources.into_iter().map(str::to_string).collect::<Vec<_>>(),
                )
            })
            .collect())
    }

    #[pyo3(signature = (start_ms, end_ms, *, source, confidence=None))]
    /// 添加指定 source 的 Speech prediction。
    ///
    /// Args:
    ///     start_ms: 起始时间。
    ///     end_ms: 结束时间。
    ///     source: 模型或流程名称。
    ///     confidence: 可选置信度。
    ///
    /// Returns:
    ///     新建或已有的 Annotation。
    ///
    /// Raises:
    ///     ValueError: source 或时间范围无效。
    ///     AsrDataError: 与同 source Speech 重叠。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> timeline = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
    ///     >>> item = timeline.prediction.add_speech(
    ///     ...     0, timeline.duration_ms, source="vad"
    ///     ... )
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

    #[pyo3(signature = (start_ms, end_ms, transcription, *, source, confidence=None))]
    #[allow(clippy::too_many_arguments)]
    /// 添加指定 source 的完整转写 prediction。
    ///
    /// Args:
    ///     start_ms: 起始时间。
    ///     end_ms: 结束时间。
    ///     transcription: Transcription payload。
    ///     source: 模型或流程名称。
    ///     confidence: 可选置信度。
    ///
    /// Returns:
    ///     新建或已有的 Annotation。
    ///
    /// Raises:
    ///     ValueError: source、时间或 token 范围无效。
    ///     AsrDataError: 与同 source 文本冲突。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> from asr_data.annotation import Transcription
    ///     >>> timeline = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
    ///     >>> item = timeline.prediction.add_transcription(
    ///     ...     0, timeline.duration_ms, Transcription("你好"), source="asr"
    ///     ... )
    fn add_transcription(
        &self,
        start_ms: u64,
        end_ms: u64,
        transcription: PyRef<'_, PyTranscription>,
        source: &str,
        confidence: Option<f32>,
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
        )
    }

    #[pyo3(signature = (start_ms, end_ms, speaker, *, source, confidence=None))]
    #[allow(clippy::too_many_arguments)]
    /// 添加指定 source 的说话人发话 prediction。
    ///
    /// Args:
    ///     start_ms: 起始时间。
    ///     end_ms: 结束时间。
    ///     speaker: Speaker payload。
    ///     source: 模型或流程名称。
    ///     confidence: 可选置信度。
    ///
    /// Returns:
    ///     新建或已有的 Annotation。
    ///
    /// Raises:
    ///     ValueError: source、时间或内嵌 token 范围无效。
    ///     AsrDataError: 与同 source、同名说话人发话冲突。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> from asr_data.annotation import Speaker
    ///     >>> timeline = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
    ///     >>> item = timeline.prediction.add_speaker(
    ///     ...     0, timeline.duration_ms, Speaker("agent"), source="diarization"
    ///     ... )
    fn add_speaker(
        &self,
        start_ms: u64,
        end_ms: u64,
        speaker: PyRef<'_, PySpeaker>,
        source: &str,
        confidence: Option<f32>,
    ) -> PyResult<PyAnnotation> {
        validate_source(source)?;
        add_speaker(
            &self.core,
            start_ms,
            end_ms,
            &speaker.inner,
            Some(source),
            confidence,
        )
    }

    /// 返回指定 source 的全部 prediction annotation。
    ///
    /// Args:
    ///     source: 要查询的来源。
    ///
    /// Returns:
    ///     保持存储顺序的 Annotation 列表。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> timeline = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
    ///     >>> items = timeline.prediction.by_source("asr")
    fn by_source(&self, source: &str) -> PyResult<Vec<PyAnnotation>> {
        let audio = self.core.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self
            .core
            .selected(&audio)?
            .predictions_by_source(source)
            .map(|annotation| self.core.annotation_handle(annotation.id.clone()))
            .collect())
    }

    /// 按时间顺序组合指定 source 的全部预测文本。
    ///
    /// Args:
    ///     source: 要组合的来源。
    ///
    /// Returns:
    ///     组合后的 Transcript。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> timeline = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
    ///     >>> text = timeline.prediction.transcript("asr").text
    fn transcript(&self, source: &str) -> PyResult<PyTranscript> {
        let audio = self.core.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(PyTranscript {
            inner: self.core.selected(&audio)?.prediction_transcript(source),
        })
    }

    /// 删除指定 source 的全部 prediction 并返回数量。
    ///
    /// Args:
    ///     source: 要删除的来源。
    ///
    /// Returns:
    ///     删除的 annotation 数量。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> timeline = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
    ///     >>> removed = timeline.prediction.remove_by_source("asr")
    fn remove_by_source(&self, source: &str) -> PyResult<usize> {
        let mut audio = self.core.audio.write().map_err(|_| poisoned("audio"))?;
        Ok(self
            .core
            .selected_mut(&mut audio)?
            .remove_predictions_by_source(source))
    }

    /// 原子重命名 prediction source 并返回修改数量。
    ///
    /// Args:
    ///     from_source: 原来源。
    ///     to_source: 新来源。
    ///
    /// Returns:
    ///     修改的 annotation 数量。
    ///
    /// Raises:
    ///     ValueError: 新来源为空。
    ///     AsrDataError: 重命名后会产生重叠冲突。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioDoc, AudioSource
    ///     >>> timeline = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
    ///     >>> changed = timeline.prediction.relabel_source("asr", "asr-v2")
    fn relabel_source(&self, from_source: &str, to_source: &str) -> PyResult<usize> {
        validate_source(to_source)?;
        let mut audio = self.core.audio.write().map_err(|_| poisoned("audio"))?;
        self.core
            .selected_mut(&mut audio)?
            .relabel_prediction_source(from_source, to_source)
            .map_err(py_error)
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
        self.core
            .add_payload(start_ms, end_ms, payload, Some(source), confidence)
    }
}

fn add_speaker(
    core: &AnnotationCollectionCore,
    start_ms: u64,
    end_ms: u64,
    speaker: &SpeakerPayload,
    source: Option<&str>,
    confidence: Option<f32>,
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
    module.add_class::<PyAnnotation>()?;
    module.add_class::<PyTranscript>()?;
    module.add_class::<PyTranscriptionEvaluation>()?;
    module.add_class::<PySpeechEvaluation>()?;
    module.add_class::<PyTimelineEvaluation>()?;
    module.add_class::<PyReferenceAnnotations>()?;
    module.add_class::<PyPredictionAnnotations>()?;
    module.add_class::<PyTimeline>()?;
    Ok(())
}
