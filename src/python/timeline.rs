use crate::audio::AudioChannel as RustAudioChannel;
use crate::doc::Audio as RustAudio;
use crate::timeline::{
    ActivityEvaluation as RustActivityEvaluation,
    ActivityEventEvaluation as RustActivityEventEvaluation, Annotation, TimeSpan as RustTimeSpan,
    Timeline as RustTimeline, TimelineEvalConfig, TimelineEvaluation as RustTimelineEvaluation,
    Transcript as RustTranscript, Transcription as RustTranscription,
    TranscriptionEvaluation as RustTranscriptionEvaluation, TranscriptionNormalization,
};
use crate::utils::{DurationMs, TimeRange};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use super::annotation::{PyAudioActivity, PySpeaker, PyToken, PyTranscription};
use super::audio::PyWaveform;
use super::common::{SharedAudio, format_duration_ms, poisoned, py_error, truncate};

/// Timeline 上一条带时间范围的标注记录。
///
/// annotation 可整体替换；替换操作会重新校验类型、token 范围和重叠规则。
#[pyclass(name = "TimeSpan")]
#[derive(Clone)]
struct PyTimeSpan {
    audio: SharedAudio,
    channel: RustAudioChannel,
    group: SpanGroup,
    annotation_id: String,
}

#[derive(Clone, Copy)]
enum SpanGroup {
    Reference,
    Prediction,
}

#[pymethods]
impl PyTimeSpan {
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

    /// 当前 annotation。
    #[getter]
    fn annotation(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self.snapshot()?.annotation {
            Annotation::Activity(inner) => Py::new(py, PyAudioActivity { inner })?.into_any(),
            Annotation::Token(inner) => Py::new(py, PyToken { inner })?.into_any(),
            Annotation::Transcription(inner) => Py::new(py, PyTranscription { inner })?.into_any(),
            Annotation::Speaker(inner) => Py::new(py, PySpeaker { inner })?.into_any(),
            annotation @ (Annotation::Sentence(_) | Annotation::Language(_)) => {
                pythonize::pythonize(py, &annotation)
                    .map_err(py_error)?
                    .unbind()
            }
        })
    }

    /// 整体替换 annotation，并原子执行完整校验。
    #[setter]
    fn set_annotation(&self, annotation: &Bound<'_, PyAny>) -> PyResult<()> {
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
        match &mut annotations[index].annotation {
            Annotation::Activity(current) => {
                let activity =
                    annotation
                        .extract::<PyRef<'_, PyAudioActivity>>()
                        .map_err(|_| {
                            PyValueError::new_err("an activity annotation must be AudioActivity")
                        })?;
                *current = activity.inner.clone();
            }
            Annotation::Token(current) => {
                let token = annotation
                    .extract::<PyRef<'_, PyToken>>()
                    .map_err(|_| PyValueError::new_err("a token annotation must be Token"))?;
                if let Some(range) = token.inner.range
                    && (range.start < annotation_range.start || range.end > annotation_range.end)
                {
                    return Err(PyValueError::new_err(
                        "token range must be within the token annotation range",
                    ));
                }
                *current = token.inner.clone();
            }
            Annotation::Transcription(current) => {
                let transcription =
                    annotation
                        .extract::<PyRef<'_, PyTranscription>>()
                        .map_err(|_| {
                            PyValueError::new_err(
                                "a transcription annotation must be Transcription",
                            )
                        })?;
                validate_transcription_range(
                    annotation_range,
                    &transcription.inner,
                    "transcription annotation",
                )?;
                *current = transcription.inner.clone();
            }
            Annotation::Speaker(current) => {
                let speaker = annotation
                    .extract::<PyRef<'_, PySpeaker>>()
                    .map_err(|_| PyValueError::new_err("a speaker annotation must be Speaker"))?;
                validate_speaker_transcription(annotation_range, &speaker.inner.transcription)?;
                *current = speaker.inner.clone();
            }
            Annotation::Sentence(_) | Annotation::Language(_) => {
                return Err(PyValueError::new_err(
                    "this annotation type cannot be replaced from Python",
                ));
            }
        }

        let updated = annotations[index].clone();
        annotations
            .retain(|annotation| annotation.id == updated.id || !annotation.content_eq(&updated));
        candidate.validate_spans().map_err(py_error)?;
        *timeline = candidate;
        Ok(())
    }

    /// 返回该时间范围对应的波形视图。
    ///
    /// Returns:
    ///     从父 Timeline 截取的 Waveform。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> from asr_data.annotation import AudioActivity
    ///     >>> audio = AudioSource.from_pcm(b"\0\0" * 1600, 16000).open()
    ///     >>> timeline = audio.timeline("mono")
    ///     >>> span = timeline.reference.annotate_span(
    ///     ...     0, timeline.duration_ms, AudioActivity(event="speech")
    ///     ... )
    ///     >>> span.as_waveform().duration_ms
    ///     100.0
    fn as_waveform(&self) -> PyResult<PyWaveform> {
        let span = self.snapshot()?;
        let mut audio = self.audio.write().map_err(|_| poisoned("audio"))?;
        let waveform = audio
            .waveform_for_channel(self.channel)
            .map(|waveform| waveform.slice_ms(span.range.start.0, span.range.end.0))
            .map_err(py_error)?;
        Ok(PyWaveform::from_rust(waveform))
    }

    fn __repr__(&self) -> PyResult<String> {
        let annotation = self.snapshot()?;
        let text = match &annotation.annotation {
            Annotation::Transcription(transcription) => {
                format!(", text={:?}", truncate(&transcription.text, 60))
            }
            Annotation::Sentence(span) => format!(", text={:?}", truncate(&span.text, 60)),
            Annotation::Token(token) => {
                format!(", text={:?}", truncate(&token.text, 60))
            }
            _ => String::new(),
        };
        let confidence = annotation
            .confidence
            .map(|value| format!(", confidence={value:.3}"))
            .unwrap_or_default();
        let speaker = match &annotation.annotation {
            Annotation::Speaker(speaker) => {
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
        let event = match &annotation.annotation {
            Annotation::Activity(activity) => activity
                .event
                .as_ref()
                .map(|event| format!(", event={event:?}"))
                .unwrap_or_default(),
            _ => String::new(),
        };
        Ok(format!(
            "TimeSpan(id={:?}, annotation={}, range={}..{}ms{event}{speaker}{text}{confidence})",
            truncate(&annotation.id, 20),
            annotation_kind(&annotation.annotation),
            annotation.range.start.0,
            annotation.range.end.0,
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        let annotation = self.snapshot()?;
        let text = match &annotation.annotation {
            Annotation::Transcription(transcription) => {
                format!(": {:?}", truncate(&transcription.text, 60))
            }
            Annotation::Sentence(span) => format!(": {:?}", truncate(&span.text, 60)),
            Annotation::Token(token) => {
                format!(": {:?}", truncate(&token.text, 60))
            }
            _ => String::new(),
        };
        Ok(format!(
            "{} [{}..{}ms]{text}",
            annotation_kind(&annotation.annotation),
            annotation.range.start.0,
            annotation.range.end.0
        ))
    }
}

fn annotations(timeline: &RustTimeline, group: SpanGroup) -> &Vec<RustTimeSpan> {
    match group {
        SpanGroup::Reference => &timeline.reference,
        SpanGroup::Prediction => &timeline.prediction,
    }
}

fn annotations_mut(timeline: &mut RustTimeline, group: SpanGroup) -> &mut Vec<RustTimeSpan> {
    match group {
        SpanGroup::Reference => &mut timeline.reference,
        SpanGroup::Prediction => &mut timeline.prediction,
    }
}

impl PyTimeSpan {
    fn snapshot(&self) -> PyResult<RustTimeSpan> {
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

fn annotation_kind(annotation: &Annotation) -> &'static str {
    annotation.source_group()
}

fn annotation_from_py(value: &Bound<'_, PyAny>, range: TimeRange) -> PyResult<Annotation> {
    if let Ok(activity) = value.extract::<PyRef<'_, PyAudioActivity>>() {
        return Ok(Annotation::Activity(activity.inner.clone()));
    }
    if let Ok(transcription) = value.extract::<PyRef<'_, PyTranscription>>() {
        validate_transcription_range(range, &transcription.inner, "time span")?;
        return Ok(Annotation::Transcription(transcription.inner.clone()));
    }
    if let Ok(speaker) = value.extract::<PyRef<'_, PySpeaker>>() {
        validate_speaker_transcription(range, &speaker.inner.transcription)?;
        return Ok(Annotation::Speaker(speaker.inner.clone()));
    }
    if let Ok(token) = value.extract::<PyRef<'_, PyToken>>() {
        if let Some(token_range) = token.inner.range
            && (token_range.start < range.start || token_range.end > range.end)
        {
            return Err(PyValueError::new_err(
                "token range must be within the time span",
            ));
        }
        return Ok(Annotation::Token(token.inner.clone()));
    }
    Err(PyTypeError::new_err(
        "annotation must be AudioActivity, Transcription, Speaker, or Token",
    ))
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

/// 单个事件的 timeline 区间评测结果。
#[pyclass(name = "ActivityEventEvaluation", frozen)]
#[derive(Clone)]
struct PyActivityEventEvaluation {
    inner: RustActivityEventEvaluation,
}

#[pymethods]
impl PyActivityEventEvaluation {
    /// 事件名称。
    #[getter]
    fn event(&self) -> String {
        self.inner.event.clone()
    }

    /// Reference 事件总时长，单位为毫秒。
    #[getter]
    fn reference_ms(&self) -> u64 {
        self.inner.reference_ms
    }

    /// Prediction 事件总时长，单位为毫秒。
    #[getter]
    fn predicted_ms(&self) -> u64 {
        self.inner.predicted_ms
    }

    /// 正确预测该事件的时长。
    #[getter]
    fn true_positive_ms(&self) -> u64 {
        self.inner.true_positive_ms
    }

    /// 正确预测为非该事件的时长。
    #[getter]
    fn true_negative_ms(&self) -> u64 {
        self.inner.true_negative_ms
    }

    /// 误报该事件的时长。
    #[getter]
    fn false_positive_ms(&self) -> u64 {
        self.inner.false_positive_ms
    }

    /// 漏报该事件的时长。
    #[getter]
    fn false_negative_ms(&self) -> u64 {
        self.inner.false_negative_ms
    }

    /// 事件 precision。
    #[getter]
    fn precision(&self) -> f64 {
        self.inner.precision()
    }

    /// 事件 recall。
    #[getter]
    fn recall(&self) -> f64 {
        self.inner.recall()
    }

    /// 事件 F1。
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
            "ActivityEventEvaluation(event={:?}, precision={:.4}, recall={:.4}, f1={:.4}, iou={:.4})",
            self.inner.event,
            self.inner.precision(),
            self.inner.recall(),
            self.inner.f1(),
            self.inner.iou(),
        )
    }
}

/// 单个 prediction source 的 timeline Activity 评测结果。
#[pyclass(name = "ActivityEvaluation", frozen)]
#[derive(Clone)]
struct PyActivityEvaluation {
    inner: RustActivityEvaluation,
}

#[pymethods]
impl PyActivityEvaluation {
    /// Prediction source。
    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    /// Reference Activity 总时长，单位为毫秒。
    #[getter]
    fn reference_ms(&self) -> u64 {
        self.inner.reference_ms
    }

    /// Prediction Activity 总时长，单位为毫秒。
    #[getter]
    fn predicted_ms(&self) -> u64 {
        self.inner.predicted_ms
    }

    /// 正确预测为 Activity 的时长。
    #[getter]
    fn true_positive_ms(&self) -> u64 {
        self.inner.true_positive_ms
    }

    /// 正确预测为非 Activity 的时长。
    #[getter]
    fn true_negative_ms(&self) -> u64 {
        self.inner.true_negative_ms
    }

    /// 误报 Activity 的时长。
    #[getter]
    fn false_positive_ms(&self) -> u64 {
        self.inner.false_positive_ms
    }

    /// 漏报 Activity 的时长。
    #[getter]
    fn false_negative_ms(&self) -> u64 {
        self.inner.false_negative_ms
    }

    /// Activity precision。
    #[getter]
    fn precision(&self) -> f64 {
        self.inner.precision()
    }

    /// Activity recall。
    #[getter]
    fn recall(&self) -> f64 {
        self.inner.recall()
    }

    /// Activity F1。
    #[getter]
    fn f1(&self) -> f64 {
        self.inner.f1()
    }

    /// Reference 与 prediction 的区间 IoU。
    #[getter]
    fn iou(&self) -> f64 {
        self.inner.iou()
    }

    /// 按 event 分组的事件区间评测。
    #[getter]
    fn events(&self) -> std::collections::BTreeMap<String, PyActivityEventEvaluation> {
        self.inner
            .events
            .iter()
            .map(|(event, inner)| {
                (
                    event.clone(),
                    PyActivityEventEvaluation {
                        inner: inner.clone(),
                    },
                )
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ActivityEvaluation(source={:?}, precision={:.4}, recall={:.4}, f1={:.4}, iou={:.4}, events={})",
            self.inner.source,
            self.inner.precision(),
            self.inner.recall(),
            self.inner.f1(),
            self.inner.iou(),
            self.inner.events.len(),
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

    /// 按 prediction source 分组的 Activity 结果。
    #[getter]
    fn activity(&self) -> std::collections::BTreeMap<String, PyActivityEvaluation> {
        self.inner
            .activity
            .iter()
            .map(|(source, inner)| {
                (
                    source.clone(),
                    PyActivityEvaluation {
                        inner: inner.clone(),
                    },
                )
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TimelineEvaluation(transcription={}, activity={})",
            self.inner.transcription.len(),
            self.inner.activity.len(),
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

    /// 所属 Audio ID。
    #[getter]
    fn audio_id(&self) -> PyResult<String> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self.selected(&audio)?.audio_id.clone())
    }

    /// 修改所属 Audio ID。
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

    /// 返回当前声道的完整波形。
    ///
    /// Returns:
    ///     当前 Timeline 声道的 Waveform。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> audio = AudioSource.from_pcm(b"\0\0" * 1600, 16000).open()
    ///     >>> audio.timeline("mono").as_waveform().duration_ms
    ///     100.0
    fn as_waveform(&self) -> PyResult<PyWaveform> {
        let waveform = self
            .audio
            .write()
            .map_err(|_| poisoned("audio"))?
            .waveform_for_channel(self.channel)
            .map_err(py_error)?;
        Ok(PyWaveform::from_rust(waveform))
    }

    /// 不带 source 的 reference 标注集合。
    #[getter]
    fn reference(&self) -> PyReferenceSpans {
        PyReferenceSpans {
            core: SpanCollectionCore::new(self, SpanGroup::Reference),
        }
    }

    /// 必须带 source 的 prediction 标注集合。
    #[getter]
    fn prediction(&self) -> PyPredictionSpans {
        PyPredictionSpans {
            core: SpanCollectionCore::new(self, SpanGroup::Prediction),
        }
    }

    /// 评测一个或多个 prediction source。
    ///
    /// 不传 source 时自动发现所有具有对应 reference 的来源。只传一个任务
    /// 参数时只评测该任务。
    ///
    /// Args:
    ///     transcription: 转写来源或来源名称列表。
    ///     activity: Activity 来源或来源名称列表。
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
    ///     >>> from asr_data import Audio, AudioSource
    ///     >>> from asr_data.annotation import Transcription
    ///     >>> timeline = Audio(
    ///     ...     AudioSource.from_pcm(b"\0\0" * 10, 16000)
    ///     ... ).timeline("mono")
    ///     >>> _ = timeline.reference.annotate_span(
    ///     ...     0, timeline.duration_ms, Transcription("你好")
    ///     ... )
    ///     >>> _ = timeline.prediction.annotate_span(
    ///     ...     0, timeline.duration_ms, Transcription("你好"), source="qwen-asr"
    ///     ... )
    ///     >>> result = timeline.eval()
    ///     >>> result.transcription["qwen-asr"].cer
    ///     0.0
    #[pyo3(signature = (*, transcription=None, activity=None, normalize=true))]
    fn eval(
        &self,
        transcription: Option<&Bound<'_, PyAny>>,
        activity: Option<&Bound<'_, PyAny>>,
        normalize: bool,
    ) -> PyResult<PyTimelineEvaluation> {
        let normalization = if normalize {
            TranscriptionNormalization::ChineseTn
        } else {
            TranscriptionNormalization::None
        };
        let config = TimelineEvalConfig {
            transcription_sources: extract_eval_sources(transcription, "transcription")?,
            activity_sources: extract_eval_sources(activity, "activity")?,
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
    fn selected<'a>(&self, audio: &'a RustAudio) -> PyResult<&'a RustTimeline> {
        audio
            .timeline(self.channel)
            .map_err(py_error)?
            .ok_or_else(|| PyRuntimeError::new_err("selected timeline does not exist"))
    }
}

#[derive(Clone)]
struct SpanCollectionCore {
    audio: SharedAudio,
    channel: RustAudioChannel,
    group: SpanGroup,
}

impl SpanCollectionCore {
    fn new(timeline: &PyTimeline, group: SpanGroup) -> Self {
        Self {
            audio: timeline.audio.clone(),
            channel: timeline.channel,
            group,
        }
    }

    fn span_handle(&self, annotation_id: String) -> PyTimeSpan {
        PyTimeSpan {
            audio: self.audio.clone(),
            channel: self.channel,
            group: self.group,
            annotation_id,
        }
    }

    fn selected<'a>(&self, audio: &'a RustAudio) -> PyResult<&'a RustTimeline> {
        audio
            .timeline(self.channel)
            .map_err(py_error)?
            .ok_or_else(|| PyRuntimeError::new_err("selected timeline does not exist"))
    }

    fn selected_mut<'a>(&self, audio: &'a mut RustAudio) -> PyResult<&'a mut RustTimeline> {
        audio
            .timeline_mut(self.channel)
            .map_err(py_error)?
            .ok_or_else(|| PyRuntimeError::new_err("selected timeline does not exist"))
    }

    fn all(&self) -> PyResult<Vec<PyTimeSpan>> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(annotations(self.selected(&audio)?, self.group)
            .iter()
            .map(|annotation| self.span_handle(annotation.id.clone()))
            .collect())
    }

    fn len(&self) -> PyResult<usize> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(annotations(self.selected(&audio)?, self.group).len())
    }

    fn annotate_span_inner(
        &self,
        start_ms: u64,
        end_ms: u64,
        annotation: Annotation,
        source: Option<&str>,
        confidence: Option<f32>,
    ) -> PyResult<PyTimeSpan> {
        if end_ms < start_ms {
            return Err(PyValueError::new_err("end_ms must be >= start_ms"));
        }
        let mut annotation = RustTimeSpan::new(
            TimeRange::new(DurationMs(start_ms), DurationMs(end_ms)),
            annotation,
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
            SpanGroup::Reference => timeline.push_reference(annotation),
            SpanGroup::Prediction => timeline.push_prediction(annotation),
        }
        .map_err(py_error)?
        .id
        .clone();
        Ok(self.span_handle(annotation_id))
    }
}

/// Timeline 的参考真值标注集合。
#[pyclass(name = "ReferenceSpans")]
#[derive(Clone)]
struct PyReferenceSpans {
    core: SpanCollectionCore,
}

impl PyReferenceSpans {
    /// 按时间顺序组合全部 reference 文本。
    ///
    /// Returns:
    ///     组合后的 Transcript。
    ///
    /// Examples:
    ///     >>> from asr_data import Audio, AudioSource
    ///     >>> from asr_data.annotation import Transcription
    ///     >>> timeline = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
    ///     >>> _ = timeline.reference.annotate_span(
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
}

#[pymethods]
impl PyReferenceSpans {
    /// 当前 reference 中的全部时间范围。
    #[getter]
    fn spans(&self) -> PyResult<Vec<PyTimeSpan>> {
        self.core.all()
    }

    /// 在 reference 中添加一条带时间范围的标注。
    ///
    /// Args:
    ///     start_ms: 全局起始时间，单位为毫秒。
    ///     end_ms: 全局结束时间，单位为毫秒。
    ///     annotation: AudioActivity、Transcription、Speaker 或 Token。
    ///     confidence: 可选的 span 级置信度。
    ///
    /// Returns:
    ///     新建或去重后已有的 TimeSpan。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> from asr_data.annotation import AudioActivity
    ///     >>> timeline = AudioSource.from_pcm(b"\0\0" * 10, 1000).open().timeline("mono")
    ///     >>> span = timeline.reference.annotate_span(
    ///     ...     0, timeline.duration_ms, AudioActivity(event="speech")
    ///     ... )
    #[pyo3(signature = (start_ms, end_ms, annotation, *, confidence=None))]
    fn annotate_span(
        &self,
        start_ms: u64,
        end_ms: u64,
        annotation: &Bound<'_, PyAny>,
        confidence: Option<f32>,
    ) -> PyResult<PyTimeSpan> {
        let range = TimeRange::new(DurationMs(start_ms), DurationMs(end_ms));
        self.core.annotate_span_inner(
            start_ms,
            end_ms,
            annotation_from_py(annotation, range)?,
            None,
            confidence,
        )
    }

    #[pyo3(name = "transcript")]
    /// 按时间顺序组合 reference 文本。
    ///
    /// Returns:
    ///     组合后的 Transcript。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> from asr_data.annotation import Transcription
    ///     >>> timeline = AudioSource.from_pcm(b"\0\0" * 10, 1000).open().timeline("mono")
    ///     >>> _ = timeline.reference.annotate_span(0, 10, Transcription("你好"))
    ///     >>> timeline.reference.transcript().text
    ///     '你好'
    fn py_transcript(&self) -> PyResult<PyTranscript> {
        self.transcript()
    }

    fn __len__(&self) -> PyResult<usize> {
        self.core.len()
    }
}

/// Timeline 的模型 prediction 标注集合。
#[pyclass(name = "PredictionSpans")]
#[derive(Clone)]
struct PyPredictionSpans {
    core: SpanCollectionCore,
}

impl PyPredictionSpans {
    /// 返回指定 source 的全部 prediction annotation。
    ///
    /// Args:
    ///     source: 要查询的来源。
    ///
    /// Returns:
    ///     保持存储顺序的 TimeSpan 列表。
    ///
    /// Examples:
    ///     >>> from asr_data import Audio, AudioSource
    ///     >>> timeline = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
    ///     >>> items = timeline.prediction.by_source("asr")
    fn by_source(&self, source: &str) -> PyResult<Vec<PyTimeSpan>> {
        let audio = self.core.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self
            .core
            .selected(&audio)?
            .predictions_by_source(source)
            .map(|annotation| self.core.span_handle(annotation.id.clone()))
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
    ///     >>> from asr_data import Audio, AudioSource
    ///     >>> timeline = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
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
    ///     >>> from asr_data import Audio, AudioSource
    ///     >>> timeline = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
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
    ///     >>> from asr_data import Audio, AudioSource
    ///     >>> timeline = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000)).timeline("mono")
    ///     >>> changed = timeline.prediction.relabel_source("asr", "asr-v2")
    fn relabel_source(&self, from_source: &str, to_source: &str) -> PyResult<usize> {
        validate_source(to_source)?;
        let mut audio = self.core.audio.write().map_err(|_| poisoned("audio"))?;
        self.core
            .selected_mut(&mut audio)?
            .relabel_prediction_source(from_source, to_source)
            .map_err(py_error)
    }
}

#[pymethods]
impl PyPredictionSpans {
    /// 当前 prediction 中的全部时间范围。
    #[getter]
    fn spans(&self) -> PyResult<Vec<PyTimeSpan>> {
        self.core.all()
    }

    /// 在 prediction 中添加一条带模型来源的标注。
    ///
    /// Args:
    ///     start_ms: 全局起始时间，单位为毫秒。
    ///     end_ms: 全局结束时间，单位为毫秒。
    ///     annotation: AudioActivity、Transcription、Speaker 或 Token。
    ///     source: 模型或导入流程名称。
    ///     confidence: 可选的 span 级置信度。
    ///
    /// Returns:
    ///     新建或去重后已有的 TimeSpan。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> from asr_data.annotation import Transcription
    ///     >>> timeline = AudioSource.from_pcm(b"\0\0" * 10, 1000).open().timeline("mono")
    ///     >>> span = timeline.prediction.annotate_span(
    ///     ...     0, timeline.duration_ms, Transcription("你好"), source="asr"
    ///     ... )
    #[pyo3(signature = (start_ms, end_ms, annotation, *, source, confidence=None))]
    fn annotate_span(
        &self,
        start_ms: u64,
        end_ms: u64,
        annotation: &Bound<'_, PyAny>,
        source: &str,
        confidence: Option<f32>,
    ) -> PyResult<PyTimeSpan> {
        validate_source(source)?;
        let range = TimeRange::new(DurationMs(start_ms), DurationMs(end_ms));
        self.core.annotate_span_inner(
            start_ms,
            end_ms,
            annotation_from_py(annotation, range)?,
            Some(source),
            confidence,
        )
    }

    /// 按 Annotation 类型分组的 prediction source。
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

    #[pyo3(name = "by_source")]
    /// 返回指定 source 的全部 prediction span。
    ///
    /// Args:
    ///     source: 模型或流程名称。
    ///
    /// Returns:
    ///     保持存储顺序的 TimeSpan 列表。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> timeline = AudioSource.from_pcm(b"\0\0", 1000).open().timeline("mono")
    ///     >>> timeline.prediction.by_source("asr")
    ///     []
    fn py_by_source(&self, source: &str) -> PyResult<Vec<PyTimeSpan>> {
        self.by_source(source)
    }

    #[pyo3(name = "transcript")]
    /// 按时间顺序组合指定 source 的预测文本。
    ///
    /// Args:
    ///     source: 模型或流程名称。
    ///
    /// Returns:
    ///     组合后的 Transcript。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> timeline = AudioSource.from_pcm(b"\0\0", 1000).open().timeline("mono")
    ///     >>> timeline.prediction.transcript("asr").text
    ///     ''
    fn py_transcript(&self, source: &str) -> PyResult<PyTranscript> {
        self.transcript(source)
    }

    #[pyo3(name = "remove_by_source")]
    /// 删除指定 source 的全部 prediction span。
    ///
    /// Args:
    ///     source: 模型或流程名称。
    ///
    /// Returns:
    ///     删除数量。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> timeline = AudioSource.from_pcm(b"\0\0", 1000).open().timeline("mono")
    ///     >>> timeline.prediction.remove_by_source("asr")
    ///     0
    fn py_remove_by_source(&self, source: &str) -> PyResult<usize> {
        self.remove_by_source(source)
    }

    #[pyo3(name = "relabel_source")]
    /// 原子修改 prediction source。
    ///
    /// Args:
    ///     from_source: 原 source。
    ///     to_source: 新 source。
    ///
    /// Returns:
    ///     修改数量。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> timeline = AudioSource.from_pcm(b"\0\0", 1000).open().timeline("mono")
    ///     >>> timeline.prediction.relabel_source("asr", "asr-v2")
    ///     0
    fn py_relabel_source(&self, from_source: &str, to_source: &str) -> PyResult<usize> {
        self.relabel_source(from_source, to_source)
    }

    fn __len__(&self) -> PyResult<usize> {
        self.core.len()
    }
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
    module.add_class::<PyTimeSpan>()?;
    module.add_class::<PyTranscript>()?;
    module.add_class::<PyTranscriptionEvaluation>()?;
    module.add_class::<PyActivityEventEvaluation>()?;
    module.add_class::<PyActivityEvaluation>()?;
    module.add_class::<PyTimelineEvaluation>()?;
    module.add_class::<PyReferenceSpans>()?;
    module.add_class::<PyPredictionSpans>()?;
    module.add_class::<PyTimeline>()?;
    Ok(())
}
