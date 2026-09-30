use crate::audio::AudioChannel as RustAudioChannel;
use crate::doc::Audio as RustAudio;
use crate::timeline::{
    ActivityEvaluation as RustActivityEvaluation,
    ActivityEventEvaluation as RustActivityEventEvaluation, Timeline as RustTimeline,
    TimelineEvalConfig, TimelineEvaluation as RustTimelineEvaluation, Transcript as RustTranscript,
    TranscriptionEvaluation as RustTranscriptionEvaluation, TranscriptionNormalization,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use super::annotation::{event_from_py, event_to_py};
use super::audio::{PyWaveform, display_rust_waveform};
use super::common::{SharedAudio, format_duration_ms, poisoned, py_error, truncate};

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
    fn duration_ms(&self) -> PyResult<usize> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self.selected(&audio)?.duration_ms())
    }

    /// 返回当前声道的完整波形。
    ///
    /// Returns:
    ///     当前 Timeline 声道的 Waveform。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> audio = AudioSource.from_pcm(b"\0\0" * 1600, 16000).load()
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

    /// 在 Jupyter 中显示当前声道的音频播放器。
    ///
    /// Args:
    ///     start_ms: 可选播放起始时间。
    ///     end_ms: 可选播放结束时间。
    ///     autoplay: 是否自动播放。
    ///
    /// Returns:
    ///     None；播放器直接发送到当前 Jupyter 输出。
    ///
    /// Raises:
    ///     ValueError: 结束时间早于起始时间。
    ///     AsrDataError: IPython 不可用。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> timeline = AudioSource.from_pcm(b"\0\0" * 100, 1000).load().timeline("mono")
    ///     >>> timeline.display(end_ms=50)
    #[pyo3(signature = (start_ms=None, end_ms=None, autoplay=false))]
    fn display(
        &self,
        py: Python<'_>,
        start_ms: Option<u64>,
        end_ms: Option<u64>,
        autoplay: bool,
    ) -> PyResult<()> {
        let waveform = self
            .audio
            .write()
            .map_err(|_| poisoned("audio"))?
            .waveform_for_channel(self.channel)
            .map_err(py_error)?;
        display_rust_waveform(py, waveform, start_ms, end_ms, autoplay)
    }

    /// 参考事件。语音是 [`Speech`](crate::timeline::Speech)，其余是基类事件。
    #[getter]
    fn reference(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        self.selected(&audio)?
            .reference
            .iter()
            .cloned()
            .map(|event| event_to_py(py, event))
            .collect()
    }

    /// 预测事件。每条都带非空 `source`。
    #[getter]
    fn prediction(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        self.selected(&audio)?
            .prediction
            .iter()
            .cloned()
            .map(|event| event_to_py(py, event))
            .collect()
    }

    /// 从参考语音事件拼出的转写。
    #[getter]
    fn reference_transcript(&self) -> PyResult<PyTranscript> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(PyTranscript {
            inner: self.selected(&audio)?.reference_transcript(),
        })
    }

    /// 写入一条 [`AudioEvent`](crate::timeline::AudioEvent) 或派生的 `Speech`。
    ///
    /// Args:
    ///     start_ms: 起始时间，单位为毫秒。
    ///     end_ms: 结束时间，单位为毫秒。
    ///     event: 基类事件或 `Speech`。
    ///     is_reference: ``True`` 表示参考，``False`` 表示预测。默认为 ``True``。
    ///     source: 预测来源；reference 必须省略。
    ///
    /// Returns:
    ///     写入后的事件。语音返回 `Speech`，其他事件返回 `AudioEvent`。
    ///
    /// Raises:
    ///     ValueError: 时间范围无效、reference 携带 source，或 prediction 缺少 source。
    ///     AsrDataError: 事件与已有内容冲突。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> from asr_data.annotation import Speech
    ///     >>> timeline = AudioSource.from_pcm(b"\0\0" * 10, 1000).load().timeline("mono")
    ///     >>> timeline.annotate(0, timeline.duration_ms, Speech()).name
    ///     'speech'
    #[pyo3(signature = (start_ms, end_ms, event, *, is_reference=true, source=None))]
    fn annotate(
        &self,
        py: Python<'_>,
        start_ms: u64,
        end_ms: u64,
        event: &Bound<'_, PyAny>,
        is_reference: bool,
        source: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        if end_ms <= start_ms {
            return Err(PyValueError::new_err("end_ms must be > start_ms"));
        }
        if is_reference {
            if source.is_some() {
                return Err(PyValueError::new_err(
                    "source must be omitted when is_reference=True",
                ));
            }
        } else {
            let source = source.ok_or_else(|| {
                PyValueError::new_err("source is required when is_reference=False")
            })?;
            validate_source(source)?;
        }
        let mut audio = self.audio.write().map_err(|_| poisoned("audio"))?;
        let timeline = audio
            .timeline_mut(self.channel)
            .map_err(py_error)?
            .ok_or_else(|| PyRuntimeError::new_err("selected timeline does not exist"))?;
        if end_ms > timeline.duration_ms() as u64 {
            return Err(PyValueError::new_err(format!(
                "event end_ms ({end_ms}) must not exceed timeline duration_ms ({})",
                timeline.duration_ms()
            )));
        }
        let stored = timeline
            .annotate_with(
                start_ms as usize,
                end_ms as usize,
                event_from_py(event)?,
                is_reference,
                source,
            )
            .map_err(py_error)?
            .clone();
        event_to_py(py, stored)
    }

    /// 按 ID 删除一条事件。
    ///
    /// Args:
    ///     event_id: 要删除的事件 ID。
    ///
    /// Returns:
    ///     找到并删除时为 ``True``。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> from asr_data.annotation import Speech
    ///     >>> timeline = AudioSource.from_pcm(b"\0\0" * 10, 1000).load().timeline("mono")
    ///     >>> event = timeline.annotate(0, timeline.duration_ms, Speech())
    ///     >>> timeline.remove(event.id)
    ///     True
    fn remove(&self, event_id: &str) -> PyResult<bool> {
        let mut audio = self.audio.write().map_err(|_| poisoned("audio"))?;
        let timeline = audio
            .timeline_mut(self.channel)
            .map_err(py_error)?
            .ok_or_else(|| PyRuntimeError::new_err("selected timeline does not exist"))?;
        Ok(timeline.remove_event(event_id))
    }

    /// 某个预测来源的转写。
    ///
    /// Args:
    ///     source: 预测来源。
    ///
    /// Returns:
    ///     按句子时间拼出的 Transcript。没有文本时文本为空。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> timeline = AudioSource.from_pcm(b"\0\0" * 10, 1000).load().timeline("mono")
    ///     >>> timeline.prediction_transcript("asr").text
    ///     ''
    fn prediction_transcript(&self, source: &str) -> PyResult<PyTranscript> {
        validate_source(source)?;
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(PyTranscript {
            inner: self.selected(&audio)?.prediction_transcript(source),
        })
    }

    /// 删除某个预测来源的全部事件。
    ///
    /// Args:
    ///     source: 预测来源。
    ///
    /// Returns:
    ///     删除的事件条数。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> timeline = AudioSource.from_pcm(b"\0\0" * 10, 1000).load().timeline("mono")
    ///     >>> timeline.remove_predictions("asr")
    ///     0
    fn remove_predictions(&self, source: &str) -> PyResult<usize> {
        validate_source(source)?;
        let mut audio = self.audio.write().map_err(|_| poisoned("audio"))?;
        let timeline = audio
            .timeline_mut(self.channel)
            .map_err(py_error)?
            .ok_or_else(|| PyRuntimeError::new_err("selected timeline does not exist"))?;
        Ok(timeline.remove_predictions_by_source(source))
    }

    /// 把预测来源从 `from_source` 改成 `to_source`。
    ///
    /// Args:
    ///     from_source: 原来源。
    ///     to_source: 新来源。
    ///
    /// Returns:
    ///     修改的事件条数。
    ///
    /// Raises:
    ///     ValueError: 来源为空。
    ///     AsrDataError: 改名后同一来源里的事件会重叠。
    ///
    /// Examples:
    ///     >>> from asr_data import AudioSource
    ///     >>> timeline = AudioSource.from_pcm(b"\0\0" * 10, 1000).load().timeline("mono")
    ///     >>> timeline.relabel_prediction_source("asr", "asr-v2")
    ///     0
    fn relabel_prediction_source(&self, from_source: &str, to_source: &str) -> PyResult<usize> {
        validate_source(from_source)?;
        validate_source(to_source)?;
        let mut audio = self.audio.write().map_err(|_| poisoned("audio"))?;
        let timeline = audio
            .timeline_mut(self.channel)
            .map_err(py_error)?
            .ok_or_else(|| PyRuntimeError::new_err("selected timeline does not exist"))?;
        timeline
            .relabel_prediction_source(from_source, to_source)
            .map_err(py_error)
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
    ///     traditional_to_simple: 是否将繁体中文转换为简体中文。
    ///     full_to_half: 是否将全角字符转换为半角字符。
    ///     remove_erhua: 是否去除儿化音“儿”。
    ///     remove_interjections: 是否去除“嗯”“啊”“呃”等语气词。
    ///     remove_puncts: 是否去除标点符号。
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
    ///     >>> from asr_data.annotation import Speech, Transcription
    ///     >>> timeline = Audio(
    ///     ...     AudioSource.from_pcm(b"\0\0" * 10, 16000)
    ///     ... ).timeline("mono")
    ///     >>> end = timeline.duration_ms
    ///     >>> speech = Speech(transcription=Transcription("你好"))
    ///     >>> _ = timeline.annotate(0, end, speech)
    ///     >>> _ = timeline.annotate(0, end, speech, is_reference=False, source="qwen-asr")
    ///     >>> result = timeline.eval(transcription="qwen-asr")
    ///     >>> result.transcription["qwen-asr"].cer
    ///     0.0
    #[pyo3(signature = (
        *,
        transcription=None,
        activity=None,
        normalize=true,
        traditional_to_simple=true,
        full_to_half=true,
        remove_erhua=true,
        remove_interjections=true,
        remove_puncts=true
    ))]
    fn eval(
        &self,
        transcription: Option<&Bound<'_, PyAny>>,
        activity: Option<&Bound<'_, PyAny>>,
        normalize: bool,
        traditional_to_simple: bool,
        full_to_half: bool,
        remove_erhua: bool,
        remove_interjections: bool,
        remove_puncts: bool,
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
            traditional_to_simple,
            full_to_half,
            remove_erhua,
            remove_interjections,
            remove_puncts,
        };
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        let inner = self.selected(&audio)?.eval(&config).map_err(py_error)?;
        Ok(PyTimelineEvaluation { inner })
    }

    fn __repr__(&self) -> PyResult<String> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        let timeline = self.selected(&audio)?;
        let duration = format!("{:?}", format_duration_ms(timeline.duration as f64));
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
        Ok(self.selected(&audio)?.terminal_view().to_string())
    }

    fn _repr_html_(&self) -> PyResult<String> {
        let audio = self.audio.read().map_err(|_| poisoned("audio"))?;
        Ok(self.selected(&audio)?.notebook_html())
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

fn validate_source(source: &str) -> PyResult<()> {
    if source.trim().is_empty() {
        return Err(PyValueError::new_err(
            "prediction source must be a non-empty string",
        ));
    }
    Ok(())
}

/// 把本模块的 Python 类型和函数注册进 `_native`。
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTranscript>()?;
    module.add_class::<PyTranscriptionEvaluation>()?;
    module.add_class::<PyActivityEventEvaluation>()?;
    module.add_class::<PyActivityEvaluation>()?;
    module.add_class::<PyTimelineEvaluation>()?;
    module.add_class::<PyTimeline>()?;
    Ok(())
}
