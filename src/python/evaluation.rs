use std::collections::BTreeMap;

use crate::{
    DatasetActivityEvaluation as RustDatasetActivityEvaluation,
    DatasetActivityEventEvaluation as RustDatasetActivityEventEvaluation,
    DatasetEvaluation as RustDatasetEvaluation,
    DatasetSpeakerEvaluation as RustDatasetSpeakerEvaluation,
    DatasetTranscriptionEvaluation as RustDatasetTranscriptionEvaluation, TimelineEvalConfig,
    TranscriptionNormalization, evaluate_dataset as rust_evaluate_dataset,
};
use pyo3::prelude::*;

use super::common::py_error;
use super::doc::PyAudio;
use super::timeline::extract_eval_sources;

/// 单个 prediction source 的数据集转写聚合结果。
///
/// CER 由各 timeline 的编辑统计量累加后计算，不是单条 CER 的平均值。
#[pyclass(name = "DatasetTranscriptionEvaluation", frozen)]
#[derive(Clone)]
pub(super) struct PyDatasetTranscriptionEvaluation {
    pub(super) inner: RustDatasetTranscriptionEvaluation,
}

/// 单个 prediction source 的数据集说话人分离结果。
#[pyclass(name = "DatasetSpeakerEvaluation", frozen)]
#[derive(Clone)]
pub(super) struct PyDatasetSpeakerEvaluation {
    pub(super) inner: RustDatasetSpeakerEvaluation,
}

#[pymethods]
impl PyDatasetSpeakerEvaluation {
    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    #[getter]
    fn evaluated_documents(&self) -> usize {
        self.inner.evaluated_documents
    }

    #[getter]
    fn evaluated_timelines(&self) -> usize {
        self.inner.evaluated_timelines
    }

    #[getter]
    fn unannotated_timelines(&self) -> usize {
        self.inner.unannotated_timelines
    }

    #[getter]
    fn missing_predictions(&self) -> usize {
        self.inner.missing_predictions
    }

    #[getter]
    fn unannotated_ids(&self) -> Vec<String> {
        self.inner.unannotated_ids.clone()
    }

    #[getter]
    fn missing_prediction_ids(&self) -> Vec<String> {
        self.inner.missing_prediction_ids.clone()
    }

    #[getter]
    fn reference_speaker_ms(&self) -> u64 {
        self.inner.reference_speaker_ms
    }

    #[getter]
    fn predicted_speaker_ms(&self) -> u64 {
        self.inner.predicted_speaker_ms
    }

    #[getter]
    fn correct_speaker_ms(&self) -> u64 {
        self.inner.correct_speaker_ms
    }

    #[getter]
    fn missed_speaker_ms(&self) -> u64 {
        self.inner.missed_speaker_ms
    }

    #[getter]
    fn false_alarm_ms(&self) -> u64 {
        self.inner.false_alarm_ms
    }

    #[getter]
    fn speaker_confusion_ms(&self) -> u64 {
        self.inner.speaker_confusion_ms
    }

    #[getter]
    fn der(&self) -> f64 {
        self.inner.der()
    }

    #[getter]
    fn coverage(&self) -> f64 {
        self.inner.coverage()
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetSpeakerEvaluation(source={:?}, der={:.4}, coverage={:.4}, timelines={})",
            self.inner.source,
            self.inner.der(),
            self.inner.coverage(),
            self.inner.evaluated_timelines,
        )
    }
}

#[pymethods]
impl PyDatasetTranscriptionEvaluation {
    /// Prediction source。
    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    /// 实际参与评测的文档数。
    #[getter]
    fn evaluated_documents(&self) -> usize {
        self.inner.evaluated_documents
    }

    /// 实际参与评测的 timeline 数。
    #[getter]
    fn evaluated_timelines(&self) -> usize {
        self.inner.evaluated_timelines
    }

    /// 缺少转写 reference 的 timeline 数。
    #[getter]
    fn unannotated_timelines(&self) -> usize {
        self.inner.unannotated_timelines
    }

    /// 有 reference 但缺少该 source prediction 的 timeline 数。
    #[getter]
    fn missing_predictions(&self) -> usize {
        self.inner.missing_predictions
    }

    /// 未标注的 ``audio_id:channel`` 列表。
    #[getter]
    fn unannotated_ids(&self) -> Vec<String> {
        self.inner.unannotated_ids.clone()
    }

    /// 缺少预测的 ``audio_id:channel`` 列表。
    #[getter]
    fn missing_prediction_ids(&self) -> Vec<String> {
        self.inner.missing_prediction_ids.clone()
    }

    /// 标准化模式。
    #[getter]
    fn normalization(&self) -> &'static str {
        normalization_name(self.inner.normalization)
    }

    /// 累计替换字符数。
    #[getter]
    fn substitutions(&self) -> usize {
        self.inner.stats.substitutions
    }

    /// 累计删除字符数。
    #[getter]
    fn deletions(&self) -> usize {
        self.inner.stats.deletions
    }

    /// 累计插入字符数。
    #[getter]
    fn insertions(&self) -> usize {
        self.inner.stats.insertions
    }

    /// 累计参考字符数。
    #[getter]
    fn reference_chars(&self) -> usize {
        self.inner.stats.reference_chars
    }

    /// 累计预测字符数。
    #[getter]
    fn hypothesis_chars(&self) -> usize {
        self.inner.hypothesis_chars
    }

    /// 累计匹配字符数。
    #[getter]
    fn matches(&self) -> usize {
        self.inner.matches()
    }

    /// 标准化文本完全一致的 timeline 数。
    #[getter]
    fn exact_matches(&self) -> usize {
        self.inner.exact_matches
    }

    /// Corpus CER。
    #[getter]
    fn cer(&self) -> f64 {
        self.inner.cer()
    }

    /// 总体字符级 precision。
    #[getter]
    fn precision(&self) -> f64 {
        self.inner.precision()
    }

    /// 总体字符级 recall。
    #[getter]
    fn recall(&self) -> f64 {
        self.inner.recall()
    }

    /// 总体字符级 F1。
    #[getter]
    fn f1(&self) -> f64 {
        self.inner.f1()
    }

    /// 完全匹配 timeline 占比。
    #[getter]
    fn exact_match_rate(&self) -> f64 {
        self.inner.exact_match_rate()
    }

    /// 具有 prediction 的已标注 timeline 占比。
    #[getter]
    fn coverage(&self) -> f64 {
        self.inner.coverage()
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetTranscriptionEvaluation(source={:?}, cer={:.4}, coverage={:.4}, timelines={})",
            self.inner.source,
            self.inner.cer(),
            self.inner.coverage(),
            self.inner.evaluated_timelines,
        )
    }
}

/// 单个事件的数据集级区间聚合结果。
#[pyclass(name = "DatasetActivityEventEvaluation", frozen)]
#[derive(Clone)]
pub(super) struct PyDatasetActivityEventEvaluation {
    inner: RustDatasetActivityEventEvaluation,
}

#[pymethods]
impl PyDatasetActivityEventEvaluation {
    /// 事件名称。
    #[getter]
    fn event(&self) -> String {
        self.inner.event.clone()
    }

    /// 实际参与评测的文档数。
    #[getter]
    fn evaluated_documents(&self) -> usize {
        self.inner.evaluated_documents
    }

    /// 实际参与评测的 timeline 数。
    #[getter]
    fn evaluated_timelines(&self) -> usize {
        self.inner.evaluated_timelines
    }

    /// 累计 reference 事件时长。
    #[getter]
    fn reference_ms(&self) -> u64 {
        self.inner.reference_ms
    }

    /// 累计 prediction 事件时长。
    #[getter]
    fn predicted_ms(&self) -> u64 {
        self.inner.predicted_ms
    }

    /// 累计正确事件时长。
    #[getter]
    fn true_positive_ms(&self) -> u64 {
        self.inner.true_positive_ms
    }

    /// 累计正确非该事件时长。
    #[getter]
    fn true_negative_ms(&self) -> u64 {
        self.inner.true_negative_ms
    }

    /// 累计误报该事件时长。
    #[getter]
    fn false_positive_ms(&self) -> u64 {
        self.inner.false_positive_ms
    }

    /// 累计漏报该事件时长。
    #[getter]
    fn false_negative_ms(&self) -> u64 {
        self.inner.false_negative_ms
    }

    /// 总体事件 precision。
    #[getter]
    fn precision(&self) -> f64 {
        self.inner.precision()
    }

    /// 总体事件 recall。
    #[getter]
    fn recall(&self) -> f64 {
        self.inner.recall()
    }

    /// 总体事件 F1。
    #[getter]
    fn f1(&self) -> f64 {
        self.inner.f1()
    }

    /// 总体事件 IoU。
    #[getter]
    fn iou(&self) -> f64 {
        self.inner.iou()
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetActivityEventEvaluation(event={:?}, f1={:.4}, iou={:.4}, timelines={})",
            self.inner.event,
            self.inner.f1(),
            self.inner.iou(),
            self.inner.evaluated_timelines,
        )
    }
}

/// 单个 prediction source 的数据集 Activity 聚合结果。
#[pyclass(name = "DatasetActivityEvaluation", frozen)]
#[derive(Clone)]
pub(super) struct PyDatasetActivityEvaluation {
    pub(super) inner: RustDatasetActivityEvaluation,
}

#[pymethods]
impl PyDatasetActivityEvaluation {
    /// Prediction source。
    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    /// 实际参与评测的文档数。
    #[getter]
    fn evaluated_documents(&self) -> usize {
        self.inner.evaluated_documents
    }

    /// 实际参与评测的 timeline 数。
    #[getter]
    fn evaluated_timelines(&self) -> usize {
        self.inner.evaluated_timelines
    }

    /// 缺少 Activity reference 的 timeline 数。
    #[getter]
    fn unannotated_timelines(&self) -> usize {
        self.inner.unannotated_timelines
    }

    /// 有 reference 但缺少该 source prediction 的 timeline 数。
    #[getter]
    fn missing_predictions(&self) -> usize {
        self.inner.missing_predictions
    }

    /// 未标注的 ``audio_id:channel`` 列表。
    #[getter]
    fn unannotated_ids(&self) -> Vec<String> {
        self.inner.unannotated_ids.clone()
    }

    /// 缺少预测的 ``audio_id:channel`` 列表。
    #[getter]
    fn missing_prediction_ids(&self) -> Vec<String> {
        self.inner.missing_prediction_ids.clone()
    }

    /// 累计 reference Activity 时长。
    #[getter]
    fn reference_ms(&self) -> u64 {
        self.inner.reference_ms
    }

    /// 累计 prediction Activity 时长。
    #[getter]
    fn predicted_ms(&self) -> u64 {
        self.inner.predicted_ms
    }

    /// 累计正确 Activity 时长。
    #[getter]
    fn true_positive_ms(&self) -> u64 {
        self.inner.true_positive_ms
    }

    /// 累计正确非 Activity 时长。
    #[getter]
    fn true_negative_ms(&self) -> u64 {
        self.inner.true_negative_ms
    }

    /// 累计误报 Activity 时长。
    #[getter]
    fn false_positive_ms(&self) -> u64 {
        self.inner.false_positive_ms
    }

    /// 累计漏报 Activity 时长。
    #[getter]
    fn false_negative_ms(&self) -> u64 {
        self.inner.false_negative_ms
    }

    /// 总体 Activity precision。
    #[getter]
    fn precision(&self) -> f64 {
        self.inner.precision()
    }

    /// 总体 Activity recall。
    #[getter]
    fn recall(&self) -> f64 {
        self.inner.recall()
    }

    /// 总体 Activity F1。
    #[getter]
    fn f1(&self) -> f64 {
        self.inner.f1()
    }

    /// 总体 Activity IoU。
    #[getter]
    fn iou(&self) -> f64 {
        self.inner.iou()
    }

    /// 具有 prediction 的已标注 timeline 占比。
    #[getter]
    fn coverage(&self) -> f64 {
        self.inner.coverage()
    }

    /// 按 event 分组的数据集事件结果。
    #[getter]
    fn events(&self) -> BTreeMap<String, PyDatasetActivityEventEvaluation> {
        self.inner
            .events
            .iter()
            .map(|(event, inner)| {
                (
                    event.clone(),
                    PyDatasetActivityEventEvaluation {
                        inner: inner.clone(),
                    },
                )
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetActivityEvaluation(source={:?}, f1={:.4}, iou={:.4}, coverage={:.4}, timelines={}, events={})",
            self.inner.source,
            self.inner.f1(),
            self.inner.iou(),
            self.inner.coverage(),
            self.inner.evaluated_timelines,
            self.inner.events.len(),
        )
    }
}

/// 数据集级评测的组合结果。
#[pyclass(name = "DatasetEvaluation", frozen)]
#[derive(Clone)]
pub(super) struct PyDatasetEvaluation {
    pub(super) inner: RustDatasetEvaluation,
}

#[pymethods]
impl PyDatasetEvaluation {
    /// 输入文档数量。
    #[getter]
    fn documents(&self) -> usize {
        self.inner.documents
    }

    /// 输入 timeline 数量。
    #[getter]
    fn timelines(&self) -> usize {
        self.inner.timelines
    }

    /// 按 source 分组的 corpus 转写结果。
    #[getter]
    fn transcription(&self) -> BTreeMap<String, PyDatasetTranscriptionEvaluation> {
        self.inner
            .transcription
            .iter()
            .map(|(source, inner)| {
                (
                    source.clone(),
                    PyDatasetTranscriptionEvaluation {
                        inner: inner.clone(),
                    },
                )
            })
            .collect()
    }

    /// 按 source 分组的总体 Activity 结果。
    #[getter]
    fn activity(&self) -> BTreeMap<String, PyDatasetActivityEvaluation> {
        self.inner
            .activity
            .iter()
            .map(|(source, inner)| {
                (
                    source.clone(),
                    PyDatasetActivityEvaluation {
                        inner: inner.clone(),
                    },
                )
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetEvaluation(documents={}, timelines={}, transcription={}, activity={})",
            self.inner.documents,
            self.inner.timelines,
            self.inner.transcription.len(),
            self.inner.activity.len(),
        )
    }
}

/// 聚合内存中的多个 Audio。
///
/// Args:
///     docs: 要评测的 Audio 列表。
///     transcription: 转写来源或来源列表；省略时自动发现。
///     activity: Activity 来源或来源列表；省略时自动发现。
///     normalize: 是否在计算 CER 前执行中文文本标准化。
///     traditional_to_simple: 是否将繁体中文转换为简体中文。
///     full_to_half: 是否将全角字符转换为半角字符。
///     remove_erhua: 是否去除儿化音“儿”。
///     remove_interjections: 是否去除“嗯”“啊”“呃”等语气词。
///     remove_puncts: 是否去除标点符号。
///
/// Returns:
///     按任务和 source 分组的数据集级结果。
///
/// Raises:
///     AsrDataError: 没有可评测内容或显式 source 不存在。
///     TypeError: source 参数类型无效。
///     ValueError: source 为空字符串。
///
/// Notes:
///     每条 timeline 独立对齐后再累计统计量，不会跨文档拼接文本。
///
/// Examples:
///     >>> from asr_data import Audio, AudioSource, evaluate_dataset
///     >>> from asr_data.annotation import Transcription
///     >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
///     >>> timeline = doc.timeline("mono")
///     >>> _ = timeline.annotate_span(
///     ...     0, timeline.duration_ms, Transcription("你好"), is_reference=True
///     ... )
///     >>> _ = timeline.annotate_span(
///     ...     0, timeline.duration_ms, Transcription("你好"),
///     ...     is_reference=False, source="asr"
///     ... )
///     >>> evaluate_dataset([doc], transcription="asr").transcription["asr"].cer
///     0.0
#[pyfunction(name = "evaluate_dataset")]
#[pyo3(signature = (
    docs,
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
fn py_evaluate_dataset(
    py: Python<'_>,
    docs: Vec<Py<PyAudio>>,
    transcription: Option<&Bound<'_, PyAny>>,
    activity: Option<&Bound<'_, PyAny>>,
    normalize: bool,
    traditional_to_simple: bool,
    full_to_half: bool,
    remove_erhua: bool,
    remove_interjections: bool,
    remove_puncts: bool,
) -> PyResult<PyDatasetEvaluation> {
    let config = eval_config(
        transcription,
        activity,
        normalize,
        traditional_to_simple,
        full_to_half,
        remove_erhua,
        remove_interjections,
        remove_puncts,
    )?;
    let docs = docs
        .iter()
        .map(|doc| doc.bind(py).borrow().cloned_inner(py))
        .collect::<PyResult<Vec<_>>>()?;
    let inner = rust_evaluate_dataset(&docs, &config).map_err(py_error)?;
    Ok(PyDatasetEvaluation { inner })
}

pub(super) fn eval_config(
    transcription: Option<&Bound<'_, PyAny>>,
    activity: Option<&Bound<'_, PyAny>>,
    normalize: bool,
    traditional_to_simple: bool,
    full_to_half: bool,
    remove_erhua: bool,
    remove_interjections: bool,
    remove_puncts: bool,
) -> PyResult<TimelineEvalConfig> {
    Ok(TimelineEvalConfig {
        transcription_sources: extract_eval_sources(transcription, "transcription")?,
        activity_sources: extract_eval_sources(activity, "activity")?,
        transcription_normalization: if normalize {
            TranscriptionNormalization::ChineseTn
        } else {
            TranscriptionNormalization::None
        },
        traditional_to_simple,
        full_to_half,
        remove_erhua,
        remove_interjections,
        remove_puncts,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn transcription_eval_config(
    source: Option<&Bound<'_, PyAny>>,
    normalize: bool,
    traditional_to_simple: bool,
    full_to_half: bool,
    remove_erhua: bool,
    remove_interjections: bool,
    remove_puncts: bool,
) -> PyResult<TimelineEvalConfig> {
    Ok(TimelineEvalConfig {
        transcription_sources: Some(
            extract_eval_sources(source, "transcription")?.unwrap_or_default(),
        ),
        activity_sources: None,
        transcription_normalization: if normalize {
            TranscriptionNormalization::ChineseTn
        } else {
            TranscriptionNormalization::None
        },
        traditional_to_simple,
        full_to_half,
        remove_erhua,
        remove_interjections,
        remove_puncts,
    })
}

pub(super) fn activity_eval_config(
    source: Option<&Bound<'_, PyAny>>,
) -> PyResult<TimelineEvalConfig> {
    Ok(TimelineEvalConfig {
        transcription_sources: None,
        activity_sources: Some(extract_eval_sources(source, "activity")?.unwrap_or_default()),
        ..TimelineEvalConfig::default()
    })
}

pub(super) fn speaker_eval_sources(source: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    Ok(extract_eval_sources(source, "speaker")?.unwrap_or_default())
}

fn normalization_name(normalization: TranscriptionNormalization) -> &'static str {
    match normalization {
        TranscriptionNormalization::None => "none",
        TranscriptionNormalization::ChineseTn => "zh_tn",
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDatasetTranscriptionEvaluation>()?;
    module.add_class::<PyDatasetActivityEventEvaluation>()?;
    module.add_class::<PyDatasetActivityEvaluation>()?;
    module.add_class::<PyDatasetSpeakerEvaluation>()?;
    module.add_class::<PyDatasetEvaluation>()?;
    module.add_function(wrap_pyfunction!(py_evaluate_dataset, module)?)?;
    Ok(())
}
