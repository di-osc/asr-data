use std::collections::BTreeMap;

use crate::{
    DatasetEvaluation as RustDatasetEvaluation,
    DatasetSpeechEvaluation as RustDatasetSpeechEvaluation,
    DatasetTranscriptionEvaluation as RustDatasetTranscriptionEvaluation, TimelineEvalConfig,
    TranscriptionNormalization, evaluate_dataset as rust_evaluate_dataset,
};
use pyo3::prelude::*;

use super::common::py_error;
use super::doc::PyAudioDoc;
use super::timeline::extract_eval_sources;

/// 单个 prediction source 的数据集转写聚合结果。
///
/// CER 由各 timeline 的编辑统计量累加后计算，不是单条 CER 的平均值。
#[pyclass(name = "DatasetTranscriptionEvaluation", frozen)]
#[derive(Clone)]
pub(super) struct PyDatasetTranscriptionEvaluation {
    inner: RustDatasetTranscriptionEvaluation,
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

/// 单个 prediction source 的数据集 Speech 聚合结果。
#[pyclass(name = "DatasetSpeechEvaluation", frozen)]
#[derive(Clone)]
pub(super) struct PyDatasetSpeechEvaluation {
    inner: RustDatasetSpeechEvaluation,
}

#[pymethods]
impl PyDatasetSpeechEvaluation {
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

    /// 缺少 Speech reference 的 timeline 数。
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

    /// 累计 reference 人声时长。
    #[getter]
    fn reference_ms(&self) -> u64 {
        self.inner.reference_ms
    }

    /// 累计 prediction 人声时长。
    #[getter]
    fn predicted_ms(&self) -> u64 {
        self.inner.predicted_ms
    }

    /// 累计正确人声时长。
    #[getter]
    fn true_positive_ms(&self) -> u64 {
        self.inner.true_positive_ms
    }

    /// 累计正确静音时长。
    #[getter]
    fn true_negative_ms(&self) -> u64 {
        self.inner.true_negative_ms
    }

    /// 累计误报人声时长。
    #[getter]
    fn false_positive_ms(&self) -> u64 {
        self.inner.false_positive_ms
    }

    /// 累计漏报人声时长。
    #[getter]
    fn false_negative_ms(&self) -> u64 {
        self.inner.false_negative_ms
    }

    /// 总体 Speech precision。
    #[getter]
    fn precision(&self) -> f64 {
        self.inner.precision()
    }

    /// 总体 Speech recall。
    #[getter]
    fn recall(&self) -> f64 {
        self.inner.recall()
    }

    /// 总体 Speech F1。
    #[getter]
    fn f1(&self) -> f64 {
        self.inner.f1()
    }

    /// 总体 Speech IoU。
    #[getter]
    fn iou(&self) -> f64 {
        self.inner.iou()
    }

    /// 具有 prediction 的已标注 timeline 占比。
    #[getter]
    fn coverage(&self) -> f64 {
        self.inner.coverage()
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetSpeechEvaluation(source={:?}, f1={:.4}, iou={:.4}, coverage={:.4}, timelines={})",
            self.inner.source,
            self.inner.f1(),
            self.inner.iou(),
            self.inner.coverage(),
            self.inner.evaluated_timelines,
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

    /// 按 source 分组的总体 Speech 结果。
    #[getter]
    fn speech(&self) -> BTreeMap<String, PyDatasetSpeechEvaluation> {
        self.inner
            .speech
            .iter()
            .map(|(source, inner)| {
                (
                    source.clone(),
                    PyDatasetSpeechEvaluation {
                        inner: inner.clone(),
                    },
                )
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetEvaluation(documents={}, timelines={}, transcription={}, speech={})",
            self.inner.documents,
            self.inner.timelines,
            self.inner.transcription.len(),
            self.inner.speech.len(),
        )
    }
}

/// 聚合内存中的多个 AudioDoc。
///
/// Args:
///     docs: 要评测的 AudioDoc 列表。
///     transcription: 转写来源或来源列表；省略时自动发现。
///     speech: Speech 来源或来源列表；省略时自动发现。
///     normalize: 是否在计算 CER 前执行中文文本标准化。
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
///     >>> from asr_data import AudioDoc, AudioSource, evaluate_dataset
///     >>> from asr_data.annotation import Transcription
///     >>> doc = AudioDoc(AudioSource.from_pcm(b"\0\0" * 10, 16000))
///     >>> timeline = doc.timeline("mono")
///     >>> _ = timeline.reference.add_transcription(
///     ...     0, timeline.duration_ms, Transcription("你好")
///     ... )
///     >>> _ = timeline.prediction.add_transcription(
///     ...     0, timeline.duration_ms, Transcription("你好"), source="asr"
///     ... )
///     >>> evaluate_dataset([doc], transcription="asr").transcription["asr"].cer
///     0.0
#[pyfunction(name = "evaluate_dataset")]
#[pyo3(signature = (docs, *, transcription=None, speech=None, normalize=true))]
fn py_evaluate_dataset(
    py: Python<'_>,
    docs: Vec<Py<PyAudioDoc>>,
    transcription: Option<&Bound<'_, PyAny>>,
    speech: Option<&Bound<'_, PyAny>>,
    normalize: bool,
) -> PyResult<PyDatasetEvaluation> {
    let config = eval_config(transcription, speech, normalize)?;
    let docs = docs
        .iter()
        .map(|doc| doc.bind(py).borrow().cloned_inner(py))
        .collect::<PyResult<Vec<_>>>()?;
    let inner = rust_evaluate_dataset(&docs, &config).map_err(py_error)?;
    Ok(PyDatasetEvaluation { inner })
}

pub(super) fn eval_config(
    transcription: Option<&Bound<'_, PyAny>>,
    speech: Option<&Bound<'_, PyAny>>,
    normalize: bool,
) -> PyResult<TimelineEvalConfig> {
    Ok(TimelineEvalConfig {
        transcription_sources: extract_eval_sources(transcription, "transcription")?,
        speech_sources: extract_eval_sources(speech, "speech")?,
        transcription_normalization: if normalize {
            TranscriptionNormalization::ChineseTn
        } else {
            TranscriptionNormalization::None
        },
    })
}

fn normalization_name(normalization: TranscriptionNormalization) -> &'static str {
    match normalization {
        TranscriptionNormalization::None => "none",
        TranscriptionNormalization::ChineseTn => "zh_tn",
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDatasetTranscriptionEvaluation>()?;
    module.add_class::<PyDatasetSpeechEvaluation>()?;
    module.add_class::<PyDatasetEvaluation>()?;
    module.add_function(wrap_pyfunction!(py_evaluate_dataset, module)?)?;
    Ok(())
}
