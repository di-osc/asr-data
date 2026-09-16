use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::thread;

use thiserror::Error;

use crate::db::{AudioDb, AudioDbError, AudioQuery};
use crate::doc::Audio;
use crate::metrics::CerStats;
use crate::timeline::{
    Annotation, Timeline, TimelineEvalConfig, TimelineEvalError, TranscriptionNormalization,
    normalize_transcription_text,
};

/// 数据集评估失败：数据库、单条时间轴，或没有任何可评估标注。
#[derive(Debug, Error)]
pub enum DatasetEvalError {
    #[error(transparent)]
    Database(#[from] AudioDbError),
    #[error(transparent)]
    Timeline(#[from] TimelineEvalError),
    #[error("the dataset has no reference annotations with matching prediction sources")]
    NoEvaluableAnnotations,
}

/// 整个数据集上转写和活动检测的汇总结果。
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetEvaluation {
    /// 参与评估的文档数。
    pub documents: usize,
    /// 参与评估的时间轴数。
    pub timelines: usize,
    /// 按预测 source 汇总的转写指标。
    pub transcription: BTreeMap<String, DatasetTranscriptionEvaluation>,
    /// 按预测 source 汇总的活动检测指标。
    pub activity: BTreeMap<String, DatasetActivityEvaluation>,
}

/// 某个转写 source 在数据集上的 CER 与覆盖率。
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetTranscriptionEvaluation {
    /// 预测来源名称。
    pub source: String,
    pub evaluated_documents: usize,
    pub evaluated_timelines: usize,
    pub unannotated_timelines: usize,
    pub missing_predictions: usize,
    pub unannotated_ids: Vec<String>,
    pub missing_prediction_ids: Vec<String>,
    pub normalization: TranscriptionNormalization,
    pub stats: CerStats,
    pub hypothesis_chars: usize,
    pub exact_matches: usize,
}

impl DatasetTranscriptionEvaluation {
    /// 正确字符数：N - S - D。
    pub fn matches(&self) -> usize {
        self.stats
            .reference_chars
            .saturating_sub(self.stats.substitutions + self.stats.deletions)
    }

    /// 转写精确率：matches / (matches + S + I)。
    pub fn precision(&self) -> f64 {
        ratio(
            self.matches(),
            self.matches() + self.stats.substitutions + self.stats.insertions,
        )
    }

    /// 转写召回率：matches / N。
    pub fn recall(&self) -> f64 {
        ratio(self.matches(), self.stats.reference_chars)
    }

    /// 转写 F1。
    pub fn f1(&self) -> f64 {
        harmonic_mean(self.precision(), self.recall())
    }

    /// 字符错误率。
    pub fn cer(&self) -> f64 {
        self.stats.cer()
    }

    /// 整段完全匹配的时间轴比例。
    pub fn exact_match_rate(&self) -> f64 {
        ratio(self.exact_matches, self.evaluated_timelines)
    }

    /// 有预测的时间轴占「有参考」时间轴的比例。
    pub fn coverage(&self) -> f64 {
        ratio(
            self.evaluated_timelines,
            self.evaluated_timelines + self.missing_predictions,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetActivityEventEvaluation {
    pub event: String,
    pub evaluated_documents: usize,
    pub evaluated_timelines: usize,
    pub reference_ms: u64,
    pub predicted_ms: u64,
    pub true_positive_ms: u64,
    pub true_negative_ms: u64,
    pub false_positive_ms: u64,
    pub false_negative_ms: u64,
}

impl DatasetActivityEventEvaluation {
    /// 事件区间精确率。
    pub fn precision(&self) -> f64 {
        interval_precision(self.true_positive_ms, self.false_positive_ms)
    }

    /// 事件区间召回率。
    pub fn recall(&self) -> f64 {
        interval_recall(self.true_positive_ms, self.false_negative_ms)
    }

    /// 事件区间 F1。
    pub fn f1(&self) -> f64 {
        harmonic_mean(self.precision(), self.recall())
    }

    /// 事件区间 IoU。
    pub fn iou(&self) -> f64 {
        interval_iou(
            self.true_positive_ms,
            self.false_positive_ms,
            self.false_negative_ms,
        )
    }
}

/// 某个活动检测 source 的合并区间指标，以及按事件拆分的结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetActivityEvaluation {
    pub source: String,
    pub evaluated_documents: usize,
    pub evaluated_timelines: usize,
    pub unannotated_timelines: usize,
    pub missing_predictions: usize,
    pub unannotated_ids: Vec<String>,
    pub missing_prediction_ids: Vec<String>,
    pub reference_ms: u64,
    pub predicted_ms: u64,
    pub true_positive_ms: u64,
    pub true_negative_ms: u64,
    pub false_positive_ms: u64,
    pub false_negative_ms: u64,
    pub events: BTreeMap<String, DatasetActivityEventEvaluation>,
}

/// 某个说话人 source 的 DER 相关毫秒统计。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetSpeakerEvaluation {
    pub source: String,
    pub evaluated_documents: usize,
    pub evaluated_timelines: usize,
    pub unannotated_timelines: usize,
    pub missing_predictions: usize,
    pub unannotated_ids: Vec<String>,
    pub missing_prediction_ids: Vec<String>,
    pub reference_speaker_ms: u64,
    pub predicted_speaker_ms: u64,
    pub correct_speaker_ms: u64,
    pub missed_speaker_ms: u64,
    pub false_alarm_ms: u64,
    pub speaker_confusion_ms: u64,
}

impl DatasetSpeakerEvaluation {
    /// 说话人错误率：(漏检 + 虚警 + 混淆) / 参考说话人时长。
    pub fn der(&self) -> f64 {
        ratio(
            self.missed_speaker_ms
                .saturating_add(self.false_alarm_ms)
                .saturating_add(self.speaker_confusion_ms) as usize,
            self.reference_speaker_ms as usize,
        )
    }

    /// 有预测的时间轴占「有参考」时间轴的比例。
    pub fn coverage(&self) -> f64 {
        ratio(
            self.evaluated_timelines,
            self.evaluated_timelines + self.missing_predictions,
        )
    }
}

impl DatasetActivityEvaluation {
    /// 合并活动区间的精确率。
    pub fn precision(&self) -> f64 {
        interval_precision(self.true_positive_ms, self.false_positive_ms)
    }

    /// 合并活动区间的召回率。
    pub fn recall(&self) -> f64 {
        interval_recall(self.true_positive_ms, self.false_negative_ms)
    }

    /// 合并活动区间的 F1。
    pub fn f1(&self) -> f64 {
        harmonic_mean(self.precision(), self.recall())
    }

    /// 合并活动区间的 IoU。
    pub fn iou(&self) -> f64 {
        interval_iou(
            self.true_positive_ms,
            self.false_positive_ms,
            self.false_negative_ms,
        )
    }

    /// 有预测的时间轴占「有参考」时间轴的比例。
    pub fn coverage(&self) -> f64 {
        ratio(
            self.evaluated_timelines,
            self.evaluated_timelines + self.missing_predictions,
        )
    }
}

/// 流式累加多条时间轴评估结果的计算器。
#[derive(Debug)]
pub struct DatasetEvaluator {
    config: TimelineEvalConfig,
    transcription_selection: Option<Vec<String>>,
    activity_selection: Option<Vec<String>>,
    documents: usize,
    timelines: usize,
    transcription_eligible: BTreeSet<String>,
    activity_eligible: BTreeSet<String>,
    transcription_unannotated: BTreeSet<String>,
    activity_unannotated: BTreeSet<String>,
    transcription: BTreeMap<String, TranscriptionAccumulator>,
    activity: BTreeMap<String, ActivityAccumulator>,
    normalization_cache: HashMap<String, String>,
}

impl DatasetEvaluator {
    /// 按配置创建累加器；未指定 source 时自动评估全部可对齐来源。
    pub fn new(config: TimelineEvalConfig) -> Self {
        let auto_all = config.transcription_sources.is_none() && config.activity_sources.is_none();
        let transcription_selection = if auto_all {
            Some(Vec::new())
        } else {
            config.transcription_sources.clone()
        };
        let activity_selection = if auto_all {
            Some(Vec::new())
        } else {
            config.activity_sources.clone()
        };
        let transcription = selected_accumulators(transcription_selection.as_deref());
        let activity = selected_accumulators(activity_selection.as_deref());
        Self {
            config,
            transcription_selection,
            activity_selection,
            documents: 0,
            timelines: 0,
            transcription_eligible: BTreeSet::new(),
            activity_eligible: BTreeSet::new(),
            transcription_unannotated: BTreeSet::new(),
            activity_unannotated: BTreeSet::new(),
            transcription,
            activity,
            normalization_cache: HashMap::new(),
        }
    }

    /// 把一篇文档里的全部时间轴累加进评估。
    pub fn push(&mut self, doc: &Audio) -> Result<(), DatasetEvalError> {
        self.prewarm_normalization_cache([doc])?;
        let result = self.push_cached(doc);
        self.normalization_cache.clear();
        result
    }

    /// 使用共享归一化缓存评估一篇文档。
    fn push_cached(&mut self, doc: &Audio) -> Result<(), DatasetEvalError> {
        self.documents += 1;
        for (channel, timeline) in doc.timelines() {
            self.timelines += 1;
            let timeline_key = format!("{}:{}", doc.id, channel.name());
            self.push_transcription(doc, timeline, &timeline_key)?;
            self.push_activity(doc, timeline, &timeline_key)?;
        }
        Ok(())
    }

    /// 并行预热转写归一化后再逐篇累加。
    fn push_batch(&mut self, docs: &[&Audio]) -> Result<(), DatasetEvalError> {
        self.prewarm_normalization_cache(docs.iter().copied())?;
        let result = docs.iter().try_for_each(|doc| self.push_cached(doc));
        self.normalization_cache.clear();
        result
    }

    /// 并行归一化本批文档里出现过的转写文本，写入缓存。
    fn prewarm_normalization_cache<'a>(
        &mut self,
        docs: impl IntoIterator<Item = &'a Audio>,
    ) -> Result<(), DatasetEvalError> {
        if self.config.transcription_normalization == TranscriptionNormalization::None {
            return Ok(());
        }
        let Some(selection) = self.transcription_selection.as_deref() else {
            return Ok(());
        };
        let mut seen = HashSet::new();
        let mut texts = Vec::new();
        for doc in docs {
            for timeline in doc.timelines().values() {
                if !timeline.reference.iter().any(is_text_annotation) {
                    continue;
                }
                let sources = sources_for_timeline(selection, &timeline.transcription_sources());
                if sources.is_empty() {
                    continue;
                }
                let reference = timeline.reference_transcript().text;
                if !self.normalization_cache.contains_key(&reference)
                    && seen.insert(reference.clone())
                {
                    texts.push(reference);
                }
                for source in sources {
                    let hypothesis = timeline.prediction_transcript(&source).text;
                    if !self.normalization_cache.contains_key(&hypothesis)
                        && seen.insert(hypothesis.clone())
                    {
                        texts.push(hypothesis);
                    }
                }
            }
        }
        let normalized = normalize_transcriptions_parallel(
            &texts,
            self.config.transcription_normalization,
            self.config.chinese_tn_options(),
        )
        .map_err(TimelineEvalError::from)?;
        self.normalization_cache
            .extend(texts.into_iter().zip(normalized));
        Ok(())
    }

    /// 汇总累加器；没有任何可评估标注时返回错误。
    pub fn finish(self) -> Result<DatasetEvaluation, DatasetEvalError> {
        let transcription = finish_transcription(
            self.transcription,
            &self.transcription_eligible,
            &self.transcription_unannotated,
            self.config.transcription_normalization,
        )?;
        let activity = finish_activity(
            self.activity,
            &self.activity_eligible,
            &self.activity_unannotated,
        )?;
        if transcription.is_empty() && activity.is_empty() {
            return Err(DatasetEvalError::NoEvaluableAnnotations);
        }
        Ok(DatasetEvaluation {
            documents: self.documents,
            timelines: self.timelines,
            transcription,
            activity,
        })
    }

    /// 把一条时间轴的转写结果累加到对应 source。
    fn push_transcription(
        &mut self,
        doc: &Audio,
        timeline: &Timeline,
        timeline_key: &str,
    ) -> Result<(), DatasetEvalError> {
        let Some(selection) = self.transcription_selection.as_deref() else {
            return Ok(());
        };
        let has_reference = timeline.reference.iter().any(is_text_annotation);
        if !has_reference {
            self.transcription_unannotated
                .insert(timeline_key.to_owned());
            return Ok(());
        }
        self.transcription_eligible.insert(timeline_key.to_owned());
        let available = timeline.transcription_sources();
        let sources = sources_for_timeline(selection, &available);
        if sources.is_empty() {
            return Ok(());
        }
        let mut results = timeline
            .evaluate_with_normalization_cache(
                &TimelineEvalConfig::new()
                    .with_transcriptions(sources.iter().cloned())
                    .with_transcription_normalization(self.config.transcription_normalization),
                &mut self.normalization_cache,
            )?
            .transcription;
        for source in sources {
            let result = results
                .remove(&source)
                .expect("a selected transcription source produced a result");
            self.transcription.entry(source.clone()).or_default().add(
                &doc.id,
                timeline_key,
                &result,
            );
        }
        Ok(())
    }

    /// 把一条时间轴的活动检测结果累加到对应 source。
    fn push_activity(
        &mut self,
        doc: &Audio,
        timeline: &Timeline,
        timeline_key: &str,
    ) -> Result<(), DatasetEvalError> {
        let Some(selection) = self.activity_selection.as_deref() else {
            return Ok(());
        };
        let has_reference = timeline
            .reference
            .iter()
            .any(|annotation| matches!(annotation.annotation, Annotation::Activity(_)));
        if !has_reference {
            self.activity_unannotated.insert(timeline_key.to_owned());
            return Ok(());
        }
        self.activity_eligible.insert(timeline_key.to_owned());
        let available = timeline.activity_sources();
        for source in sources_for_timeline(selection, &available) {
            let mut result = timeline
                .eval(&TimelineEvalConfig::new().with_activity(&source))?
                .activity;
            let result = result
                .remove(&source)
                .expect("a selected activity source produced a result");
            self.activity
                .entry(source.clone())
                .or_default()
                .add(&doc.id, timeline_key, &result);
        }
        Ok(())
    }
}

/// 按查询条件读取数据库并评估全部命中文档。
pub fn evaluate_dataset<'a>(
    docs: impl IntoIterator<Item = &'a Audio>,
    config: &TimelineEvalConfig,
) -> Result<DatasetEvaluation, DatasetEvalError> {
    let mut evaluator = DatasetEvaluator::new(config.clone());
    let docs = docs.into_iter().collect::<Vec<_>>();
    for batch in docs.chunks(100) {
        evaluator.push_batch(batch)?;
    }
    evaluator.finish()
}

impl AudioDb {
    /// 用默认配置评估时间轴。
    pub fn eval(
        &self,
        query: &AudioQuery,
        config: &TimelineEvalConfig,
    ) -> Result<DatasetEvaluation, DatasetEvalError> {
        let mut evaluator = DatasetEvaluator::new(config.clone());
        let mut page_query = query.clone();
        page_query.limit = page_query.limit.max(1);
        loop {
            let page = self.query(&page_query)?;
            if page.is_empty() {
                break;
            }
            page_query.after = page.last().map(Audio::audio_id);
            let docs = page.iter().collect::<Vec<_>>();
            evaluator.push_batch(&docs)?;
            if page.len() < page_query.limit {
                break;
            }
        }
        evaluator.finish()
    }

    /// 评估数据库中的说话人任务。
    pub fn eval_speaker(
        &self,
        query: &AudioQuery,
        sources: &[String],
    ) -> Result<BTreeMap<String, DatasetSpeakerEvaluation>, DatasetEvalError> {
        let mut evaluator = SpeakerDatasetEvaluator::new(sources);
        let mut page_query = query.clone();
        page_query.limit = page_query.limit.max(1);
        loop {
            let page = self.query(&page_query)?;
            if page.is_empty() {
                break;
            }
            page_query.after = page.last().map(Audio::audio_id);
            for doc in &page {
                evaluator.push(doc);
            }
            if page.len() < page_query.limit {
                break;
            }
        }
        evaluator.finish()
    }
}

#[derive(Debug, Default)]
/// 单条时间轴上一个说话人 source 的毫秒统计。
struct SpeakerStats {
    reference_speaker_ms: u64,
    predicted_speaker_ms: u64,
    correct_speaker_ms: u64,
    missed_speaker_ms: u64,
    false_alarm_ms: u64,
    speaker_confusion_ms: u64,
}

#[derive(Debug, Default)]
/// 跨文档累加说话人评估。
struct SpeakerAccumulator {
    evaluated_documents: BTreeSet<String>,
    evaluated_timelines: BTreeSet<String>,
    stats: SpeakerStats,
}

impl SpeakerAccumulator {
    /// 累加一条时间轴的说话人统计。
    fn add(&mut self, audio_id: &str, timeline_id: &str, stats: SpeakerStats) {
        self.evaluated_documents.insert(audio_id.to_owned());
        self.evaluated_timelines.insert(timeline_id.to_owned());
        self.stats.reference_speaker_ms = self
            .stats
            .reference_speaker_ms
            .saturating_add(stats.reference_speaker_ms);
        self.stats.predicted_speaker_ms = self
            .stats
            .predicted_speaker_ms
            .saturating_add(stats.predicted_speaker_ms);
        self.stats.correct_speaker_ms = self
            .stats
            .correct_speaker_ms
            .saturating_add(stats.correct_speaker_ms);
        self.stats.missed_speaker_ms = self
            .stats
            .missed_speaker_ms
            .saturating_add(stats.missed_speaker_ms);
        self.stats.false_alarm_ms = self
            .stats
            .false_alarm_ms
            .saturating_add(stats.false_alarm_ms);
        self.stats.speaker_confusion_ms = self
            .stats
            .speaker_confusion_ms
            .saturating_add(stats.speaker_confusion_ms);
    }
}

/// 按 source 收集说话人评估。
struct SpeakerDatasetEvaluator {
    selection: Vec<String>,
    eligible: BTreeSet<String>,
    unannotated: BTreeSet<String>,
    accumulators: BTreeMap<String, SpeakerAccumulator>,
}

impl SpeakerDatasetEvaluator {
    fn new(sources: &[String]) -> Self {
        Self {
            selection: sources.to_vec(),
            eligible: BTreeSet::new(),
            unannotated: BTreeSet::new(),
            accumulators: sources
                .iter()
                .cloned()
                .map(|source| (source, SpeakerAccumulator::default()))
                .collect(),
        }
    }

    /// 把一篇文档里的全部时间轴累加进评估。
    fn push(&mut self, doc: &Audio) {
        for (channel, timeline) in doc.timelines() {
            let timeline_id = format!("{}:{}", doc.id, channel.name());
            if !timeline.reference.iter().any(is_speaker_annotation) {
                self.unannotated.insert(timeline_id);
                continue;
            }
            self.eligible.insert(timeline_id.clone());
            let available = timeline
                .prediction
                .iter()
                .filter(|span| is_speaker_annotation(span))
                .filter_map(|span| span.source.clone())
                .collect::<BTreeSet<_>>();
            let sources = sources_for_timeline(&self.selection, &available);
            for source in sources {
                let stats = evaluate_speakers(timeline, &source);
                self.accumulators
                    .entry(source)
                    .or_default()
                    .add(&doc.id, &timeline_id, stats);
            }
        }
    }

    /// 汇总累加器；没有任何可评估标注时返回错误。
    fn finish(self) -> Result<BTreeMap<String, DatasetSpeakerEvaluation>, DatasetEvalError> {
        if self.eligible.is_empty() || self.accumulators.is_empty() {
            return Err(DatasetEvalError::NoEvaluableAnnotations);
        }
        self.accumulators
            .into_iter()
            .map(|(source, accumulator)| {
                if accumulator.evaluated_timelines.is_empty() {
                    return Err(TimelineEvalError::MissingPrediction {
                        kind: "speaker",
                        prediction_source: source,
                    }
                    .into());
                }
                let missing_prediction_ids = self
                    .eligible
                    .difference(&accumulator.evaluated_timelines)
                    .cloned()
                    .collect::<Vec<_>>();
                Ok((
                    source.clone(),
                    DatasetSpeakerEvaluation {
                        source,
                        evaluated_documents: accumulator.evaluated_documents.len(),
                        evaluated_timelines: accumulator.evaluated_timelines.len(),
                        unannotated_timelines: self.unannotated.len(),
                        missing_predictions: missing_prediction_ids.len(),
                        unannotated_ids: self.unannotated.iter().cloned().collect(),
                        missing_prediction_ids,
                        reference_speaker_ms: accumulator.stats.reference_speaker_ms,
                        predicted_speaker_ms: accumulator.stats.predicted_speaker_ms,
                        correct_speaker_ms: accumulator.stats.correct_speaker_ms,
                        missed_speaker_ms: accumulator.stats.missed_speaker_ms,
                        false_alarm_ms: accumulator.stats.false_alarm_ms,
                        speaker_confusion_ms: accumulator.stats.speaker_confusion_ms,
                    },
                ))
            })
            .collect()
    }
}

/// 是否为说话人标注。
fn is_speaker_annotation(span: &crate::timeline::TimeSpan) -> bool {
    matches!(span.annotation, Annotation::Speaker(_))
}

/// 用最大权匹配对齐参考/预测说话人区间，计算 DER 分量。
fn evaluate_speakers(timeline: &Timeline, source: &str) -> SpeakerStats {
    let reference = timeline
        .reference
        .iter()
        .filter_map(speaker_span)
        .collect::<Vec<_>>();
    let prediction = timeline
        .prediction
        .iter()
        .filter(|span| span.source.as_deref() == Some(source))
        .filter_map(speaker_span)
        .collect::<Vec<_>>();
    let reference_labels = reference
        .iter()
        .map(|(label, _, _)| (*label).to_owned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let prediction_labels = prediction
        .iter()
        .map(|(label, _, _)| (*label).to_owned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let reference_index = reference_labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.as_str(), index))
        .collect::<HashMap<_, _>>();
    let prediction_index = prediction_labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut boundaries = reference
        .iter()
        .chain(&prediction)
        .flat_map(|(_, start, end)| [*start, *end])
        .collect::<Vec<_>>();
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut overlap = vec![vec![0_u64; prediction_labels.len()]; reference_labels.len()];
    let mut stats = SpeakerStats::default();
    for window in boundaries.windows(2) {
        let start = window[0];
        let end = window[1];
        let duration = end.saturating_sub(start);
        if duration == 0 {
            continue;
        }
        let active_reference = reference
            .iter()
            .filter(|(_, span_start, span_end)| *span_start < end && *span_end > start)
            .map(|(label, _, _)| reference_index[label])
            .collect::<Vec<_>>();
        let active_prediction = prediction
            .iter()
            .filter(|(_, span_start, span_end)| *span_start < end && *span_end > start)
            .map(|(label, _, _)| prediction_index[label])
            .collect::<Vec<_>>();
        let reference_count = active_reference.len() as u64;
        let prediction_count = active_prediction.len() as u64;
        stats.reference_speaker_ms = stats
            .reference_speaker_ms
            .saturating_add(duration.saturating_mul(reference_count));
        stats.predicted_speaker_ms = stats
            .predicted_speaker_ms
            .saturating_add(duration.saturating_mul(prediction_count));
        stats.missed_speaker_ms = stats.missed_speaker_ms.saturating_add(
            duration.saturating_mul(reference_count.saturating_sub(prediction_count)),
        );
        stats.false_alarm_ms = stats.false_alarm_ms.saturating_add(
            duration.saturating_mul(prediction_count.saturating_sub(reference_count)),
        );
        stats.speaker_confusion_ms = stats
            .speaker_confusion_ms
            .saturating_add(duration.saturating_mul(reference_count.min(prediction_count)));
        for reference in &active_reference {
            for prediction in &active_prediction {
                overlap[*reference][*prediction] =
                    overlap[*reference][*prediction].saturating_add(duration);
            }
        }
    }
    stats.correct_speaker_ms = maximum_weight_assignment(&overlap);
    stats.speaker_confusion_ms = stats
        .speaker_confusion_ms
        .saturating_sub(stats.correct_speaker_ms);
    stats
}

/// 取出说话人姓名和起止毫秒。
fn speaker_span(span: &crate::timeline::TimeSpan) -> Option<(&str, u64, u64)> {
    let Annotation::Speaker(speaker) = &span.annotation else {
        return None;
    };
    Some((
        speaker.name.as_str(),
        span.range.start_ms as u64,
        span.range.end_ms as u64,
    ))
}

/// 匈牙利算法求二分图最大权和，用于说话人对齐。
fn maximum_weight_assignment(weights: &[Vec<u64>]) -> u64 {
    let rows = weights.len();
    let columns = weights.first().map_or(0, Vec::len);
    let size = rows.max(columns);
    if size == 0 {
        return 0;
    }
    let maximum = weights.iter().flatten().copied().max().unwrap_or_default() as i128;
    let mut u = vec![0_i128; size + 1];
    let mut v = vec![0_i128; size + 1];
    let mut p = vec![0_usize; size + 1];
    let mut way = vec![0_usize; size + 1];
    for row in 1..=size {
        p[0] = row;
        let mut column = 0;
        let mut min_value = vec![i128::MAX; size + 1];
        let mut used = vec![false; size + 1];
        loop {
            used[column] = true;
            let current_row = p[column];
            let mut delta = i128::MAX;
            let mut next_column = 0;
            for candidate in 1..=size {
                if used[candidate] {
                    continue;
                }
                let weight = if current_row <= rows && candidate <= columns {
                    weights[current_row - 1][candidate - 1] as i128
                } else {
                    0
                };
                let cost = maximum - weight - u[current_row] - v[candidate];
                if cost < min_value[candidate] {
                    min_value[candidate] = cost;
                    way[candidate] = column;
                }
                if min_value[candidate] < delta {
                    delta = min_value[candidate];
                    next_column = candidate;
                }
            }
            for candidate in 0..=size {
                if used[candidate] {
                    u[p[candidate]] += delta;
                    v[candidate] -= delta;
                } else {
                    min_value[candidate] -= delta;
                }
            }
            column = next_column;
            if p[column] == 0 {
                break;
            }
        }
        loop {
            let previous = way[column];
            p[column] = p[previous];
            column = previous;
            if column == 0 {
                break;
            }
        }
    }
    (1..=size)
        .filter_map(|column| {
            let row = p[column];
            (row > 0 && row <= rows && column <= columns).then_some(weights[row - 1][column - 1])
        })
        .sum()
}

/// 在线程池里并行做转写文本归一化。
fn normalize_transcriptions_parallel(
    texts: &[String],
    normalization: TranscriptionNormalization,
    options: crate::metrics::ChineseTextNormalizationOptions,
) -> Result<Vec<String>, crate::metrics::TextNormalizationError> {
    let available = thread::available_parallelism().map_or(1, usize::from);
    let workers = available.min(4).min(texts.len());
    if workers <= 1 {
        return texts
            .iter()
            .map(|text| normalize_transcription_text(text, normalization, options))
            .collect();
    }
    let chunk_size = texts.len().div_ceil(workers);
    thread::scope(|scope| {
        let handles = texts
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|text| normalize_transcription_text(text, normalization, options))
                        .collect::<Result<Vec<_>, _>>()
                })
            })
            .collect::<Vec<_>>();
        let mut normalized = Vec::with_capacity(texts.len());
        for handle in handles {
            normalized.extend(handle.join().expect("normalization worker panicked")?);
        }
        Ok(normalized)
    })
}

#[derive(Debug, Default)]
/// 跨文档累加某个转写 source 的 CER。
struct TranscriptionAccumulator {
    evaluated_documents: BTreeSet<String>,
    evaluated_timelines: BTreeSet<String>,
    stats: CerStats,
    hypothesis_chars: usize,
    exact_matches: usize,
}

impl TranscriptionAccumulator {
    fn add(
        &mut self,
        audio_id: &str,
        timeline_id: &str,
        result: &crate::timeline::TranscriptionEvaluation,
    ) {
        self.evaluated_documents.insert(audio_id.to_owned());
        self.evaluated_timelines.insert(timeline_id.to_owned());
        self.stats.substitutions += result.stats.substitutions;
        self.stats.deletions += result.stats.deletions;
        self.stats.insertions += result.stats.insertions;
        self.stats.reference_chars += result.stats.reference_chars;
        self.hypothesis_chars += result.hypothesis_chars;
        self.exact_matches += usize::from(result.exact_match());
    }
}

#[derive(Debug, Default)]
/// 跨文档累加某个活动事件的区间计数。
struct ActivityEventAccumulator {
    evaluated_documents: BTreeSet<String>,
    evaluated_timelines: BTreeSet<String>,
    reference_ms: u64,
    predicted_ms: u64,
    true_positive_ms: u64,
    true_negative_ms: u64,
    false_positive_ms: u64,
    false_negative_ms: u64,
}

impl ActivityEventAccumulator {
    fn add(
        &mut self,
        audio_id: &str,
        timeline_id: &str,
        result: &crate::timeline::ActivityEventEvaluation,
    ) {
        self.evaluated_documents.insert(audio_id.to_owned());
        self.evaluated_timelines.insert(timeline_id.to_owned());
        self.reference_ms = self.reference_ms.saturating_add(result.reference_ms);
        self.predicted_ms = self.predicted_ms.saturating_add(result.predicted_ms);
        self.true_positive_ms = self
            .true_positive_ms
            .saturating_add(result.true_positive_ms);
        self.true_negative_ms = self
            .true_negative_ms
            .saturating_add(result.true_negative_ms);
        self.false_positive_ms = self
            .false_positive_ms
            .saturating_add(result.false_positive_ms);
        self.false_negative_ms = self
            .false_negative_ms
            .saturating_add(result.false_negative_ms);
    }
}

#[derive(Debug, Default)]
/// 跨文档累加某个活动 source 的合并区间与事件。
struct ActivityAccumulator {
    evaluated_documents: BTreeSet<String>,
    evaluated_timelines: BTreeSet<String>,
    reference_ms: u64,
    predicted_ms: u64,
    true_positive_ms: u64,
    true_negative_ms: u64,
    false_positive_ms: u64,
    false_negative_ms: u64,
    events: BTreeMap<String, ActivityEventAccumulator>,
}

impl ActivityAccumulator {
    fn add(
        &mut self,
        audio_id: &str,
        timeline_id: &str,
        result: &crate::timeline::ActivityEvaluation,
    ) {
        self.evaluated_documents.insert(audio_id.to_owned());
        self.evaluated_timelines.insert(timeline_id.to_owned());
        self.reference_ms = self.reference_ms.saturating_add(result.reference_ms);
        self.predicted_ms = self.predicted_ms.saturating_add(result.predicted_ms);
        self.true_positive_ms = self
            .true_positive_ms
            .saturating_add(result.true_positive_ms);
        self.true_negative_ms = self
            .true_negative_ms
            .saturating_add(result.true_negative_ms);
        self.false_positive_ms = self
            .false_positive_ms
            .saturating_add(result.false_positive_ms);
        self.false_negative_ms = self
            .false_negative_ms
            .saturating_add(result.false_negative_ms);
        for (event, evaluation) in &result.events {
            self.events
                .entry(event.clone())
                .or_default()
                .add(audio_id, timeline_id, evaluation);
        }
    }
}

/// 按选择列表预创建累加器；空选择表示稍后按出现的 source 动态创建。
fn selected_accumulators<T: Default>(selection: Option<&[String]>) -> BTreeMap<String, T> {
    selection
        .into_iter()
        .flatten()
        .map(|source| (source.clone(), T::default()))
        .collect()
}

/// 空选择表示使用时间轴上全部可用 source。
fn sources_for_timeline(selection: &[String], available: &BTreeSet<String>) -> BTreeSet<String> {
    if selection.is_empty() {
        available.clone()
    } else {
        selection
            .iter()
            .filter(|source| available.contains(*source))
            .cloned()
            .collect()
    }
}

/// 把转写累加器收成最终评估结果。
fn finish_transcription(
    accumulators: BTreeMap<String, TranscriptionAccumulator>,
    eligible: &BTreeSet<String>,
    unannotated: &BTreeSet<String>,
    normalization: TranscriptionNormalization,
) -> Result<BTreeMap<String, DatasetTranscriptionEvaluation>, DatasetEvalError> {
    if eligible.is_empty() {
        return Ok(BTreeMap::new());
    }
    accumulators
        .into_iter()
        .map(|(source, accumulator)| {
            if accumulator.evaluated_timelines.is_empty() {
                return Err(TimelineEvalError::MissingPrediction {
                    kind: "transcription",
                    prediction_source: source,
                }
                .into());
            }
            let missing_prediction_ids = eligible
                .difference(&accumulator.evaluated_timelines)
                .cloned()
                .collect::<Vec<_>>();
            let result = DatasetTranscriptionEvaluation {
                source: source.clone(),
                evaluated_documents: accumulator.evaluated_documents.len(),
                evaluated_timelines: accumulator.evaluated_timelines.len(),
                unannotated_timelines: unannotated.len(),
                missing_predictions: missing_prediction_ids.len(),
                unannotated_ids: unannotated.iter().cloned().collect(),
                missing_prediction_ids,
                normalization,
                stats: accumulator.stats,
                hypothesis_chars: accumulator.hypothesis_chars,
                exact_matches: accumulator.exact_matches,
            };
            Ok((source, result))
        })
        .collect()
}

/// 把活动累加器收成最终评估结果。
fn finish_activity(
    accumulators: BTreeMap<String, ActivityAccumulator>,
    eligible: &BTreeSet<String>,
    unannotated: &BTreeSet<String>,
) -> Result<BTreeMap<String, DatasetActivityEvaluation>, DatasetEvalError> {
    if eligible.is_empty() {
        return Ok(BTreeMap::new());
    }
    accumulators
        .into_iter()
        .map(|(source, accumulator)| {
            if accumulator.evaluated_timelines.is_empty() {
                return Err(TimelineEvalError::MissingPrediction {
                    kind: "activity",
                    prediction_source: source,
                }
                .into());
            }
            let missing_prediction_ids = eligible
                .difference(&accumulator.evaluated_timelines)
                .cloned()
                .collect::<Vec<_>>();
            let events = accumulator
                .events
                .into_iter()
                .map(|(event, value)| {
                    (
                        event.clone(),
                        DatasetActivityEventEvaluation {
                            event,
                            evaluated_documents: value.evaluated_documents.len(),
                            evaluated_timelines: value.evaluated_timelines.len(),
                            reference_ms: value.reference_ms,
                            predicted_ms: value.predicted_ms,
                            true_positive_ms: value.true_positive_ms,
                            true_negative_ms: value.true_negative_ms,
                            false_positive_ms: value.false_positive_ms,
                            false_negative_ms: value.false_negative_ms,
                        },
                    )
                })
                .collect();
            let result = DatasetActivityEvaluation {
                source: source.clone(),
                evaluated_documents: accumulator.evaluated_documents.len(),
                evaluated_timelines: accumulator.evaluated_timelines.len(),
                unannotated_timelines: unannotated.len(),
                missing_predictions: missing_prediction_ids.len(),
                unannotated_ids: unannotated.iter().cloned().collect(),
                missing_prediction_ids,
                reference_ms: accumulator.reference_ms,
                predicted_ms: accumulator.predicted_ms,
                true_positive_ms: accumulator.true_positive_ms,
                true_negative_ms: accumulator.true_negative_ms,
                false_positive_ms: accumulator.false_positive_ms,
                false_negative_ms: accumulator.false_negative_ms,
                events,
            };
            Ok((source, result))
        })
        .collect()
}

/// 是否为转写或句子类文本标注。
fn is_text_annotation(annotation: &crate::timeline::TimeSpan) -> bool {
    match &annotation.annotation {
        Annotation::Transcription(_) | Annotation::Sentence(_) => true,
        Annotation::Speaker(speaker) => speaker.transcription.is_some(),
        _ => false,
    }
}

/// numerator/denominator；分母为 0 时返回 0。
fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        if numerator == 0 { 1.0 } else { 0.0 }
    } else {
        numerator as f64 / denominator as f64
    }
}

/// 两个比率的调和平均；任一为 0 则返回 0。
fn harmonic_mean(left: f64, right: f64) -> f64 {
    if left + right == 0.0 {
        0.0
    } else {
        2.0 * left * right / (left + right)
    }
}

/// TP / (TP + FP)。
fn interval_precision(true_positive_ms: u64, false_positive_ms: u64) -> f64 {
    ratio(
        true_positive_ms as usize,
        (true_positive_ms + false_positive_ms) as usize,
    )
}

/// TP / (TP + FN)。
fn interval_recall(true_positive_ms: u64, false_negative_ms: u64) -> f64 {
    ratio(
        true_positive_ms as usize,
        (true_positive_ms + false_negative_ms) as usize,
    )
}

/// TP / (TP + FP + FN)。
fn interval_iou(true_positive_ms: u64, false_positive_ms: u64, false_negative_ms: u64) -> f64 {
    ratio(
        true_positive_ms as usize,
        (true_positive_ms + false_positive_ms + false_negative_ms) as usize,
    )
}
