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

#[derive(Debug, Error)]
pub enum DatasetEvalError {
    #[error(transparent)]
    Database(#[from] AudioDbError),
    #[error(transparent)]
    Timeline(#[from] TimelineEvalError),
    #[error("the dataset has no reference annotations with matching prediction sources")]
    NoEvaluableAnnotations,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetEvaluation {
    pub documents: usize,
    pub timelines: usize,
    pub transcription: BTreeMap<String, DatasetTranscriptionEvaluation>,
    pub activity: BTreeMap<String, DatasetActivityEvaluation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetTranscriptionEvaluation {
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
    pub fn matches(&self) -> usize {
        self.stats
            .reference_chars
            .saturating_sub(self.stats.substitutions + self.stats.deletions)
    }

    pub fn precision(&self) -> f64 {
        ratio(
            self.matches(),
            self.matches() + self.stats.substitutions + self.stats.insertions,
        )
    }

    pub fn recall(&self) -> f64 {
        ratio(self.matches(), self.stats.reference_chars)
    }

    pub fn f1(&self) -> f64 {
        harmonic_mean(self.precision(), self.recall())
    }

    pub fn cer(&self) -> f64 {
        self.stats.cer()
    }

    pub fn exact_match_rate(&self) -> f64 {
        ratio(self.exact_matches, self.evaluated_timelines)
    }

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
    pub fn precision(&self) -> f64 {
        interval_precision(self.true_positive_ms, self.false_positive_ms)
    }

    pub fn recall(&self) -> f64 {
        interval_recall(self.true_positive_ms, self.false_negative_ms)
    }

    pub fn f1(&self) -> f64 {
        harmonic_mean(self.precision(), self.recall())
    }

    pub fn iou(&self) -> f64 {
        interval_iou(
            self.true_positive_ms,
            self.false_positive_ms,
            self.false_negative_ms,
        )
    }
}

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
    pub fn der(&self) -> f64 {
        ratio(
            self.missed_speaker_ms
                .saturating_add(self.false_alarm_ms)
                .saturating_add(self.speaker_confusion_ms) as usize,
            self.reference_speaker_ms as usize,
        )
    }

    pub fn coverage(&self) -> f64 {
        ratio(
            self.evaluated_timelines,
            self.evaluated_timelines + self.missing_predictions,
        )
    }
}

impl DatasetActivityEvaluation {
    pub fn precision(&self) -> f64 {
        interval_precision(self.true_positive_ms, self.false_positive_ms)
    }

    pub fn recall(&self) -> f64 {
        interval_recall(self.true_positive_ms, self.false_negative_ms)
    }

    pub fn f1(&self) -> f64 {
        harmonic_mean(self.precision(), self.recall())
    }

    pub fn iou(&self) -> f64 {
        interval_iou(
            self.true_positive_ms,
            self.false_positive_ms,
            self.false_negative_ms,
        )
    }

    pub fn coverage(&self) -> f64 {
        ratio(
            self.evaluated_timelines,
            self.evaluated_timelines + self.missing_predictions,
        )
    }
}

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

    pub fn push(&mut self, doc: &Audio) -> Result<(), DatasetEvalError> {
        self.prewarm_normalization_cache([doc])?;
        let result = self.push_cached(doc);
        self.normalization_cache.clear();
        result
    }

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

    fn push_batch(&mut self, docs: &[&Audio]) -> Result<(), DatasetEvalError> {
        self.prewarm_normalization_cache(docs.iter().copied())?;
        let result = docs.iter().try_for_each(|doc| self.push_cached(doc));
        self.normalization_cache.clear();
        result
    }

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
struct SpeakerStats {
    reference_speaker_ms: u64,
    predicted_speaker_ms: u64,
    correct_speaker_ms: u64,
    missed_speaker_ms: u64,
    false_alarm_ms: u64,
    speaker_confusion_ms: u64,
}

#[derive(Debug, Default)]
struct SpeakerAccumulator {
    evaluated_documents: BTreeSet<String>,
    evaluated_timelines: BTreeSet<String>,
    stats: SpeakerStats,
}

impl SpeakerAccumulator {
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

fn is_speaker_annotation(span: &crate::timeline::TimeSpan) -> bool {
    matches!(span.annotation, Annotation::Speaker(_))
}

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

fn speaker_span(span: &crate::timeline::TimeSpan) -> Option<(&str, u64, u64)> {
    let Annotation::Speaker(speaker) = &span.annotation else {
        return None;
    };
    Some((speaker.name.as_str(), span.range.start.0, span.range.end.0))
}

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

fn selected_accumulators<T: Default>(selection: Option<&[String]>) -> BTreeMap<String, T> {
    selection
        .into_iter()
        .flatten()
        .map(|source| (source.clone(), T::default()))
        .collect()
}

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

fn is_text_annotation(annotation: &crate::timeline::TimeSpan) -> bool {
    match &annotation.annotation {
        Annotation::Transcription(_) | Annotation::Sentence(_) => true,
        Annotation::Speaker(speaker) => speaker.transcription.is_some(),
        _ => false,
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        if numerator == 0 { 1.0 } else { 0.0 }
    } else {
        numerator as f64 / denominator as f64
    }
}

fn harmonic_mean(left: f64, right: f64) -> f64 {
    if left + right == 0.0 {
        0.0
    } else {
        2.0 * left * right / (left + right)
    }
}

fn interval_precision(true_positive_ms: u64, false_positive_ms: u64) -> f64 {
    ratio(
        true_positive_ms as usize,
        (true_positive_ms + false_positive_ms) as usize,
    )
}

fn interval_recall(true_positive_ms: u64, false_negative_ms: u64) -> f64 {
    ratio(
        true_positive_ms as usize,
        (true_positive_ms + false_negative_ms) as usize,
    )
}

fn interval_iou(true_positive_ms: u64, false_positive_ms: u64, false_negative_ms: u64) -> f64 {
    ratio(
        true_positive_ms as usize,
        (true_positive_ms + false_positive_ms + false_negative_ms) as usize,
    )
}
