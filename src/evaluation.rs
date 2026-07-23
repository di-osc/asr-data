use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::db::{AudioDb, AudioDbError, AudioQuery};
use crate::doc::AudioDoc;
use crate::metrics::CerStats;
use crate::timeline::{
    AnnotationPayload, Timeline, TimelineEvalConfig, TimelineEvalError, TranscriptionNormalization,
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
    pub speech: BTreeMap<String, DatasetSpeechEvaluation>,
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
pub struct DatasetSpeechEvaluation {
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
}

impl DatasetSpeechEvaluation {
    pub fn precision(&self) -> f64 {
        ratio(
            self.true_positive_ms as usize,
            (self.true_positive_ms + self.false_positive_ms) as usize,
        )
    }

    pub fn recall(&self) -> f64 {
        ratio(
            self.true_positive_ms as usize,
            (self.true_positive_ms + self.false_negative_ms) as usize,
        )
    }

    pub fn f1(&self) -> f64 {
        harmonic_mean(self.precision(), self.recall())
    }

    pub fn iou(&self) -> f64 {
        ratio(
            self.true_positive_ms as usize,
            (self.true_positive_ms + self.false_positive_ms + self.false_negative_ms) as usize,
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
    speech_selection: Option<Vec<String>>,
    documents: usize,
    timelines: usize,
    transcription_eligible: BTreeSet<String>,
    speech_eligible: BTreeSet<String>,
    transcription_unannotated: BTreeSet<String>,
    speech_unannotated: BTreeSet<String>,
    transcription: BTreeMap<String, TranscriptionAccumulator>,
    speech: BTreeMap<String, SpeechAccumulator>,
}

impl DatasetEvaluator {
    pub fn new(config: TimelineEvalConfig) -> Self {
        let auto_all = config.transcription_sources.is_none() && config.speech_sources.is_none();
        let transcription_selection = if auto_all {
            Some(Vec::new())
        } else {
            config.transcription_sources.clone()
        };
        let speech_selection = if auto_all {
            Some(Vec::new())
        } else {
            config.speech_sources.clone()
        };
        let transcription = selected_accumulators(transcription_selection.as_deref());
        let speech = selected_accumulators(speech_selection.as_deref());
        Self {
            config,
            transcription_selection,
            speech_selection,
            documents: 0,
            timelines: 0,
            transcription_eligible: BTreeSet::new(),
            speech_eligible: BTreeSet::new(),
            transcription_unannotated: BTreeSet::new(),
            speech_unannotated: BTreeSet::new(),
            transcription,
            speech,
        }
    }

    pub fn push(&mut self, doc: &AudioDoc) -> Result<(), DatasetEvalError> {
        self.documents += 1;
        for (channel, timeline) in doc.timelines() {
            self.timelines += 1;
            let timeline_key = format!("{}:{}", doc.id, channel.name());
            self.push_transcription(doc, timeline, &timeline_key)?;
            self.push_speech(doc, timeline, &timeline_key)?;
        }
        Ok(())
    }

    pub fn finish(self) -> Result<DatasetEvaluation, DatasetEvalError> {
        let transcription = finish_transcription(
            self.transcription,
            &self.transcription_eligible,
            &self.transcription_unannotated,
            self.config.transcription_normalization,
        )?;
        let speech = finish_speech(self.speech, &self.speech_eligible, &self.speech_unannotated)?;
        if transcription.is_empty() && speech.is_empty() {
            return Err(DatasetEvalError::NoEvaluableAnnotations);
        }
        Ok(DatasetEvaluation {
            documents: self.documents,
            timelines: self.timelines,
            transcription,
            speech,
        })
    }

    fn push_transcription(
        &mut self,
        doc: &AudioDoc,
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
        for source in sources_for_timeline(selection, &available) {
            let mut result = timeline
                .eval(
                    &TimelineEvalConfig::new()
                        .with_transcription(&source)
                        .with_transcription_normalization(self.config.transcription_normalization),
                )?
                .transcription;
            let result = result
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

    fn push_speech(
        &mut self,
        doc: &AudioDoc,
        timeline: &Timeline,
        timeline_key: &str,
    ) -> Result<(), DatasetEvalError> {
        let Some(selection) = self.speech_selection.as_deref() else {
            return Ok(());
        };
        let has_reference = timeline
            .reference
            .iter()
            .any(|annotation| matches!(annotation.payload, AnnotationPayload::Speech));
        if !has_reference {
            self.speech_unannotated.insert(timeline_key.to_owned());
            return Ok(());
        }
        self.speech_eligible.insert(timeline_key.to_owned());
        let available = timeline.speech_sources();
        for source in sources_for_timeline(selection, &available) {
            let mut result = timeline
                .eval(&TimelineEvalConfig::new().with_speech(&source))?
                .speech;
            let result = result
                .remove(&source)
                .expect("a selected speech source produced a result");
            self.speech
                .entry(source.clone())
                .or_default()
                .add(&doc.id, timeline_key, &result);
        }
        Ok(())
    }
}

pub fn evaluate_dataset<'a>(
    docs: impl IntoIterator<Item = &'a AudioDoc>,
    config: &TimelineEvalConfig,
) -> Result<DatasetEvaluation, DatasetEvalError> {
    let mut evaluator = DatasetEvaluator::new(config.clone());
    for doc in docs {
        evaluator.push(doc)?;
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
            page_query.after = page.last().map(AudioDoc::audio_id);
            for doc in &page {
                evaluator.push(doc)?;
            }
            if page.len() < page_query.limit {
                break;
            }
        }
        evaluator.finish()
    }
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
struct SpeechAccumulator {
    evaluated_documents: BTreeSet<String>,
    evaluated_timelines: BTreeSet<String>,
    reference_ms: u64,
    predicted_ms: u64,
    true_positive_ms: u64,
    true_negative_ms: u64,
    false_positive_ms: u64,
    false_negative_ms: u64,
}

impl SpeechAccumulator {
    fn add(
        &mut self,
        audio_id: &str,
        timeline_id: &str,
        result: &crate::timeline::SpeechEvaluation,
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

fn finish_speech(
    accumulators: BTreeMap<String, SpeechAccumulator>,
    eligible: &BTreeSet<String>,
    unannotated: &BTreeSet<String>,
) -> Result<BTreeMap<String, DatasetSpeechEvaluation>, DatasetEvalError> {
    if eligible.is_empty() {
        return Ok(BTreeMap::new());
    }
    accumulators
        .into_iter()
        .map(|(source, accumulator)| {
            if accumulator.evaluated_timelines.is_empty() {
                return Err(TimelineEvalError::MissingPrediction {
                    kind: "speech",
                    prediction_source: source,
                }
                .into());
            }
            let missing_prediction_ids = eligible
                .difference(&accumulator.evaluated_timelines)
                .cloned()
                .collect::<Vec<_>>();
            let result = DatasetSpeechEvaluation {
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
            };
            Ok((source, result))
        })
        .collect()
}

fn is_text_annotation(annotation: &crate::timeline::Annotation) -> bool {
    match &annotation.payload {
        AnnotationPayload::Transcription(_) | AnnotationPayload::Sentence(_) => true,
        AnnotationPayload::Speaker(speaker) => speaker.transcription.is_some(),
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
