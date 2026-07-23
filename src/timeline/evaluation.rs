use thiserror::Error;

use super::{Annotation, AnnotationPayload, Timeline};
use crate::metrics::{
    CerStats, TextNormalizationError, compute_cer, normalize_for_cer, normalize_zh,
};
use crate::utils::TimeRange;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TranscriptionNormalization {
    None,
    #[default]
    ChineseTn,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TimelineEvalConfig {
    pub transcription_source: Option<String>,
    pub speech_source: Option<String>,
    pub transcription_normalization: TranscriptionNormalization,
}

impl TimelineEvalConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_transcription(mut self, source: impl Into<String>) -> Self {
        self.transcription_source = Some(source.into());
        self
    }

    pub fn with_speech(mut self, source: impl Into<String>) -> Self {
        self.speech_source = Some(source.into());
        self
    }

    pub fn with_transcription_normalization(
        mut self,
        normalization: TranscriptionNormalization,
    ) -> Self {
        self.transcription_normalization = normalization;
        self
    }
}

#[derive(Debug, Error)]
pub enum TimelineEvalError {
    #[error("at least one evaluation source must be provided")]
    NoTaskRequested,
    #[error("{kind} reference annotations are missing")]
    MissingReference { kind: &'static str },
    #[error("{kind} predictions from source {prediction_source:?} are missing")]
    MissingPrediction {
        kind: &'static str,
        prediction_source: String,
    },
    #[error(transparent)]
    TextNormalization(#[from] TextNormalizationError),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimelineEvaluation {
    pub transcription: Option<TranscriptionEvaluation>,
    pub speech: Option<SpeechEvaluation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionEvaluation {
    pub source: String,
    pub reference: String,
    pub hypothesis: String,
    pub normalized_reference: String,
    pub normalized_hypothesis: String,
    pub normalization: TranscriptionNormalization,
    pub stats: CerStats,
    pub hypothesis_chars: usize,
}

impl TranscriptionEvaluation {
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

    pub fn exact_match(&self) -> bool {
        self.normalized_reference == self.normalized_hypothesis
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechEvaluation {
    pub source: String,
    pub reference_ms: u64,
    pub predicted_ms: u64,
    pub true_positive_ms: u64,
    pub true_negative_ms: u64,
    pub false_positive_ms: u64,
    pub false_negative_ms: u64,
}

impl SpeechEvaluation {
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
}

impl Timeline {
    pub fn eval(
        &self,
        config: &TimelineEvalConfig,
    ) -> Result<TimelineEvaluation, TimelineEvalError> {
        self.evaluate(config)
    }

    pub fn evaluate(
        &self,
        config: &TimelineEvalConfig,
    ) -> Result<TimelineEvaluation, TimelineEvalError> {
        if config.transcription_source.is_none() && config.speech_source.is_none() {
            return Err(TimelineEvalError::NoTaskRequested);
        }
        let transcription = config
            .transcription_source
            .as_deref()
            .map(|source| self.evaluate_transcription(source, config.transcription_normalization))
            .transpose()?;
        let speech = config
            .speech_source
            .as_deref()
            .map(|source| self.evaluate_speech(source))
            .transpose()?;
        Ok(TimelineEvaluation {
            transcription,
            speech,
        })
    }

    fn evaluate_transcription(
        &self,
        source: &str,
        normalization: TranscriptionNormalization,
    ) -> Result<TranscriptionEvaluation, TimelineEvalError> {
        if !self.reference.iter().any(is_final_text_annotation) {
            return Err(TimelineEvalError::MissingReference {
                kind: "transcription",
            });
        }
        if !self
            .predictions_by_source(source)
            .any(is_final_text_annotation)
        {
            return Err(TimelineEvalError::MissingPrediction {
                kind: "transcription",
                prediction_source: source.to_owned(),
            });
        }
        let reference = self.reference_transcript().text;
        let hypothesis = self.prediction_transcript(source).text;
        let normalize = |text: &str| -> Result<String, TextNormalizationError> {
            match normalization {
                TranscriptionNormalization::None => Ok(text.to_owned()),
                TranscriptionNormalization::ChineseTn => {
                    normalize_zh(text).map(|text| normalize_for_cer(&text, true))
                }
            }
        };
        let normalized_reference = normalize(&reference)?;
        let normalized_hypothesis = normalize(&hypothesis)?;
        let stats = compute_cer(&normalized_reference, &normalized_hypothesis);
        let hypothesis_chars = normalized_hypothesis.chars().count();
        Ok(TranscriptionEvaluation {
            source: source.to_owned(),
            reference,
            hypothesis,
            normalized_reference,
            normalized_hypothesis,
            normalization,
            stats,
            hypothesis_chars,
        })
    }

    fn evaluate_speech(&self, source: &str) -> Result<SpeechEvaluation, TimelineEvalError> {
        let reference = merged_speech_ranges(self.reference.iter());
        if reference.is_empty() {
            return Err(TimelineEvalError::MissingReference { kind: "speech" });
        }
        let prediction = merged_speech_ranges(self.predictions_by_source(source));
        if prediction.is_empty() {
            return Err(TimelineEvalError::MissingPrediction {
                kind: "speech",
                prediction_source: source.to_owned(),
            });
        }
        let reference_ms = ranges_duration(&reference);
        let predicted_ms = ranges_duration(&prediction);
        let true_positive_ms = intersection_duration(&reference, &prediction);
        let false_positive_ms = predicted_ms.saturating_sub(true_positive_ms);
        let false_negative_ms = reference_ms.saturating_sub(true_positive_ms);
        let covered_ms = true_positive_ms + false_positive_ms + false_negative_ms;
        let true_negative_ms = self.duration.0.saturating_sub(covered_ms);
        Ok(SpeechEvaluation {
            source: source.to_owned(),
            reference_ms,
            predicted_ms,
            true_positive_ms,
            true_negative_ms,
            false_positive_ms,
            false_negative_ms,
        })
    }
}

fn is_final_text_annotation(annotation: &Annotation) -> bool {
    match &annotation.payload {
        AnnotationPayload::Transcription(_) | AnnotationPayload::Sentence(_) => true,
        AnnotationPayload::Speaker(speaker) => speaker.transcription.is_some(),
        _ => false,
    }
}

fn merged_speech_ranges<'a>(annotations: impl Iterator<Item = &'a Annotation>) -> Vec<TimeRange> {
    let mut ranges = annotations
        .filter(|annotation| matches!(annotation.payload, AnnotationPayload::Speech))
        .map(|annotation| annotation.range)
        .filter(|range| range.end > range.start)
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| (range.start, range.end));
    let mut merged: Vec<TimeRange> = Vec::new();
    for range in ranges {
        if let Some(previous) = merged.last_mut()
            && range.start <= previous.end
        {
            previous.end = previous.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    merged
}

fn ranges_duration(ranges: &[TimeRange]) -> u64 {
    ranges
        .iter()
        .map(|range| range.end.0.saturating_sub(range.start.0))
        .sum()
}

fn intersection_duration(left: &[TimeRange], right: &[TimeRange]) -> u64 {
    let mut left_index = 0;
    let mut right_index = 0;
    let mut duration = 0u64;
    while left_index < left.len() && right_index < right.len() {
        let start = left[left_index].start.max(right[right_index].start);
        let end = left[left_index].end.min(right[right_index].end);
        duration += end.0.saturating_sub(start.0);
        if left[left_index].end <= right[right_index].end {
            left_index += 1;
        } else {
            right_index += 1;
        }
    }
    duration
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline::Annotation;
    use crate::utils::DurationMs;

    fn speech(start: u64, end: u64, source: Option<&str>) -> Annotation {
        Annotation::new(
            TimeRange::new(DurationMs(start), DurationMs(end)),
            AnnotationPayload::Speech,
            source.map(str::to_owned),
        )
    }

    #[test]
    fn evaluates_merged_speech_ranges() {
        let mut timeline = Timeline::new("audio", DurationMs(1_000));
        timeline.reference.push(speech(100, 500, None));
        timeline.reference.push(speech(400, 600, None));
        timeline.prediction.push(speech(200, 700, Some("vad")));
        let result = timeline
            .evaluate(&TimelineEvalConfig::new().with_speech("vad"))
            .unwrap()
            .speech
            .unwrap();
        assert_eq!(result.reference_ms, 500);
        assert_eq!(result.predicted_ms, 500);
        assert_eq!(result.true_positive_ms, 400);
        assert_eq!(result.false_positive_ms, 100);
        assert_eq!(result.false_negative_ms, 100);
        assert_eq!(result.true_negative_ms, 400);
        assert_eq!(result.precision(), 0.8);
        assert_eq!(result.recall(), 0.8);
        assert!((result.f1() - 0.8).abs() < f64::EPSILON);
        assert_eq!(result.iou(), 2.0 / 3.0);
    }

    #[test]
    fn can_evaluate_transcription_without_normalization() {
        use crate::timeline::Transcription;

        let mut timeline = Timeline::new("audio", DurationMs(1_000));
        timeline
            .push_reference(Annotation::new(
                TimeRange::new(DurationMs(0), DurationMs(1_000)),
                AnnotationPayload::Transcription(Transcription::new("交易停滞")),
                None,
            ))
            .unwrap();
        timeline
            .push_prediction(Annotation::new(
                TimeRange::new(DurationMs(0), DurationMs(1_000)),
                AnnotationPayload::Transcription(Transcription::new("交易停止")),
                Some("asr".to_owned()),
            ))
            .unwrap();
        let config = TimelineEvalConfig::new()
            .with_transcription("asr")
            .with_transcription_normalization(TranscriptionNormalization::None);
        let result = timeline.evaluate(&config).unwrap().transcription.unwrap();
        assert_eq!(result.stats.substitutions, 1);
        assert_eq!(result.matches(), 3);
        assert_eq!(result.stats.cer(), 0.25);
        assert!(!result.exact_match());
    }
}
