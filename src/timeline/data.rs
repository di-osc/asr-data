use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use uuid::Uuid;

use super::annotation::{Annotation, AudioId, TimeSpan, TimeSpanId, TimelineId};
use super::segment::{Sentence, Transcript};
use crate::utils::{DurationMs, TimeRange};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeSpanConflictKind {
    Activity,
    Speaker,
    Transcription,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Timeline {
    pub id: TimelineId,
    pub audio_id: AudioId,
    pub duration: DurationMs,
    pub reference: Vec<TimeSpan>,
    pub prediction: Vec<TimeSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeSpanOverlap {
    pub kind: TimeSpanConflictKind,
    pub source: Option<String>,
    pub speaker: Option<String>,
    pub first_id: TimeSpanId,
    pub first_range: TimeRange,
    pub second_id: TimeSpanId,
    pub second_range: TimeRange,
}

impl fmt::Display for TimeSpanOverlap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?} annotation {:?} at {:?} overlaps annotation {:?} at {:?} (source={:?}, speaker={:?})",
            self.kind,
            self.second_id,
            self.second_range,
            self.first_id,
            self.first_range,
            self.source,
            self.speaker,
        )
    }
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum TimelineSpanError {
    #[error("reference annotation {annotation_id:?} must not have a source")]
    ReferenceHasSource { annotation_id: TimeSpanId },
    #[error("prediction annotation {annotation_id:?} must have a non-empty source")]
    PredictionMissingSource { annotation_id: TimeSpanId },
    #[error("prediction source must contain at least one non-whitespace character")]
    InvalidPredictionSource,
    #[error("activity event must contain at least one non-whitespace character")]
    InvalidActivityEvent,
    #[error("{0}")]
    Overlap(Box<TimeSpanOverlap>),
}

impl Timeline {
    pub fn new(audio_id: impl Into<AudioId>, duration: DurationMs) -> Self {
        Self {
            id: format!("tl_{}", Uuid::new_v4().simple()),
            audio_id: audio_id.into(),
            duration,
            reference: Vec::new(),
            prediction: Vec::new(),
        }
    }

    pub fn annotate_span(
        &mut self,
        is_reference: bool,
        annotation: TimeSpan,
    ) -> Result<&TimeSpan, TimelineSpanError> {
        if is_reference {
            if annotation.source.is_some() {
                return Err(TimelineSpanError::ReferenceHasSource {
                    annotation_id: annotation.id,
                });
            }
            return push_validated(&mut self.reference, annotation, false);
        }
        if annotation
            .source
            .as_deref()
            .is_none_or(|source| source.trim().is_empty())
        {
            return Err(TimelineSpanError::PredictionMissingSource {
                annotation_id: annotation.id,
            });
        }
        push_validated(&mut self.prediction, annotation, true)
    }

    pub fn all_spans(&self) -> impl Iterator<Item = &TimeSpan> {
        self.reference.iter().chain(&self.prediction)
    }

    pub fn span_count(&self) -> usize {
        self.reference.len() + self.prediction.len()
    }

    pub fn predictions_by_source<'a>(
        &'a self,
        source: &'a str,
    ) -> impl Iterator<Item = &'a TimeSpan> + 'a {
        self.prediction
            .iter()
            .filter(move |annotation| annotation.source.as_deref() == Some(source))
    }

    pub fn prediction_sources(&self) -> BTreeMap<&'static str, Vec<&str>> {
        let mut sources = [
            "activity",
            "token",
            "transcription",
            "sentence",
            "speaker",
            "language",
        ]
        .into_iter()
        .map(|kind| (kind, Vec::new()))
        .collect::<BTreeMap<_, _>>();
        for annotation in &self.prediction {
            if let Some(source) = annotation.source.as_deref() {
                sources
                    .get_mut(annotation.annotation.source_group())
                    .expect("every annotation kind has a source group")
                    .push(source);
            }
        }
        for values in sources.values_mut() {
            values.sort_unstable();
            values.dedup();
        }
        sources
    }

    pub fn reference_transcript(&self) -> Transcript {
        transcript_from_annotations(self.reference.iter())
    }

    pub fn prediction_transcript(&self, source: &str) -> Transcript {
        transcript_from_annotations(self.predictions_by_source(source))
    }

    pub fn remove_predictions_by_source(&mut self, source: &str) -> usize {
        let old_len = self.prediction.len();
        self.prediction
            .retain(|annotation| annotation.source.as_deref() != Some(source));
        old_len - self.prediction.len()
    }

    pub fn relabel_prediction_source(
        &mut self,
        from: &str,
        to: &str,
    ) -> Result<usize, TimelineSpanError> {
        if to.trim().is_empty() {
            return Err(TimelineSpanError::InvalidPredictionSource);
        }
        let mut candidate = self.prediction.clone();
        let mut changed = 0;
        for annotation in &mut candidate {
            if annotation.source.as_deref() == Some(from) {
                annotation.source = Some(to.to_string());
                changed += 1;
            }
        }
        validate_prediction_slice(&candidate)?;
        self.prediction = candidate;
        Ok(changed)
    }

    pub fn validate_spans(&self) -> Result<(), TimelineSpanError> {
        validate_reference_slice(&self.reference)?;
        validate_prediction_slice(&self.prediction)
    }

    pub(crate) fn extend_to(&mut self, duration: DurationMs) {
        if duration > self.duration {
            self.duration = duration;
        }
    }
}

fn push_validated(
    annotations: &mut Vec<TimeSpan>,
    annotation: TimeSpan,
    prediction: bool,
) -> Result<&TimeSpan, TimelineSpanError> {
    validate_activity_event(&annotation)?;
    if let Some(index) = annotations
        .iter()
        .position(|existing| existing.content_eq(&annotation))
    {
        return Ok(&annotations[index]);
    }
    for existing in annotations.iter() {
        if let Some((kind, speaker)) = overlap_conflict(existing, &annotation, prediction) {
            return Err(TimelineSpanError::Overlap(Box::new(TimeSpanOverlap {
                kind,
                source: annotation.source.clone(),
                speaker,
                first_id: existing.id.clone(),
                first_range: existing.range,
                second_id: annotation.id.clone(),
                second_range: annotation.range,
            })));
        }
    }
    annotations.push(annotation);
    Ok(annotations
        .last()
        .expect("the annotation was just inserted"))
}

fn validate_reference_slice(annotations: &[TimeSpan]) -> Result<(), TimelineSpanError> {
    validate_slice(annotations, false)
}

fn validate_prediction_slice(annotations: &[TimeSpan]) -> Result<(), TimelineSpanError> {
    for annotation in annotations {
        if annotation
            .source
            .as_deref()
            .is_none_or(|source| source.trim().is_empty())
        {
            return Err(TimelineSpanError::PredictionMissingSource {
                annotation_id: annotation.id.clone(),
            });
        }
    }
    validate_slice(annotations, true)
}

fn validate_slice(annotations: &[TimeSpan], prediction: bool) -> Result<(), TimelineSpanError> {
    for annotation in annotations {
        validate_activity_event(annotation)?;
    }
    for (index, first) in annotations.iter().enumerate() {
        for second in &annotations[index + 1..] {
            if let Some((kind, speaker)) = overlap_conflict(first, second, prediction) {
                return Err(TimelineSpanError::Overlap(Box::new(TimeSpanOverlap {
                    kind,
                    source: second.source.clone(),
                    speaker,
                    first_id: first.id.clone(),
                    first_range: first.range,
                    second_id: second.id.clone(),
                    second_range: second.range,
                })));
            }
        }
    }
    Ok(())
}

fn validate_activity_event(annotation: &TimeSpan) -> Result<(), TimelineSpanError> {
    if let Annotation::Activity(activity) = &annotation.annotation
        && activity
            .event
            .as_deref()
            .is_some_and(|event| event.trim().is_empty())
    {
        return Err(TimelineSpanError::InvalidActivityEvent);
    }
    Ok(())
}

fn overlap_conflict(
    first: &TimeSpan,
    second: &TimeSpan,
    prediction: bool,
) -> Option<(TimeSpanConflictKind, Option<String>)> {
    if !first.range.overlaps(&second.range)
        || prediction && first.source.as_deref() != second.source.as_deref()
    {
        return None;
    }
    match (&first.annotation, &second.annotation) {
        (Annotation::Activity(first), Annotation::Activity(second))
            if first.event == second.event =>
        {
            Some((TimeSpanConflictKind::Activity, None))
        }
        (Annotation::Speaker(first), Annotation::Speaker(second)) if first.name == second.name => {
            Some((TimeSpanConflictKind::Speaker, Some(first.name.clone())))
        }
        (Annotation::Transcription(_), Annotation::Transcription(_)) => {
            Some((TimeSpanConflictKind::Transcription, None))
        }
        (Annotation::Transcription(_), Annotation::Speaker(speaker))
        | (Annotation::Speaker(speaker), Annotation::Transcription(_))
            if speaker.transcription.is_some() =>
        {
            Some((
                TimeSpanConflictKind::Transcription,
                Some(speaker.name.clone()),
            ))
        }
        _ => None,
    }
}

fn transcript_from_annotations<'a>(annotations: impl Iterator<Item = &'a TimeSpan>) -> Transcript {
    let mut segments = annotations
        .filter_map(|annotation| match &annotation.annotation {
            Annotation::Transcription(transcription) => Some((
                annotation.range.start,
                Sentence {
                    text: transcription.text.clone(),
                    tokens: transcription.tokens.clone(),
                    language: transcription.language.clone(),
                },
            )),
            Annotation::Sentence(segment) => Some((annotation.range.start, segment.clone())),
            Annotation::Speaker(speaker) => speaker.transcription.as_ref().map(|value| {
                (
                    annotation.range.start,
                    Sentence {
                        text: value.text.clone(),
                        tokens: value.tokens.clone(),
                        language: value.language.clone(),
                    },
                )
            }),
            _ => None,
        })
        .collect::<Vec<(DurationMs, Sentence)>>();

    segments.sort_by_key(|(start, _)| *start);
    let segments = segments
        .into_iter()
        .map(|(_, segment)| segment)
        .collect::<Vec<_>>();
    let text = segments
        .iter()
        .map(|segment| segment.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let language = segments.iter().find_map(|segment| segment.language.clone());

    Transcript {
        text,
        language,
        segments,
    }
}

impl Default for Timeline {
    fn default() -> Self {
        Self::new(String::new(), DurationMs(0))
    }
}
