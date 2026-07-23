use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

use super::annotation::{Annotation, AnnotationId, AnnotationPayload, AudioId, TimelineId};
use super::segment::{TextSpan, Transcript};
use crate::utils::{DurationMs, TimeRange};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationConflictKind {
    Speech,
    Speaker,
    Transcription,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Timeline {
    pub id: TimelineId,
    pub audio_id: AudioId,
    pub duration: DurationMs,
    pub reference: Vec<Annotation>,
    pub prediction: Vec<Annotation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnotationOverlap {
    pub kind: AnnotationConflictKind,
    pub source: Option<String>,
    pub speaker: Option<String>,
    pub first_id: AnnotationId,
    pub first_range: TimeRange,
    pub second_id: AnnotationId,
    pub second_range: TimeRange,
}

impl fmt::Display for AnnotationOverlap {
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
pub enum TimelineAnnotationError {
    #[error("prediction annotation {annotation_id:?} must have a non-empty source")]
    PredictionMissingSource { annotation_id: AnnotationId },
    #[error("prediction source must contain at least one non-whitespace character")]
    InvalidPredictionSource,
    #[error("{0}")]
    Overlap(Box<AnnotationOverlap>),
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

    pub fn push_reference(
        &mut self,
        mut annotation: Annotation,
    ) -> Result<&Annotation, TimelineAnnotationError> {
        annotation.source = None;
        push_validated(&mut self.reference, annotation, false)
    }

    pub fn push_prediction(
        &mut self,
        annotation: Annotation,
    ) -> Result<&Annotation, TimelineAnnotationError> {
        if annotation
            .source
            .as_deref()
            .is_none_or(|source| source.trim().is_empty())
        {
            return Err(TimelineAnnotationError::PredictionMissingSource {
                annotation_id: annotation.id,
            });
        }
        push_validated(&mut self.prediction, annotation, true)
    }

    pub fn all_annotations(&self) -> impl Iterator<Item = &Annotation> {
        self.reference.iter().chain(&self.prediction)
    }

    pub fn annotation_count(&self) -> usize {
        self.reference.len() + self.prediction.len()
    }

    pub fn predictions_by_source<'a>(
        &'a self,
        source: &'a str,
    ) -> impl Iterator<Item = &'a Annotation> + 'a {
        self.prediction
            .iter()
            .filter(move |annotation| annotation.source.as_deref() == Some(source))
    }

    pub fn prediction_sources(&self) -> Vec<&str> {
        let mut sources = self
            .prediction
            .iter()
            .filter_map(|annotation| annotation.source.as_deref())
            .collect::<Vec<_>>();
        sources.sort_unstable();
        sources.dedup();
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
    ) -> Result<usize, TimelineAnnotationError> {
        if to.trim().is_empty() {
            return Err(TimelineAnnotationError::InvalidPredictionSource);
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

    pub fn validate_annotations(&self) -> Result<(), TimelineAnnotationError> {
        validate_reference_slice(&self.reference)?;
        validate_prediction_slice(&self.prediction)
    }
}

fn push_validated(
    annotations: &mut Vec<Annotation>,
    annotation: Annotation,
    prediction: bool,
) -> Result<&Annotation, TimelineAnnotationError> {
    if let Some(index) = annotations
        .iter()
        .position(|existing| existing.content_eq(&annotation))
    {
        return Ok(&annotations[index]);
    }
    for existing in annotations.iter() {
        if let Some((kind, speaker)) = overlap_conflict(existing, &annotation, prediction) {
            return Err(TimelineAnnotationError::Overlap(Box::new(
                AnnotationOverlap {
                    kind,
                    source: annotation.source.clone(),
                    speaker,
                    first_id: existing.id.clone(),
                    first_range: existing.range,
                    second_id: annotation.id.clone(),
                    second_range: annotation.range,
                },
            )));
        }
    }
    annotations.push(annotation);
    Ok(annotations
        .last()
        .expect("the annotation was just inserted"))
}

fn validate_reference_slice(annotations: &[Annotation]) -> Result<(), TimelineAnnotationError> {
    validate_slice(annotations, false)
}

fn validate_prediction_slice(annotations: &[Annotation]) -> Result<(), TimelineAnnotationError> {
    for annotation in annotations {
        if annotation
            .source
            .as_deref()
            .is_none_or(|source| source.trim().is_empty())
        {
            return Err(TimelineAnnotationError::PredictionMissingSource {
                annotation_id: annotation.id.clone(),
            });
        }
    }
    validate_slice(annotations, true)
}

fn validate_slice(
    annotations: &[Annotation],
    prediction: bool,
) -> Result<(), TimelineAnnotationError> {
    for (index, first) in annotations.iter().enumerate() {
        for second in &annotations[index + 1..] {
            if let Some((kind, speaker)) = overlap_conflict(first, second, prediction) {
                return Err(TimelineAnnotationError::Overlap(Box::new(
                    AnnotationOverlap {
                        kind,
                        source: second.source.clone(),
                        speaker,
                        first_id: first.id.clone(),
                        first_range: first.range,
                        second_id: second.id.clone(),
                        second_range: second.range,
                    },
                )));
            }
        }
    }
    Ok(())
}

fn overlap_conflict(
    first: &Annotation,
    second: &Annotation,
    prediction: bool,
) -> Option<(AnnotationConflictKind, Option<String>)> {
    if !first.range.overlaps(&second.range)
        || prediction && first.source.as_deref() != second.source.as_deref()
    {
        return None;
    }
    match (&first.payload, &second.payload) {
        (AnnotationPayload::Speech, AnnotationPayload::Speech) => {
            Some((AnnotationConflictKind::Speech, None))
        }
        (AnnotationPayload::Speaker(first), AnnotationPayload::Speaker(second))
            if first.name == second.name =>
        {
            Some((AnnotationConflictKind::Speaker, Some(first.name.clone())))
        }
        (AnnotationPayload::Transcription(_), AnnotationPayload::Transcription(_)) => {
            Some((AnnotationConflictKind::Transcription, None))
        }
        (AnnotationPayload::Transcription(_), AnnotationPayload::Speaker(speaker))
        | (AnnotationPayload::Speaker(speaker), AnnotationPayload::Transcription(_))
            if speaker.transcription.is_some() =>
        {
            Some((
                AnnotationConflictKind::Transcription,
                Some(speaker.name.clone()),
            ))
        }
        _ => None,
    }
}

fn transcript_from_annotations<'a>(
    annotations: impl Iterator<Item = &'a Annotation>,
) -> Transcript {
    let mut segments = annotations
        .filter_map(|annotation| match &annotation.payload {
            AnnotationPayload::Transcription(transcription) => Some((
                annotation.range.start,
                TextSpan {
                    text: transcription.text.clone(),
                    tokens: transcription.tokens.clone(),
                    language: transcription.language.clone(),
                },
            )),
            AnnotationPayload::Sentence(segment) => Some((annotation.range.start, segment.clone())),
            AnnotationPayload::Speaker(speaker) => speaker.transcription.as_ref().map(|value| {
                (
                    annotation.range.start,
                    TextSpan {
                        text: value.text.clone(),
                        tokens: value.tokens.clone(),
                        language: value.language.clone(),
                    },
                )
            }),
            _ => None,
        })
        .collect::<Vec<(DurationMs, TextSpan)>>();

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
