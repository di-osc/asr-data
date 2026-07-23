use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::annotation::{Annotation, AnnotationId, AnnotationPayload, AudioId, TimelineId};
use super::segment::{TextSpan, Transcript};
use crate::utils::DurationMs;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Timeline {
    pub id: TimelineId,
    pub audio_id: AudioId,
    pub duration: DurationMs,
    pub reference: Vec<Annotation>,
    pub prediction: Vec<Annotation>,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum TimelineAnnotationError {
    #[error("prediction annotation {annotation_id:?} must have a non-empty source")]
    PredictionMissingSource { annotation_id: AnnotationId },
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
        Ok(push_unique(&mut self.reference, annotation))
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
        Ok(push_unique(&mut self.prediction, annotation))
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

    pub fn relabel_prediction_source(&mut self, from: &str, to: &str) -> usize {
        let mut changed = 0;
        for annotation in &mut self.prediction {
            if annotation.source.as_deref() == Some(from) {
                annotation.source = Some(to.to_string());
                changed += 1;
            }
        }
        changed
    }
}

fn push_unique(annotations: &mut Vec<Annotation>, annotation: Annotation) -> &Annotation {
    if let Some(index) = annotations
        .iter()
        .position(|existing| existing.content_eq(&annotation))
    {
        return &annotations[index];
    }
    annotations.push(annotation);
    annotations
        .last()
        .expect("the annotation was just inserted")
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
