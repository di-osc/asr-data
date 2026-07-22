use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::annotation::{Annotation, AnnotationPayload, AnnotationStatus, AudioId, TimelineId};
use super::segment::{TextSpan, Transcript};
use crate::utils::DurationMs;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Timeline {
    pub id: TimelineId,
    pub audio_id: AudioId,
    pub duration: DurationMs,
    pub reference: Vec<Annotation>,
    pub prediction: Vec<Annotation>,
}

impl<'de> Deserialize<'de> for Timeline {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            #[serde(default)]
            id: TimelineId,
            #[serde(default)]
            audio_id: AudioId,
            #[serde(default, deserialize_with = "deserialize_duration")]
            duration: DurationMs,
            #[serde(default)]
            reference: Vec<Annotation>,
            #[serde(default)]
            prediction: Vec<Annotation>,
            #[serde(default)]
            annotations: Vec<Annotation>,
        }

        let mut wire = Wire::deserialize(deserializer)?;
        // Timelines written before reference/prediction separation had one flat
        // collection. Preserve those records as predictions because their role
        // was not represented explicitly.
        for annotation in &mut wire.annotations {
            if annotation
                .source
                .as_deref()
                .is_none_or(|source| source.trim().is_empty())
            {
                annotation.source = Some("import".to_string());
            }
        }
        wire.prediction.append(&mut wire.annotations);
        Ok(Self {
            id: wire.id,
            audio_id: wire.audio_id,
            duration: wire.duration,
            reference: wire.reference,
            prediction: wire.prediction,
        })
    }
}

fn deserialize_duration<'de, D>(deserializer: D) -> Result<DurationMs, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<DurationMs>::deserialize(deserializer)?.unwrap_or_default())
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

    pub fn push_reference(&mut self, mut annotation: Annotation) {
        annotation.source = None;
        self.reference.push(annotation);
    }

    pub fn push_prediction(&mut self, annotation: Annotation) {
        self.prediction.push(annotation);
    }

    pub fn push_reference_unique(&mut self, mut annotation: Annotation) -> &Annotation {
        annotation.source = None;
        push_unique(&mut self.reference, annotation)
    }

    pub fn push_prediction_unique(&mut self, annotation: Annotation) -> &Annotation {
        push_unique(&mut self.prediction, annotation)
    }

    pub fn all_annotations(&self) -> impl Iterator<Item = &Annotation> {
        self.reference.iter().chain(&self.prediction)
    }

    pub fn annotation_count(&self) -> usize {
        self.reference.len() + self.prediction.len()
    }

    pub fn by_status(&self, status: AnnotationStatus) -> Vec<&Annotation> {
        self.all_annotations()
            .filter(|annotation| annotation.status == status)
            .collect()
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
        .filter(|annotation| annotation.status == AnnotationStatus::Final)
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
