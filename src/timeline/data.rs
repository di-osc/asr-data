use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, AudioId, TimelineId,
    Transcript,
};
use crate::utils::DurationMs;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Timeline {
    pub id: TimelineId,
    pub audio_id: AudioId,
    #[serde(default, deserialize_with = "deserialize_duration")]
    pub duration: DurationMs,
    #[serde(default)]
    pub annotations: Vec<Annotation>,
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
            annotations: Vec::new(),
        }
    }

    pub fn push(&mut self, annotation: Annotation) {
        self.annotations.push(annotation);
    }

    pub fn extend(&mut self, annotations: impl IntoIterator<Item = Annotation>) {
        self.annotations.extend(annotations);
    }

    pub fn by_status(&self, status: AnnotationStatus) -> Vec<&Annotation> {
        self.annotations
            .iter()
            .filter(|annotation| annotation.status == status)
            .collect()
    }

    pub fn annotations_by_source<'a>(
        &'a self,
        source: &'a AnnotationSource,
    ) -> impl Iterator<Item = &'a Annotation> + 'a {
        self.annotations
            .iter()
            .filter(move |annotation| &annotation.source == source)
    }

    pub fn transcript_by_source(&self, source: &AnnotationSource) -> Transcript {
        transcript_from_annotations(self.annotations_by_source(source))
    }

    pub fn remove_annotations_by_source(&mut self, source: &AnnotationSource) -> usize {
        let old_len = self.annotations.len();
        self.annotations
            .retain(|annotation| &annotation.source != source);
        old_len - self.annotations.len()
    }

    pub fn relabel_annotations_source(
        &mut self,
        from: &AnnotationSource,
        to: AnnotationSource,
    ) -> usize {
        let mut changed = 0;
        for annotation in &mut self.annotations {
            if &annotation.source == from {
                annotation.source = to.clone();
                changed += 1;
            }
        }
        changed
    }

    pub fn transcript(&self) -> Transcript {
        transcript_from_annotations(self.annotations.iter())
    }
}

fn transcript_from_annotations<'a>(
    annotations: impl Iterator<Item = &'a Annotation>,
) -> Transcript {
    let mut segments = annotations
        .filter(|annotation| annotation.status == AnnotationStatus::Final)
        .filter_map(|annotation| match &annotation.payload {
            AnnotationPayload::Transcription(segment) | AnnotationPayload::Sentence(segment) => {
                Some((annotation.range.start, segment.clone()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();

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
