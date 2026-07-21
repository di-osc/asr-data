use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{TextSpan, TimeRange, Token, Transcript};

pub type AudioId = String;
pub type TimelineId = String;
pub type AnnotationId = String;
pub type SpeakerId = String;
pub type LanguageTag = String;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnnotationStatus {
    Partial,
    Final,
    Revised,
    Deleted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnnotationSource {
    User,
    Model(String),
    Stage(String),
    System,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HotwordMatch {
    pub text: String,
    pub normalized: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticEvent {
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    pub component: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnnotationPayload {
    Speech,
    Silence,
    Token(Token),
    #[serde(alias = "Segment")]
    Transcription(TextSpan),
    Sentence(TextSpan),
    Speaker(SpeakerId),
    Language(LanguageTag),
    Hotword(HotwordMatch),
    AcousticEvent(AcousticEvent),
    Diagnostic(Diagnostic),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    pub id: AnnotationId,
    pub range: TimeRange,
    pub status: AnnotationStatus,
    pub source: AnnotationSource,
    pub confidence: Option<f32>,
    pub payload: AnnotationPayload,
}

impl Annotation {
    pub fn new(
        range: TimeRange,
        payload: AnnotationPayload,
        source: AnnotationSource,
        status: AnnotationStatus,
    ) -> Self {
        Self {
            id: format!("ann_{}", Uuid::new_v4().simple()),
            range,
            status,
            source,
            confidence: None,
            payload,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Timeline {
    pub id: TimelineId,
    pub audio_id: AudioId,
    #[serde(default, deserialize_with = "deserialize_duration")]
    pub duration: crate::DurationMs,
    #[serde(default)]
    pub annotations: Vec<Annotation>,
}

fn deserialize_duration<'de, D>(deserializer: D) -> Result<crate::DurationMs, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<crate::DurationMs>::deserialize(deserializer)?.unwrap_or_default())
}

impl Timeline {
    pub fn new(audio_id: impl Into<AudioId>, duration: crate::DurationMs) -> Self {
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
        Self::new(String::new(), crate::DurationMs(0))
    }
}
