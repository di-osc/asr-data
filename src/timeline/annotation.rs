use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::segment::TextSpan;
use super::token::Token;
use crate::utils::TimeRange;

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
