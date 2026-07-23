use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::segment::Sentence;
use crate::utils::TimeRange;

pub type AudioId = String;
pub type TimelineId = String;
pub type TimeSpanId = String;
pub type SpeakerId = String;
pub type LanguageTag = String;

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AudioActivity {
    #[serde(default)]
    pub event: Option<String>,
}

impl AudioActivity {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_event(mut self, event: impl Into<String>) -> Self {
        self.event = Some(event.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    pub text: String,
    pub range: Option<TimeRange>,
    pub confidence: Option<f32>,
}

impl Token {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            range: None,
            confidence: None,
        }
    }

    pub fn with_range(mut self, range: TimeRange) -> Self {
        self.range = Some(range);
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcription {
    pub text: String,
    #[serde(default)]
    pub tokens: Vec<Token>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

impl Transcription {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens: Vec::new(),
            language: None,
            confidence: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerPayload {
    pub name: SpeakerId,
    #[serde(default)]
    pub transcription: Option<Transcription>,
}

impl SpeakerPayload {
    pub fn new(name: impl Into<SpeakerId>) -> Self {
        Self {
            name: name.into(),
            transcription: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Annotation {
    Activity(AudioActivity),
    Token(Token),
    Transcription(Transcription),
    Sentence(Sentence),
    Speaker(SpeakerPayload),
    Language(LanguageTag),
}

impl Annotation {
    pub(crate) fn source_group(&self) -> &'static str {
        match self {
            Self::Activity(_) => "activity",
            Self::Token(_) => "token",
            Self::Transcription(_) => "transcription",
            Self::Sentence(_) => "sentence",
            Self::Speaker(_) => "speaker",
            Self::Language(_) => "language",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeSpan {
    pub id: TimeSpanId,
    pub range: TimeRange,
    pub source: Option<String>,
    pub confidence: Option<f32>,
    pub annotation: Annotation,
}

impl TimeSpan {
    pub fn new(range: TimeRange, annotation: Annotation, source: Option<String>) -> Self {
        Self {
            id: format!("span_{}", Uuid::new_v4().simple()),
            range,
            source,
            confidence: None,
            annotation,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Compares annotation content while ignoring the generated identity.
    pub fn content_eq(&self, other: &Self) -> bool {
        self.range == other.range
            && self.source == other.source
            && self.confidence == other.confidence
            && self.annotation == other.annotation
    }
}
