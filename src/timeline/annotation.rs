use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::segment::Sentence;
use crate::utils::TimeRange;

pub type AudioId = String;
pub type TimelineId = String;
pub type TimeSpanId = String;
pub type SpeakerId = String;
pub type LanguageTag = String;

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AudioActivity {
    #[serde(default)]
    pub event: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

impl AudioActivity {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_event(mut self, event: impl Into<String>) -> Self {
        self.event = Some(event.into());
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
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
    #[serde(default)]
    pub confidence: Option<f32>,
}

impl SpeakerPayload {
    pub fn new(name: impl Into<SpeakerId>) -> Self {
        Self {
            name: name.into(),
            transcription: None,
            confidence: None,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
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

    pub fn confidence(&self) -> Option<f32> {
        match self {
            Self::Activity(value) => value.confidence,
            Self::Token(value) => value.confidence,
            Self::Transcription(value) => value.confidence,
            Self::Speaker(value) => value.confidence,
            Self::Sentence(_) | Self::Language(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeSpan {
    pub id: TimeSpanId,
    pub range: TimeRange,
    pub source: Option<String>,
    pub annotation: Annotation,
}

impl TimeSpan {
    pub fn new(range: TimeRange, annotation: Annotation, source: Option<String>) -> Self {
        Self {
            id: format!("span_{}", Uuid::new_v4().simple()),
            range,
            source,
            annotation,
        }
    }

    /// Compares annotation content while ignoring the generated identity.
    pub fn content_eq(&self, other: &Self) -> bool {
        self.range == other.range
            && self.source == other.source
            && self.annotation == other.annotation
    }
}

#[cfg(test)]
mod tests {
    use super::{Annotation, AudioActivity, TimeSpan};
    use crate::utils::{DurationMs, TimeRange};

    #[test]
    fn confidence_is_serialized_inside_the_annotation_payload() {
        let span = TimeSpan::new(
            TimeRange::new(DurationMs(0), DurationMs(1_000)),
            Annotation::Activity(
                AudioActivity::new()
                    .with_event("speech")
                    .with_confidence(0.98),
            ),
            Some("vad".to_owned()),
        );

        let value = serde_json::to_value(span).expect("serialize span");
        assert!(value.get("confidence").is_none());
        let confidence = value["annotation"]["Activity"]["confidence"]
            .as_f64()
            .expect("numeric confidence");
        assert!((confidence - 0.98).abs() < 1e-6);
    }
}
