use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::segment::TextSpan;
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticEvent {
    pub label: String,
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

#[derive(Debug, Clone, PartialEq, Serialize)]
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

impl<'de> Deserialize<'de> for SpeakerPayload {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire {
            LegacyName(SpeakerId),
            Structured {
                name: SpeakerId,
                #[serde(default)]
                transcription: Option<Transcription>,
            },
        }

        match Wire::deserialize(deserializer)? {
            Wire::LegacyName(name) => Ok(Self::new(name)),
            Wire::Structured {
                name,
                transcription,
            } => Ok(Self {
                name,
                transcription,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnnotationPayload {
    Speech,
    Token(Token),
    #[serde(alias = "Segment")]
    Transcription(Transcription),
    Sentence(TextSpan),
    Speaker(SpeakerPayload),
    Language(LanguageTag),
    AcousticEvent(AcousticEvent),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    pub id: AnnotationId,
    pub range: TimeRange,
    pub status: AnnotationStatus,
    #[serde(default, deserialize_with = "deserialize_annotation_source")]
    pub source: Option<String>,
    pub confidence: Option<f32>,
    pub payload: AnnotationPayload,
}

impl Annotation {
    pub fn new(
        range: TimeRange,
        payload: AnnotationPayload,
        source: Option<String>,
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

    /// Compares annotation content while ignoring the generated identity.
    pub fn content_eq(&self, other: &Self) -> bool {
        self.range == other.range
            && self.status == other.status
            && self.source == other.source
            && self.confidence == other.confidence
            && self.payload == other.payload
    }
}

fn deserialize_annotation_source<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    enum LegacyNamed {
        User(String),
        Model(String),
        Stage(String),
    }

    #[derive(Deserialize)]
    enum LegacyUnit {
        User,
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Value {
        LegacyUnit(LegacyUnit),
        LegacyNamed(LegacyNamed),
        Name(String),
    }

    Ok(
        Option::<Value>::deserialize(deserializer)?.map(|value| match value {
            Value::Name(name)
            | Value::LegacyNamed(LegacyNamed::Model(name))
            | Value::LegacyNamed(LegacyNamed::Stage(name)) => name,
            Value::LegacyNamed(LegacyNamed::User(name)) => {
                drop(name);
                "import".to_string()
            }
            Value::LegacyUnit(LegacyUnit::User) => "import".to_string(),
        }),
    )
}

#[cfg(test)]
mod tests {
    use serde::Serialize;

    use super::{Annotation, AnnotationPayload, AnnotationStatus, SpeakerPayload};
    use crate::utils::{DurationMs, TimeRange};

    #[test]
    fn legacy_speaker_string_deserializes_to_structured_payload() {
        let payload: AnnotationPayload =
            serde_json::from_str(r#"{"Speaker":"alice"}"#).expect("legacy speaker payload");

        assert_eq!(
            payload,
            AnnotationPayload::Speaker(SpeakerPayload::new("alice"))
        );
    }

    #[test]
    fn legacy_user_source_deserializes_as_import() {
        #[derive(Serialize)]
        enum LegacySource {
            User,
        }

        #[derive(Serialize)]
        struct LegacyAnnotation {
            id: String,
            range: TimeRange,
            status: AnnotationStatus,
            source: LegacySource,
            confidence: Option<f32>,
            payload: AnnotationPayload,
        }

        let bytes = rmp_serde::to_vec_named(&LegacyAnnotation {
            id: "ann_1".to_string(),
            range: TimeRange::new(DurationMs(0), DurationMs(1)),
            status: AnnotationStatus::Final,
            source: LegacySource::User,
            confidence: None,
            payload: AnnotationPayload::Speech,
        })
        .expect("legacy annotation");
        let annotation: Annotation = rmp_serde::from_slice(&bytes).expect("annotation");

        assert_eq!(annotation.source.as_deref(), Some("import"));
    }
}
