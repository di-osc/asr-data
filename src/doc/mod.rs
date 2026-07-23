use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

use crate::audio::{AudioChannel, AudioSource};
use crate::timeline::{Timeline, TimelineAnnotationError};
use crate::utils::DurationMs;

/// An audio source together with all annotations and per-audio metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioDoc {
    #[serde(default)]
    pub id: String,
    pub source: AudioSource,
    pub(crate) timelines: BTreeMap<AudioChannel, Timeline>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl AudioDoc {
    pub fn new(source: impl Into<AudioSource>) -> Self {
        Self::with_id(format!("audio_{}", uuid::Uuid::new_v4().simple()), source)
    }

    pub fn with_id(audio_id: impl Into<String>, source: impl Into<AudioSource>) -> Self {
        let audio_id = audio_id.into();
        Self {
            id: audio_id,
            source: source.into(),
            timelines: BTreeMap::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_timeline(mut self, timeline: Timeline) -> Self {
        let audio_id = timeline.audio_id.clone();
        self.set_audio_id(audio_id);
        self.timelines.insert(AudioChannel::Mono, timeline);
        self
    }

    pub fn timeline(&self, channel: AudioChannel) -> Result<Option<&Timeline>, AudioChannelError> {
        validate_channel(channel)?;
        Ok(self.timelines.get(&channel))
    }

    pub fn timeline_mut(
        &mut self,
        channel: AudioChannel,
    ) -> Result<Option<&mut Timeline>, AudioChannelError> {
        validate_channel(channel)?;
        Ok(self.timelines.get_mut(&channel))
    }

    pub fn ensure_timeline(
        &mut self,
        channel: AudioChannel,
        duration: Option<DurationMs>,
    ) -> Result<&mut Timeline, AudioTimelineError> {
        validate_channel(channel).map_err(AudioTimelineError::InvalidChannel)?;
        let expected = self
            .timelines
            .values()
            .next()
            .map(|timeline| timeline.duration);
        let duration = match (expected, duration) {
            (None, None) => return Err(AudioTimelineError::MissingDuration),
            (None, Some(duration)) | (Some(duration), None) => duration,
            (Some(expected), Some(found)) if expected == found => expected,
            (Some(expected), Some(found)) => {
                return Err(AudioTimelineError::DurationMismatch { expected, found });
            }
        };
        let audio_id = self.id.clone();
        Ok(self
            .timelines
            .entry(channel)
            .or_insert_with(|| Timeline::new(audio_id, duration)))
    }

    pub fn mono_timeline(&self) -> Option<&Timeline> {
        self.timelines.get(&AudioChannel::Mono)
    }

    pub fn mono_timeline_mut(&mut self) -> Option<&mut Timeline> {
        self.timelines.get_mut(&AudioChannel::Mono)
    }

    pub fn timelines(&self) -> &BTreeMap<AudioChannel, Timeline> {
        &self.timelines
    }

    pub fn remove_timeline(
        &mut self,
        channel: AudioChannel,
    ) -> Result<Option<Timeline>, AudioChannelError> {
        validate_channel(channel)?;
        Ok(self.timelines.remove(&channel))
    }

    pub fn with_metadata_value(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn id_for_index(index: usize) -> String {
        format!("audio-{index}")
    }

    pub fn audio_id(&self) -> String {
        sanitize_audio_id(&self.id)
    }

    pub fn set_audio_id(&mut self, audio_id: impl Into<String>) {
        let audio_id = audio_id.into();
        self.id.clone_from(&audio_id);
        for timeline in self.timelines.values_mut() {
            timeline.audio_id.clone_from(&audio_id);
        }
    }

    pub fn timeline_duration(&self) -> Option<DurationMs> {
        self.timelines
            .values()
            .next()
            .map(|timeline| timeline.duration)
    }

    pub fn validate(&self) -> Result<(), AudioValidationError> {
        if self.id.trim().is_empty() {
            return Err(AudioValidationError::EmptyAudioId);
        }
        let expected_duration = self.timeline_duration();
        for (channel, timeline) in &self.timelines {
            if !channel.is_canonical() {
                return Err(AudioValidationError::NonCanonicalChannel { channel: *channel });
            }
            if timeline.audio_id != self.id {
                return Err(AudioValidationError::TimelineAudioIdMismatch {
                    channel: *channel,
                    expected: self.id.clone(),
                    found: timeline.audio_id.clone(),
                });
            }
            if Some(timeline.duration) != expected_duration {
                return Err(AudioValidationError::TimelineDurationMismatch {
                    channel: *channel,
                    expected: expected_duration.expect("a timeline established the duration"),
                    found: timeline.duration,
                });
            }
            for annotation in &timeline.reference {
                if annotation.source.is_some() {
                    return Err(AudioValidationError::ReferenceAnnotationHasSource {
                        channel: *channel,
                        annotation_id: annotation.id.clone(),
                    });
                }
            }
            for annotation in &timeline.prediction {
                if annotation
                    .source
                    .as_deref()
                    .is_none_or(|source| source.trim().is_empty())
                {
                    return Err(AudioValidationError::PredictionAnnotationMissingSource {
                        channel: *channel,
                        annotation_id: annotation.id.clone(),
                    });
                }
            }
            for annotation in timeline.all_annotations() {
                if annotation.range.end > timeline.duration {
                    return Err(AudioValidationError::AnnotationOutOfBounds {
                        channel: *channel,
                        annotation_id: annotation.id.clone(),
                        end: annotation.range.end,
                        duration: timeline.duration,
                    });
                }
            }
            timeline.validate_annotations().map_err(|error| {
                AudioValidationError::InvalidAnnotations {
                    channel: *channel,
                    error,
                }
            })?;
        }
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AudioValidationError {
    #[error("audio id must not be empty")]
    EmptyAudioId,
    #[error("audio channel {channel:?} is not canonical")]
    NonCanonicalChannel { channel: AudioChannel },
    #[error("timeline {channel:?} audio id mismatch: expected {expected:?}, found {found:?}")]
    TimelineAudioIdMismatch {
        channel: AudioChannel,
        expected: String,
        found: String,
    },
    #[error("timeline {channel:?} duration mismatch: expected {expected:?}, found {found:?}")]
    TimelineDurationMismatch {
        channel: AudioChannel,
        expected: DurationMs,
        found: DurationMs,
    },
    #[error(
        "annotation {annotation_id:?} on {channel:?} ends at {end:?}, past audio duration {duration:?}"
    )]
    AnnotationOutOfBounds {
        channel: AudioChannel,
        annotation_id: String,
        end: DurationMs,
        duration: DurationMs,
    },
    #[error("reference annotation {annotation_id:?} on {channel:?} must not have a source")]
    ReferenceAnnotationHasSource {
        channel: AudioChannel,
        annotation_id: String,
    },
    #[error("prediction annotation {annotation_id:?} on {channel:?} must have a non-empty source")]
    PredictionAnnotationMissingSource {
        channel: AudioChannel,
        annotation_id: String,
    },
    #[error("invalid annotations on {channel:?}: {error}")]
    InvalidAnnotations {
        channel: AudioChannel,
        error: TimelineAnnotationError,
    },
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("channel index {index} has a named representation")]
pub struct AudioChannelError {
    pub index: u16,
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum AudioTimelineError {
    #[error(transparent)]
    InvalidChannel(AudioChannelError),
    #[error("duration is required when creating the first timeline")]
    MissingDuration,
    #[error("timeline duration mismatch: expected {expected:?}, found {found:?}")]
    DurationMismatch {
        expected: DurationMs,
        found: DurationMs,
    },
}

fn validate_channel(channel: AudioChannel) -> Result<(), AudioChannelError> {
    match channel {
        AudioChannel::Channel(index @ 0..=1) => Err(AudioChannelError { index }),
        _ => Ok(()),
    }
}

impl From<AudioSource> for AudioDoc {
    fn from(source: AudioSource) -> Self {
        Self::new(source)
    }
}

impl From<&str> for AudioDoc {
    fn from(source: &str) -> Self {
        Self::new(AudioSource::new(source))
    }
}

impl From<String> for AudioDoc {
    fn from(source: String) -> Self {
        Self::new(AudioSource::new(source))
    }
}

fn sanitize_audio_id(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}
