use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

use crate::audio::{AudioChannel, AudioEncoding, AudioFormat, AudioInfo, AudioSource, Waveform};
use crate::timeline::{Timeline, TimelineSpanError};
use crate::utils::DurationMs;

/// An audio source together with all annotations and per-audio metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Audio {
    #[serde(default)]
    pub id: String,
    pub source: AudioSource,
    pub info: AudioInfo,
    pub(crate) timelines: BTreeMap<AudioChannel, Timeline>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(skip)]
    pub(crate) waveform: Option<Waveform>,
    #[serde(skip)]
    pub(crate) stream_active: bool,
}

impl Audio {
    pub fn new(source: impl Into<AudioSource>) -> anyhow::Result<Self> {
        Self::with_id(format!("audio_{}", uuid::Uuid::new_v4().simple()), source)
    }

    pub fn with_id(
        audio_id: impl Into<String>,
        source: impl Into<AudioSource>,
    ) -> anyhow::Result<Self> {
        Self::with_id_from_source(audio_id, source)
    }

    pub fn from_source(source: impl Into<AudioSource>) -> anyhow::Result<Self> {
        Self::new(source)
    }

    pub fn with_id_from_source(
        audio_id: impl Into<String>,
        source: impl Into<AudioSource>,
    ) -> anyhow::Result<Self> {
        let source = source.into();
        let info = source.probe()?;
        Ok(Self::with_id_from_info(audio_id, source, &info))
    }

    pub async fn afrom_source(source: impl Into<AudioSource>) -> anyhow::Result<Self> {
        let source = source.into();
        let info = source.aprobe().await?;
        Ok(Self::from_info(source, &info))
    }

    pub async fn with_id_afrom_source(
        audio_id: impl Into<String>,
        source: impl Into<AudioSource>,
    ) -> anyhow::Result<Self> {
        let audio_id = audio_id.into();
        let source = source.into();
        let info = source.aprobe().await?;
        Ok(Self::with_id_from_info(audio_id, source, &info))
    }

    pub fn from_info(source: impl Into<AudioSource>, info: &AudioInfo) -> Self {
        Self::with_id_from_info(
            format!("audio_{}", uuid::Uuid::new_v4().simple()),
            source,
            info,
        )
    }

    pub fn with_id_from_info(
        audio_id: impl Into<String>,
        source: impl Into<AudioSource>,
        info: &AudioInfo,
    ) -> Self {
        let mut doc = Self {
            id: audio_id.into(),
            source: source.into(),
            info: info.clone(),
            timelines: BTreeMap::new(),
            metadata: BTreeMap::new(),
            waveform: None,
            stream_active: false,
        };
        let duration = DurationMs(info.timeline_duration_ms());
        if info.channels == 1 {
            doc.timelines
                .insert(AudioChannel::Mono, Timeline::new(doc.id.clone(), duration));
        } else {
            for index in 0..info.channels {
                doc.timelines.insert(
                    AudioChannel::from_index(index),
                    Timeline::new(doc.id.clone(), duration),
                );
            }
        }
        doc
    }

    pub(crate) fn with_loaded_waveform(
        audio_id: impl Into<String>,
        source: impl Into<AudioSource>,
        waveform: Waveform,
    ) -> Self {
        let info = AudioInfo {
            sample_rate: waveform.sample_rate,
            channels: waveform.channels,
            frame_count: waveform.frame_count() as u64,
            source_format: waveform.source_format.clone().unwrap_or(AudioFormat {
                encoding: AudioEncoding::Unknown,
                sample_rate: waveform.sample_rate,
                channels: waveform.channels,
            }),
        };
        let mut audio = Self::with_id_from_info(audio_id, source, &info);
        audio.waveform = Some(waveform);
        audio
    }

    pub fn is_loaded(&self) -> bool {
        self.waveform.is_some()
    }

    pub fn load(&mut self) -> anyhow::Result<&Waveform> {
        if self.waveform.is_none() {
            self.waveform = Some(self.source.decode_waveform()?);
        }
        Ok(self.waveform.as_ref().expect("waveform was just loaded"))
    }

    pub fn as_waveform(&mut self) -> anyhow::Result<Waveform> {
        Ok(self.load()?.clone())
    }

    pub fn waveform_for_channel(&mut self, channel: AudioChannel) -> anyhow::Result<Waveform> {
        validate_channel(channel)?;
        let waveform = self.load()?;
        match channel {
            AudioChannel::Mono => waveform.to_mono().map_err(Into::into),
            AudioChannel::Left => waveform.channel(0).map_err(Into::into),
            AudioChannel::Right => waveform.channel(1).map_err(Into::into),
            AudioChannel::Channel(index) => waveform.channel(index).map_err(Into::into),
        }
    }

    pub fn unload(&mut self) {
        self.waveform = None;
    }

    #[cfg(feature = "python-bindings")]
    pub(crate) fn begin_stream(&mut self) -> anyhow::Result<()> {
        if self.stream_active {
            anyhow::bail!("audio already has an active stream");
        }
        self.stream_active = true;
        Ok(())
    }

    #[cfg(feature = "python-bindings")]
    pub(crate) fn end_stream(&mut self) {
        self.stream_active = false;
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
        let expected = Some(DurationMs(self.info.timeline_duration_ms()));
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
        Some(DurationMs(self.info.timeline_duration_ms()))
    }

    pub fn validate(&self) -> Result<(), AudioValidationError> {
        if self.id.trim().is_empty() {
            return Err(AudioValidationError::EmptyAudioId);
        }
        if self.info.sample_rate == 0 {
            return Err(AudioValidationError::InvalidAudioInfoSampleRate);
        }
        if self.info.channels == 0 {
            return Err(AudioValidationError::InvalidAudioInfoChannels);
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
            for annotation in timeline.all_spans() {
                if annotation.range.end > timeline.duration {
                    return Err(AudioValidationError::AnnotationOutOfBounds {
                        channel: *channel,
                        annotation_id: annotation.id.clone(),
                        end: annotation.range.end,
                        duration: timeline.duration,
                    });
                }
            }
            timeline.validate_spans().map_err(|error| {
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
    #[error("audio info sample rate must be greater than zero")]
    InvalidAudioInfoSampleRate,
    #[error("audio info channel count must be greater than zero")]
    InvalidAudioInfoChannels,
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
        error: TimelineSpanError,
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
