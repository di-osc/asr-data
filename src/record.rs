use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

use crate::{Annotation, AudioChannel, AudioEncoding, AudioSource, DurationMs, Timeline, Waveform};

/// An audio source together with all annotations and per-audio metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Audio {
    pub source: AudioSource,
    pub(crate) timelines: BTreeMap<AudioChannel, Timeline>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl Audio {
    pub fn new(source: impl Into<AudioSource>) -> Self {
        Self::with_id(format!("audio_{}", uuid::Uuid::new_v4().simple()), source)
    }

    pub fn with_id(audio_id: impl Into<String>, source: impl Into<AudioSource>) -> Self {
        let timeline = Timeline::new(audio_id);
        Self {
            source: source.into(),
            timelines: BTreeMap::from([(AudioChannel::Mono, timeline)]),
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
    ) -> Result<&mut Timeline, AudioChannelError> {
        validate_channel(channel)?;
        let audio_id = self.mono_timeline().audio_id.clone();
        Ok(self
            .timelines
            .entry(channel)
            .or_insert_with(|| Timeline::new(audio_id)))
    }

    pub fn mono_timeline(&self) -> &Timeline {
        self.timelines
            .get(&AudioChannel::Mono)
            .expect("Audio always contains a mono timeline")
    }

    pub fn mono_timeline_mut(&mut self) -> &mut Timeline {
        self.timelines
            .get_mut(&AudioChannel::Mono)
            .expect("Audio always contains a mono timeline")
    }

    pub fn timelines(&self) -> &BTreeMap<AudioChannel, Timeline> {
        &self.timelines
    }

    pub fn with_metadata_value(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn id_for_index(index: usize) -> String {
        format!("audio-{index}")
    }

    pub fn audio_id(&self) -> String {
        sanitize_audio_id(&self.mono_timeline().audio_id)
    }

    pub fn set_audio_id(&mut self, audio_id: impl Into<String>) {
        let audio_id = audio_id.into();
        for timeline in self.timelines.values_mut() {
            timeline.audio_id.clone_from(&audio_id);
        }
    }

    pub fn load(&self) -> anyhow::Result<Waveform> {
        self.source.load()
    }

    #[cfg(feature = "audio-loading")]
    pub async fn aload(&self) -> anyhow::Result<Waveform> {
        self.source.aload().await
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("channel index {index} has a named representation")]
pub struct AudioChannelError {
    pub index: u16,
}

fn validate_channel(channel: AudioChannel) -> Result<(), AudioChannelError> {
    match channel {
        AudioChannel::Channel(index @ 0..=1) => Err(AudioChannelError { index }),
        _ => Ok(()),
    }
}

impl From<AudioSource> for Audio {
    fn from(source: AudioSource) -> Self {
        Self::new(source)
    }
}

impl From<&str> for Audio {
    fn from(source: &str) -> Self {
        Self::new(AudioSource::new(source))
    }
}

impl From<String> for Audio {
    fn from(source: String) -> Self {
        Self::new(AudioSource::new(source))
    }
}

#[derive(Debug, Error)]
pub enum LegacyImportError {
    #[error("failed to decode legacy MessagePack audio data: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
    #[error("legacy audio data I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Reads the original MessagePack list formats for migration into [`crate::AudioDb`].
/// New data should be stored in an `AudioDb`, not written back to MessagePack.
pub fn read_legacy_msgpack(path: impl AsRef<Path>) -> Result<Vec<Audio>, LegacyImportError> {
    let path = path.as_ref();
    let list_result = LegacyAudioList::deserialize(&mut rmp_serde::Deserializer::new(
        BufReader::new(File::open(path)?),
    ));
    match list_result {
        Ok(list) => Ok(list.audios.or(list.records).unwrap_or_default()),
        Err(list_error) => {
            let single = LegacyAudio::deserialize(&mut rmp_serde::Deserializer::new(
                BufReader::new(File::open(path)?),
            ));
            single
                .map(|audio| vec![audio.0])
                .map_err(|_| LegacyImportError::Decode(list_error))
        }
    }
}

struct LegacyAudioList {
    audios: Option<Vec<Audio>>,
    records: Option<Vec<Audio>>,
}

struct LegacyAudio(Audio);

#[derive(Deserialize)]
struct LegacyAudioWire {
    #[serde(default)]
    source: Option<AudioSource>,
    #[serde(default)]
    timeline: Option<LegacyTimeline>,
    #[serde(default)]
    metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    audio: Option<Box<LegacyAudio>>,
    #[serde(default)]
    media: Option<LegacyAudioAsset>,
}

#[derive(Deserialize)]
struct LegacyTimeline {
    #[serde(default)]
    id: String,
    #[serde(default, alias = "media_id")]
    audio_id: String,
    #[serde(default)]
    duration: Option<DurationMs>,
    #[serde(default)]
    annotations: Vec<Annotation>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LegacyAudioAsset {
    Uri {
        uri: String,
        format: LegacyAudioFormat,
        duration: Option<DurationMs>,
        sha256: Option<String>,
    },
    Embedded {
        #[serde(with = "serde_bytes")]
        bytes: Vec<u8>,
        format: LegacyAudioFormat,
        duration: Option<DurationMs>,
        sha256: Option<String>,
    },
}

#[derive(Deserialize)]
struct LegacyAudioFormat {
    sample_rate: Option<u32>,
    channels: Option<u16>,
    encoding: AudioEncoding,
}

impl<'de> Deserialize<'de> for LegacyAudio {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LegacyAudioWire::deserialize(deserializer)?;
        if let Some(audio) = wire.audio {
            let mut audio = audio.0;
            for (key, value) in wire.metadata {
                audio.metadata.entry(key).or_insert(value);
            }
            return Ok(Self(audio));
        }

        let mut timeline = wire.timeline.unwrap_or(LegacyTimeline {
            id: String::new(),
            audio_id: String::new(),
            duration: None,
            annotations: Vec::new(),
        });
        if timeline.audio_id.trim().is_empty() {
            timeline.audio_id = if timeline.id.trim().is_empty() {
                "audio".to_string()
            } else {
                timeline.id.clone()
            };
        }
        if timeline.id.trim().is_empty() {
            timeline.id = format!("tl_{}", uuid::Uuid::new_v4().simple());
        }

        let mut metadata = wire.metadata;
        let source = match (wire.source, wire.media) {
            (Some(source), _) => source,
            (None, Some(asset)) => {
                let migrated = migrate_legacy_asset(asset);
                if timeline.duration.is_none() {
                    timeline.duration = migrated.duration;
                }
                if let Some(sha256) = migrated.sha256 {
                    metadata.insert("sha256".to_string(), serde_json::Value::String(sha256));
                }
                metadata.insert("legacy_format".to_string(), migrated.format);
                migrated.source
            }
            (None, None) => return Err(serde::de::Error::missing_field("source")),
        };

        let timeline = Timeline {
            id: timeline.id,
            audio_id: timeline.audio_id,
            duration: timeline.duration,
            annotations: timeline.annotations,
        };
        Ok(Self(Audio {
            source,
            timelines: BTreeMap::from([(AudioChannel::Mono, timeline)]),
            metadata,
        }))
    }
}

impl<'de> Deserialize<'de> for LegacyAudioList {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            #[serde(default)]
            audios: Option<Vec<LegacyAudio>>,
            #[serde(default)]
            records: Option<Vec<LegacyAudio>>,
        }

        let wire = Wire::deserialize(deserializer)?;
        Ok(Self {
            audios: wire
                .audios
                .map(|audios| audios.into_iter().map(|audio| audio.0).collect()),
            records: wire
                .records
                .map(|audios| audios.into_iter().map(|audio| audio.0).collect()),
        })
    }
}

struct MigratedAsset {
    source: AudioSource,
    duration: Option<DurationMs>,
    sha256: Option<String>,
    format: serde_json::Value,
}

fn migrate_legacy_asset(asset: LegacyAudioAsset) -> MigratedAsset {
    match asset {
        LegacyAudioAsset::Uri {
            uri,
            format,
            duration,
            sha256,
        } => MigratedAsset {
            source: AudioSource::new(uri),
            duration,
            sha256,
            format: legacy_format_value(&format),
        },
        LegacyAudioAsset::Embedded {
            bytes,
            format,
            duration,
            sha256,
        } => {
            let source = if format.encoding == AudioEncoding::PcmS16Le {
                match (format.sample_rate, format.channels) {
                    (Some(sample_rate), Some(channels)) => {
                        AudioSource::from_pcm_s16le(bytes, sample_rate, channels)
                    }
                    _ => AudioSource::from_encoded_bytes(bytes),
                }
            } else {
                AudioSource::from_encoded_bytes(bytes)
            };
            MigratedAsset {
                source,
                duration,
                sha256,
                format: legacy_format_value(&format),
            }
        }
    }
}

fn legacy_format_value(format: &LegacyAudioFormat) -> serde_json::Value {
    serde_json::json!({
        "encoding": format!("{:?}", format.encoding),
        "sample_rate": format.sample_rate,
        "channels": format.channels,
    })
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
