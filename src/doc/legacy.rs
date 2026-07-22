use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde::{Deserialize, Deserializer};
use thiserror::Error;

use super::AudioDoc;
use crate::audio::{AudioChannel, AudioEncoding, AudioSource};
use crate::timeline::{Annotation, Timeline};
use crate::utils::DurationMs;

#[derive(Debug, Error)]
pub enum LegacyImportError {
    #[error("failed to decode legacy MessagePack audio data: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
    #[error("legacy audio data I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Reads the original MessagePack list formats for migration into [`crate::AudioDb`].
/// New data should be stored in an `AudioDb`, not written back to MessagePack.
pub fn read_legacy_msgpack(path: impl AsRef<Path>) -> Result<Vec<AudioDoc>, LegacyImportError> {
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
    audios: Option<Vec<AudioDoc>>,
    records: Option<Vec<AudioDoc>>,
}

struct LegacyAudio(AudioDoc);

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
            if audio.id.trim().is_empty() {
                audio.id = audio
                    .timelines
                    .values()
                    .next()
                    .map(|timeline| timeline.audio_id.clone())
                    .unwrap_or_else(|| "audio".to_string());
            }
            audio.set_audio_id(audio.id.clone());
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

        for annotation in &mut timeline.annotations {
            if annotation
                .source
                .as_deref()
                .is_none_or(|source| source.trim().is_empty())
            {
                annotation.source = Some("import".to_string());
            }
        }
        let timeline = Timeline {
            id: timeline.id,
            audio_id: timeline.audio_id,
            duration: timeline.duration.unwrap_or_default(),
            reference: Vec::new(),
            prediction: timeline.annotations,
        };
        Ok(Self(AudioDoc {
            id: timeline.audio_id.clone(),
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
