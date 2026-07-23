use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{Waveform, data, decode, local_path_from_urlish};
use crate::doc::Audio;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AudioChannel {
    Mono,
    Left,
    Right,
    Channel(u16),
}

impl AudioChannel {
    pub fn from_index(index: u16) -> Self {
        match index {
            0 => Self::Left,
            1 => Self::Right,
            index => Self::Channel(index),
        }
    }

    pub fn index(self) -> Option<u16> {
        match self {
            Self::Mono => None,
            Self::Left => Some(0),
            Self::Right => Some(1),
            Self::Channel(index) => Some(index),
        }
    }

    pub fn name(self) -> String {
        match self {
            Self::Mono => "mono".to_string(),
            Self::Left => "left".to_string(),
            Self::Right => "right".to_string(),
            Self::Channel(index) => index.to_string(),
        }
    }

    pub fn is_canonical(self) -> bool {
        !matches!(self, Self::Channel(0 | 1))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioEncoding {
    Wav,
    Flac,
    Mp3,
    Ogg,
    PcmS16Le,
    Other(String),
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioFormat {
    pub encoding: AudioEncoding,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub frame_count: u64,
    pub source_format: AudioFormat,
}

impl AudioInfo {
    pub fn duration_ms(&self) -> f64 {
        self.frame_count as f64 * 1000.0 / f64::from(self.sample_rate)
    }

    pub fn timeline_duration_ms(&self) -> u64 {
        let millis = u128::from(self.frame_count)
            .saturating_mul(1000)
            .div_ceil(u128::from(self.sample_rate));
        millis.min(u128::from(u64::MAX)) as u64
    }
}

impl AudioFormat {
    pub fn pcm16_mono(sample_rate: u32) -> Self {
        Self {
            encoding: AudioEncoding::PcmS16Le,
            sample_rate,
            channels: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioSource {
    Path(PathBuf),
    Url(String),
    Base64(String),
    EncodedBytes(#[serde(with = "serde_bytes")] Vec<u8>),
    PcmS16Le {
        #[serde(with = "serde_bytes")]
        bytes: Vec<u8>,
        sample_rate: u32,
        channels: u16,
    },
}

impl AudioSource {
    pub fn new(path_or_url: impl Into<String>) -> Self {
        let value = path_or_url.into();
        if value.contains("://") {
            Self::Url(value)
        } else {
            Self::Path(PathBuf::from(value))
        }
    }

    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self::Path(path.into())
    }

    pub fn from_url(url: impl Into<String>) -> Self {
        Self::Url(url.into())
    }

    pub fn from_base64(data: impl Into<String>) -> Self {
        Self::Base64(data.into())
    }

    pub fn from_encoded_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self::EncodedBytes(bytes.into())
    }

    pub fn from_pcm_s16le(bytes: impl Into<Vec<u8>>, sample_rate: u32, channels: u16) -> Self {
        Self::PcmS16Le {
            bytes: bytes.into(),
            sample_rate,
            channels,
        }
    }

    pub(crate) fn decode_waveform(&self) -> anyhow::Result<Waveform> {
        let waveform = match self {
            Self::Path(path) => decode::decode_path_audio(path)?,
            Self::Url(url) => {
                if let Some(path) = local_path_from_urlish(url) {
                    decode::decode_path_audio(&path)?
                } else {
                    decode::decode_url_audio(url)?
                }
            }
            Self::Base64(b64) => decode::decode_base64_audio(b64)?,
            Self::EncodedBytes(bytes) => decode::decode_bytes_audio(bytes.clone())?,
            Self::PcmS16Le {
                bytes,
                sample_rate,
                channels,
            } => Waveform::from_i16_pcm_bytes_with_channels(bytes, *sample_rate, *channels)?,
        };
        let mut waveform = waveform;
        data::sanitize_samples(&mut waveform.samples);
        Ok(waveform)
    }

    pub fn open(&self) -> anyhow::Result<Audio> {
        Audio::from_source(self.clone())
    }

    pub fn open_with_id(&self, audio_id: impl Into<String>) -> anyhow::Result<Audio> {
        Audio::with_id_from_source(audio_id, self.clone())
    }

    pub fn load(&self) -> anyhow::Result<Audio> {
        self.load_with_id(format!("audio_{}", uuid::Uuid::new_v4().simple()))
    }

    pub fn load_with_id(&self, audio_id: impl Into<String>) -> anyhow::Result<Audio> {
        let waveform = self.decode_waveform()?;
        Ok(Audio::with_loaded_waveform(
            audio_id,
            self.clone(),
            waveform,
        ))
    }

    pub fn probe(&self) -> anyhow::Result<AudioInfo> {
        match self {
            Self::Path(path) => decode::probe_path(path),
            Self::Url(url) => {
                if let Some(path) = local_path_from_urlish(url) {
                    decode::probe_path(&path)
                } else {
                    decode::probe_url(url)
                }
            }
            Self::Base64(data) => decode::probe_base64(data),
            Self::EncodedBytes(bytes) => decode::probe_bytes(bytes.clone()),
            Self::PcmS16Le {
                bytes,
                sample_rate,
                channels,
            } => {
                if *sample_rate == 0 {
                    anyhow::bail!("sample rate must be greater than zero");
                }
                if *channels == 0 {
                    anyhow::bail!("channel count must be greater than zero");
                }
                if !bytes.len().is_multiple_of(2 * usize::from(*channels)) {
                    anyhow::bail!(
                        "PCM byte length {} is not a whole number of {}-channel frames",
                        bytes.len(),
                        channels
                    );
                }
                Ok(AudioInfo {
                    sample_rate: *sample_rate,
                    channels: *channels,
                    frame_count: (bytes.len() / 2 / usize::from(*channels)) as u64,
                    source_format: AudioFormat {
                        encoding: AudioEncoding::PcmS16Le,
                        sample_rate: *sample_rate,
                        channels: *channels,
                    },
                })
            }
        }
    }

    pub async fn aprobe(&self) -> anyhow::Result<AudioInfo> {
        match self {
            Self::Url(url) if url.starts_with("http://") || url.starts_with("https://") => {
                let bytes = decode::download_url_bytes(url).await?;
                tokio::task::spawn_blocking(move || decode::probe_bytes(bytes))
                    .await
                    .map_err(|error| anyhow::anyhow!("audio probe worker failed: {error}"))?
            }
            source => {
                let source = source.clone();
                tokio::task::spawn_blocking(move || source.probe())
                    .await
                    .map_err(|error| anyhow::anyhow!("audio probe worker failed: {error}"))?
            }
        }
    }

    pub async fn aload(&self) -> anyhow::Result<Audio> {
        let source = self.clone();
        tokio::task::spawn_blocking(move || source.load())
            .await
            .map_err(|error| anyhow::anyhow!("audio loader worker failed: {error}"))?
    }

    pub async fn aopen(&self) -> anyhow::Result<Audio> {
        let source = self.clone();
        let info = source.aprobe().await?;
        Ok(Audio::from_info(source, &info))
    }
}

impl From<&str> for AudioSource {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for AudioSource {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<PathBuf> for AudioSource {
    fn from(value: PathBuf) -> Self {
        Self::Path(value)
    }
}
