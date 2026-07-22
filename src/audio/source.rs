use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{Audio, data, decode, local_path_from_urlish, transform_loaded_audio};

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

    pub fn load(&self) -> anyhow::Result<Audio> {
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
            } => Audio::from_i16_pcm_bytes_with_channels(bytes, *sample_rate, *channels)?,
        };
        let mut waveform = waveform;
        data::sanitize_samples(&mut waveform.samples);
        Ok(waveform)
    }

    pub fn load_with(&self, sample_rate: Option<u32>, mono: Option<bool>) -> anyhow::Result<Audio> {
        transform_loaded_audio(self.load()?, sample_rate, mono)
    }

    pub async fn aload(&self) -> anyhow::Result<Audio> {
        self.aload_with(None, None).await
    }

    pub async fn aload_with(
        &self,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> anyhow::Result<Audio> {
        let waveform = match self {
            Self::Url(url) if url.starts_with("http://") || url.starts_with("https://") => {
                let bytes = decode::download_url_bytes(url).await?;
                tokio::task::spawn_blocking(move || decode::decode_bytes_audio(bytes))
                    .await
                    .map_err(|error| anyhow::anyhow!("audio decoder worker failed: {error}"))?
            }
            source => {
                let source = source.clone();
                tokio::task::spawn_blocking(move || source.load())
                    .await
                    .map_err(|error| anyhow::anyhow!("audio loader worker failed: {error}"))?
            }
        }?;
        transform_loaded_audio(waveform, sample_rate, mono)
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
