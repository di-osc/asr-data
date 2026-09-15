//! 音频来源描述，以及声道、编码、探测信息等轻量类型。
//!
//! [`AudioSource`] 只保存怎么找到音频，真正的 I/O 和解码发生在
//! [`AudioSource::probe`]、[`AudioSource::load`] 或 [`AudioSource::stream`]。

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{Waveform, decode, local_path_from_urlish, waveform};
use crate::doc::{Audio, AudioStream};

/// 时间轴和声道选择使用的声道标识。
///
/// `Left` / `Right` 是 0 / 1 的规范名称；更大的下标用 [`AudioChannel::Channel`]。
/// `Channel(0)` 和 `Channel(1)` 不是规范形式，校验时会被拒绝。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AudioChannel {
    /// 单声道时间轴。
    Mono,
    /// 立体声左声道，对应下标 0。
    Left,
    /// 立体声右声道，对应下标 1。
    Right,
    /// 第 `n` 个声道；`n >= 2` 才是规范形式。
    Channel(u16),
}

impl AudioChannel {
    /// 用声道下标构造标识：0 → `Left`，1 → `Right`，其余 → `Channel(index)`。
    pub fn from_index(index: u16) -> Self {
        match index {
            0 => Self::Left,
            1 => Self::Right,
            index => Self::Channel(index),
        }
    }

    /// 返回声道下标；`Mono` 没有下标，返回 `None`。
    pub fn index(self) -> Option<u16> {
        match self {
            Self::Mono => None,
            Self::Left => Some(0),
            Self::Right => Some(1),
            Self::Channel(index) => Some(index),
        }
    }

    /// 用于 timeline 键和展示的稳定名称。
    pub fn name(self) -> String {
        match self {
            Self::Mono => "mono".to_string(),
            Self::Left => "left".to_string(),
            Self::Right => "right".to_string(),
            Self::Channel(index) => index.to_string(),
        }
    }

    /// 是否为规范声道：`Channel(0|1)` 应写成 `Left` / `Right`。
    pub fn is_canonical(self) -> bool {
        !matches!(self, Self::Channel(0 | 1))
    }
}

/// 源容器或原始 PCM 的编码种类。
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

/// 解码前的源格式：编码、采样率和声道数。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioFormat {
    /// 容器或 PCM 编码种类。
    pub encoding: AudioEncoding,
    /// 源采样率。
    pub sample_rate: u32,
    /// 源声道数。
    pub channels: u16,
}

/// 不包含样本本身的音频元信息，通常来自 probe。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioInfo {
    /// 解码后（或 PCM 声明的）采样率。
    pub sample_rate: u32,
    /// 声道数。
    pub channels: u16,
    /// 整段音频的帧数。
    pub frame_count: u64,
    /// 解码前的源格式。
    pub source_format: AudioFormat,
}

impl AudioInfo {
    /// 按时长公式 `frames * 1000 / sample_rate` 计算毫秒数（浮点）。
    pub fn duration_ms(&self) -> f64 {
        self.frame_count as f64 * 1000.0 / f64::from(self.sample_rate)
    }

    /// 时间轴使用的整毫秒时长，对帧数向上取整。
    pub fn timeline_duration_ms(&self) -> u64 {
        let millis = u128::from(self.frame_count)
            .saturating_mul(1000)
            .div_ceil(u128::from(self.sample_rate));
        millis.min(u128::from(u64::MAX)) as u64
    }
}

impl AudioFormat {
    /// 构造单声道 PCM S16LE 格式描述。
    pub fn pcm16_mono(sample_rate: u32) -> Self {
        Self {
            encoding: AudioEncoding::PcmS16Le,
            sample_rate,
            channels: 1,
        }
    }
}

/// 尚未解码的音频来源。
///
/// 真正读文件 / 下载 / 解码发生在 [`Self::probe`]、[`Self::load`] 或
/// [`Self::stream`]。PCM 变体已经是样本字节，只是还没转成 `f32`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioSource {
    /// 本地文件路径。
    Path(PathBuf),
    /// HTTP(S) 或 `file://` URL。
    Url(String),
    /// base64 字符串，可带 `data:` 前缀。
    Base64(String),
    /// 带容器或编码头的原始字节。
    EncodedBytes(#[serde(with = "serde_bytes")] Vec<u8>),
    /// little-endian `i16` PCM，采样率和声道数由调用方提供。
    PcmS16Le {
        #[serde(with = "serde_bytes")]
        bytes: Vec<u8>,
        sample_rate: u32,
        channels: u16,
    },
}

impl AudioSource {
    /// 含 `://` 的字符串当成 URL，否则当成本地路径。
    pub fn new(path_or_url: impl Into<String>) -> Self {
        let value = path_or_url.into();
        if value.contains("://") {
            Self::Url(value)
        } else {
            Self::Path(PathBuf::from(value))
        }
    }

    /// 从本地路径构造来源。
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self::Path(path.into())
    }

    /// 从 URL 字符串构造来源。
    pub fn from_url(url: impl Into<String>) -> Self {
        Self::Url(url.into())
    }

    /// 从 base64 字符串构造来源。
    pub fn from_base64(data: impl Into<String>) -> Self {
        Self::Base64(data.into())
    }

    /// 从带容器的编码字节构造来源。
    pub fn from_encoded_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self::EncodedBytes(bytes.into())
    }

    /// 从 PCM S16LE 字节构造来源。
    pub fn from_pcm_s16le(bytes: impl Into<Vec<u8>>, sample_rate: u32, channels: u16) -> Self {
        Self::PcmS16Le {
            bytes: bytes.into(),
            sample_rate,
            channels,
        }
    }

    /// 按变体选择解码路径，得到清洗后的 [`Waveform`]。
    ///
    /// HTTP URL 若其实是本地 `file://` 或普通路径，会改走文件解码。
    ///
    /// # Errors
    ///
    /// 打开、下载、解码或 PCM 对齐失败时返回错误。
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
        waveform::sanitize_samples(&mut waveform.samples);
        Ok(waveform)
    }

    /// 解码完整波形并包装成带随机 ID 的 [`Audio`] 文档。
    ///
    /// # Errors
    ///
    /// 解码失败时返回错误。
    pub fn load(&self) -> anyhow::Result<Audio> {
        self.load_with_id(format!("audio_{}", uuid::Uuid::new_v4().simple()))
    }

    /// 解码完整波形并包装成指定 ID 的 [`Audio`] 文档。
    ///
    /// # Errors
    ///
    /// 解码失败时返回错误。
    pub fn load_with_id(&self, audio_id: impl Into<String>) -> anyhow::Result<Audio> {
        let waveform = self.decode_waveform()?;
        Ok(Audio::with_loaded_waveform(
            audio_id,
            self.clone(),
            waveform,
        ))
    }

    /// 创建 timeline 随迭代增长的 [`AudioStream`]，自动生成文档 ID。
    ///
    /// # Errors
    ///
    /// 探测来源或创建流失败时返回错误。
    pub fn stream(&self, chunk_size_ms: u64) -> anyhow::Result<AudioStream> {
        self.stream_with_id(
            format!("audio_{}", uuid::Uuid::new_v4().simple()),
            chunk_size_ms,
        )
    }

    /// 创建指定 ID 的 [`AudioStream`]。
    ///
    /// 先 [`Self::probe`] 拿到时长和声道，再把 PCM 生产器交给文档层。
    ///
    /// # Errors
    ///
    /// 探测失败或 `chunk_size_ms` 无效时返回错误。
    pub fn stream_with_id(
        &self,
        audio_id: impl Into<String>,
        chunk_size_ms: u64,
    ) -> anyhow::Result<AudioStream> {
        let info = self.probe()?;
        AudioStream::new(audio_id, self.clone(), info, chunk_size_ms)
    }

    /// 探测采样率、声道和帧数，不把波形整段读进内存（PCM 除外）。
    ///
    /// # Errors
    ///
    /// 打开、下载、探测失败，或 PCM 参数非法时返回错误。
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

    /// 异步探测 [`AudioInfo`]。
    ///
    /// HTTP(S) URL 先异步下载再在阻塞线程里 probe；其它来源整段放到 worker。
    ///
    /// # Errors
    ///
    /// 下载、worker 崩溃或探测失败时返回错误。
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

    /// 在阻塞线程中异步执行 [`Self::load`]。
    ///
    /// # Errors
    ///
    /// worker 崩溃或解码失败时返回错误。
    pub async fn aload(&self) -> anyhow::Result<Audio> {
        let source = self.clone();
        tokio::task::spawn_blocking(move || source.load())
            .await
            .map_err(|error| anyhow::anyhow!("audio loader worker failed: {error}"))?
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
