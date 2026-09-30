//! 音频来源描述，以及声道、编码、探测信息等轻量类型。
//!
//! [`AudioSource`] 只保存怎么找到音频，真正的 I/O 和解码发生在
//! [`AudioSource::probe`]、[`AudioSource::load`] 或 [`AudioSource::stream`]。

use std::future::Future;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use super::{Waveform, decode, local_path_from_urlish, waveform};
use crate::doc::{Audio, AudioStream, new_audio_id};

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
    pub fn timeline_duration_ms(&self) -> usize {
        let millis = u128::from(self.frame_count)
            .saturating_mul(1000)
            .div_ceil(u128::from(self.sample_rate));
        millis.min(usize::MAX as u128) as usize
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
    /// ModelScope 数据集仓库中的单个音频文件。
    ///
    /// 只保存仓库身份。下载发生在 [`Self::probe`]、[`Self::load`] 或
    /// [`Self::stream`]，缓存由 modelhub 管理。
    ModelScope {
        /// 数据集仓库 ID，例如 `org/name`。
        repo_id: String,
        /// 仓库内相对路径，例如 `wav/a.wav`。
        file_path: String,
        /// 仓库 revision；构造时缺省为 `master`。
        revision: String,
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

    /// 从 ModelScope 数据集中的单个文件构造来源，不立即下载。
    ///
    /// `revision` 为 `None` 时使用 `master`。`repo_id`、`file_path` 和显式
    /// `revision` 去掉首尾空白后不能为空。
    ///
    /// # Errors
    ///
    /// 仓库 ID、文件路径或 revision 为空时返回错误。
    pub fn from_modelscope(
        repo_id: impl Into<String>,
        file_path: impl Into<String>,
        revision: Option<&str>,
    ) -> anyhow::Result<Self> {
        let (repo_id, file_path, revision) =
            normalize_modelscope_identity(repo_id, file_path, revision)?;
        Ok(Self::ModelScope {
            repo_id,
            file_path,
            revision,
        })
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
            Self::ModelScope {
                repo_id,
                file_path,
                revision,
            } => {
                // 先落到 modelhub 缓存，再按本地文件解码。
                let path = materialize_modelscope_file(repo_id, file_path, revision)?;
                decode::decode_path_audio(&path)?
            }
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
        self.load_with_id(new_audio_id())
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
        self.stream_with(chunk_size_ms, None, None)
    }

    /// 创建 [`AudioStream`]，并可在吐块前重采样或转单声道。
    ///
    /// `sample_rate` 为 `None` 时保持源采样率。`mono` 为 `Some(true)` 时把多声道
    /// 平均成单声道；默认保持原声道，便于按声道标注。
    ///
    /// # Errors
    ///
    /// 探测失败、目标采样率为 0，或 `chunk_size_ms` 无效时返回错误。
    pub fn stream_with(
        &self,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> anyhow::Result<AudioStream> {
        self.open_stream(new_audio_id(), chunk_size_ms, sample_rate, mono)
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
        self.open_stream(audio_id, chunk_size_ms, None, None)
    }

    /// 探测并构造文档流，把输出格式变换传给 PCM 生产器。
    ///
    /// # Errors
    ///
    /// 探测失败、目标采样率为 0，或无法创建上游生产器时返回错误。
    fn open_stream(
        &self,
        audio_id: impl Into<String>,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> anyhow::Result<AudioStream> {
        let info = crate::audio::stream_output_info(self.probe()?, sample_rate, mono)?;
        AudioStream::new(
            audio_id,
            self.clone(),
            info,
            chunk_size_ms,
            sample_rate,
            mono,
        )
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
            Self::ModelScope {
                repo_id,
                file_path,
                revision,
            } => {
                let path = materialize_modelscope_file(repo_id, file_path, revision)?;
                decode::probe_path(&path)
            }
        }
    }

    /// 异步探测 [`AudioInfo`]。
    ///
    /// HTTP(S) URL 和 ModelScope 文件先异步下载，再在阻塞线程里 probe；其它来源整段放到 worker。
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
            Self::ModelScope {
                repo_id,
                file_path,
                revision,
            } => {
                let path = download_modelscope_file(repo_id, file_path, revision).await?;
                tokio::task::spawn_blocking(move || decode::probe_path(&path))
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

const DEFAULT_MODELSCOPE_REVISION: &str = "master";

/// 校验并规范化 ModelScope 身份，缺省 revision 为 `master`。
///
/// # Errors
///
/// 仓库 ID、文件路径或 revision 去掉空白后为空时返回错误。
fn normalize_modelscope_identity(
    repo_id: impl Into<String>,
    file_path: impl Into<String>,
    revision: Option<&str>,
) -> anyhow::Result<(String, String, String)> {
    let repo_id = repo_id.into().trim().to_owned();
    let file_path = file_path.into().trim().to_owned();
    let revision = revision
        .unwrap_or(DEFAULT_MODELSCOPE_REVISION)
        .trim()
        .to_owned();
    ensure_modelscope_identity(&repo_id, &file_path, &revision)?;
    Ok((repo_id, file_path, revision))
}

/// 拒绝空的仓库 ID、文件路径或 revision。
///
/// # Errors
///
/// 任一字段去掉空白后为空时返回错误。
fn ensure_modelscope_identity(
    repo_id: &str,
    file_path: &str,
    revision: &str,
) -> anyhow::Result<()> {
    if repo_id.trim().is_empty() {
        anyhow::bail!("ModelScope repository id must not be empty");
    }
    if file_path.trim().is_empty() {
        anyhow::bail!("ModelScope file path must not be empty");
    }
    if revision.trim().is_empty() {
        anyhow::bail!("ModelScope revision must not be empty");
    }
    Ok(())
}

/// 同步下载 ModelScope 音频文件，返回 modelhub 缓存中的本地路径。
///
/// # Errors
///
/// 身份为空、未启用 `modelscope` feature、下载失败或仓库不是数据集时返回错误。
pub(crate) fn materialize_modelscope_file(
    repo_id: &str,
    file_path: &str,
    revision: &str,
) -> anyhow::Result<PathBuf> {
    block_on(download_modelscope_file(repo_id, file_path, revision))
}

/// 通过 modelhub 下载数据集中的单个文件。
///
/// 只请求这一个 ModelScope 数据集文件。缓存已有时直接返回，不再访问网络。
/// 返回值是 modelhub 给出的落盘路径，不会链到 ModelScope 原生缓存。
///
/// # Errors
///
/// 身份为空、未启用 `modelscope` feature、下载失败，或仓库不是数据集时返回错误。
pub(crate) async fn download_modelscope_file(
    repo_id: &str,
    file_path: &str,
    revision: &str,
) -> anyhow::Result<PathBuf> {
    ensure_modelscope_identity(repo_id, file_path, revision)?;
    download_modelscope_dataset_file(repo_id, file_path, revision, None).await
}

/// 下载单个数据集文件。`cache_dir` 为 `None` 时使用 modelhub 默认缓存根目录。
///
/// # Errors
///
/// 未启用 `modelscope` feature、下载失败、仓库不是数据集，或 modelhub 没有返回文件路径时返回错误。
async fn download_modelscope_dataset_file(
    repo_id: &str,
    file_path: &str,
    revision: &str,
    cache_dir: Option<&Path>,
) -> anyhow::Result<PathBuf> {
    #[cfg(not(feature = "modelscope"))]
    {
        let _ = (repo_id, file_path, revision, cache_dir);
        anyhow::bail!("ModelScope audio sources require the `modelscope` feature");
    }
    #[cfg(feature = "modelscope")]
    {
        let mut options = modelhub::DownloadOptions::new(repo_id);
        // 已知是 ModelScope 数据集，跳过对 Hugging Face 和模型仓库的探测。
        options.kind = Some(modelhub::RepoKind::Dataset);
        options.backend = Some(modelhub::Backend::ModelScope);
        options.revision = Some(revision.to_owned());
        options.file = Some(file_path.to_owned());
        options.progress = false;
        if let Some(cache_dir) = cache_dir {
            options.cache_root = cache_dir.to_path_buf();
        }
        let downloaded = modelhub::download(&options).await.with_context(|| {
            format!(
                "failed to download ModelScope dataset {repo_id:?} file {file_path:?} at revision {revision:?}"
            )
        })?;
        if downloaded.kind != modelhub::RepoKind::Dataset {
            anyhow::bail!("ModelScope repository {repo_id:?} is not a dataset");
        }
        downloaded.file.with_context(|| {
            format!(
                "modelhub did not return a local path for ModelScope dataset {repo_id:?} file {file_path:?}"
            )
        })
    }
}

/// 若当前已在 tokio runtime 内，则到新线程里 `block_on`，避免嵌套 runtime。
fn block_on<F>(future: F) -> F::Output
where
    F: Future + Send,
    F::Output: Send,
{
    if tokio::runtime::Handle::try_current().is_ok() {
        return std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    tokio::runtime::Runtime::new()
                        .expect("create tokio runtime")
                        .block_on(future)
                })
                .join()
                .expect("join download thread")
        });
    }
    tokio::runtime::Runtime::new()
        .expect("create tokio runtime")
        .block_on(future)
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

#[cfg(test)]
mod tests {
    use super::AudioSource;

    #[test]
    fn modelscope_source_rejects_empty_identity() {
        assert!(AudioSource::from_modelscope(" ", "wav/a.wav", None).is_err());
        assert!(AudioSource::from_modelscope("org/name", "", None).is_err());
        assert!(AudioSource::from_modelscope("org/name", "wav/a.wav", Some(" ")).is_err());
    }

    #[test]
    fn modelscope_source_defaults_revision_and_roundtrips() {
        let source =
            AudioSource::from_modelscope(" org/name ", " wav/a.wav ", None).expect("identity");
        match &source {
            AudioSource::ModelScope {
                repo_id,
                file_path,
                revision,
            } => {
                assert_eq!(repo_id, "org/name");
                assert_eq!(file_path, "wav/a.wav");
                assert_eq!(revision, "master");
            }
            _ => panic!("expected ModelScope source"),
        }
        let encoded = serde_json::to_string(&source).expect("encode");
        let decoded: AudioSource = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(source, decoded);
    }
}
