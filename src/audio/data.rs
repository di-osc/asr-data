//! 内存中的波形、切块迭代器，以及 PCM 清洗 / 归一化辅助函数。

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{AudioEncoding, AudioFormat, AudioInfo, AudioSource};
use crate::utils::DurationMs;

/// 低能量切分时，在目标切点之前回看的最大窗口（毫秒）。
const LOW_ENERGY_SEARCH_WINDOW_MS: u64 = 5_000;
/// 计算局部能量所用的最小窗长（毫秒）。
const LOW_ENERGY_MIN_WINDOW_MS: u64 = 100;

/// 波形构造、切块和声道操作中的可恢复错误。
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AudioError {
    #[error("sample rate must be greater than zero")]
    InvalidSampleRate,
    #[error("audio byte input length must be divisible by two")]
    OddPcmByteLength,
    #[error("channel count must be greater than zero")]
    InvalidChannelCount,
    #[error("chunk size must be greater than zero")]
    InvalidChunkSize,
    #[error("chunk-local range must be ordered and contained in the chunk")]
    InvalidChunkRange,
    #[error("channel index is out of range")]
    ChannelIndexOutOfRange,
    #[error("sample count {samples} is not divisible by channel count {channels}")]
    IncompleteFrame { samples: usize, channels: u16 },
}

/// 已解码到内存中的交错 PCM 波形。
///
/// `samples` 按帧交错排列，取值范围约为 `[-1.0, 1.0]`。`source_format` 记录
/// 解码前的容器/编码信息，重采样或改声道后采样率、声道数字段会变，但来源格式
/// 仍保留。
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Waveform {
    /// 交错 `f32` 样本。长度为 `frame_count * channels`。
    pub samples: Vec<f32>,
    /// 每秒每个声道的采样帧数。
    pub sample_rate: u32,
    /// 声道数；1 表示单声道。
    pub channels: u16,
    /// 解码前的源格式；构造自原始 PCM 或容器探测结果。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_format: Option<AudioFormat>,
}

impl fmt::Debug for Waveform {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Waveform")
            .field("sample_count", &self.samples.len())
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("source_format", &self.source_format)
            .finish()
    }
}

/// 源音频中一段帧对齐的 PCM 窗口，带有在整段中的位置。
///
/// 由 [`AudioChunks`]、[`DecodedAudioChunks`](super::decode::DecodedAudioChunks)
/// 或 [`crate::doc::AudioStream`] 迭代产生。最后一块不补零，`is_final` 标记
/// 是否已经到达来源结尾。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioChunk {
    /// 本块的交错 `f32` 样本。
    pub samples: Vec<f32>,
    /// 本块样本的采样率。
    pub sample_rate: u32,
    /// 本块声道数。
    pub channels: u16,
    /// 源容器/编码格式；流式路径上通常从第一块开始带上。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_format: Option<AudioFormat>,
    /// 在父流中从零开始的块序号。
    pub index: usize,
    /// 本块起始位置，相对于源/父时间轴的毫秒偏移。
    pub offset_ms: u64,
    /// 是否为来源的最后一块。
    pub is_final: bool,
}

/// 把已经在内存中的 [`Waveform`] 按固定时长惰性切成 [`AudioChunk`]。
///
/// 这是 PCM 路径上的切窗迭代器：样本已经解码，不再读容器。编码音频的对应
/// 类型是 [`DecodedAudioChunks`](super::decode::DecodedAudioChunks)。
/// 由 [`Waveform::into_chunks_ms`] 创建；需要一次性拿到全部块时用
/// [`Waveform::chunks_ms`]。
pub struct AudioChunks {
    /// 尚未切出的剩余交错样本。
    samples: std::vec::IntoIter<f32>,
    sample_rate: u32,
    channels: u16,
    source_format: Option<AudioFormat>,
    /// 每块包含的帧数（由 `chunk_size_ms` 和采样率向上取整）。
    frames_per_chunk: usize,
    /// 下一块在源波形中的起始帧。
    next_frame: usize,
    /// 下一块的从零开始序号。
    next_index: usize,
}

impl Waveform {
    /// 用单声道样本构造波形。
    ///
    /// 多声道请用 [`Self::new_with_channels`] 或 [`Self::try_new_with_channels`]。
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::new_with_channels(samples, sample_rate, 1)
    }

    /// 用指定声道数构造波形，不检查样本长度是否整除声道数。
    ///
    /// 调试构建下若 `channels != 0` 且样本数无法整除声道数会断言失败。
    /// 需要校验时用 [`Self::try_new_with_channels`]。
    pub fn new_with_channels(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        debug_assert!(channels == 0 || samples.len().is_multiple_of(usize::from(channels)));
        Self {
            samples,
            sample_rate,
            channels,
            source_format: None,
        }
    }

    /// 附上解码前的源格式，便于后续保留容器编码信息。
    pub fn with_source_format(mut self, source_format: AudioFormat) -> Self {
        self.source_format = Some(source_format);
        self
    }

    /// 校验声道数和样本对齐后构造波形。
    ///
    /// # Errors
    ///
    /// 声道数为 0，或样本数无法被声道数整除时返回错误。
    pub fn try_new_with_channels(
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Self, AudioError> {
        if channels == 0 {
            return Err(AudioError::InvalidChannelCount);
        }
        if !samples.len().is_multiple_of(usize::from(channels)) {
            return Err(AudioError::IncompleteFrame {
                samples: samples.len(),
                channels,
            });
        }
        Ok(Self::new_with_channels(samples, sample_rate, channels))
    }

    /// 返回帧数：`samples.len() / channels`。声道数为 0 时返回 0。
    pub fn frame_count(&self) -> usize {
        let channels = usize::from(self.channels);
        if channels == 0 {
            return 0;
        }
        self.samples.len() / channels
    }

    /// 把 `i16` PCM 转成单声道 `f32` 波形，幅度除以 32768。
    pub fn from_i16_pcm(samples: &[i16], sample_rate: u32) -> Self {
        let samples = samples
            .iter()
            .map(|sample| f32::from(*sample) / 32768.0)
            .collect();
        Self::new(samples, sample_rate)
    }

    /// 把交错 `i16` PCM 转成指定声道数的 `f32` 波形，并标记为 `PcmS16Le`。
    pub fn from_i16_pcm_with_channels(samples: &[i16], sample_rate: u32, channels: u16) -> Self {
        let samples = samples
            .iter()
            .map(|sample| f32::from(*sample) / 32768.0)
            .collect();
        Self::new_with_channels(samples, sample_rate, channels).with_source_format(AudioFormat {
            encoding: AudioEncoding::PcmS16Le,
            sample_rate,
            channels,
        })
    }

    /// 把 little-endian `i16` 字节解码为单声道波形。
    ///
    /// # Errors
    ///
    /// 字节长度为奇数时返回 [`AudioError::OddPcmByteLength`]。
    pub fn from_i16_pcm_bytes(bytes: &[u8], sample_rate: u32) -> Result<Self, AudioError> {
        Self::from_i16_pcm_bytes_with_channels(bytes, sample_rate, 1)
    }

    /// 把 little-endian 交错 `i16` 字节解码为指定声道数的波形。
    ///
    /// # Errors
    ///
    /// 字节长度为奇数、声道数为 0，或样本无法对齐成完整帧时返回错误。
    pub fn from_i16_pcm_bytes_with_channels(
        bytes: &[u8],
        sample_rate: u32,
        channels: u16,
    ) -> Result<Self, AudioError> {
        if !bytes.len().is_multiple_of(2) {
            return Err(AudioError::OddPcmByteLength);
        }
        if channels == 0 {
            return Err(AudioError::InvalidChannelCount);
        }

        let samples = bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();
        let samples = samples
            .iter()
            .map(|sample| f32::from(*sample) / 32768.0)
            .collect();
        Self::try_new_with_channels(samples, sample_rate, channels).map(|waveform| {
            waveform.with_source_format(AudioFormat {
                encoding: AudioEncoding::PcmS16Le,
                sample_rate,
                channels,
            })
        })
    }

    /// 从本地路径解码完整波形。
    ///
    /// # Errors
    ///
    /// 打开或解码失败时返回错误。
    pub fn from_path(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        AudioSource::from_path(path.as_ref().to_path_buf()).decode_waveform()
    }

    /// 从 URL 下载并解码完整波形。
    ///
    /// # Errors
    ///
    /// 下载或解码失败时返回错误。
    pub fn from_url(url: impl Into<String>) -> anyhow::Result<Self> {
        AudioSource::from_url(url).decode_waveform()
    }

    /// 解码带容器的音频字节为波形。
    ///
    /// # Errors
    ///
    /// 字节无法探测或解码时返回错误。
    pub fn from_encoded_bytes(bytes: impl Into<Vec<u8>>) -> anyhow::Result<Self> {
        AudioSource::from_encoded_bytes(bytes).decode_waveform()
    }

    /// 解码 base64 音频为波形。
    ///
    /// # Errors
    ///
    /// base64 非法或解码失败时返回错误。
    pub fn from_base64(data: impl Into<String>) -> anyhow::Result<Self> {
        AudioSource::from_base64(data).decode_waveform()
    }

    /// 从 PCM S16LE 字节构造波形。
    ///
    /// # Errors
    ///
    /// 字节无法对齐成完整帧时返回错误。
    pub fn from_pcm_s16le(
        bytes: impl Into<Vec<u8>>,
        sample_rate: u32,
        channels: u16,
    ) -> anyhow::Result<Self> {
        AudioSource::from_pcm_s16le(bytes, sample_rate, channels).decode_waveform()
    }

    /// 按 [`AudioSource`] 类型选择对应解码路径，得到完整波形。
    ///
    /// # Errors
    ///
    /// 打开、下载或解码失败时返回错误。
    pub fn from_source(source: &AudioSource) -> anyhow::Result<Self> {
        source.decode_waveform()
    }

    /// 在阻塞线程中异步解码 [`AudioSource`]。
    ///
    /// # Errors
    ///
    /// worker 崩溃或解码失败时返回错误。
    pub async fn aload_from_source(source: &AudioSource) -> anyhow::Result<Self> {
        let source = source.clone();
        tokio::task::spawn_blocking(move || source.decode_waveform())
            .await
            .map_err(|error| anyhow::anyhow!("waveform loader worker failed: {error}"))?
    }

    /// 波形时长（毫秒）。采样率或声道数为 0 时返回 0。
    pub fn duration_ms(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.frame_count() as f64 * 1000.0 / f64::from(self.sample_rate)
    }

    /// 波形时长（秒）。
    pub fn duration_seconds(&self) -> f64 {
        self.duration_ms() / 1000.0
    }

    /// 把波形切成固定时长、帧对齐的 [`AudioChunk`] 列表。
    ///
    /// 最后一块不补零。若不再需要原波形，优先用 [`Self::into_chunks_ms`] 避免克隆。
    ///
    /// # Errors
    ///
    /// `chunk_size_ms`、采样率或声道数无效时返回错误。
    pub fn chunks_ms(&self, chunk_size_ms: u64) -> Result<Vec<AudioChunk>, AudioError> {
        self.clone()
            .into_chunks_ms(chunk_size_ms)
            .map(Iterator::collect)
    }

    /// 消耗波形，返回按固定时长惰性吐块的 [`AudioChunks`]。
    ///
    /// 适合 PCM 已经在内存、且希望和流式解码走同一套 [`AudioChunk`] 接口的场景。
    ///
    /// # Errors
    ///
    /// `chunk_size_ms`、采样率或声道数无效时返回错误。
    pub fn into_chunks_ms(self, chunk_size_ms: u64) -> Result<AudioChunks, AudioError> {
        if chunk_size_ms == 0 {
            return Err(AudioError::InvalidChunkSize);
        }
        if self.sample_rate == 0 {
            return Err(AudioError::InvalidSampleRate);
        }
        if self.channels == 0 {
            return Err(AudioError::InvalidChannelCount);
        }

        let frames_per_chunk = (u128::from(chunk_size_ms)
            .saturating_mul(u128::from(self.sample_rate))
            .div_ceil(1000))
        .max(1)
        .min(usize::MAX as u128) as usize;
        Ok(AudioChunks {
            samples: self.samples.into_iter(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            source_format: self.source_format,
            frames_per_chunk,
            next_frame: 0,
            next_index: 0,
        })
    }

    /// 在低能量边界切开超长波形，不改动样本本身。
    ///
    /// 每段不超过 `max_duration`，且保持完整帧。切点优先选目标时长附近能量最低
    /// 的位置，避免把说话人截在音节中间。
    ///
    /// # Errors
    ///
    /// `max_duration`、采样率或声道数无效时返回错误。
    pub fn split_at_low_energy(&self, max_duration: DurationMs) -> Result<Vec<Self>, AudioError> {
        if max_duration.0 == 0 {
            return Err(AudioError::InvalidChunkSize);
        }
        if self.sample_rate == 0 {
            return Err(AudioError::InvalidSampleRate);
        }
        if self.channels == 0 {
            return Err(AudioError::InvalidChannelCount);
        }

        let total_frames = self.frame_count();
        if total_frames == 0 {
            return Ok(Vec::new());
        }
        let max_frames = frames_for_ms(max_duration.0, self.sample_rate);
        if total_frames <= max_frames {
            return Ok(vec![self.clone()]);
        }

        let channels = usize::from(self.channels);
        let frame_energy = self
            .samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().map(|sample| sample.abs()).sum::<f32>() / channels as f32)
            .collect::<Vec<_>>();
        let search_frames =
            frames_for_ms(LOW_ENERGY_SEARCH_WINDOW_MS, self.sample_rate).min(max_frames / 2);
        let energy_window = frames_for_ms(LOW_ENERGY_MIN_WINDOW_MS, self.sample_rate).max(1);

        let mut chunks = Vec::new();
        let mut start = 0;
        while total_frames - start > max_frames {
            let cut = start + max_frames;
            // 只在切点前的搜索窗里找最低能量帧，避免切到正在说话的位置。
            let search_start = cut.saturating_sub(search_frames).max(start + 1);
            let boundary = lowest_energy_boundary(&frame_energy, search_start, cut, energy_window)
                .unwrap_or(cut)
                .clamp(start + 1, cut);
            chunks.push(self.frame_slice(start, boundary));
            start = boundary;
        }
        chunks.push(self.frame_slice(start, total_frames));
        Ok(chunks)
    }

    /// 按帧下标切出 `[start, end)` 区间，保留采样率和源格式。
    fn frame_slice(&self, start: usize, end: usize) -> Self {
        let channels = usize::from(self.channels);
        let mut waveform = Self::new_with_channels(
            self.samples[start * channels..end * channels].to_vec(),
            self.sample_rate,
            self.channels,
        );
        waveform.source_format = self.source_format.clone();
        waveform
    }

    /// 把 `f32` 样本量化回 `i16` PCM。
    ///
    /// 先钳到 `[-1, 1]` 再乘 32768；`1.0` 会被钳到 `i16::MAX`，避免溢出。
    pub fn to_i16_pcm(&self) -> Vec<i16> {
        self.samples
            .iter()
            .map(|sample| {
                let scaled = sample.clamp(-1.0, 1.0) * 32768.0;
                scaled.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
            })
            .collect()
    }

    /// 把另一段相同采样率、声道数的波形追加到末尾。
    ///
    /// 源格式不一致时会丢掉 `source_format`，避免误标编码。
    ///
    /// # Errors
    ///
    /// 采样率或声道数不匹配（含为 0）时返回错误。
    pub fn append(&mut self, other: &Waveform) -> Result<(), AudioError> {
        if self.sample_rate == 0 || other.sample_rate == 0 || self.sample_rate != other.sample_rate
        {
            return Err(AudioError::InvalidSampleRate);
        }
        if self.channels == 0 || other.channels == 0 || self.channels != other.channels {
            return Err(AudioError::InvalidChannelCount);
        }
        if self.source_format != other.source_format {
            self.source_format = None;
        }
        self.samples.extend_from_slice(&other.samples);
        Ok(())
    }

    /// 按毫秒区间切出子波形；`end_ms <= start_ms` 时返回空样本。
    ///
    /// 起始帧向下取整、结束帧向上取整，保证覆盖请求的时间范围。
    pub fn slice_ms(&self, start_ms: u64, end_ms: u64) -> Self {
        if end_ms <= start_ms || self.sample_rate == 0 || self.channels == 0 {
            let mut waveform = Self::new_with_channels(Vec::new(), self.sample_rate, self.channels);
            waveform.source_format = self.source_format.clone();
            return waveform;
        }

        let channels = usize::from(self.channels);
        let start_frame = (start_ms as usize).saturating_mul(self.sample_rate as usize) / 1000;
        let end_frame = (end_ms as usize)
            .saturating_mul(self.sample_rate as usize)
            .div_ceil(1000)
            .min(self.frame_count());
        let start = start_frame.saturating_mul(channels).min(self.samples.len());
        let end = end_frame.saturating_mul(channels).min(self.samples.len());
        let mut waveform = Self::new_with_channels(
            self.samples[start.min(self.samples.len())..end].to_vec(),
            self.sample_rate,
            self.channels,
        );
        waveform.source_format = self.source_format.clone();
        waveform
    }

    /// 抽出指定声道，返回单声道波形。
    ///
    /// # Errors
    ///
    /// 声道数为 0 或 `index` 越界时返回错误。
    pub fn channel(&self, index: u16) -> Result<Self, AudioError> {
        if self.channels == 0 {
            return Err(AudioError::InvalidChannelCount);
        }
        if index >= self.channels {
            return Err(AudioError::ChannelIndexOutOfRange);
        }
        if self.channels == 1 {
            return Ok(self.clone());
        }

        let channels = usize::from(self.channels);
        let index = usize::from(index);
        let samples = self
            .samples
            .chunks_exact(channels)
            .map(|frame| frame[index])
            .collect();
        let mut waveform = Self::new_with_channels(samples, self.sample_rate, 1);
        waveform.source_format = self.source_format.clone();
        Ok(waveform)
    }

    /// 把各声道平均成单声道；已经是单声道时直接克隆。
    ///
    /// # Errors
    ///
    /// 声道数为 0 时返回错误。
    pub fn to_mono(&self) -> Result<Self, AudioError> {
        if self.channels == 0 {
            return Err(AudioError::InvalidChannelCount);
        }
        if self.channels == 1 {
            return Ok(self.clone());
        }

        let channels = usize::from(self.channels);
        let samples = self
            .samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
            .collect();
        let mut waveform = Self::new_with_channels(samples, self.sample_rate, 1);
        waveform.source_format = self.source_format.clone();
        Ok(waveform)
    }

    /// 重采样到 `target_sample_rate`，声道布局不变。
    ///
    /// # Errors
    ///
    /// 采样率为 0 或重采样器失败时返回错误。
    pub fn resample(&self, target_sample_rate: u32) -> anyhow::Result<Self> {
        if self.sample_rate == 0 || target_sample_rate == 0 {
            anyhow::bail!(
                "invalid sample rate conversion: {} -> {}",
                self.sample_rate,
                target_sample_rate
            );
        }
        if self.sample_rate == target_sample_rate {
            return Ok(self.clone());
        }
        let samples = super::stream::resample_interleaved(
            &self.samples,
            self.sample_rate,
            target_sample_rate,
            self.channels,
        )?;
        let mut waveform = Self::new_with_channels(samples, target_sample_rate, self.channels);
        waveform.source_format = self.source_format.clone();
        Ok(waveform)
    }

    /// 把峰值幅度缩放到不超过 1.0。
    ///
    /// 非有限值先替换为 `0.0`。剩余峰值大于 1.0 时整段除以该峰值，再钳到
    /// `[-1, 1]`。解码时的样本清洗只钳位，不缩放。
    pub fn peak_normalize(&mut self) {
        peak_normalize_samples(&mut self.samples);
    }

    /// [`Self::peak_normalize`] 的链式版本。
    pub fn with_peak_normalize(mut self) -> Self {
        self.peak_normalize();
        self
    }
}

/// 把非有限样本置 0，并把有限值钳到 `[-1, 1]`，不做幅度缩放。
pub(crate) fn sanitize_samples(samples: &mut [f32]) {
    for sample in samples {
        *sample = if sample.is_finite() {
            sample.clamp(-1.0, 1.0)
        } else {
            0.0
        };
    }
}

/// 按峰值缩放样本，使最大绝对值不超过 1.0。
fn peak_normalize_samples(samples: &mut [f32]) {
    for sample in samples.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }
    let peak = samples
        .iter()
        .fold(0.0f32, |max, sample| sample.abs().max(max));
    if peak.is_finite() && peak > 1.0 {
        for sample in samples.iter_mut() {
            *sample /= peak;
        }
    }
    for sample in samples {
        *sample = sample.clamp(-1.0, 1.0);
    }
}

/// 把毫秒时长换成帧数，至少 1 帧，并防止溢出 `usize`。
fn frames_for_ms(duration_ms: u64, sample_rate: u32) -> usize {
    (u128::from(duration_ms)
        .saturating_mul(u128::from(sample_rate))
        .div_ceil(1000))
    .max(1)
    .min(usize::MAX as u128) as usize
}

/// 在 `[start, end)` 里找滑动能量窗最低的位置，再返回窗内能量最低的那一帧。
///
/// 用于 [`Waveform::split_at_low_energy`]：先用窗和定位安静区，再把切点落到
/// 窗内最安静的单帧，尽量避开语音。
fn lowest_energy_boundary(
    energy: &[f32],
    start: usize,
    end: usize,
    window: usize,
) -> Option<usize> {
    if start >= end || end > energy.len() {
        return None;
    }
    let window = window.min(end - start);
    if window == 0 {
        return None;
    }

    let mut sum = energy[start..start + window].iter().sum::<f32>();
    let mut best_sum = sum;
    let mut best_start = start;
    // 滑动窗口：每次只加减边界两帧，O(n) 找最低能量区间。
    for position in start + 1..=end - window {
        sum += energy[position + window - 1] - energy[position - 1];
        if sum < best_sum {
            best_sum = sum;
            best_start = position;
        }
    }

    energy[best_start..best_start + window]
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(offset, _)| best_start + offset)
}

impl Default for Waveform {
    fn default() -> Self {
        Self::new(Vec::new(), 16_000)
    }
}

impl Iterator for AudioChunks {
    type Item = AudioChunk;

    /// 切出下一块；最后一块可能短于 `frames_per_chunk`，且不补零。
    fn next(&mut self) -> Option<Self::Item> {
        if self.samples.len() == 0 {
            return None;
        }
        let sample_count = self
            .frames_per_chunk
            .saturating_mul(usize::from(self.channels))
            .min(self.samples.len());
        let samples = self.samples.by_ref().take(sample_count).collect();
        let offset_ms = self.next_frame as u64 * 1_000 / u64::from(self.sample_rate);
        self.next_frame = self
            .next_frame
            .saturating_add(sample_count / usize::from(self.channels));
        let chunk = AudioChunk {
            samples,
            sample_rate: self.sample_rate,
            channels: self.channels,
            source_format: self.source_format.clone(),
            index: self.next_index,
            offset_ms,
            is_final: self.samples.len() == 0,
        };
        self.next_index = self.next_index.saturating_add(1);
        Some(chunk)
    }

    /// 剩余块数可精确计算，因此同时实现 [`ExactSizeIterator`]。
    fn size_hint(&self) -> (usize, Option<usize>) {
        let samples_per_chunk = self
            .frames_per_chunk
            .saturating_mul(usize::from(self.channels));
        let chunks = self.samples.len().div_ceil(samples_per_chunk);
        (chunks, Some(chunks))
    }
}

impl ExactSizeIterator for AudioChunks {}

impl AudioChunk {
    /// 替换样本、采样率或声道，但保留 `index` / `offset_ms` / `is_final`。
    fn with_samples(&self, samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
            source_format: self.source_format.clone(),
            index: self.index,
            offset_ms: self.offset_ms,
            is_final: self.is_final,
        }
    }

    /// 只描述本块的 [`AudioInfo`]，`frame_count` 是本块帧数而不是整段音频。
    pub fn info(&self) -> AudioInfo {
        AudioInfo {
            sample_rate: self.sample_rate,
            channels: self.channels,
            frame_count: self.frame_count() as u64,
            source_format: self.source_format.clone().unwrap_or(AudioFormat {
                encoding: AudioEncoding::Unknown,
                sample_rate: self.sample_rate,
                channels: self.channels,
            }),
        }
    }

    /// 把本块样本拷成独立 [`Waveform`]。
    pub fn as_waveform(&self) -> Waveform {
        Waveform {
            samples: self.samples.clone(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            source_format: self.source_format.clone(),
        }
    }

    /// 本块结束位置，使用父时间轴的全局毫秒坐标。
    pub fn end_ms(&self) -> u64 {
        self.offset_ms
            .saturating_add(self.duration_ms().ceil() as u64)
    }

    /// 把块内毫秒区间换成父时间轴坐标。
    ///
    /// # Errors
    ///
    /// 区间无序，或结束点超出本块时长时返回 [`AudioError::InvalidChunkRange`]。
    pub fn to_timeline_range(&self, start_ms: u64, end_ms: u64) -> Result<(u64, u64), AudioError> {
        if end_ms < start_ms || end_ms as f64 > self.duration_ms().ceil() {
            return Err(AudioError::InvalidChunkRange);
        }
        Ok((
            self.offset_ms.saturating_add(start_ms),
            self.offset_ms.saturating_add(end_ms),
        ))
    }

    /// 本块帧数；声道数为 0 时返回 0。
    pub fn frame_count(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / usize::from(self.channels)
        }
    }

    /// 本块时长（毫秒）。
    pub fn duration_ms(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.frame_count() as f64 * 1000.0 / f64::from(self.sample_rate)
    }

    /// 把本块样本量化为 `i16` PCM。
    pub fn to_i16_pcm(&self) -> Vec<i16> {
        self.samples
            .iter()
            .map(|sample| {
                let scaled = sample.clamp(-1.0, 1.0) * 32768.0;
                scaled.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
            })
            .collect()
    }

    /// 抽出指定声道，保留 `index` / `offset_ms` / `is_final`。
    ///
    /// # Errors
    ///
    /// 声道数为 0 或 `index` 越界时返回错误。
    pub fn channel(&self, index: u16) -> Result<Self, AudioError> {
        if self.channels == 0 {
            return Err(AudioError::InvalidChannelCount);
        }
        if index >= self.channels {
            return Err(AudioError::ChannelIndexOutOfRange);
        }
        if self.channels == 1 {
            return Ok(self.clone());
        }
        let channels = usize::from(self.channels);
        let index = usize::from(index);
        let samples = self
            .samples
            .chunks_exact(channels)
            .map(|frame| frame[index])
            .collect();
        Ok(self.with_samples(samples, self.sample_rate, 1))
    }

    /// 把本块平均成单声道，保留时间位置元数据。
    ///
    /// # Errors
    ///
    /// 声道数为 0 时返回错误。
    pub fn to_mono(&self) -> Result<Self, AudioError> {
        if self.channels == 0 {
            return Err(AudioError::InvalidChannelCount);
        }
        if self.channels == 1 {
            return Ok(self.clone());
        }
        let channels = usize::from(self.channels);
        let samples = self
            .samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
            .collect();
        Ok(self.with_samples(samples, self.sample_rate, 1))
    }

    /// 重采样本块；`index` / `offset_ms` / `is_final` 不变。
    ///
    /// # Errors
    ///
    /// 采样率为 0 或重采样失败时返回错误。
    pub fn resample(&self, sample_rate: u32) -> anyhow::Result<Self> {
        if self.sample_rate == 0 || sample_rate == 0 {
            anyhow::bail!(
                "invalid sample rate conversion: {} -> {}",
                self.sample_rate,
                sample_rate
            );
        }
        if self.sample_rate == sample_rate {
            return Ok(self.clone());
        }
        let samples = super::stream::resample_interleaved(
            &self.samples,
            self.sample_rate,
            sample_rate,
            self.channels,
        )?;
        Ok(self.with_samples(samples, sample_rate, self.channels))
    }

    /// 把本块峰值幅度缩放到不超过 1.0。
    ///
    /// 行为与 [`Waveform::peak_normalize`] 相同。
    pub fn peak_normalize(&mut self) {
        peak_normalize_samples(&mut self.samples);
    }

    /// [`Self::peak_normalize`] 的链式版本。
    pub fn with_peak_normalize(mut self) -> Self {
        self.peak_normalize();
        self
    }

    /// 按块内毫秒区间切出子块，并平移 `offset_ms`。
    ///
    /// 切到原块末尾且原块是最后一块时，结果仍标记 `is_final`。
    pub fn slice_ms(&self, start_ms: u64, end_ms: u64) -> Self {
        let duration_ms = self.duration_ms().ceil() as u64;
        let effective_start = start_ms.min(duration_ms);
        let channels = usize::from(self.channels);
        let start_frame = (start_ms as usize).saturating_mul(self.sample_rate as usize) / 1000;
        let end_frame = (end_ms as usize)
            .saturating_mul(self.sample_rate as usize)
            .div_ceil(1000)
            .min(self.frame_count());
        let start = start_frame.saturating_mul(channels).min(self.samples.len());
        let end = end_frame.saturating_mul(channels).min(self.samples.len());
        let samples = if end_ms <= start_ms || start >= end {
            Vec::new()
        } else {
            self.samples[start..end].to_vec()
        };
        let mut chunk = self.with_samples(samples, self.sample_rate, self.channels);
        chunk.offset_ms = self.offset_ms.saturating_add(effective_start);
        chunk.is_final = self.is_final && end_ms >= duration_ms;
        chunk
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioChunk, Waveform, peak_normalize_samples, sanitize_samples};

    #[test]
    fn waveform_samples_are_sanitized() {
        let mut samples = vec![f32::NAN, f32::NEG_INFINITY, -1.5, 0.5, 1.5, f32::INFINITY];

        sanitize_samples(&mut samples);

        assert_eq!(samples, vec![0.0, 0.0, -1.0, 0.5, 1.0, 0.0]);
    }

    #[test]
    fn peak_normalize_scales_when_peak_exceeds_one() {
        let mut waveform = Waveform::new(vec![0.0, 2.0, -2.0], 16_000);
        waveform.peak_normalize();
        assert_eq!(waveform.samples, vec![0.0, 1.0, -1.0]);
    }

    #[test]
    fn peak_normalize_leaves_in_range_samples_unchanged() {
        let original = vec![-1.0, -0.5, 0.0, 0.25, 1.0];
        let waveform = Waveform::new(original.clone(), 16_000).with_peak_normalize();
        assert_eq!(waveform.samples, original);
    }

    #[test]
    fn peak_normalize_zeros_non_finite_values_before_scaling() {
        let mut samples = vec![f32::NAN, f32::INFINITY, 2.0, -2.0];
        peak_normalize_samples(&mut samples);
        assert_eq!(samples, vec![0.0, 0.0, 1.0, -1.0]);
    }

    #[test]
    fn audio_chunk_peak_normalize_matches_waveform() {
        let mut chunk = AudioChunk {
            samples: vec![0.5, 2.0],
            sample_rate: 16_000,
            channels: 1,
            source_format: None,
            index: 0,
            offset_ms: 0,
            is_final: true,
        };
        chunk.peak_normalize();
        assert_eq!(chunk.samples, vec![0.25, 1.0]);
    }
}
