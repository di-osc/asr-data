//! 带时间位置的帧对齐 PCM 窗口。

use serde::{Deserialize, Serialize};

use super::waveform::{AudioError, Waveform, peak_normalize_samples};
use super::{AudioEncoding, AudioFormat, AudioInfo};

/// 源音频中一段帧对齐的 PCM 窗口，带有在整段中的位置。
///
/// 由 [`Waveform::chunk`](super::Waveform::chunk)、
/// [`DecodedAudioChunks`](super::decode::DecodedAudioChunks)
/// 或 [`crate::doc::AudioStream`] 产生。最后一块不补零，`is_final` 标记
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
