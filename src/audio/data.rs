use anyhow::Context;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{AudioEncoding, AudioFormat, AudioSource, DurationMs};

const LOW_ENERGY_SEARCH_WINDOW_MS: u64 = 5_000;
const LOW_ENERGY_MIN_WINDOW_MS: u64 = 100;

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
    #[error("channel index is out of range")]
    ChannelIndexOutOfRange,
    #[error("sample count {samples} is not divisible by channel count {channels}")]
    IncompleteFrame { samples: usize, channels: u16 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_format: Option<AudioFormat>,
    #[serde(default)]
    pub is_normalized: bool,
}

/// A frame-aligned piece of streamed audio with its position in the source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_format: Option<AudioFormat>,
    #[serde(default)]
    pub is_normalized: bool,
    pub offset_ms: u64,
    pub is_final: bool,
}

pub struct AudioChunks {
    samples: std::vec::IntoIter<f32>,
    sample_rate: u32,
    channels: u16,
    source_format: Option<AudioFormat>,
    is_normalized: bool,
    frames_per_chunk: usize,
    next_frame: usize,
}

impl Audio {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::new_with_channels(samples, sample_rate, 1)
    }

    pub fn new_with_channels(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        debug_assert!(channels == 0 || samples.len().is_multiple_of(usize::from(channels)));
        Self {
            samples,
            sample_rate,
            channels,
            source_format: None,
            is_normalized: false,
        }
    }

    pub fn with_source_format(mut self, source_format: AudioFormat) -> Self {
        self.source_format = Some(source_format);
        self
    }

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

    pub fn frame_count(&self) -> usize {
        let channels = usize::from(self.channels);
        if channels == 0 {
            return 0;
        }
        self.samples.len() / channels
    }

    pub fn from_i16_pcm(samples: &[i16], sample_rate: u32) -> Self {
        let samples = samples
            .iter()
            .map(|sample| f32::from(*sample) / 32768.0)
            .collect();
        Self::new(samples, sample_rate)
    }

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

    pub fn from_i16_pcm_bytes(bytes: &[u8], sample_rate: u32) -> Result<Self, AudioError> {
        Self::from_i16_pcm_bytes_with_channels(bytes, sample_rate, 1)
    }

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

    pub fn from_path(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        AudioSource::from_path(path.as_ref().to_path_buf()).load()
    }

    pub fn from_url(url: impl Into<String>) -> anyhow::Result<Self> {
        AudioSource::from_url(url).load()
    }

    pub fn from_encoded_bytes(bytes: impl Into<Vec<u8>>) -> anyhow::Result<Self> {
        AudioSource::from_encoded_bytes(bytes).load()
    }

    pub fn from_base64(data: impl Into<String>) -> anyhow::Result<Self> {
        AudioSource::from_base64(data).load()
    }

    pub fn from_pcm_s16le(
        bytes: impl Into<Vec<u8>>,
        sample_rate: u32,
        channels: u16,
    ) -> anyhow::Result<Self> {
        AudioSource::from_pcm_s16le(bytes, sample_rate, channels).load()
    }

    pub fn from_source(source: &AudioSource) -> anyhow::Result<Self> {
        source.load()
    }

    pub async fn aload_from_source(source: &AudioSource) -> anyhow::Result<Self> {
        source.aload().await
    }

    pub fn duration_ms(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.frame_count() as f64 * 1000.0 / f64::from(self.sample_rate)
    }

    pub fn duration_seconds(&self) -> f64 {
        self.duration_ms() / 1000.0
    }

    /// Splits the waveform into fixed-duration, frame-aligned chunks.
    /// The final chunk is not padded.
    pub fn chunks_ms(&self, chunk_size_ms: u64) -> Result<Vec<AudioChunk>, AudioError> {
        self.clone()
            .into_chunks_ms(chunk_size_ms)
            .map(Iterator::collect)
    }

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
            is_normalized: self.is_normalized,
            frames_per_chunk,
            next_frame: 0,
        })
    }

    /// Splits a long waveform at low-energy boundaries without changing its samples.
    /// Every returned waveform is at most `max_duration` long and preserves complete frames.
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

    fn frame_slice(&self, start: usize, end: usize) -> Self {
        let channels = usize::from(self.channels);
        let mut waveform = Self::new_with_channels(
            self.samples[start * channels..end * channels].to_vec(),
            self.sample_rate,
            self.channels,
        );
        waveform.source_format = self.source_format.clone();
        waveform.is_normalized = self.is_normalized;
        waveform
    }

    pub fn to_i16_pcm(&self) -> Vec<i16> {
        self.samples
            .iter()
            .map(|sample| {
                let scaled = sample.clamp(-1.0, 1.0) * 32768.0;
                scaled.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
            })
            .collect()
    }

    pub fn append(&mut self, other: &Audio) -> Result<(), AudioError> {
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

        if self.channels == 1 {
            let samples = resample_mono_f32(&self.samples, self.sample_rate, target_sample_rate)?;
            let mut audio = Self::new(samples, target_sample_rate);
            audio.source_format = self.source_format.clone();
            audio.is_normalized = self.is_normalized;
            return Ok(audio);
        }

        let channels = usize::from(self.channels);
        let mut deinterleaved = (0..channels)
            .map(|_| Vec::with_capacity(self.frame_count()))
            .collect::<Vec<_>>();
        for frame in self.samples.chunks_exact(channels) {
            for (channel, sample) in deinterleaved.iter_mut().zip(frame) {
                channel.push(*sample);
            }
        }
        let channel_samples = deinterleaved
            .iter()
            .map(|samples| resample_mono_f32(samples, self.sample_rate, target_sample_rate))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let frames = channel_samples
            .iter()
            .map(Vec::len)
            .min()
            .unwrap_or_default();
        let mut samples = Vec::with_capacity(frames * channel_samples.len());
        for frame in 0..frames {
            for channel in &channel_samples {
                samples.push(channel[frame]);
            }
        }
        let mut waveform = Self::new_with_channels(samples, target_sample_rate, self.channels);
        waveform.source_format = self.source_format.clone();
        waveform.is_normalized = self.is_normalized;
        Ok(waveform)
    }

    pub fn normalize(mut self) -> Self {
        self.normalize_in_place();
        self
    }

    pub fn normalize_in_place(&mut self) {
        normalize_samples_in_place(&mut self.samples);
        self.is_normalized = true;
    }
}

fn normalize_samples_in_place(samples: &mut [f32]) {
    let mut peak = 0.0_f32;
    for sample in &mut *samples {
        if !sample.is_finite() {
            *sample = 0.0;
        }
        peak = peak.max(sample.abs());
    }
    if peak > 0.0 {
        let scale = peak.recip();
        for sample in &mut *samples {
            *sample = (*sample * scale).clamp(-1.0, 1.0);
        }
    } else {
        samples.fill(0.0);
    }
}

fn frames_for_ms(duration_ms: u64, sample_rate: u32) -> usize {
    (u128::from(duration_ms)
        .saturating_mul(u128::from(sample_rate))
        .div_ceil(1000))
    .max(1)
    .min(usize::MAX as u128) as usize
}

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

fn resample_mono_f32(samples: &[f32], from_hz: u32, to_hz: u32) -> anyhow::Result<Vec<f32>> {
    use rubato::audioadapter_buffers::direct::InterleavedSlice;
    use rubato::{
        Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
        WindowFunction,
    };

    if from_hz == 0 || to_hz == 0 {
        anyhow::bail!("invalid sample rates: from_hz={from_hz} to_hz={to_hz}");
    }
    if from_hz == to_hz {
        return Ok(samples.to_vec());
    }
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let ratio = f64::from(to_hz) / f64::from(from_hz);
    let chunk_size = 1024.min(samples.len().max(1));
    let mut resampler =
        Async::<f32>::new_sinc(ratio, 2.0, &params, chunk_size, 1, FixedAsync::Input).map_err(
            |error| {
                anyhow::anyhow!(
                    "resampler creation failed for from_hz={from_hz} to_hz={to_hz}: {error}"
                )
            },
        )?;

    let input_frames = samples.len();
    let input = InterleavedSlice::new(samples, 1, input_frames)
        .map_err(|error| anyhow::anyhow!("failed to create resampler input buffer: {error}"))?;
    let output_len = resampler.process_all_needed_output_len(input_frames);
    let mut output_samples = vec![0.0_f32; output_len];
    let mut output = InterleavedSlice::new_mut(&mut output_samples, 1, output_len)
        .map_err(|error| anyhow::anyhow!("failed to create resampler output buffer: {error}"))?;
    let (_, produced) = resampler
        .process_all_into_buffer(&input, &mut output, input_frames, None)
        .context("resampling failed")?;

    output_samples.truncate(produced);
    Ok(output_samples)
}

impl Default for Audio {
    fn default() -> Self {
        Self::new(Vec::new(), 16_000)
    }
}

impl Iterator for AudioChunks {
    type Item = AudioChunk;

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
        Some(AudioChunk {
            samples,
            sample_rate: self.sample_rate,
            channels: self.channels,
            source_format: self.source_format.clone(),
            is_normalized: self.is_normalized,
            offset_ms,
            is_final: self.samples.len() == 0,
        })
    }

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
    fn with_samples(&self, samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
            source_format: self.source_format.clone(),
            is_normalized: self.is_normalized,
            offset_ms: self.offset_ms,
            is_final: self.is_final,
        }
    }

    pub fn frame_count(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / usize::from(self.channels)
        }
    }

    pub fn duration_ms(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.frame_count() as f64 * 1000.0 / f64::from(self.sample_rate)
    }

    pub fn to_i16_pcm(&self) -> Vec<i16> {
        self.samples
            .iter()
            .map(|sample| {
                let scaled = sample.clamp(-1.0, 1.0) * 32768.0;
                scaled.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
            })
            .collect()
    }

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
        let channels = usize::from(self.channels);
        let mut deinterleaved = (0..channels)
            .map(|_| Vec::with_capacity(self.frame_count()))
            .collect::<Vec<_>>();
        for frame in self.samples.chunks_exact(channels) {
            for (channel, sample) in deinterleaved.iter_mut().zip(frame) {
                channel.push(*sample);
            }
        }
        let channel_samples = deinterleaved
            .iter()
            .map(|samples| resample_mono_f32(samples, self.sample_rate, sample_rate))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let frames = channel_samples.iter().map(Vec::len).min().unwrap_or(0);
        let mut samples = Vec::with_capacity(frames.saturating_mul(channels));
        for frame in 0..frames {
            for channel in &channel_samples {
                samples.push(channel[frame]);
            }
        }
        Ok(self.with_samples(samples, sample_rate, self.channels))
    }

    pub fn normalize(&self) -> Self {
        let mut chunk = self.clone();
        normalize_samples_in_place(&mut chunk.samples);
        chunk.is_normalized = true;
        chunk
    }

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
