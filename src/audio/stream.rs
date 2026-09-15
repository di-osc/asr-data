//! 从 [`AudioSource`] 按块拉取 PCM，并可在流上做重采样与声道变换。
//!
//! 分层关系：
//! - [`RawAudioStream`]：源格式 PCM 窗口，只负责「下一块从哪来」
//! - [`SourceAudioStream`]：可选转单声道 / 重采样后，再按目标时长切块
//! - 公开的 [`crate::doc::AudioStream`] 再包一层文档上下文（id、timeline、累积波形）

use std::collections::VecDeque;

use anyhow::{Context, Result, bail};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

use super::{AudioChunk, AudioChunks, AudioFormat, AudioSource};

/// 把两种源格式 PCM 拉取路径合成一个 iterator。
///
/// 这是流式管线的最底层：不做重采样、不混音、不建文档，只按源采样率吐
/// [`AudioChunk`]。编码音频走 [`DecodedAudioChunks`](super::decode::DecodedAudioChunks)
/// 边解边切；原始 PCM 已经在内存里，走 [`AudioChunks`] 只切窗。
enum RawAudioStream {
    /// 压缩容器：Symphonia 按 packet 解码后再切块。
    Decoded(super::decode::DecodedAudioChunks),
    /// 原始 PCM：先整段解码成 [`Waveform`](super::Waveform)，再惰性切块。
    Pcm(AudioChunks),
}

impl Iterator for RawAudioStream {
    type Item = Result<AudioChunk>;

    /// 从当前变体拉取下一块源格式 PCM。
    ///
    /// PCM 路径本身不会失败，因此把 [`AudioChunks`] 的输出包成 `Ok`。
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Decoded(chunks) => chunks.next(),
            Self::Pcm(chunks) => chunks.next().map(Ok),
        }
    }
}

/// 有状态的交错 PCM 重采样器，可跨多个输入块复用。
///
/// `rubato::Async` 需要固定大小的输入窗，并且开头有输出延迟。本类型把不足一块的
/// 样本缓存在 `input` 里，丢掉延迟帧，并在最后一个输入块上冲刷尾巴，使输出帧数
/// 对齐 `ceil(input_frames * to_hz / from_hz)`。
struct StreamingResampler {
    /// rubato 异步 sinc 重采样器，按固定输入帧数工作。
    inner: Async<f32>,
    /// 交错样本的声道数。
    channels: usize,
    /// `to_hz / from_hz`，用于估算最终输出帧数。
    ratio: f64,
    /// 仍需从输出开头丢掉的延迟帧数。
    frames_to_trim: usize,
    /// 已经送入重采样器的总输入帧数。
    total_input_frames: usize,
    /// 已经交给调用方的总输出帧数（不含仍待 trim 的延迟）。
    total_output_frames: usize,
    /// 还不够一次 `process_once` 的交错输入样本。
    input: VecDeque<f32>,
}

/// 把交错 PCM 从 `from_hz` 重采样到 `to_hz`。
///
/// 采样率相同或输入为空时直接复制。该函数一次性处理整段样本，内部仍走
/// [`StreamingResampler`]，并在结束时冲刷延迟。
///
/// # Errors
///
/// 采样率为 0，或 rubato 创建 / 处理失败时返回错误。
pub(crate) fn resample_interleaved(
    samples: &[f32],
    from_hz: u32,
    to_hz: u32,
    channels: u16,
) -> Result<Vec<f32>> {
    if from_hz == 0 || to_hz == 0 {
        bail!("invalid sample rates: from_hz={from_hz} to_hz={to_hz}");
    }
    if from_hz == to_hz || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let frames = samples.len() / usize::from(channels);
    StreamingResampler::new(from_hz, to_hz, channels)?.process(samples, frames, true)
}

impl StreamingResampler {
    /// 创建从 `from_hz` 到 `to_hz` 的 sinc 重采样器。
    ///
    /// `frames_to_trim` 取自 rubato 的 `output_delay()`，后续输出会先丢掉这些
    /// 预热帧，避免把滤波器延迟当成真实音频。
    ///
    /// # Errors
    ///
    /// rubato 无法按给定参数创建重采样器时返回错误。
    fn new(from_hz: u32, to_hz: u32, channels: u16) -> Result<Self> {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let ratio = f64::from(to_hz) / f64::from(from_hz);
        let inner = Async::<f32>::new_sinc(
            ratio,
            2.0,
            &params,
            1024,
            usize::from(channels),
            FixedAsync::Input,
        )
        .with_context(|| format!("resampler creation failed for {from_hz}Hz -> {to_hz}Hz"))?;
        let frames_to_trim = inner.output_delay();
        Ok(Self {
            inner,
            channels: usize::from(channels),
            ratio,
            frames_to_trim,
            total_input_frames: 0,
            total_output_frames: 0,
            input: VecDeque::new(),
        })
    }

    /// 处理一段交错输入，返回本次可以交出的输出样本。
    ///
    /// 输入先进入 `self.input`；凑够 rubato 当前要求的帧数才真正调用一次
    /// [`Self::process_once`]。`final_chunk` 为真时会冲刷剩余样本和滤波器尾巴，
    /// 并把总输出截到期望帧数，避免多吐延迟补零。
    ///
    /// # Errors
    ///
    /// 单次重采样失败时返回错误。
    fn process(&mut self, samples: &[f32], frames: usize, final_chunk: bool) -> Result<Vec<f32>> {
        self.total_input_frames = self.total_input_frames.saturating_add(frames);
        self.input.extend(samples.iter().copied());
        let mut output = Vec::new();
        let input_frames = self.inner.input_frames_next();
        let input_samples = input_frames.saturating_mul(self.channels);
        while self.input.len() >= input_samples {
            let samples = self.input.drain(..input_samples).collect::<Vec<_>>();
            output.extend(self.process_once(&samples, input_frames)?);
        }
        if final_chunk {
            // 最后一块：不足一个输入窗的尾巴也要送进去，再空跑冲刷延迟。
            let remaining_frames = self.input.len() / self.channels;
            if remaining_frames > 0 {
                let samples = self.input.drain(..).collect::<Vec<_>>();
                output.extend(self.process_once(&samples, remaining_frames)?);
            }
            let expected = (self.ratio * self.total_input_frames as f64).ceil() as usize;
            while self.total_output_frames + output.len() / self.channels < expected {
                output.extend(self.process_once(&[], 0)?);
            }
            output.truncate(
                expected
                    .saturating_sub(self.total_output_frames)
                    .saturating_mul(self.channels),
            );
        }
        self.total_output_frames = self
            .total_output_frames
            .saturating_add(output.len() / self.channels);
        Ok(output)
    }

    /// 把至多一个输入窗送进 rubato，并丢掉尚未消耗的输出延迟。
    ///
    /// 输入不足 `input_frames_next` 时右侧补零，同时通过 `partial_len` 告诉
    /// rubato 真实帧数，避免把补零当成有效音频。
    ///
    /// # Errors
    ///
    /// 构造交错缓冲或 rubato 处理失败时返回错误。
    fn process_once(&mut self, samples: &[f32], frames: usize) -> Result<Vec<f32>> {
        let input_frames = self.inner.input_frames_next();
        let mut padded = vec![0.0_f32; input_frames.saturating_mul(self.channels)];
        let sample_count = frames.saturating_mul(self.channels).min(samples.len());
        padded[..sample_count].copy_from_slice(&samples[..sample_count]);
        let input = InterleavedSlice::new(&padded, self.channels, input_frames)
            .context("failed to create streaming resampler input")?;

        let output_frames = self.inner.output_frames_max();
        let mut output_samples = vec![0.0_f32; output_frames.saturating_mul(self.channels)];
        let mut output =
            InterleavedSlice::new_mut(&mut output_samples, self.channels, output_frames)
                .context("failed to create streaming resampler output")?;
        let indexing = Indexing {
            input_offset: 0,
            output_offset: 0,
            active_channels_mask: None,
            // 不足一整窗时声明真实长度，rubato 才不会把右侧补零算进有效输入。
            partial_len: (frames < input_frames).then_some(frames),
        };
        let (_, produced) = self
            .inner
            .process_into_buffer(&input, &mut output, Some(&indexing))
            .context("streaming resampling failed")?;
        output_samples.truncate(produced.saturating_mul(self.channels));

        let trim = self.frames_to_trim.min(produced);
        self.frames_to_trim -= trim;
        if trim > 0 {
            output_samples.drain(..trim.saturating_mul(self.channels));
        }
        Ok(output_samples)
    }
}

/// [`crate::doc::AudioStream`] 使用的按需 PCM 生产器。
///
/// 内部握着 [`RawAudioStream`]，可按需转单声道、重采样，再把变换后的样本按
/// `chunk_size_ms` 重新切窗。真正缓存样本的是 `output` 队列；本类型本身是
/// iterator，不是已经切好的 chunk 列表。
pub(crate) struct SourceAudioStream {
    /// 源格式 PCM 拉取器。
    raw: RawAudioStream,
    /// 目标采样率；`None` 表示保持源采样率。
    sample_rate: Option<u32>,
    /// `Some(true)` 时把多声道块平均成单声道。
    mono: Option<bool>,
    /// 输出块的目标时长（毫秒）。
    chunk_size_ms: u64,
    /// 变换后、尚未组成完整输出块的交错样本。
    output: VecDeque<f32>,
    /// 第一块到来后锁定的输出采样率。
    output_sample_rate: Option<u32>,
    /// 第一块到来后锁定的输出声道数。
    output_channels: Option<u16>,
    /// 源容器/编码格式，原样传到每个输出块。
    source_format: Option<AudioFormat>,
    /// 仅在需要改变采样率时创建，并跨块复用。
    resampler: Option<StreamingResampler>,
    /// 下一个输出块在输出时间轴上的起始帧。
    next_output_frame: usize,
    /// 下一个输出块的从零开始序号。
    next_output_index: usize,
    /// 上游已经耗尽，或处理出错后不再拉取。
    finished: bool,
}

impl SourceAudioStream {
    /// 从 `source` 构造按 `chunk_size_ms` 吐块的生产器。
    ///
    /// 原始 PCM 没有容器可读，先整段解码再切块；其余来源走流式解码。
    /// `sample_rate` / `mono` 在第一块被处理后才真正生效。
    ///
    /// # Errors
    ///
    /// `chunk_size_ms` 或目标采样率为 0，或无法打开 / 解码来源时返回错误。
    pub(crate) fn new(
        source: AudioSource,
        chunk_size_ms: u64,
        sample_rate: Option<u32>,
        mono: Option<bool>,
    ) -> Result<Self> {
        if chunk_size_ms == 0 {
            bail!("chunk size must be greater than zero");
        }
        if sample_rate == Some(0) {
            bail!("sample rate must be greater than zero");
        }
        // PCM 没有 packet 边界，只能先变成 Waveform；编码音频可以边读边解。
        let raw = match &source {
            AudioSource::PcmS16Le { .. } => {
                RawAudioStream::Pcm(source.decode_waveform()?.into_chunks_ms(chunk_size_ms)?)
            }
            _ => RawAudioStream::Decoded(super::decode::stream_source(&source, chunk_size_ms)?),
        };
        Ok(Self {
            raw,
            sample_rate,
            mono,
            chunk_size_ms,
            output: VecDeque::new(),
            output_sample_rate: None,
            output_channels: None,
            source_format: None,
            resampler: None,
            next_output_frame: 0,
            next_output_index: 0,
            finished: false,
        })
    }

    /// 把上游一块源格式 PCM 变换后追加进 `output`。
    ///
    /// 顺序是：可选转单声道 → 需要时重采样 → 清洗非有限值 → 写入输出队列。
    /// 第一块会锁定输出采样率和声道数；重采样器只创建一次并跨后续块复用。
    ///
    /// # Errors
    ///
    /// 转单声道或重采样失败时返回错误。
    fn process_chunk(&mut self, chunk: AudioChunk) -> Result<()> {
        let chunk = if self.mono == Some(true) {
            chunk.to_mono()?
        } else {
            chunk
        };
        let target_rate = self.sample_rate.unwrap_or(chunk.sample_rate);
        let target_channels = chunk.channels;
        self.output_sample_rate.get_or_insert(target_rate);
        self.output_channels.get_or_insert(target_channels);
        if self.source_format.is_none() {
            self.source_format = chunk.source_format.clone();
        }

        let frames = chunk.frame_count();
        let mut samples = if target_rate == chunk.sample_rate {
            chunk.samples
        } else {
            if self.resampler.is_none() {
                self.resampler = Some(StreamingResampler::new(
                    chunk.sample_rate,
                    target_rate,
                    target_channels,
                )?);
            }
            self.resampler
                .as_mut()
                .expect("resampler initialized")
                .process(&chunk.samples, frames, chunk.is_final)?
        };
        super::data::sanitize_samples(&mut samples);
        self.output.extend(samples);
        if chunk.is_final {
            self.finished = true;
        }
        Ok(())
    }
}

impl Iterator for SourceAudioStream {
    type Item = Result<AudioChunk>;

    /// 吐出下一块目标时长的 PCM。
    ///
    /// 输出队列还不够一块、且上游未结束时，继续 `process_chunk`。采样率变换会
    /// 改变块边界，所以这里按变换后的采样率重新计算每块帧数，而不是沿用上游块。
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let channels = usize::from(self.output_channels.unwrap_or(1));
            let frames_per_chunk = self.output_sample_rate.map(|sample_rate| {
                (u128::from(self.chunk_size_ms) * u128::from(sample_rate))
                    .div_ceil(1000)
                    .max(1) as usize
            });
            let samples_per_chunk = frames_per_chunk
                .unwrap_or(usize::MAX)
                .saturating_mul(channels);
            // 队列已超过一块，或上游结束只剩尾巴时，切出当前输出块。
            if (frames_per_chunk.is_some() && self.output.len() > samples_per_chunk)
                || (self.finished && !self.output.is_empty())
            {
                let sample_count = samples_per_chunk.min(self.output.len());
                let samples = self.output.drain(..sample_count).collect::<Vec<_>>();
                let sample_rate = self
                    .output_sample_rate
                    .expect("stream metadata initialized");
                let channels = self.output_channels.expect("stream metadata initialized");
                let offset_ms = self.next_output_frame as u64 * 1000 / u64::from(sample_rate);
                self.next_output_frame = self
                    .next_output_frame
                    .saturating_add(samples.len() / usize::from(channels));
                let chunk = AudioChunk {
                    samples,
                    sample_rate,
                    channels,
                    source_format: self.source_format.clone(),
                    index: self.next_output_index,
                    offset_ms,
                    is_final: self.finished && self.output.is_empty(),
                };
                self.next_output_index = self.next_output_index.saturating_add(1);
                return Some(Ok(chunk));
            }
            if self.finished {
                return None;
            }
            match self.raw.next() {
                None => {
                    self.finished = true;
                }
                Some(Ok(chunk)) => {
                    if let Err(error) = self.process_chunk(chunk) {
                        self.finished = true;
                        return Some(Err(error));
                    }
                }
                Some(Err(error)) => {
                    self.finished = true;
                    return Some(Err(error));
                }
            }
        }
    }
}
