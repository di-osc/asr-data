use std::collections::VecDeque;

use anyhow::{Context, Result, bail};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

#[cfg(feature = "python-bindings")]
use super::{AudioChunk, AudioChunks, AudioLoadOptions, AudioSource};

#[cfg(feature = "python-bindings")]
enum RawAudioStream {
    Decoded(super::decode::DecodedAudioChunks),
    Pcm(AudioChunks),
}

#[cfg(feature = "python-bindings")]
impl Iterator for RawAudioStream {
    type Item = Result<AudioChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Decoded(chunks) => chunks.next(),
            Self::Pcm(chunks) => chunks.next().map(Ok),
        }
    }
}

struct StreamingResampler {
    inner: Async<f32>,
    channels: usize,
    ratio: f64,
    frames_to_trim: usize,
    total_input_frames: usize,
    total_output_frames: usize,
    input: VecDeque<f32>,
}

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

#[cfg(feature = "python-bindings")]
pub(crate) struct SourceAudioStream {
    raw: RawAudioStream,
    options: AudioLoadOptions,
    chunk_size_ms: u64,
    output: VecDeque<f32>,
    output_sample_rate: Option<u32>,
    output_channels: Option<u16>,
    source_format: Option<crate::AudioFormat>,
    resampler: Option<StreamingResampler>,
    next_output_frame: usize,
    finished: bool,
}

#[cfg(feature = "python-bindings")]
impl SourceAudioStream {
    pub(crate) fn new(
        source: AudioSource,
        chunk_size_ms: u64,
        options: AudioLoadOptions,
    ) -> Result<Self> {
        if chunk_size_ms == 0 {
            bail!("chunk size must be greater than zero");
        }
        if options.sample_rate == Some(0) {
            bail!("sample rate must be greater than zero");
        }
        let raw = match &source {
            AudioSource::PcmS16Le { .. } => {
                RawAudioStream::Pcm(source.load()?.into_chunks_ms(chunk_size_ms)?)
            }
            _ => RawAudioStream::Decoded(super::decode::stream_source(&source, chunk_size_ms)?),
        };
        Ok(Self {
            raw,
            options,
            chunk_size_ms,
            output: VecDeque::new(),
            output_sample_rate: None,
            output_channels: None,
            source_format: None,
            resampler: None,
            next_output_frame: 0,
            finished: false,
        })
    }

    fn process_chunk(&mut self, chunk: AudioChunk) -> Result<()> {
        let chunk = if self.options.mono == Some(true) {
            chunk.to_mono()?
        } else {
            chunk
        };
        let target_rate = self.options.sample_rate.unwrap_or(chunk.sample_rate);
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

#[cfg(feature = "python-bindings")]
impl Iterator for SourceAudioStream {
    type Item = Result<AudioChunk>;

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
                return Some(Ok(AudioChunk {
                    samples,
                    sample_rate,
                    channels,
                    source_format: self.source_format.clone(),
                    offset_ms,
                    is_final: self.finished && self.output.is_empty(),
                }));
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
