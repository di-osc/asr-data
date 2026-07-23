//! Decoding audio bytes into a waveform.

use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};

use super::{Audio, AudioChunk, AudioEncoding, AudioFormat, AudioInfo, AudioSource};

pub struct DecodedAudioChunks {
    format: Box<dyn symphonia::core::formats::FormatReader>,
    decoder: Box<dyn symphonia::core::codecs::audio::AudioDecoder>,
    track_id: u32,
    sample_rate: u32,
    channels: u16,
    source_format: AudioFormat,
    samples_per_chunk: usize,
    buffered: VecDeque<f32>,
    offset_frames: usize,
    finished: bool,
}

impl DecodedAudioChunks {
    fn new(
        mss: symphonia::core::io::MediaSourceStream<'static>,
        hint: symphonia::core::formats::probe::Hint,
        encoding: AudioEncoding,
        chunk_size_ms: u64,
    ) -> Result<Self> {
        use symphonia::core::codecs::audio::AudioDecoderOptions;
        use symphonia::core::formats::{FormatOptions, TrackType};
        use symphonia::core::meta::MetadataOptions;
        if chunk_size_ms == 0 {
            bail!("chunk size must be greater than zero");
        }
        let format = symphonia::default::get_probe()
            .probe(
                &hint,
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|e| anyhow::anyhow!("failed to probe audio format: {e}"))?;
        let track = format
            .default_track(TrackType::Audio)
            .ok_or_else(|| anyhow::anyhow!("no audio tracks found"))?;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or_else(|| anyhow::anyhow!("track has no audio codec parameters"))?;
        let sample_rate = params
            .sample_rate
            .ok_or_else(|| anyhow::anyhow!("unknown sample rate"))?;
        let channels = params.channels.as_ref().map(|c| c.count()).unwrap_or(1) as u16;
        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(params, &AudioDecoderOptions::default())
            .map_err(|e| anyhow::anyhow!("failed to create decoder: {e}"))?;
        let track_id = track.id;
        let frames = (u128::from(chunk_size_ms) * u128::from(sample_rate))
            .div_ceil(1000)
            .max(1) as usize;
        Ok(Self {
            format,
            decoder,
            track_id,
            sample_rate,
            channels,
            source_format: AudioFormat {
                encoding,
                sample_rate,
                channels,
            },
            samples_per_chunk: frames.saturating_mul(usize::from(channels)),
            buffered: VecDeque::new(),
            offset_frames: 0,
            finished: false,
        })
    }

    fn decode_packet(&mut self) -> Result<()> {
        use symphonia::core::errors::Error as SymphoniaError;
        loop {
            let packet = match self.format.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => {
                    self.finished = true;
                    return Ok(());
                }
                Err(error) => return Err(anyhow::anyhow!("failed to read audio packet: {error}")),
            };
            if packet.track_id != self.track_id {
                continue;
            }
            match self.decoder.decode(&packet) {
                Ok(decoded) => {
                    let mut samples: Vec<f32> = Vec::new();
                    decoded.copy_to_vec_interleaved(&mut samples);
                    crate::audio::data::sanitize_samples(&mut samples);
                    self.buffered.extend(samples);
                    return Ok(());
                }
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(error) => {
                    return Err(anyhow::anyhow!("failed to decode audio packet: {error}"));
                }
            }
        }
    }
}

impl Iterator for DecodedAudioChunks {
    type Item = Result<AudioChunk>;
    fn next(&mut self) -> Option<Self::Item> {
        while self.buffered.len() < self.samples_per_chunk && !self.finished {
            if let Err(error) = self.decode_packet() {
                self.finished = true;
                return Some(Err(error));
            }
        }
        if self.buffered.is_empty() {
            return None;
        }
        let count = self.samples_per_chunk.min(self.buffered.len());
        let samples = self.buffered.drain(..count).collect::<Vec<_>>();
        let offset_ms = self.offset_frames as u64 * 1000 / u64::from(self.sample_rate);
        self.offset_frames += count / usize::from(self.channels);
        Some(Ok(AudioChunk {
            samples,
            sample_rate: self.sample_rate,
            channels: self.channels,
            source_format: Some(self.source_format.clone()),
            offset_ms,
            is_final: self.finished && self.buffered.is_empty(),
        }))
    }
}

struct HttpMediaSource(reqwest::blocking::Response);
impl Read for HttpMediaSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}
impl Seek for HttpMediaSource {
    fn seek(&mut self, _: SeekFrom) -> std::io::Result<u64> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "HTTP stream is not seekable",
        ))
    }
}
impl symphonia::core::io::MediaSource for HttpMediaSource {
    fn is_seekable(&self) -> bool {
        false
    }
    fn byte_len(&self) -> Option<u64> {
        self.0.content_length()
    }
}

pub fn stream_source(source: &AudioSource, chunk_size_ms: u64) -> Result<DecodedAudioChunks> {
    use base64::Engine;
    use std::fs::File;
    use std::io::Cursor;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::io::MediaSourceStream;
    let (media, hint, encoding): (
        Box<dyn symphonia::core::io::MediaSource>,
        Hint,
        AudioEncoding,
    ) = match source {
        AudioSource::Path(path) => {
            let mut hint = Hint::new();
            if let Some(ext) = path.extension().and_then(|v| v.to_str()) {
                hint.with_extension(ext);
            }
            (
                Box::new(File::open(path)?),
                hint,
                path.extension()
                    .and_then(|v| v.to_str())
                    .map(encoding_from_extension)
                    .unwrap_or(AudioEncoding::Unknown),
            )
        }
        AudioSource::Url(url) => {
            if let Ok(parsed) = reqwest::Url::parse(url)
                && parsed.scheme() == "file"
            {
                let path = parsed
                    .to_file_path()
                    .map_err(|_| anyhow::anyhow!("invalid file URL {url:?}"))?;
                let mut hint = Hint::new();
                if let Some(ext) = path.extension().and_then(|v| v.to_str()) {
                    hint.with_extension(ext);
                }
                let encoding = path
                    .extension()
                    .and_then(|v| v.to_str())
                    .map(encoding_from_extension)
                    .unwrap_or(AudioEncoding::Unknown);
                return DecodedAudioChunks::new(
                    MediaSourceStream::new(Box::new(File::open(path)?), Default::default()),
                    hint,
                    encoding,
                    chunk_size_ms,
                );
            }
            let response = reqwest::blocking::get(url)?;
            if !response.status().is_success() {
                bail!("HTTP error fetching {url:?}: {}", response.status());
            }
            (
                Box::new(HttpMediaSource(response)),
                Hint::new(),
                encoding_from_extension(
                    url.split(['?', '#'])
                        .next()
                        .unwrap_or(url)
                        .rsplit('.')
                        .next()
                        .unwrap_or(""),
                ),
            )
        }
        AudioSource::EncodedBytes(bytes) => (
            Box::new(Cursor::new(bytes.clone())),
            Hint::new(),
            detect_encoding(bytes),
        ),
        AudioSource::Base64(data) => {
            let raw = data
                .strip_prefix("data:")
                .and_then(|v| v.split_once(',').map(|v| v.1))
                .unwrap_or(data);
            let bytes = base64::engine::general_purpose::STANDARD.decode(raw)?;
            let encoding = detect_encoding(&bytes);
            (Box::new(Cursor::new(bytes)), Hint::new(), encoding)
        }
        AudioSource::PcmS16Le { .. } => bail!("raw PCM uses the direct chunk iterator"),
    };
    DecodedAudioChunks::new(
        MediaSourceStream::new(media, Default::default()),
        hint,
        encoding,
        chunk_size_ms,
    )
}

pub fn decode_path(path: &Path) -> Result<(Vec<f32>, u32)> {
    let waveform = decode_path_audio(path)?;
    let mono = waveform.to_mono()?;
    Ok((mono.samples, mono.sample_rate))
}

pub fn probe_path(path: &Path) -> Result<AudioInfo> {
    use std::fs::File;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::io::MediaSourceStream;

    let file = File::open(path).with_context(|| format!("failed to open audio file {path:?}"))?;
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        hint.with_extension(extension);
    }
    let encoding = path
        .extension()
        .and_then(|value| value.to_str())
        .map(encoding_from_extension)
        .unwrap_or(AudioEncoding::Unknown);
    probe_audio_stream(
        MediaSourceStream::new(Box::new(file), Default::default()),
        hint,
        encoding,
    )
}

pub fn probe_url(url: &str) -> Result<AudioInfo> {
    let response = reqwest::blocking::get(url)
        .with_context(|| format!("failed to fetch audio from URL {url:?}"))?;
    if !response.status().is_success() {
        bail!("HTTP error fetching {url:?}: {}", response.status());
    }
    let bytes = response
        .bytes()
        .with_context(|| format!("failed to read response body for {url:?}"))?
        .to_vec();
    probe_bytes_with_encoding(bytes.clone(), encoding_from_url(url, &bytes))
}

pub fn probe_base64(data: &str) -> Result<AudioInfo> {
    use base64::Engine;
    let raw = data
        .strip_prefix("data:")
        .and_then(|value| value.split_once(',').map(|value| value.1))
        .unwrap_or(data);
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(raw)
        .map_err(|error| anyhow::anyhow!("base64 decode error: {error}"))?;
    probe_bytes(bytes)
}

pub fn probe_bytes(bytes: impl Into<Vec<u8>>) -> Result<AudioInfo> {
    let bytes = bytes.into();
    let encoding = detect_encoding(&bytes);
    probe_bytes_with_encoding(bytes, encoding)
}

fn probe_bytes_with_encoding(bytes: Vec<u8>, encoding: AudioEncoding) -> Result<AudioInfo> {
    use std::io::Cursor;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::io::MediaSourceStream;
    probe_audio_stream(
        MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default()),
        Hint::new(),
        encoding,
    )
}

fn probe_audio_stream(
    mss: symphonia::core::io::MediaSourceStream,
    hint: symphonia::core::formats::probe::Hint,
    encoding: AudioEncoding,
) -> Result<AudioInfo> {
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::meta::MetadataOptions;

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|error| anyhow::anyhow!("failed to probe audio format: {error}"))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| anyhow::anyhow!("no audio tracks found"))?;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| anyhow::anyhow!("track has no audio codec parameters"))?;
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| anyhow::anyhow!("unknown sample rate"))?;
    let channels = params
        .channels
        .as_ref()
        .map(|value| value.count())
        .unwrap_or(1) as u16;
    let track_id = track.id;
    let time_base = track.time_base;
    let frame_count = track.num_frames.or_else(|| {
        let duration = track.duration?;
        let time_base = time_base?;
        let numerator = u128::from(duration.get())
            .saturating_mul(u128::from(time_base.numer.get()))
            .saturating_mul(u128::from(sample_rate));
        Some(
            numerator
                .div_ceil(u128::from(time_base.denom.get()))
                .min(u128::from(u64::MAX)) as u64,
        )
    });
    let frame_count = match frame_count {
        Some(frame_count) => frame_count,
        None => {
            let time_base =
                time_base.ok_or_else(|| anyhow::anyhow!("audio duration is unavailable"))?;
            let mut ticks = 0_u64;
            loop {
                match format.next_packet() {
                    Ok(Some(packet)) if packet.track_id == track_id => {
                        ticks = ticks.saturating_add(packet.dur.get());
                    }
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(error) => {
                        return Err(anyhow::anyhow!(
                            "failed to scan audio packets for duration: {error}"
                        ));
                    }
                }
            }
            let numerator = u128::from(ticks)
                .saturating_mul(u128::from(time_base.numer.get()))
                .saturating_mul(u128::from(sample_rate));
            numerator
                .div_ceil(u128::from(time_base.denom.get()))
                .min(u128::from(u64::MAX)) as u64
        }
    };
    Ok(AudioInfo {
        sample_rate,
        channels,
        frame_count,
        source_format: AudioFormat {
            encoding,
            sample_rate,
            channels,
        },
    })
}

pub fn decode_path_audio(path: &Path) -> Result<Audio> {
    use std::fs::File;

    use symphonia::core::formats::probe::Hint;
    use symphonia::core::io::MediaSourceStream;

    let file = File::open(path).with_context(|| format!("failed to open audio file {path:?}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    let encoding = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(encoding_from_extension)
        .unwrap_or(AudioEncoding::Unknown);
    let (samples, sr, channels) = decode_audio_stream(mss, hint)?;
    Ok(
        Audio::try_new_with_channels(samples, sr, channels as u16)?.with_source_format(
            AudioFormat {
                encoding,
                sample_rate: sr,
                channels: channels as u16,
            },
        ),
    )
}

pub fn decode_url(url: &str) -> Result<(Vec<f32>, u32)> {
    let waveform = decode_url_audio(url)?;
    let mono = waveform.to_mono()?;
    Ok((mono.samples, mono.sample_rate))
}

pub fn decode_url_audio(url: &str) -> Result<Audio> {
    use std::io::Cursor;

    use symphonia::core::formats::probe::Hint;
    use symphonia::core::io::MediaSourceStream;

    let resp = reqwest::blocking::get(url)
        .with_context(|| format!("failed to fetch audio from URL {url:?}"))?;
    if !resp.status().is_success() {
        bail!("HTTP error fetching {url:?}: {}", resp.status());
    }

    let bytes = resp
        .bytes()
        .with_context(|| format!("failed to read response body for {url:?}"))?;
    let bytes = bytes.to_vec();
    let encoding = encoding_from_url(url, &bytes);
    let cursor = Cursor::new(bytes);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = url.rsplit('.').next() {
        let ext = ext.to_lowercase();
        if ["wav", "mp3", "flac", "ogg", "m4a", "aac", "opus", "webm"].contains(&ext.as_str()) {
            hint.with_extension(ext.as_str());
        }
    }

    let (samples, sr, channels) = decode_audio_stream(mss, hint)?;
    Ok(
        Audio::try_new_with_channels(samples, sr, channels as u16)?.with_source_format(
            AudioFormat {
                encoding,
                sample_rate: sr,
                channels: channels as u16,
            },
        ),
    )
}

pub async fn download_url_bytes(url: &str) -> Result<Vec<u8>> {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    static NO_PROXY_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

    let parsed = reqwest::Url::parse(url).with_context(|| format!("invalid audio URL {url:?}"))?;
    let local = matches!(parsed.host_str(), Some("localhost" | "127.0.0.1" | "::1"));
    let client = if local {
        NO_PROXY_CLIENT.get_or_init(|| {
            reqwest::Client::builder()
                .no_proxy()
                .build()
                .unwrap_or_else(|error| panic!("failed to build no-proxy HTTP client: {error}"))
        })
    } else {
        CLIENT.get_or_init(reqwest::Client::new)
    };
    let response = client
        .get(parsed)
        .send()
        .await
        .with_context(|| format!("failed to fetch audio from URL {url:?}"))?;
    if !response.status().is_success() {
        bail!("HTTP error fetching {url:?}: {}", response.status());
    }
    Ok(response
        .bytes()
        .await
        .with_context(|| format!("failed to read response body for {url:?}"))?
        .to_vec())
}

pub fn decode_base64(b64: &str) -> Result<(Vec<f32>, u32)> {
    let waveform = decode_base64_audio(b64)?;
    let mono = waveform.to_mono()?;
    Ok((mono.samples, mono.sample_rate))
}

pub fn decode_base64_audio(b64: &str) -> Result<Audio> {
    use base64::Engine;

    let data = if b64.contains(',') && b64.trim().starts_with("data:") {
        b64.split(',')
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("invalid data URL base64 format"))?
    } else {
        b64
    };

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|e| anyhow::anyhow!("base64 decode error: {e}"))?;

    decode_bytes_audio(bytes)
}

pub fn decode_bytes_audio(bytes: impl Into<Vec<u8>>) -> Result<Audio> {
    use std::io::Cursor;

    use symphonia::core::formats::probe::Hint;
    use symphonia::core::io::MediaSourceStream;

    let bytes = bytes.into();
    let encoding = detect_encoding(&bytes);
    let cursor = Cursor::new(bytes);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let (samples, sr, channels) = decode_audio_stream(mss, Hint::new())?;
    Ok(
        Audio::try_new_with_channels(samples, sr, channels as u16)?.with_source_format(
            AudioFormat {
                encoding,
                sample_rate: sr,
                channels: channels as u16,
            },
        ),
    )
}

fn encoding_from_url(url: &str, bytes: &[u8]) -> AudioEncoding {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let extension = path.rsplit_once('.').map(|(_, extension)| extension);
    extension
        .map(encoding_from_extension)
        .filter(|encoding| *encoding != AudioEncoding::Unknown)
        .unwrap_or_else(|| detect_encoding(bytes))
}

fn encoding_from_extension(extension: &str) -> AudioEncoding {
    match extension.to_ascii_lowercase().as_str() {
        "wav" | "wave" => AudioEncoding::Wav,
        "flac" => AudioEncoding::Flac,
        "mp3" | "mpeg" => AudioEncoding::Mp3,
        "ogg" | "oga" | "opus" => AudioEncoding::Ogg,
        _ => AudioEncoding::Unknown,
    }
}

fn detect_encoding(bytes: &[u8]) -> AudioEncoding {
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        AudioEncoding::Wav
    } else if bytes.starts_with(b"fLaC") {
        AudioEncoding::Flac
    } else if bytes.starts_with(b"OggS") {
        AudioEncoding::Ogg
    } else if bytes.starts_with(b"ID3")
        || bytes
            .get(..2)
            .is_some_and(|prefix| prefix[0] == 0xff && prefix[1] & 0xe0 == 0xe0)
    {
        AudioEncoding::Mp3
    } else {
        AudioEncoding::Unknown
    }
}

fn decode_audio_stream(
    mss: symphonia::core::io::MediaSourceStream,
    hint: symphonia::core::formats::probe::Hint,
) -> Result<(Vec<f32>, u32, usize)> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::formats::TrackType;
    use symphonia::core::meta::MetadataOptions;

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| anyhow::anyhow!("failed to probe audio format: {e}"))?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| anyhow::anyhow!("no audio tracks found"))?;

    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| anyhow::anyhow!("track has no audio codec parameters"))?;

    let sample_rate = audio_params
        .sample_rate
        .ok_or_else(|| anyhow::anyhow!("unknown sample rate"))?;
    let channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|e| anyhow::anyhow!("failed to create decoder: {e}"))?;

    let track_id = track.id;
    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(e) => return Err(anyhow::anyhow!("failed to read audio packet: {e}")),
        };

        if packet.track_id != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(anyhow::anyhow!("failed to decode audio packet: {e}")),
        };
        let decoded_channels = decoded.spec().channels().count();
        if decoded_channels != channels {
            bail!(
                "decoded channel count changed from {} to {}",
                channels,
                decoded_channels
            );
        }

        let mut chunk = Vec::new();
        decoded.copy_to_vec_interleaved(&mut chunk);
        samples.extend_from_slice(&chunk);
    }

    crate::audio::data::sanitize_samples(&mut samples);
    Ok((samples, sample_rate, channels))
}

#[cfg(test)]
mod tests {
    use super::{Audio, decode_bytes_audio, decode_path, decode_path_audio};
    use std::io::Write;

    #[test]
    fn test_decode_path_wav() -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let path_48k = root.join("fixtures").join("audio").join("asr_en.wav");
        let path_16k = root.join("fixtures").join("audio").join("asr_en_16k.wav");
        if !path_48k.exists() || !path_16k.exists() {
            return Ok(());
        }

        let (wav_48k, sr_48k) = decode_path(&path_48k)?;
        if sr_48k != 48_000 {
            anyhow::bail!("expected 48kHz wav, got sr={sr_48k}");
        }
        if wav_48k.is_empty() {
            anyhow::bail!("expected non-empty decode output for 48k wav");
        }

        let (wav_16k, sr_16k) = decode_path(&path_16k)?;
        if sr_16k != 16_000 {
            anyhow::bail!("expected 16kHz wav, got sr={sr_16k}");
        }
        if wav_16k.len() != 240_820 {
            anyhow::bail!("expected 240_820 samples, got {}", wav_16k.len());
        }

        Ok(())
    }

    #[test]
    fn stereo_samples_are_downmixed_by_averaging_channels() -> anyhow::Result<()> {
        let stereo = Audio::new_with_channels(
            vec![
                1.0, 3.0, // frame 0
                2.0, 4.0, // frame 1
            ],
            16_000,
            2,
        );

        let mono = stereo.to_mono()?;

        assert_eq!(mono.samples, vec![2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn decode_path_downmixes_stereo_wav_instead_of_dropping_a_channel() -> anyhow::Result<()> {
        let path = std::env::temp_dir().join(format!(
            "asr-stereo-downmix-{}.wav",
            uuid::Uuid::new_v4().simple()
        ));
        write_pcm16_wav(
            &path,
            16_000,
            2,
            &[
                8192, 24576, // frame 0 => 0.5
                16384, -8192, // frame 1 => 0.125
            ],
        )?;

        let (samples, sample_rate) = decode_path(&path)?;
        let waveform = decode_path_audio(&path)?;
        std::fs::remove_file(&path).ok();

        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 0.5).abs() < 1e-6, "{:?}", samples);
        assert!((samples[1] - 0.125).abs() < 1e-6, "{:?}", samples);
        assert_eq!(waveform.channels, 2);
        Ok(())
    }

    #[test]
    fn decode_bytes_audio_decodes_encoded_audio_bytes() -> anyhow::Result<()> {
        let path = std::env::temp_dir().join(format!(
            "asr-encoded-bytes-{}.wav",
            uuid::Uuid::new_v4().simple()
        ));
        write_pcm16_wav(&path, 16_000, 1, &[0, 16_384, -16_384])?;
        let bytes = std::fs::read(&path)?;
        std::fs::remove_file(&path).ok();

        let waveform = decode_bytes_audio(bytes)?;

        assert_eq!(waveform.sample_rate, 16_000);
        assert_eq!(waveform.channels, 1);
        assert_eq!(waveform.samples.len(), 3);
        Ok(())
    }

    fn write_pcm16_wav(
        path: &std::path::Path,
        sample_rate: u32,
        channels: u16,
        samples: &[i16],
    ) -> anyhow::Result<()> {
        let mut file = std::fs::File::create(path)?;
        let data_len = samples.len() as u32 * 2;
        let byte_rate = sample_rate * u32::from(channels) * 2;
        let block_align = channels * 2;

        file.write_all(b"RIFF")?;
        file.write_all(&(36 + data_len).to_le_bytes())?;
        file.write_all(b"WAVEfmt ")?;
        file.write_all(&16u32.to_le_bytes())?;
        file.write_all(&1u16.to_le_bytes())?;
        file.write_all(&channels.to_le_bytes())?;
        file.write_all(&sample_rate.to_le_bytes())?;
        file.write_all(&byte_rate.to_le_bytes())?;
        file.write_all(&block_align.to_le_bytes())?;
        file.write_all(&16u16.to_le_bytes())?;
        file.write_all(b"data")?;
        file.write_all(&data_len.to_le_bytes())?;
        for sample in samples {
            file.write_all(&sample.to_le_bytes())?;
        }
        Ok(())
    }
}
