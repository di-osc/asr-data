//! Decoding audio bytes into a waveform.
//!
//! This is intentionally feature-gated: production ASR can accept paths/URLs/base64,
//! but for model/math bring-up you can start with in-memory waveform inputs only.

use anyhow::{Result, bail};
use std::path::Path;
#[cfg(feature = "audio-loading")]
use std::sync::OnceLock;

#[cfg(feature = "audio-loading")]
use anyhow::Context;

#[cfg(feature = "audio-loading")]
pub fn decode_path(path: &Path) -> Result<(Vec<f32>, u32)> {
    let waveform = decode_path_waveform(path)?;
    let mono = waveform.to_mono()?;
    Ok((mono.samples, mono.sample_rate))
}

#[cfg(feature = "audio-loading")]
pub fn decode_path_waveform(path: &Path) -> Result<crate::Waveform> {
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
        .unwrap_or(crate::AudioEncoding::Unknown);
    let (samples, sr, channels) = decode_audio_stream(mss, hint)?;
    Ok(
        crate::Waveform::try_new_with_channels(samples, sr, channels as u16)?.with_source_format(
            crate::AudioFormat {
                encoding,
                sample_rate: sr,
                channels: channels as u16,
            },
        ),
    )
}

#[cfg(not(feature = "audio-loading"))]
pub fn decode_path(_path: &Path) -> Result<(Vec<f32>, u32)> {
    bail!("decode_path requires the `audio-loading` feature")
}

#[cfg(not(feature = "audio-loading"))]
pub fn decode_path_waveform(_path: &Path) -> Result<crate::Waveform> {
    bail!("decode_path_waveform requires the `audio-loading` feature")
}

#[cfg(feature = "audio-loading")]
pub fn decode_url(url: &str) -> Result<(Vec<f32>, u32)> {
    let waveform = decode_url_waveform(url)?;
    let mono = waveform.to_mono()?;
    Ok((mono.samples, mono.sample_rate))
}

#[cfg(feature = "audio-loading")]
pub fn decode_url_waveform(url: &str) -> Result<crate::Waveform> {
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
        crate::Waveform::try_new_with_channels(samples, sr, channels as u16)?.with_source_format(
            crate::AudioFormat {
                encoding,
                sample_rate: sr,
                channels: channels as u16,
            },
        ),
    )
}

#[cfg(feature = "audio-loading")]
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

#[cfg(not(feature = "audio-loading"))]
pub fn decode_url(_url: &str) -> Result<(Vec<f32>, u32)> {
    bail!("decode_url requires the `audio-loading` feature")
}

#[cfg(not(feature = "audio-loading"))]
pub fn decode_url_waveform(_url: &str) -> Result<crate::Waveform> {
    bail!("decode_url_waveform requires the `audio-loading` feature")
}

#[cfg(feature = "audio-loading")]
pub fn decode_base64(b64: &str) -> Result<(Vec<f32>, u32)> {
    let waveform = decode_base64_waveform(b64)?;
    let mono = waveform.to_mono()?;
    Ok((mono.samples, mono.sample_rate))
}

#[cfg(feature = "audio-loading")]
pub fn decode_base64_waveform(b64: &str) -> Result<crate::Waveform> {
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

    decode_bytes_waveform(bytes)
}

#[cfg(feature = "audio-loading")]
pub fn decode_bytes_waveform(bytes: impl Into<Vec<u8>>) -> Result<crate::Waveform> {
    use std::io::Cursor;

    use symphonia::core::formats::probe::Hint;
    use symphonia::core::io::MediaSourceStream;

    let bytes = bytes.into();
    let encoding = detect_encoding(&bytes);
    let cursor = Cursor::new(bytes);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let (samples, sr, channels) = decode_audio_stream(mss, Hint::new())?;
    Ok(
        crate::Waveform::try_new_with_channels(samples, sr, channels as u16)?.with_source_format(
            crate::AudioFormat {
                encoding,
                sample_rate: sr,
                channels: channels as u16,
            },
        ),
    )
}

#[cfg(feature = "audio-loading")]
fn encoding_from_url(url: &str, bytes: &[u8]) -> crate::AudioEncoding {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let extension = path.rsplit_once('.').map(|(_, extension)| extension);
    extension
        .map(encoding_from_extension)
        .filter(|encoding| *encoding != crate::AudioEncoding::Unknown)
        .unwrap_or_else(|| detect_encoding(bytes))
}

#[cfg(feature = "audio-loading")]
fn encoding_from_extension(extension: &str) -> crate::AudioEncoding {
    match extension.to_ascii_lowercase().as_str() {
        "wav" | "wave" => crate::AudioEncoding::Wav,
        "flac" => crate::AudioEncoding::Flac,
        "mp3" | "mpeg" => crate::AudioEncoding::Mp3,
        "ogg" | "oga" | "opus" => crate::AudioEncoding::Ogg,
        _ => crate::AudioEncoding::Unknown,
    }
}

#[cfg(feature = "audio-loading")]
fn detect_encoding(bytes: &[u8]) -> crate::AudioEncoding {
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        crate::AudioEncoding::Wav
    } else if bytes.starts_with(b"fLaC") {
        crate::AudioEncoding::Flac
    } else if bytes.starts_with(b"OggS") {
        crate::AudioEncoding::Ogg
    } else if bytes.starts_with(b"ID3")
        || bytes
            .get(..2)
            .is_some_and(|prefix| prefix[0] == 0xff && prefix[1] & 0xe0 == 0xe0)
    {
        crate::AudioEncoding::Mp3
    } else {
        crate::AudioEncoding::Unknown
    }
}

#[cfg(not(feature = "audio-loading"))]
pub fn decode_base64(_b64: &str) -> Result<(Vec<f32>, u32)> {
    bail!("decode_base64 requires the `audio-loading` feature")
}

#[cfg(not(feature = "audio-loading"))]
pub fn decode_base64_waveform(_b64: &str) -> Result<crate::Waveform> {
    bail!("decode_base64_waveform requires the `audio-loading` feature")
}

#[cfg(not(feature = "audio-loading"))]
pub fn decode_bytes_waveform(_bytes: impl Into<Vec<u8>>) -> Result<crate::Waveform> {
    bail!("decode_bytes_waveform requires the `audio-loading` feature")
}

#[cfg(feature = "audio-loading")]
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

    Ok((samples, sample_rate, channels))
}

#[cfg(all(test, feature = "audio-loading"))]
mod tests {
    use super::{decode_bytes_waveform, decode_path, decode_path_waveform};
    use crate::Waveform;
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
        let stereo = Waveform::new_with_channels(
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
        let waveform = decode_path_waveform(&path)?;
        std::fs::remove_file(&path).ok();

        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 0.5).abs() < 1e-6, "{:?}", samples);
        assert!((samples[1] - 0.125).abs() < 1e-6, "{:?}", samples);
        assert_eq!(waveform.channels, 2);
        Ok(())
    }

    #[test]
    fn decode_bytes_waveform_decodes_encoded_audio_bytes() -> anyhow::Result<()> {
        let path = std::env::temp_dir().join(format!(
            "asr-encoded-bytes-{}.wav",
            uuid::Uuid::new_v4().simple()
        ));
        write_pcm16_wav(&path, 16_000, 1, &[0, 16_384, -16_384])?;
        let bytes = std::fs::read(&path)?;
        std::fs::remove_file(&path).ok();

        let waveform = decode_bytes_waveform(bytes)?;

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
