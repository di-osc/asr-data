//! Audio input decoding, loading, normalization, and resampling.

use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

use crate::AudioSource;

pub(crate) mod data;
pub mod decode;
pub(crate) mod stream;
pub use data::{Audio, AudioChunk, AudioChunks, AudioError};

/// Target ASR/VAD sample rate used by the offline pipeline.
pub const SAMPLE_RATE_HZ: u32 = 16_000;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AudioLoadOptions {
    pub sample_rate: Option<u32>,
    pub mono: Option<bool>,
}

#[derive(Debug, Default, Clone)]
pub struct AudioLoader;

impl AudioLoader {
    pub fn load_raw(&self, source: &AudioSource) -> Result<Audio> {
        let waveform = match source {
            AudioSource::Path(path) => decode::decode_path_audio(path)?,
            AudioSource::Url(url) => {
                if let Some(path) = local_path_from_urlish(url) {
                    decode::decode_path_audio(&path)?
                } else {
                    decode::decode_url_audio(url)?
                }
            }
            AudioSource::Base64(b64) => decode::decode_base64_audio(b64)?,
            AudioSource::EncodedBytes(bytes) => decode::decode_bytes_audio(bytes.clone())?,
            AudioSource::PcmS16Le {
                bytes,
                sample_rate,
                channels,
            } => Audio::from_i16_pcm_bytes_with_channels(bytes, *sample_rate, *channels)?,
        };
        let mut waveform = waveform;
        data::sanitize_samples(&mut waveform.samples);
        Ok(waveform)
    }

    pub fn load(&self, source: &AudioSource, options: &AudioLoadOptions) -> Result<Audio> {
        transform_loaded_audio(self.load_raw(source)?, options)
    }
}

pub(crate) fn transform_loaded_audio(
    mut waveform: Audio,
    options: &AudioLoadOptions,
) -> Result<Audio> {
    if waveform.channels == 0 {
        bail!("invalid channel count: 0");
    }
    if options.mono == Some(true) && waveform.channels != 1 {
        waveform = waveform.to_mono()?;
    }
    if let Some(sample_rate) = options.sample_rate {
        if sample_rate == 0 {
            bail!("sample rate must be greater than zero");
        }
        if waveform.sample_rate != sample_rate {
            waveform = waveform.resample(sample_rate)?;
        }
    }
    data::sanitize_samples(&mut waveform.samples);
    Ok(waveform)
}

fn local_path_from_urlish(value: &str) -> Option<PathBuf> {
    if let Some(rest) = value.strip_prefix("file://") {
        return Some(PathBuf::from(percent_decode_path(rest)));
    }
    if value.starts_with("http://") || value.starts_with("https://") {
        return None;
    }
    Some(Path::new(value).to_path_buf())
}

fn percent_decode_path(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(hi), Some(lo)) = (hex_value(bytes[i + 1]), hex_value(bytes[i + 2]))
        {
            out.push((hi << 4) | lo);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::local_path_from_urlish;
    use std::path::Path;

    #[test]
    fn urlish_local_path_detection_supports_file_urls_and_plain_paths() {
        assert_eq!(
            local_path_from_urlish("file:///tmp/audio.wav").as_deref(),
            Some(Path::new("/tmp/audio.wav"))
        );
        assert_eq!(
            local_path_from_urlish("file:///tmp/audio%20%281%29.wav").as_deref(),
            Some(Path::new("/tmp/audio (1).wav"))
        );
        assert_eq!(
            local_path_from_urlish("/tmp/audio.wav").as_deref(),
            Some(Path::new("/tmp/audio.wav"))
        );
        assert!(local_path_from_urlish("https://example.com/audio.wav").is_none());
    }
}
