//! Waveform input decoding, loading, normalization, and resampling.

#[cfg(feature = "python-bindings")]
use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

pub(crate) mod data;
pub mod decode;
mod source;
pub(crate) mod stream;
pub use data::{AudioChunk, AudioChunks, AudioError, Waveform};
pub use source::{AudioChannel, AudioEncoding, AudioFormat, AudioInfo, AudioSource};

/// Python 加载路径上的可选重采样与转单声道。
///
/// # Errors
///
/// 声道数为 0、目标采样率为 0 或变换失败时返回错误。
#[cfg(feature = "python-bindings")]
pub(crate) fn transform_loaded_audio(
    mut waveform: Waveform,
    sample_rate: Option<u32>,
    mono: Option<bool>,
) -> Result<Waveform> {
    if waveform.channels == 0 {
        bail!("invalid channel count: 0");
    }
    if mono == Some(true) && waveform.channels != 1 {
        waveform = waveform.to_mono()?;
    }
    if let Some(sample_rate) = sample_rate {
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

/// 把 `file://` URL 或普通本地路径转成 [`PathBuf`]；HTTP(S) 返回 `None`。
fn local_path_from_urlish(value: &str) -> Option<PathBuf> {
    if let Some(rest) = value.strip_prefix("file://") {
        return Some(PathBuf::from(percent_decode_path(rest)));
    }
    if value.starts_with("http://") || value.starts_with("https://") {
        return None;
    }
    Some(Path::new(value).to_path_buf())
}

/// 解码路径中的 `%HH` 转义，非法序列原样保留。
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

/// 解析一位十六进制字符；非法字符返回 `None`。
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
