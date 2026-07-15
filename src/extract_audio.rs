use std::fs;
use std::path::Path;

use crate::{Audio, AudioDb, AudioDbError, AudioSource};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractAudioSummary {
    pub extracted: usize,
    pub skipped: usize,
}

pub fn extract_embedded_audio(
    audios: &[Audio],
    dir: impl AsRef<Path>,
) -> Result<ExtractAudioSummary, std::io::Error> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;

    let mut extracted = 0usize;
    let mut skipped = 0usize;
    let mut used_names = std::collections::HashSet::new();

    for audio in audios {
        let Some((bytes, extension)) = embedded_audio_bytes(audio) else {
            skipped += 1;
            continue;
        };

        let stem = audio.audio_id();
        let mut filename = format!("{stem}.{extension}");
        let mut suffix = 1usize;
        while used_names.contains(&filename) {
            filename = format!("{stem}_{suffix}.{extension}");
            suffix += 1;
        }
        used_names.insert(filename.clone());

        let path = dir.join(filename);
        fs::write(path, bytes)?;
        extracted += 1;
    }

    Ok(ExtractAudioSummary { extracted, skipped })
}

fn embedded_audio_bytes(audio: &Audio) -> Option<(&[u8], &'static str)> {
    match &audio.source {
        AudioSource::EncodedBytes(bytes) => Some((bytes, detect_extension(bytes))),
        AudioSource::PcmS16Le { bytes, .. } => Some((bytes, "pcm")),
        AudioSource::Path(_) | AudioSource::Url(_) | AudioSource::Base64(_) => None,
    }
}

fn detect_extension(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        "wav"
    } else if bytes.starts_with(b"fLaC") {
        "flac"
    } else if bytes.starts_with(b"OggS") {
        "ogg"
    } else if bytes.starts_with(b"ID3")
        || bytes
            .get(..2)
            .is_some_and(|prefix| prefix[0] == 0xff && prefix[1] & 0xe0 == 0xe0)
    {
        "mp3"
    } else {
        "audio"
    }
}

pub fn extract_embedded_audio_from_db(
    input: impl AsRef<Path>,
    dir: impl AsRef<Path>,
) -> Result<ExtractAudioSummary, AudioDbError> {
    let audios = AudioDb::open(input, crate::AudioDbMode::ReadOnly)?.load_all()?;
    extract_embedded_audio(&audios, dir).map_err(|error| {
        AudioDbError::Sqlite(rusqlite::Error::ToSqlConversionFailure(Box::new(error)))
    })
}

#[cfg(test)]
mod tests {
    use crate::{Audio, AudioSource};

    #[test]
    fn exported_filename_uses_audio_id() {
        let audio = Audio::with_id("record-12", AudioSource::from_encoded_bytes(vec![1, 2, 3]));
        assert_eq!(audio.audio_id(), "record-12");
    }
}
