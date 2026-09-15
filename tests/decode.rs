use std::io::Write;
use std::path::Path;

use asr_data::audio::decode::{decode_bytes_audio, decode_path, decode_path_audio};

#[test]
fn test_decode_path_wav() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let path_48k = root.join("fixtures").join("audio").join("asr_en.wav");
    let path_16k = root.join("fixtures").join("audio").join("asr_en_16k.wav");
    if !path_48k.exists() || !path_16k.exists() {
        return;
    }

    let (wav_48k, sr_48k) = decode_path(&path_48k).expect("decode 48k wav");
    assert_eq!(sr_48k, 48_000);
    assert!(!wav_48k.is_empty());

    let (wav_16k, sr_16k) = decode_path(&path_16k).expect("decode 16k wav");
    assert_eq!(sr_16k, 16_000);
    assert_eq!(wav_16k.len(), 240_820);
}

#[test]
fn decode_path_downmixes_stereo_wav_instead_of_dropping_a_channel() {
    let path = std::env::temp_dir().join(format!(
        "asr-stereo-downmix-{}.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    write_pcm16_wav(
        &path,
        16_000,
        2,
        &[
            8192, 24576, // frame 0 => 0.5
            16384, -8192, // frame 1 => 0.125
        ],
    )
    .expect("write wav");

    let (samples, sample_rate) = decode_path(&path).expect("decode_path");
    let waveform = decode_path_audio(&path).expect("decode_path_audio");
    std::fs::remove_file(&path).ok();

    assert_eq!(sample_rate, 16_000);
    assert_eq!(samples.len(), 2);
    assert!((samples[0] - 0.5).abs() < 1e-6, "{samples:?}");
    assert!((samples[1] - 0.125).abs() < 1e-6, "{samples:?}");
    assert_eq!(waveform.channels, 2);
}

#[test]
fn decode_bytes_audio_decodes_encoded_audio_bytes() {
    let path = std::env::temp_dir().join(format!(
        "asr-encoded-bytes-{}.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    write_pcm16_wav(&path, 16_000, 1, &[0, 16_384, -16_384]).expect("write wav");
    let bytes = std::fs::read(&path).expect("read wav");
    std::fs::remove_file(&path).ok();

    let waveform = decode_bytes_audio(bytes).expect("decode bytes");

    assert_eq!(waveform.sample_rate, 16_000);
    assert_eq!(waveform.channels, 1);
    assert_eq!(waveform.samples.len(), 3);
}

fn write_pcm16_wav(
    path: &Path,
    sample_rate: u32,
    channels: u16,
    samples: &[i16],
) -> std::io::Result<()> {
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
