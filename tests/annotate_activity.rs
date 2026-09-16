use asr_data::{AudioActivity, AudioChannel, AudioSource, TimeRange};

#[test]
fn annotate_activity_writes_per_channel_predictions() {
    let mut audio = AudioSource::from_pcm_s16le(vec![0_u8; 64_000], 16_000, 2)
        .load()
        .expect("load stereo pcm");

    audio
        .annotate_activity(|channel, waveform| {
            assert_eq!(waveform.channels, 1);
            assert_eq!(waveform.sample_rate, 16_000);
            Ok(vec![
                AudioActivity::new()
                    .with_event("speech")
                    .with_confidence(0.9)
                    .into_span(
                        TimeRange::new(0, 100),
                        format!("test-vad-{}", channel.name()),
                    ),
            ])
        })
        .expect("annotate each channel");

    let left = audio
        .timeline(asr_data::AudioChannel::Left)
        .expect("left channel")
        .expect("left timeline");
    assert_eq!(left.prediction.len(), 1);
    assert_eq!(left.prediction[0].source.as_deref(), Some("test-vad-left"));

    let right = audio
        .timeline(asr_data::AudioChannel::Right)
        .expect("right channel")
        .expect("right timeline");
    assert_eq!(right.prediction.len(), 1);
    assert_eq!(
        right.prediction[0].source.as_deref(),
        Some("test-vad-right")
    );
}

#[test]
fn stream_with_resamples_without_downmixing_channels() {
    // 8 kHz stereo, 250 ms → 16 kHz stereo chunks, still two channels.
    let pcm = vec![0_u8; 8_000];
    let mut stream = AudioSource::from_pcm_s16le(pcm, 8_000, 2)
        .stream_with(250, Some(16_000), None)
        .expect("create resampled stereo stream");

    assert_eq!(stream.info.sample_rate, 16_000);
    assert_eq!(stream.info.channels, 2);
    assert!(stream.timeline(AudioChannel::Left).unwrap().is_some());
    assert!(stream.timeline(AudioChannel::Right).unwrap().is_some());
    assert!(stream.timeline(AudioChannel::Mono).unwrap().is_none());

    let chunks = stream
        .by_ref()
        .collect::<Result<Vec<_>, _>>()
        .expect("chunks");
    assert!(!chunks.is_empty());
    assert!(chunks.iter().all(|chunk| chunk.sample_rate == 16_000));
    assert!(chunks.iter().all(|chunk| chunk.channels == 2));
    assert!(stream.is_complete());
    assert_eq!(stream.as_waveform().sample_rate, 16_000);
    assert_eq!(stream.as_waveform().channels, 2);
}

#[test]
fn stream_annotate_activity_writes_per_chunk_predictions() {
    let pcm = vec![0_u8; 64_000];
    let mut stream = AudioSource::from_pcm_s16le(pcm, 16_000, 2)
        .stream_with(250, None, None)
        .expect("create stereo stream");

    let mut calls = Vec::new();
    stream
        .annotate_activity(|channel, waveform, is_final| {
            assert_eq!(waveform.channels, 1);
            assert_eq!(waveform.sample_rate, 16_000);
            calls.push((channel, waveform.frame_count(), is_final));
            if is_final {
                Ok(vec![
                    AudioActivity::new()
                        .with_event("speech")
                        .with_confidence(0.8)
                        .into_span(
                            TimeRange::new(0, 100),
                            format!("stream-vad-{}", channel.name()),
                        ),
                ])
            } else {
                Ok(Vec::new())
            }
        })
        .expect("annotate stream");

    assert!(
        calls
            .iter()
            .any(|(channel, _, is_final)| { *channel == AudioChannel::Left && *is_final })
    );
    assert!(
        calls
            .iter()
            .any(|(channel, _, is_final)| { *channel == AudioChannel::Right && *is_final })
    );

    let left = stream
        .timeline(AudioChannel::Left)
        .expect("left channel")
        .expect("left timeline");
    assert_eq!(left.prediction.len(), 1);
    assert_eq!(
        left.prediction[0].source.as_deref(),
        Some("stream-vad-left")
    );
}

#[test]
fn stream_annotate_activity_chunk_keeps_intermediate_predictions() {
    let pcm = vec![0_u8; 16_000];
    let mut stream = AudioSource::from_pcm_s16le(pcm, 16_000, 1)
        .stream(250)
        .expect("create mono stream");

    let mut seen_prediction_counts = Vec::new();
    while let Some(chunk) = stream.next() {
        let chunk = chunk.expect("chunk");
        let start = chunk.offset_ms as usize;
        let end = (chunk.end_ms() as usize).max(start.saturating_add(1));
        stream
            .annotate_activity_chunk(&chunk, |_channel, _waveform, _is_final| {
                Ok(vec![
                    AudioActivity::new()
                        .with_event("speech")
                        .with_confidence(0.7)
                        .into_span(TimeRange::new(start, end), "chunk-vad"),
                ])
            })
            .expect("annotate one chunk");
        let timeline = stream
            .timeline(AudioChannel::Mono)
            .expect("mono channel")
            .expect("mono timeline");
        seen_prediction_counts.push(timeline.prediction.len());
    }

    assert!(seen_prediction_counts.len() >= 2);
    assert!(
        seen_prediction_counts
            .windows(2)
            .any(|pair| pair[1] > pair[0])
    );
}
