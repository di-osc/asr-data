use asr_data::{Audio, AudioActivity, Speaker, Token, Transcription};

fn main() -> anyhow::Result<()> {
    let mut audio = Audio::from_url("https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav")?;
    let timeline = audio
        .mono_timeline_mut()
        .expect("example.wav is mono and has a timeline");
    // 标注VAD模型结果
    let duration_ms = timeline.duration_ms();
    timeline.annotate_span(
        0,
        duration_ms,
        AudioActivity::new()
            .with_event("speech")
            .with_confidence(0.9),
    )?;
    // 标注语音识别结果
    const TEXT: &str = "甚至出现交易几乎停滞的情况。";
    let chars: Vec<char> = TEXT.chars().collect();
    let token_count = chars.len();
    let tokens: Vec<Token> = chars
        .into_iter()
        .enumerate()
        .map(|(index, character)| {
            Token::new(character.to_string()).with_range(
                duration_ms * index / token_count,
                duration_ms * (index + 1) / token_count,
            )
        })
        .collect();
    timeline.annotate_span(
        0,
        duration_ms,
        Speaker::new("female0")
            .with_confidence(0.9)
            .with_transcription(Transcription::new(TEXT).with_tokens(tokens)),
    )?;

    println!("{timeline}");
    Ok(())
}
