use asr_data::{Audio, AudioActivity, DurationMs, Speaker, TimeRange, Token, Transcription};

const TEXT: &str = "甚至出现交易几乎停滞的情况。";

fn main() -> anyhow::Result<()> {
    let mut audio = Audio::from_url("https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav")?;

    // 示例 wav 为单声道，加载时已按探测结果创建 timeline。
    let timeline = audio
        .mono_timeline_mut()
        .expect("example.wav is mono and has a timeline");
    // 示例 wav 就是这句转写，span 覆盖整段真实时长。
    let duration_ms = timeline.duration.0;
    let chars: Vec<char> = TEXT.chars().collect();
    let token_count = chars.len() as u64;
    let tokens: Vec<Token> = chars
        .into_iter()
        .enumerate()
        .map(|(index, character)| {
            let index = index as u64;
            Token::new(character.to_string()).with_range(TimeRange::new(
                DurationMs(duration_ms * index / token_count),
                DurationMs(duration_ms * (index + 1) / token_count),
            ))
        })
        .collect();

    timeline.annotate_span(
        0,
        duration_ms,
        AudioActivity::new()
            .with_event("speech")
            .with_confidence(0.9),
    )?;
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
