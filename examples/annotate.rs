use asr_data::{Audio, Gender, Sentence, Speaker, Speech, Token, Transcription};

fn main() -> anyhow::Result<()> {
    let mut audio = Audio::from_url("https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav")?;
    // 标注需要 timeline 的可变借用，写完后立刻结束借用，才能整体打印 audio 的波形和标注。
    {
        let timeline = audio
            .mono_timeline_mut()
            .expect("example.wav is mono and has a timeline");
        let duration_ms = timeline.duration_ms();
        const TEXT: &str = "甚至出现交易几乎停滞的情况。";
        let chars: Vec<char> = TEXT.chars().collect();
        let token_count = chars.len().max(1);
        // 每个字在整段语音里各占一段绝对时间。
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
        // 整句覆盖整段语音，token 挂在这一句下面。
        let sentences = vec![Sentence::new(TEXT, 0, duration_ms).with_tokens(tokens)];
        // 一段 speech 同时带上活动置信度、说话人和转写。
        timeline.annotate(
            0,
            duration_ms,
            Speech::new()
                .with_confidence(0.9)
                .with_speaker(Speaker::new("female0").with_gender(Gender::Female))
                .with_transcription(Transcription::new(TEXT).with_sentences(sentences)),
        )?;
    }
    // 打印整个 audio，显示波形 + 标注的终端卡片，与 Python `print(audio)` 完全一致。
    println!("{audio}");
    Ok(())
}
