from asr_data import Audio
from asr_data.annotation import Sentence, Speaker, Speech, Token, Transcription


def main() -> None:
    """加载示例音频，并把整段标成一条带说话人和转写的语音。"""
    audio = Audio.from_url("https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav")
    timeline = audio.timeline()
    if timeline is None:
        raise RuntimeError("example.wav is mono and has a timeline")
    duration_ms = timeline.duration_ms
    text = "甚至出现交易几乎停滞的情况。"
    chars = list(text)
    token_count = max(len(chars), 1)
    # 每个字在整段语音里各占一段绝对时间。
    tokens = [
        Token(
            char,
            start_ms=duration_ms * index // token_count,
            end_ms=duration_ms * (index + 1) // token_count,
        )
        for index, char in enumerate(chars)
    ]
    # 一段 speech 同时带上活动置信度、说话人和转写。
    timeline.annotate(
        0,
        duration_ms,
        Speech(
            confidence=0.9,
            speaker=Speaker("female0", gender="female"),
            transcription=Transcription(
                text,
                sentences=[Sentence(text, 0, duration_ms, tokens=tokens)],
            ),
        ),
    )
    # 打印整个 audio，显示波形 + 标注的终端卡片，与 Rust `println!("{audio}")` 完全一致。
    print(audio)


if __name__ == "__main__":
    main()
