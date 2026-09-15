from asr_data import Audio
from asr_data.annotation import AudioActivity, Speaker, Token, Transcription


def main() -> None:

    TEXT = "甚至出现交易几乎停滞的情况。"

    audio = Audio.from_url("https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav")
    timeline = audio.timeline()
    # 示例 wav 就是这句转写，span 覆盖整段真实时长。
    duration_ms = timeline.duration_ms
    chars = list(TEXT)
    token_count = len(chars)
    tokens = [
        Token(
            text=char,
            start_ms=duration_ms * index // token_count,
            end_ms=duration_ms * (index + 1) // token_count,
        )
        for index, char in enumerate(chars)
    ]

    timeline.annotate_span(
        start_ms=0,
        end_ms=duration_ms,
        annotation=AudioActivity(event="speech", confidence=0.9),
    )
    timeline.annotate_span(
        start_ms=0,
        end_ms=duration_ms,
        annotation=Speaker(
            name="female0",
            confidence=0.9,
            transcription=Transcription(text=TEXT, tokens=tokens),
        ),
    )

    print(timeline)


if __name__ == "__main__":
    main()
