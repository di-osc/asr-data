from asr_data import Audio


def main() -> None:
    """加载示例音频，打印音频卡片和波形卡片。"""
    url = "https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav"
    audio = Audio.from_url(url)
    print(audio)

    waveform = audio.as_waveform()
    print(waveform)


if __name__ == "__main__":
    main()
