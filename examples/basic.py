from pathlib import Path

from asr_data import Audio


def main() -> None:

    example_path = Path(__file__).resolve().parents[1] / "assets" / "example.wav"
    audio = Audio.from_path(example_path)

    url = "https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav"
    audio = Audio.from_url(url)

    waveform = audio.as_waveform()
    print(waveform)


if __name__ == "__main__":
    main()
