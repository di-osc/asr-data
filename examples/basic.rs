use asr_data::Audio;

fn main() {
    let mut audio =
        Audio::from_url("https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav").unwrap();
    println!("{audio}");

    let waveform = audio.as_waveform().unwrap();

    println!("{waveform}");
}
