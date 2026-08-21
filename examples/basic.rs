use asr_data::Audio;

fn main() {
    let audio =
        Audio::from_url("https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav").unwrap();
    println!("{audio}");
}
