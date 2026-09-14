use asr_data::AudioDataset;

fn main() {
    let dataset = AudioDataset::from_modelscope("WMD1992/fangchan-call-user", None, None).unwrap();
    println!("{dataset:?}");
}
