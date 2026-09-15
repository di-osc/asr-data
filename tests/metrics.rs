use asr_data::{
    CerStats, ChineseTextNormalizationOptions, compute_cer, normalize_for_cer, normalize_zh,
    normalize_zh_with_options,
};

#[test]
fn cer_formula_matches_edit_counts() {
    let stats = compute_cer("kitten", "sitting");
    assert_eq!(
        stats,
        CerStats {
            substitutions: 2,
            deletions: 0,
            insertions: 1,
            reference_chars: 6,
        }
    );
    assert!((stats.cer() - 0.5).abs() < f64::EPSILON);
}

#[test]
fn cer_handles_chinese_characters() {
    let stats = compute_cer("你好世界", "你好世");
    assert_eq!(
        stats,
        CerStats {
            substitutions: 0,
            deletions: 1,
            insertions: 0,
            reference_chars: 4,
        }
    );
    assert!((stats.cer() - 0.25).abs() < f64::EPSILON);
}

#[test]
fn normalize_for_cer_strips_whitespace_by_default() {
    let normalized = normalize_for_cer("你 好\n世界", true);
    assert_eq!(normalized, "你好世界");
}

#[test]
fn normalize_for_cer_strips_punctuation() {
    let normalized = normalize_for_cer("你好，世界！How are you?", true);
    assert_eq!(normalized, "你好世界Howareyou");
}

#[test]
fn normalizes_chinese_numbers_from_embedded_fsts() {
    assert_eq!(normalize_zh("2024年"), Ok("二零二四年".to_owned()));
}

#[test]
fn removes_erhua_by_default() {
    assert_eq!(normalize_zh("花儿"), Ok("花".to_owned()));
    assert_eq!(
        normalize_zh_with_options(
            "花儿",
            ChineseTextNormalizationOptions {
                remove_erhua: false,
                ..ChineseTextNormalizationOptions::default()
            },
        ),
        Ok("花儿".to_owned()),
    );
}

#[test]
fn removes_interjections_by_default() {
    let options = ChineseTextNormalizationOptions {
        remove_interjections: false,
        ..ChineseTextNormalizationOptions::default()
    };
    assert_eq!(
        normalize_zh_with_options("嗯啊呃你好", options),
        Ok("嗯啊呃你好".to_owned()),
    );
    assert_eq!(normalize_zh("嗯啊呃你好"), Ok("你好".to_owned()));
}

#[test]
fn applies_optional_pre_and_postprocessors() {
    let options = ChineseTextNormalizationOptions {
        traditional_to_simple: true,
        full_to_half: true,
        remove_puncts: true,
        ..ChineseTextNormalizationOptions::default()
    };
    assert_eq!(
        normalize_zh_with_options("這是ＡＢＣ！", options),
        Ok("这是ABC".to_owned()),
    );
}
