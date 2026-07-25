//! Chinese text normalization used by ASR evaluation.
//!
//! The FST execution pipeline is adapted from wetext-rs (Apache-2.0), but is intentionally
//! limited to Chinese text normalization and embedded resources.

use std::collections::HashMap;
use std::sync::OnceLock;

use rustfst::algorithms::compose::compose;
use rustfst::algorithms::shortest_path;
use rustfst::fst_impls::VectorFst;
use rustfst::fst_traits::SerializableFst;
use rustfst::prelude::*;
use rustfst::semirings::TropicalWeight;
use rustfst::utils::{acceptor, decode_linear_fst};
use rustfst::{EPS_LABEL, Label};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TextNormalizationError {
    #[error("failed to load embedded Chinese TN resource {resource}: {message}")]
    FstLoad {
        resource: &'static str,
        message: String,
    },
    #[error("Chinese TN operation failed: {0}")]
    FstOperation(String),
    #[error("failed to parse Chinese TN token stream: {0}")]
    TokenParse(String),
}

struct FstTextNormalizer {
    fst: VectorFst<TropicalWeight>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChineseTextNormalizationOptions {
    pub traditional_to_simple: bool,
    pub full_to_half: bool,
    pub remove_erhua: bool,
    pub remove_interjections: bool,
    pub remove_puncts: bool,
}

impl Default for ChineseTextNormalizationOptions {
    fn default() -> Self {
        Self {
            traditional_to_simple: true,
            full_to_half: true,
            remove_erhua: true,
            remove_interjections: true,
            remove_puncts: true,
        }
    }
}

impl FstTextNormalizer {
    fn from_bytes(resource: &'static str, bytes: &[u8]) -> Result<Self, TextNormalizationError> {
        let fst = VectorFst::<TropicalWeight>::load(bytes).map_err(|error| {
            TextNormalizationError::FstLoad {
                resource,
                message: error.to_string(),
            }
        })?;
        Ok(Self { fst })
    }

    fn normalize(&self, input: &str) -> Result<String, TextNormalizationError> {
        if input.is_empty() {
            return Ok(String::new());
        }
        let labels = input
            .as_bytes()
            .iter()
            .map(|byte| Label::from(*byte))
            .collect::<Vec<_>>();
        let input_fst: VectorFst<TropicalWeight> = acceptor(&labels, TropicalWeight::one());
        let composed: VectorFst<TropicalWeight> = compose::<
            TropicalWeight,
            VectorFst<TropicalWeight>,
            VectorFst<TropicalWeight>,
            VectorFst<TropicalWeight>,
            _,
            _,
        >(&input_fst, &self.fst)
        .map_err(|error| TextNormalizationError::FstOperation(error.to_string()))?;
        if composed.num_states() == 0 {
            return Ok(input.to_owned());
        }
        let best_path: VectorFst<TropicalWeight> = shortest_path(&composed)
            .map_err(|error| TextNormalizationError::FstOperation(error.to_string()))?;
        if best_path.num_states() == 0 {
            return Ok(input.to_owned());
        }
        fst_output(&best_path)
    }
}

fn fst_output(fst: &VectorFst<TropicalWeight>) -> Result<String, TextNormalizationError> {
    let path = decode_linear_fst(fst)
        .map_err(|error| TextNormalizationError::FstOperation(error.to_string()))?;
    let uses_codepoints = path
        .olabels
        .iter()
        .any(|label| *label != EPS_LABEL && *label > 255);
    if uses_codepoints {
        return path
            .olabels
            .iter()
            .filter(|label| **label != EPS_LABEL)
            .map(|label| {
                char::from_u32(*label).ok_or_else(|| {
                    TextNormalizationError::FstOperation(format!(
                        "invalid Unicode code point in FST output: {label}"
                    ))
                })
            })
            .collect();
    }
    let bytes = path
        .olabels
        .iter()
        .filter(|label| **label != EPS_LABEL)
        .map(|label| *label as u8)
        .collect::<Vec<_>>();
    String::from_utf8(bytes)
        .map_err(|error| TextNormalizationError::FstOperation(error.to_string()))
}

struct ChineseTn {
    tagger: FstTextNormalizer,
    verbalizer: FstTextNormalizer,
    verbalizer_remove_erhua: FstTextNormalizer,
    traditional_to_simple: FstTextNormalizer,
    full_to_half: FstTextNormalizer,
    remove_interjections: FstTextNormalizer,
    remove_puncts: FstTextNormalizer,
}

impl ChineseTn {
    fn embedded() -> Result<Self, TextNormalizationError> {
        let tagger = FstTextNormalizer::from_bytes(
            "zh/tn/tagger.fst",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/wetext/tagger.fst"
            )),
        )?;
        let verbalizer = FstTextNormalizer::from_bytes(
            "zh/tn/verbalizer.fst",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/wetext/verbalizer.fst"
            )),
        )?;
        let verbalizer_remove_erhua = FstTextNormalizer::from_bytes(
            "zh/tn/verbalizer_remove_erhua.fst",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/wetext/verbalizer_remove_erhua.fst"
            )),
        )?;
        let remove_interjections = FstTextNormalizer::from_bytes(
            "remove_interjections.fst",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/wetext/remove_interjections.fst"
            )),
        )?;
        let traditional_to_simple = FstTextNormalizer::from_bytes(
            "traditional_to_simple.fst",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/wetext/traditional_to_simple.fst"
            )),
        )?;
        let full_to_half = FstTextNormalizer::from_bytes(
            "full_to_half.fst",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/wetext/full_to_half.fst"
            )),
        )?;
        let remove_puncts = FstTextNormalizer::from_bytes(
            "remove_puncts.fst",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/wetext/remove_puncts.fst"
            )),
        )?;
        Ok(Self {
            tagger,
            verbalizer,
            verbalizer_remove_erhua,
            traditional_to_simple,
            full_to_half,
            remove_interjections,
            remove_puncts,
        })
    }

    fn normalize(
        &self,
        text: &str,
        options: ChineseTextNormalizationOptions,
    ) -> Result<String, TextNormalizationError> {
        let text = self.preprocess(text.trim(), options)?;
        if text.is_empty() {
            return Ok(String::new());
        }
        let tagged = self.tagger.normalize(&text)?;
        let reordered = reorder_zh_tn_tokens(&tagged).unwrap_or(tagged);
        let normalized = if options.remove_erhua {
            self.verbalizer_remove_erhua.normalize(reordered.trim())?
        } else {
            self.verbalizer.normalize(reordered.trim())?
        };
        self.postprocess(&normalized, options)
    }

    fn preprocess(
        &self,
        text: &str,
        options: ChineseTextNormalizationOptions,
    ) -> Result<String, TextNormalizationError> {
        if options.traditional_to_simple {
            self.traditional_to_simple.normalize(text)
        } else {
            Ok(text.to_owned())
        }
    }

    fn postprocess(
        &self,
        text: &str,
        options: ChineseTextNormalizationOptions,
    ) -> Result<String, TextNormalizationError> {
        let mut normalized = text.to_owned();
        if options.full_to_half {
            normalized = self.full_to_half.normalize(&normalized)?;
        }
        if options.remove_interjections {
            normalized = self.remove_interjections(&normalized)?;
        }
        if options.remove_puncts {
            normalized = self.remove_puncts.normalize(&normalized)?;
        }
        Ok(normalized.trim().to_owned())
    }

    fn remove_interjections(&self, text: &str) -> Result<String, TextNormalizationError> {
        // The upstream FST currently removes `呃` and `啊`; its public option also
        // documents `嗯`, so keep that documented behavior explicit here.
        self.remove_interjections
            .normalize(text)
            .map(|text| text.replace('嗯', ""))
    }
}

/// Normalize Chinese written text to its spoken form with the embedded WeText FSTs.
pub fn normalize_zh(text: &str) -> Result<String, TextNormalizationError> {
    normalize_zh_with_options(text, ChineseTextNormalizationOptions::default())
}

pub fn normalize_zh_with_options(
    text: &str,
    options: ChineseTextNormalizationOptions,
) -> Result<String, TextNormalizationError> {
    chinese_tn()?.normalize(text, options)
}

#[cfg(test)]
fn remove_zh_interjections(text: &str) -> Result<String, TextNormalizationError> {
    chinese_tn()?.remove_interjections(text)
}

pub(crate) fn normalize_zh_without_tn(
    text: &str,
    options: ChineseTextNormalizationOptions,
) -> Result<String, TextNormalizationError> {
    let normalizer = chinese_tn()?;
    let preprocessed = normalizer.preprocess(text.trim(), options)?;
    normalizer.postprocess(&preprocessed, options)
}

fn chinese_tn() -> Result<&'static ChineseTn, TextNormalizationError> {
    static NORMALIZER: OnceLock<Result<ChineseTn, TextNormalizationError>> = OnceLock::new();
    match NORMALIZER.get_or_init(ChineseTn::embedded) {
        Ok(normalizer) => Ok(normalizer),
        Err(error) => Err(error.clone()),
    }
}

#[derive(Debug)]
struct TaggedToken {
    name: String,
    order: Vec<String>,
    members: HashMap<String, String>,
}

impl TaggedToken {
    fn render(&self) -> String {
        let preferred = preferred_order(&self.name);
        let order = preferred.as_deref().unwrap_or(&self.order);
        let mut output = format!("{} {{", self.name);
        for key in order {
            if let Some(value) = self.members.get(key) {
                output.push_str(&format!(" {key}: \"{value}\""));
            }
        }
        output.push_str(" }");
        output
    }
}

fn preferred_order(name: &str) -> Option<Vec<String>> {
    let keys: &[&str] = match name {
        "date" => &["year", "month", "day"],
        "fraction" => &["denominator", "numerator"],
        "measure" => &["denominator", "numerator", "value"],
        "money" => &["value", "currency"],
        "time" => &["noon", "hour", "minute", "second"],
        _ => return None,
    };
    Some(keys.iter().map(|key| (*key).to_owned()).collect())
}

fn reorder_zh_tn_tokens(input: &str) -> Result<String, TextNormalizationError> {
    if !input.contains('{') {
        return Ok(input.to_owned());
    }
    parse_tagged_tokens(input).map(|tokens| {
        tokens
            .iter()
            .map(TaggedToken::render)
            .collect::<Vec<_>>()
            .join(" ")
    })
}

fn parse_tagged_tokens(input: &str) -> Result<Vec<TaggedToken>, TextNormalizationError> {
    let chars = input.chars().collect::<Vec<_>>();
    let mut index = 0;
    let mut tokens = Vec::new();
    while index < chars.len() {
        skip_whitespace(&chars, &mut index);
        if index == chars.len() {
            break;
        }
        let name = parse_identifier(&chars, &mut index);
        if name.is_empty() {
            return Err(TextNormalizationError::TokenParse(format!(
                "expected token name at character {index}"
            )));
        }
        skip_whitespace(&chars, &mut index);
        expect(&chars, &mut index, '{')?;
        let mut order = Vec::new();
        let mut members = HashMap::new();
        loop {
            skip_whitespace(&chars, &mut index);
            if chars.get(index) == Some(&'}') {
                index += 1;
                break;
            }
            let key = parse_identifier(&chars, &mut index);
            if key.is_empty() {
                return Err(TextNormalizationError::TokenParse(format!(
                    "expected field name at character {index}"
                )));
            }
            skip_whitespace(&chars, &mut index);
            expect(&chars, &mut index, ':')?;
            skip_whitespace(&chars, &mut index);
            let value = parse_quoted_value(&chars, &mut index)?;
            order.push(key.clone());
            members.insert(key, value);
        }
        tokens.push(TaggedToken {
            name,
            order,
            members,
        });
    }
    Ok(tokens)
}

fn skip_whitespace(chars: &[char], index: &mut usize) {
    while chars
        .get(*index)
        .is_some_and(|character| character.is_whitespace())
    {
        *index += 1;
    }
}

fn parse_identifier(chars: &[char], index: &mut usize) -> String {
    let start = *index;
    while chars
        .get(*index)
        .is_some_and(|character| character.is_ascii_alphanumeric() || *character == '_')
    {
        *index += 1;
    }
    chars[start..*index].iter().collect()
}

fn expect(chars: &[char], index: &mut usize, expected: char) -> Result<(), TextNormalizationError> {
    if chars.get(*index) != Some(&expected) {
        return Err(TextNormalizationError::TokenParse(format!(
            "expected {expected:?} at character {index}"
        )));
    }
    *index += 1;
    Ok(())
}

fn parse_quoted_value(chars: &[char], index: &mut usize) -> Result<String, TextNormalizationError> {
    expect(chars, index, '"')?;
    let mut value = String::new();
    let mut escaped = false;
    while let Some(character) = chars.get(*index).copied() {
        *index += 1;
        if escaped {
            value.push(character);
            escaped = false;
        } else if character == '\\' {
            value.push(character);
            escaped = true;
        } else if character == '"' {
            return Ok(value);
        } else {
            value.push(character);
        }
    }
    Err(TextNormalizationError::TokenParse(
        "unterminated quoted value".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        ChineseTextNormalizationOptions, normalize_zh, normalize_zh_with_options,
        remove_zh_interjections, reorder_zh_tn_tokens,
    };

    #[test]
    fn reorders_chinese_tn_token_fields() {
        let tagged = r#"date { day: "15" year: "2024" month: "1" }"#;
        assert_eq!(
            reorder_zh_tn_tokens(tagged).unwrap(),
            r#"date { year: "2024" month: "1" day: "15" }"#
        );
    }

    #[test]
    fn leaves_plain_text_unchanged() {
        assert_eq!(reorder_zh_tn_tokens("普通文本").unwrap(), "普通文本");
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
        assert_eq!(remove_zh_interjections("嗯啊呃你好"), Ok("你好".to_owned()));
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
}
