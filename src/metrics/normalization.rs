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
        Ok(Self { tagger, verbalizer })
    }

    fn normalize(&self, text: &str) -> Result<String, TextNormalizationError> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(String::new());
        }
        let tagged = self.tagger.normalize(text)?;
        let reordered = reorder_zh_tn_tokens(&tagged).unwrap_or(tagged);
        self.verbalizer.normalize(reordered.trim())
    }
}

/// Normalize Chinese written text to its spoken form with the embedded WeText FSTs.
pub fn normalize_zh_tn(text: &str) -> Result<String, TextNormalizationError> {
    static NORMALIZER: OnceLock<Result<ChineseTn, TextNormalizationError>> = OnceLock::new();
    match NORMALIZER.get_or_init(ChineseTn::embedded) {
        Ok(normalizer) => normalizer.normalize(text),
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
    use super::{normalize_zh_tn, reorder_zh_tn_tokens};

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
        assert_eq!(normalize_zh_tn("2024年"), Ok("二零二四年".to_owned()));
    }
}
