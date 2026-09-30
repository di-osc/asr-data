//! 时间轴上的转写分段。

use serde::{Deserialize, Serialize};

use super::annotation::Token;
use crate::utils::TimeRange;

/// 一句转写文本。时间是时间轴上的绝对毫秒，并且必须落在所属语音区间内。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sentence {
    /// 整句文本。
    pub text: String,
    /// 这句覆盖的绝对时间范围。
    pub range: TimeRange,
    /// 可选的词级切分。空列表表示这句没有 token。
    #[serde(default)]
    pub tokens: Vec<Token>,
}

impl Sentence {
    /// 用文本和绝对时间构造句子，token 列表为空。
    pub fn new(text: impl Into<String>, start_ms: usize, end_ms: usize) -> Self {
        Self {
            text: text.into(),
            range: TimeRange::new(start_ms, end_ms),
            tokens: Vec::new(),
        }
    }

    /// 附上词级 token。
    pub fn with_tokens(mut self, tokens: Vec<Token>) -> Self {
        self.tokens = tokens;
        self
    }
}

/// 从若干语音段抽出的转写视图。
///
/// 这是读取结果，不是可以写回时间轴的标注。语种来自贡献句子的第一段语音。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    /// 按语音时间拼接的各段转写全文。
    pub text: String,
    /// 第一段带转写且标了语种的语音的语种。
    pub language: Option<String>,
    /// 句级分段，已按起点排序。
    #[serde(default)]
    pub segments: Vec<Sentence>,
}
