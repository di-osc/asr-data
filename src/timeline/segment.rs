//! 时间轴上的转写分段。

use serde::{Deserialize, Serialize};

use super::annotation::Token;

/// 一句转写文本，可附带 token 和语种。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sentence {
    /// 整句文本。
    pub text: String,
    /// 可选的词级切分。
    #[serde(default)]
    pub tokens: Vec<Token>,
    /// BCP-47 语种标签。
    pub language: Option<String>,
}

impl Sentence {
    /// 用纯文本构造句子，token 列表为空。
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens: Vec::new(),
            language: None,
        }
    }
}

/// 一整段转写，由若干 [`Sentence`] 组成。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    /// 拼接后的全文。
    pub text: String,
    /// 整段语种；各句可以另有自己的 language。
    pub language: Option<String>,
    /// 句级分段。
    #[serde(default)]
    pub segments: Vec<Sentence>,
}
