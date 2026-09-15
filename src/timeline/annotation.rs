//! 时间轴标注 payload：活动、token、转写、说话人和语种。

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::segment::Sentence;
use crate::utils::TimeRange;

/// 音频文档 ID。
pub type AudioId = String;
/// 时间轴 ID，通常与音频 ID 相同。
pub type TimelineId = String;
/// 单个时间段标注的 ID。
pub type TimeSpanId = String;
/// 说话人名称或标识。
pub type SpeakerId = String;
/// BCP-47 语种标签。
pub type LanguageTag = String;

/// 音频活动（VAD / 事件检测）标注。
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AudioActivity {
    /// 可选事件名，例如 `speech`；缺省只表示存在活动。
    #[serde(default)]
    pub event: Option<String>,
    /// 可选置信度。
    #[serde(default)]
    pub confidence: Option<f32>,
}

impl AudioActivity {
    /// 构造没有事件名和置信度的活动标注。
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置事件名。
    pub fn with_event(mut self, event: impl Into<String>) -> Self {
        self.event = Some(event.into());
        self
    }

    /// 设置检测置信度。
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

/// 词级转写 token，可带局部时间和置信度。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    /// token 文本。
    pub text: String,
    /// 相对于父 span 或全局时间轴的时间范围。
    pub range: Option<TimeRange>,
    /// 可选置信度。
    pub confidence: Option<f32>,
}

impl Token {
    /// 用纯文本构造 token。
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            range: None,
            confidence: None,
        }
    }

    /// 附上时间范围。
    pub fn with_range(mut self, range: TimeRange) -> Self {
        self.range = Some(range);
        self
    }

    /// 附上置信度。
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

/// 一段转写文本，可带 token、语种和置信度。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcription {
    /// 转写全文。
    pub text: String,
    /// 词级切分。
    #[serde(default)]
    pub tokens: Vec<Token>,
    /// 语种标签。
    #[serde(default)]
    pub language: Option<String>,
    /// 整段置信度。
    #[serde(default)]
    pub confidence: Option<f32>,
}

impl Transcription {
    /// 用纯文本构造转写。
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens: Vec::new(),
            language: None,
            confidence: None,
        }
    }

    /// 附上词级 token。
    pub fn with_tokens(mut self, tokens: Vec<Token>) -> Self {
        self.tokens = tokens;
        self
    }

    /// 附上语种标签。
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// 附上整段置信度。
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

/// 说话人标注，可附带该段转写。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerPayload {
    /// 说话人标识。
    pub name: SpeakerId,
    /// 该说话人在此时间段内的转写。
    #[serde(default)]
    pub transcription: Option<Transcription>,
    /// 说话人分配置信度。
    #[serde(default)]
    pub confidence: Option<f32>,
}

impl SpeakerPayload {
    /// 用说话人名称构造 payload。
    pub fn new(name: impl Into<SpeakerId>) -> Self {
        Self {
            name: name.into(),
            transcription: None,
            confidence: None,
        }
    }

    /// 附上置信度。
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// 附上该说话人在此时间段内的转写。
    pub fn with_transcription(mut self, transcription: Transcription) -> Self {
        self.transcription = Some(transcription);
        self
    }
}

/// 可放进 [`TimeSpan`] 的标注内容。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Annotation {
    /// 活动 / 事件检测。
    Activity(AudioActivity),
    /// 词级 token。
    Token(Token),
    /// 转写文本。
    Transcription(Transcription),
    /// 句级分段。
    Sentence(Sentence),
    /// 说话人。
    Speaker(SpeakerPayload),
    /// 语种。
    Language(LanguageTag),
}

impl From<AudioActivity> for Annotation {
    fn from(value: AudioActivity) -> Self {
        Self::Activity(value)
    }
}

impl From<Token> for Annotation {
    fn from(value: Token) -> Self {
        Self::Token(value)
    }
}

impl From<Transcription> for Annotation {
    fn from(value: Transcription) -> Self {
        Self::Transcription(value)
    }
}

impl From<Sentence> for Annotation {
    fn from(value: Sentence) -> Self {
        Self::Sentence(value)
    }
}

impl From<SpeakerPayload> for Annotation {
    fn from(value: SpeakerPayload) -> Self {
        Self::Speaker(value)
    }
}

impl Annotation {
    /// 用于分组和评估的稳定类别名。
    pub(crate) fn source_group(&self) -> &'static str {
        match self {
            Self::Activity(_) => "activity",
            Self::Token(_) => "token",
            Self::Transcription(_) => "transcription",
            Self::Sentence(_) => "sentence",
            Self::Speaker(_) => "speaker",
            Self::Language(_) => "language",
        }
    }

    /// 取出 payload 上的置信度；句子和语种没有该字段。
    pub fn confidence(&self) -> Option<f32> {
        match self {
            Self::Activity(value) => value.confidence,
            Self::Token(value) => value.confidence,
            Self::Transcription(value) => value.confidence,
            Self::Speaker(value) => value.confidence,
            Self::Sentence(_) | Self::Language(_) => None,
        }
    }
}

/// 时间轴上的一段标注：区间、来源和 payload。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeSpan {
    /// 自动生成的 span ID。
    pub id: TimeSpanId,
    /// 覆盖的时间范围。
    pub range: TimeRange,
    /// 预测来源；参考标注必须为 `None`。
    pub source: Option<String>,
    /// 标注内容。
    pub annotation: Annotation,
}

impl TimeSpan {
    /// 构造 span 并生成随机 ID。
    pub fn new(range: TimeRange, annotation: Annotation, source: Option<String>) -> Self {
        Self {
            id: format!("span_{}", Uuid::new_v4().simple()),
            range,
            source,
            annotation,
        }
    }

    /// 比较标注内容，忽略自动生成的 ID。
    pub fn content_eq(&self, other: &Self) -> bool {
        self.range == other.range
            && self.source == other.source
            && self.annotation == other.annotation
    }
}
