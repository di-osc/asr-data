//! 时间轴标注：语音树，以及其他音频事件。

use serde::{Deserialize, Serialize};

use super::segment::Sentence;
use crate::utils::TimeRange;

/// 音频文档 ID。
pub type AudioId = String;
/// 时间轴 ID，通常与音频 ID 相同。
pub type TimelineId = String;
/// 单个事件的 ID。
pub type EventId = String;
/// 说话人名称或标识。
pub type SpeakerId = String;
/// BCP-47 语种标签。
pub type LanguageTag = String;

/// 说话人性别。
///
/// 未标注时用 [`Option`] 的 `None`，不要用 [`Gender::Unknown`] 代替。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gender {
    /// 男性。
    Male,
    /// 女性。
    Female,
    /// 已标注，但性别未知。
    Unknown,
}

impl Gender {
    /// 解析 `male` / `female` / `unknown`。
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "male" => Some(Self::Male),
            "female" => Some(Self::Female),
            "unknown" => Some(Self::Unknown),
            _ => None,
        }
    }

    /// 稳定的小写标签，与 serde 和 Python 绑定一致。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Male => "male",
            Self::Female => "female",
            Self::Unknown => "unknown",
        }
    }
}

/// 一段语音上的说话人。
///
/// 一段 [`Speech`] 最多挂一个说话人。两人同时说话时写成两段重叠的语音。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Speaker {
    /// 说话人标识。
    pub name: SpeakerId,
    /// 性别；`None` 表示还没标。
    #[serde(default)]
    pub gender: Option<Gender>,
}

impl Speaker {
    /// 用说话人名称构造标注，性别留空。
    pub fn new(name: impl Into<SpeakerId>) -> Self {
        Self {
            name: name.into(),
            gender: None,
        }
    }

    /// 附上性别。
    pub fn with_gender(mut self, gender: Gender) -> Self {
        self.gender = Some(gender);
        self
    }
}

/// 词级转写 token，时间是时间轴上的绝对毫秒。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    /// token 文本。
    pub text: String,
    /// 可选的绝对时间范围，必须落在所属句子内。
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

    /// 附上绝对时间范围，参数与 Python `Token(..., start_ms, end_ms)` 对齐。
    pub fn with_range(mut self, start_ms: usize, end_ms: usize) -> Self {
        self.range = Some(TimeRange::new(start_ms, end_ms));
        self
    }

    /// 附上置信度。
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

/// 一段语音里的转写。
///
/// 全文 [`Transcription::text`] 是最小内容，单独存在即可。句级 [`Sentence`]
/// 和句内 token 都是可选切分，列表可以为空。评测和 transcript 使用这里的全文，
/// 不从句子再拼一次。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcription {
    /// 整段转写文本。可以是空字符串，表示这段假设被整段删除。
    pub text: String,
    /// 句级切分。空列表表示没有句子；非空时句子时间不能重叠。
    #[serde(default)]
    pub sentences: Vec<Sentence>,
    /// 整段转写置信度。
    #[serde(default)]
    pub confidence: Option<f32>,
}

impl Transcription {
    /// 用整段文本构造转写。句子和置信度留空。
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            sentences: Vec::new(),
            confidence: None,
        }
    }

    /// 附上句级切分。传入空列表等于不切分。
    pub fn with_sentences(mut self, sentences: Vec<Sentence>) -> Self {
        self.sentences = sentences;
        self
    }

    /// 附上整段置信度。
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

/// 按给定顺序拼接非空文本，中间用一个空格隔开。
pub(crate) fn join_transcript_text<'a>(parts: impl IntoIterator<Item = &'a str>) -> String {
    parts
        .into_iter()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// 语音事件固定使用的名字。
pub const SPEECH_EVENT_NAME: &str = "speech";

/// 从 [`AudioEvent`] 派生的语音。
///
/// 名字固定为 [`SPEECH_EVENT_NAME`]。语种、说话人和转写都可以空着，
/// 所以只有活动区间的 VAD 结果也是一条合法语音。`id`、`range` 和 `source`
/// 由时间轴在写入时覆盖。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Speech {
    /// 写入时生成的事件 ID。
    pub id: EventId,
    /// 这段语音在时间轴上的绝对时间。
    pub range: TimeRange,
    /// 预测来源。参考事件必须为 `None`。
    #[serde(default)]
    pub source: Option<String>,
    /// 可选活动检测置信度。
    #[serde(default)]
    pub confidence: Option<f32>,
    /// BCP-47 语种标签。语种只挂在语音上。
    #[serde(default)]
    pub language: Option<LanguageTag>,
    /// 这段语音的唯一说话人。
    #[serde(default)]
    pub speaker: Option<Speaker>,
    /// 这段语音的转写。
    #[serde(default)]
    pub transcription: Option<Transcription>,
}

impl Speech {
    /// 构造一段还没有身份的语音。语种、说话人和转写都留空。
    pub fn new() -> Self {
        Self {
            id: String::new(),
            range: TimeRange::default(),
            source: None,
            confidence: None,
            language: None,
            speaker: None,
            transcription: None,
        }
    }

    /// 附上活动检测置信度。
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// 附上语种。
    pub fn with_language(mut self, language: impl Into<LanguageTag>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// 附上说话人。一段语音只有一个。
    pub fn with_speaker(mut self, speaker: Speaker) -> Self {
        self.speaker = Some(speaker);
        self
    }

    /// 附上转写。
    pub fn with_transcription(mut self, transcription: Transcription) -> Self {
        self.transcription = Some(transcription);
        self
    }
}

impl Default for Speech {
    fn default() -> Self {
        Self::new()
    }
}

/// 时间轴上的一条音频事件。
///
/// [`AudioEvent::Base`] 是基类，用来表示音乐、噪声、静音以及其他非语音事件。
/// [`Speech`] 是派生类型，包在 [`AudioEvent::Speech`] 里。写入 [`crate::timeline::Timeline::annotate`]
/// 时两种值都能直接传入。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AudioEvent {
    /// 基类事件。`name` 不能是 `speech`。
    Base {
        /// 写入时生成的事件 ID。
        id: EventId,
        /// 事件在时间轴上的绝对时间。
        range: TimeRange,
        /// 预测来源。参考事件必须为 `None`。
        #[serde(default)]
        source: Option<String>,
        /// 事件名，例如 `music`、`noise`、`silence`。
        name: String,
        /// 可选检测置信度。
        #[serde(default)]
        confidence: Option<f32>,
    },
    /// 派生的语音事件。
    Speech(Speech),
}

impl From<Speech> for AudioEvent {
    fn from(speech: Speech) -> Self {
        Self::Speech(speech)
    }
}

impl AudioEvent {
    /// 构造一个还没有身份的基类事件。
    ///
    /// `id`、`range` 和 `source` 由 [`crate::timeline::Timeline::annotate`] 填入。
    pub fn new(name: impl Into<String>) -> Self {
        Self::Base {
            id: String::new(),
            range: TimeRange::default(),
            source: None,
            name: name.into(),
            confidence: None,
        }
    }

    /// 附上检测置信度。语音请改用 [`Speech::with_confidence`]。
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        match &mut self {
            Self::Base {
                confidence: slot, ..
            } => *slot = Some(confidence),
            Self::Speech(speech) => speech.confidence = Some(confidence),
        }
        self
    }

    /// 事件 ID。尚未写入时间轴时为空字符串。
    pub fn id(&self) -> &str {
        match self {
            Self::Base { id, .. } => id,
            Self::Speech(speech) => speech.id.as_str(),
        }
    }

    /// 事件覆盖的绝对时间。
    pub fn range(&self) -> TimeRange {
        match self {
            Self::Base { range, .. } => *range,
            Self::Speech(speech) => speech.range,
        }
    }

    /// 预测来源。参考事件返回 `None`。
    pub fn source(&self) -> Option<&str> {
        match self {
            Self::Base { source, .. } => source.as_deref(),
            Self::Speech(speech) => speech.source.as_deref(),
        }
    }

    /// 事件名。语音事件固定返回 [`SPEECH_EVENT_NAME`]。
    pub fn name(&self) -> &str {
        match self {
            Self::Base { name, .. } => name.as_str(),
            Self::Speech(_) => SPEECH_EVENT_NAME,
        }
    }

    /// 这段事件自身的检测置信度。
    pub fn confidence(&self) -> Option<f32> {
        match self {
            Self::Base { confidence, .. } => *confidence,
            Self::Speech(speech) => speech.confidence,
        }
    }

    /// 语音事件的语种。基类事件返回 `None`。
    pub fn language(&self) -> Option<&str> {
        match self {
            Self::Speech(speech) => speech.language.as_deref(),
            Self::Base { .. } => None,
        }
    }

    /// 语音事件的说话人。基类事件返回 `None`。
    pub fn speaker(&self) -> Option<&Speaker> {
        match self {
            Self::Speech(speech) => speech.speaker.as_ref(),
            Self::Base { .. } => None,
        }
    }

    /// 语音事件的转写。基类事件返回 `None`。
    pub fn transcription(&self) -> Option<&Transcription> {
        match self {
            Self::Speech(speech) => speech.transcription.as_ref(),
            Self::Base { .. } => None,
        }
    }

    /// 是否为派生的 [`Speech`]。
    pub fn is_speech(&self) -> bool {
        matches!(self, Self::Speech(_))
    }

    /// 用于分组预测来源的稳定类别名。
    pub(crate) fn source_group(&self) -> &'static str {
        if self.is_speech() { "speech" } else { "event" }
    }

    /// 比较事件内容，忽略自动生成的 ID。
    pub fn content_eq(&self, other: &Self) -> bool {
        self.range() == other.range()
            && self.source() == other.source()
            && self.name() == other.name()
            && self.confidence() == other.confidence()
            && self.language() == other.language()
            && self.speaker() == other.speaker()
            && self.transcription() == other.transcription()
    }

    /// 写入时间轴时填上身份。调用方传入的内容里这些字段会被覆盖。
    pub(crate) fn assign_identity(
        &mut self,
        id: EventId,
        range: TimeRange,
        source: Option<String>,
    ) {
        match self {
            Self::Base {
                id: slot_id,
                range: slot_range,
                source: slot_source,
                ..
            } => {
                *slot_id = id;
                *slot_range = range;
                *slot_source = source;
            }
            Self::Speech(speech) => {
                speech.id = id;
                speech.range = range;
                speech.source = source;
            }
        }
    }

    /// 改写预测来源，不改动其余字段。
    pub(crate) fn set_source(&mut self, source: Option<String>) {
        match self {
            Self::Base {
                source: slot_source,
                ..
            } => *slot_source = source,
            Self::Speech(speech) => speech.source = source,
        }
    }
}
