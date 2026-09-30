//! 时间轴容器：参考/预测 span，以及重叠校验。

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use uuid::Uuid;

use super::annotation::{AudioEvent, AudioId, EventId, TimelineId, join_transcript_text};
use super::segment::{Sentence, Transcript};
use crate::utils::TimeRange;

/// 不允许重叠的标注类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventConflictKind {
    /// 相同事件名的非语音区间重叠。
    Event,
    /// 语音区间重叠：同一说话人，或至少一段没有说话人。
    Speech,
}

/// 一条声道上的参考与预测标注集合。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Timeline {
    /// 时间轴 ID。
    pub id: TimelineId,
    /// 所属音频文档 ID。
    pub audio_id: AudioId,
    /// 覆盖的音频时长，单位毫秒。
    pub duration: usize,
    /// 人工 / 参考事件，`source` 必须为空。
    pub reference: Vec<AudioEvent>,
    /// 模型预测，必须带非空 `source`。
    pub prediction: Vec<AudioEvent>,
}

/// 终端打印时使用 [`Timeline::terminal_view`] 的卡片、时间刻度和标注树。
impl fmt::Display for Timeline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.terminal_view().fmt(formatter)
    }
}

/// 一次非法重叠的详细信息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventOverlap {
    pub kind: EventConflictKind,
    pub source: Option<String>,
    pub speaker: Option<String>,
    pub first_id: EventId,
    pub first_range: TimeRange,
    pub second_id: EventId,
    pub second_range: TimeRange,
}

impl fmt::Display for EventOverlap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?} annotation {:?} at {:?} overlaps annotation {:?} at {:?} (source={:?}, speaker={:?})",
            self.kind,
            self.second_id,
            self.second_range,
            self.first_id,
            self.first_range,
            self.source,
            self.speaker,
        )
    }
}

/// 向时间轴写入或校验 span 时的错误。
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum TimelineEventError {
    #[error("reference event {event_id:?} must not have a source")]
    ReferenceHasSource { event_id: EventId },
    #[error("prediction event {event_id:?} must have a non-empty source")]
    PredictionMissingSource { event_id: EventId },
    #[error("prediction source must contain at least one non-whitespace character")]
    InvalidPredictionSource,
    #[error("event name must contain at least one non-whitespace character")]
    InvalidEventName,
    #[error("an event named \"speech\" must be stored as a speech annotation")]
    SpeechEventName,
    #[error("speaker name must contain at least one non-whitespace character")]
    InvalidSpeakerName,
    #[error("sentence range {range:?} must end after it starts")]
    InvalidSentenceRange { range: TimeRange },
    #[error("token range {range:?} must end after it starts")]
    InvalidTokenRange { range: TimeRange },
    #[error("sentence range {sentence:?} is outside speech span {speech:?}")]
    SentenceOutsideSpeech {
        sentence: TimeRange,
        speech: TimeRange,
    },
    #[error("token range {token:?} is outside sentence {sentence:?}")]
    TokenOutsideSentence {
        token: TimeRange,
        sentence: TimeRange,
    },
    #[error("sentence range {first:?} overlaps sentence range {second:?}")]
    OverlappingSentences { first: TimeRange, second: TimeRange },
    #[error("{0}")]
    Overlap(Box<EventOverlap>),
}

impl Timeline {
    /// 渲染紧凑的终端摘要：时长卡片和 Reference / Prediction 标注轨道。
    ///
    /// 宽度读取 `COLUMNS`（夹在 56–120），颜色在 TTY 且未设置 `NO_COLOR` 时开启。
    pub fn terminal_view(&self) -> impl fmt::Display + '_ {
        crate::doc::TimelineTerminalView::auto(self)
    }

    /// 渲染带或不带 ANSI 颜色的终端摘要，宽度仍随终端。
    pub fn terminal_view_with_color(&self, color: bool) -> impl fmt::Display + '_ {
        crate::doc::TimelineTerminalView::with_color(self, color)
    }

    /// 构造空时间轴，生成随机 ID。
    pub fn new(audio_id: impl Into<AudioId>, duration: usize) -> Self {
        Self {
            id: format!("tl_{}", Uuid::new_v4().simple()),
            audio_id: audio_id.into(),
            duration,
            reference: Vec::new(),
            prediction: Vec::new(),
        }
    }

    /// 时间轴覆盖时长，单位毫秒。
    ///
    /// 与 Python `Timeline.duration_ms` 对齐，便于和 `usize` 下标一起做分段计算。
    pub fn duration_ms(&self) -> usize {
        self.duration
    }

    /// 写入一条事件，默认作为 reference。
    ///
    /// 时间轴会填上 `id` 和 `range`，并清掉 `source`。内容相同的事件会去重。
    ///
    /// # Errors
    ///
    /// 事件内容不合法，或与已有事件非法重叠时返回错误。
    pub fn annotate(
        &mut self,
        start_ms: usize,
        end_ms: usize,
        event: impl Into<AudioEvent>,
    ) -> Result<&AudioEvent, TimelineEventError> {
        self.annotate_with(start_ms, end_ms, event, true, None)
    }

    /// 写入一条事件，并指定它属于 reference 还是 prediction。
    ///
    /// 参考禁止 `source`，预测必须有非空 `source`。调用方带进来的 `id`、`range`
    /// 和 `source` 会被这次写入覆盖。内容相同的事件会去重。
    ///
    /// # Errors
    ///
    /// source 约束不满足、事件内容不合法，或与已有事件非法重叠时返回错误。
    pub fn annotate_with(
        &mut self,
        start_ms: usize,
        end_ms: usize,
        event: impl Into<AudioEvent>,
        is_reference: bool,
        source: Option<&str>,
    ) -> Result<&AudioEvent, TimelineEventError> {
        let mut event = event.into();
        let id = format!("evt_{}", Uuid::new_v4().simple());
        if is_reference {
            if source.is_some() {
                return Err(TimelineEventError::ReferenceHasSource { event_id: id });
            }
            event.assign_identity(id, TimeRange::new(start_ms, end_ms), None);
            return push_validated(&mut self.reference, event, false);
        }
        if source.is_none_or(|source| source.trim().is_empty()) {
            return Err(TimelineEventError::PredictionMissingSource { event_id: id });
        }
        event.assign_identity(
            id,
            TimeRange::new(start_ms, end_ms),
            source.map(str::to_owned),
        );
        push_validated(&mut self.prediction, event, true)
    }

    /// 按 ID 删除一条事件，返回是否确实删掉了。
    pub fn remove_event(&mut self, id: &str) -> bool {
        let old_len = self.reference.len() + self.prediction.len();
        self.reference.retain(|event| event.id() != id);
        self.prediction.retain(|event| event.id() != id);
        old_len != self.reference.len() + self.prediction.len()
    }

    /// 先参考后预测地遍历全部 span。
    pub fn all_spans(&self) -> impl Iterator<Item = &AudioEvent> {
        self.reference.iter().chain(&self.prediction)
    }

    /// 参考加预测的 span 总数。
    pub fn span_count(&self) -> usize {
        self.reference.len() + self.prediction.len()
    }

    /// 筛出指定预测来源的 span。
    pub fn predictions_by_source<'a>(
        &'a self,
        source: &'a str,
    ) -> impl Iterator<Item = &'a AudioEvent> + 'a {
        self.prediction
            .iter()
            .filter(move |event| event.source() == Some(source))
    }

    /// 按标注类别汇总去重后的预测 source 列表。
    pub fn prediction_sources(&self) -> BTreeMap<&'static str, Vec<&str>> {
        let mut sources = ["speech", "event"]
            .into_iter()
            .map(|kind| (kind, Vec::new()))
            .collect::<BTreeMap<_, _>>();
        for event in &self.prediction {
            if let Some(source) = event.source() {
                sources
                    .get_mut(event.source_group())
                    .expect("every event kind has a source group")
                    .push(source);
            }
        }
        for values in sources.values_mut() {
            values.sort_unstable();
            values.dedup();
        }
        sources
    }

    /// 从参考标注拼出转写文本。
    pub fn reference_transcript(&self) -> Transcript {
        transcript_from_annotations(self.reference.iter())
    }

    /// 从指定预测来源拼出转写文本。
    pub fn prediction_transcript(&self, source: &str) -> Transcript {
        transcript_from_annotations(self.predictions_by_source(source))
    }

    /// 删除某个 source 的全部预测，返回删除条数。
    pub fn remove_predictions_by_source(&mut self, source: &str) -> usize {
        let old_len = self.prediction.len();
        self.prediction
            .retain(|event| event.source() != Some(source));
        old_len - self.prediction.len()
    }

    /// 把预测 source 从 `from` 改成 `to`，并重新做重叠校验。
    ///
    /// # Errors
    ///
    /// `to` 为空或改名后出现非法重叠时返回错误。
    pub fn relabel_prediction_source(
        &mut self,
        from: &str,
        to: &str,
    ) -> Result<usize, TimelineEventError> {
        if to.trim().is_empty() {
            return Err(TimelineEventError::InvalidPredictionSource);
        }
        let mut candidate = self.prediction.clone();
        let mut changed = 0;
        for event in &mut candidate {
            if event.source() == Some(from) {
                event.set_source(Some(to.to_string()));
                changed += 1;
            }
        }
        validate_prediction_slice(&candidate)?;
        self.prediction = candidate;
        Ok(changed)
    }

    /// 校验当前参考和预测是否满足 source 与重叠约束。
    pub fn validate_spans(&self) -> Result<(), TimelineEventError> {
        validate_reference_slice(&self.reference)?;
        validate_prediction_slice(&self.prediction)
    }

    /// 把时间轴时长延长到至少 `duration`；不会缩短。
    pub(crate) fn extend_to(&mut self, duration: usize) {
        if duration > self.duration {
            self.duration = duration;
        }
    }
}

/// 去重后写入；与已有同类标注重叠则报错。
fn push_validated(
    annotations: &mut Vec<AudioEvent>,
    annotation: AudioEvent,
    prediction: bool,
) -> Result<&AudioEvent, TimelineEventError> {
    validate_span_payload(&annotation)?;
    if let Some(index) = annotations
        .iter()
        .position(|existing| existing.content_eq(&annotation))
    {
        return Ok(&annotations[index]);
    }
    for existing in annotations.iter() {
        if let Some((kind, speaker)) = overlap_conflict(existing, &annotation, prediction) {
            return Err(TimelineEventError::Overlap(Box::new(EventOverlap {
                kind,
                source: annotation.source().map(str::to_owned),
                speaker,
                first_id: existing.id().to_owned(),
                first_range: existing.range(),
                second_id: annotation.id().to_owned(),
                second_range: annotation.range(),
            })));
        }
    }
    annotations.push(annotation);
    Ok(annotations
        .last()
        .expect("the annotation was just inserted"))
}

/// 校验参考列表：禁止 source，并检查重叠。
fn validate_reference_slice(annotations: &[AudioEvent]) -> Result<(), TimelineEventError> {
    validate_slice(annotations, false)
}

/// 校验预测列表：必须有 source，并检查重叠。
fn validate_prediction_slice(annotations: &[AudioEvent]) -> Result<(), TimelineEventError> {
    for annotation in annotations {
        if annotation
            .source()
            .is_none_or(|source| source.trim().is_empty())
        {
            return Err(TimelineEventError::PredictionMissingSource {
                event_id: annotation.id().to_owned(),
            });
        }
    }
    validate_slice(annotations, true)
}

/// 两两检查同类标注是否非法重叠，并检查语音树内部的时间。
fn validate_slice(annotations: &[AudioEvent], prediction: bool) -> Result<(), TimelineEventError> {
    for annotation in annotations {
        validate_span_payload(annotation)?;
    }
    for (index, first) in annotations.iter().enumerate() {
        for second in &annotations[index + 1..] {
            if let Some((kind, speaker)) = overlap_conflict(first, second, prediction) {
                return Err(TimelineEventError::Overlap(Box::new(EventOverlap {
                    kind,
                    source: second.source().map(str::to_owned),
                    speaker,
                    first_id: first.id().to_owned(),
                    first_range: first.range(),
                    second_id: second.id().to_owned(),
                    second_range: second.range(),
                })));
            }
        }
    }
    Ok(())
}

/// 检查事件名、说话人，以及语音内部的句子和 token 时间。
fn validate_span_payload(event: &AudioEvent) -> Result<(), TimelineEventError> {
    match event {
        AudioEvent::Base { name, .. } => validate_event_name(name),
        AudioEvent::Speech(speech) => {
            if let Some(speaker) = &speech.speaker
                && speaker.name.trim().is_empty()
            {
                return Err(TimelineEventError::InvalidSpeakerName);
            }
            if let Some(transcription) = &speech.transcription {
                validate_transcription(speech.range, transcription)?;
            }
            Ok(())
        }
    }
}

/// 事件名去掉空白后不能为空，也不能叫 `speech`。
fn validate_event_name(event: &str) -> Result<(), TimelineEventError> {
    let event = event.trim();
    if event.is_empty() {
        return Err(TimelineEventError::InvalidEventName);
    }
    if event == "speech" {
        return Err(TimelineEventError::SpeechEventName);
    }
    Ok(())
}

/// 句子可以为空。非空时，每句落在语音内且互不重叠，token 落在句子内。
///
/// token 列表同样可以为空。没有时间的 token 跳过区间检查。
fn validate_transcription(
    speech_range: TimeRange,
    transcription: &super::annotation::Transcription,
) -> Result<(), TimelineEventError> {
    for sentence in &transcription.sentences {
        validate_sentence(speech_range, sentence)?;
    }
    for (index, sentence) in transcription.sentences.iter().enumerate() {
        for other in &transcription.sentences[index + 1..] {
            if sentence.range.overlaps(&other.range) {
                return Err(TimelineEventError::OverlappingSentences {
                    first: sentence.range,
                    second: other.range,
                });
            }
        }
    }
    Ok(())
}

/// 句子必须有正时长，并且连同 token 一起落在语音区间内。
fn validate_sentence(
    speech_range: TimeRange,
    sentence: &Sentence,
) -> Result<(), TimelineEventError> {
    if sentence.range.end_ms <= sentence.range.start_ms {
        return Err(TimelineEventError::InvalidSentenceRange {
            range: sentence.range,
        });
    }
    if !speech_range.contains_range(&sentence.range) {
        return Err(TimelineEventError::SentenceOutsideSpeech {
            sentence: sentence.range,
            speech: speech_range,
        });
    }
    for token in &sentence.tokens {
        let Some(range) = token.range else {
            continue;
        };
        if range.end_ms <= range.start_ms {
            return Err(TimelineEventError::InvalidTokenRange { range });
        }
        if !sentence.range.contains_range(&range) {
            return Err(TimelineEventError::TokenOutsideSentence {
                token: range,
                sentence: sentence.range,
            });
        }
    }
    Ok(())
}

/// 判断两条 span 是否构成需要拒绝的重叠。
///
/// 预测只和相同 source 比较。同名非语音事件不能重叠。
/// 两段语音只有在双方都有说话人、且名字不同时可以重叠。
fn overlap_conflict(
    first: &AudioEvent,
    second: &AudioEvent,
    prediction: bool,
) -> Option<(EventConflictKind, Option<String>)> {
    if !first.range().overlaps(&second.range()) || prediction && first.source() != second.source() {
        return None;
    }
    match (first, second) {
        (AudioEvent::Base { name: left, .. }, AudioEvent::Base { name: right, .. })
            if left == right =>
        {
            Some((EventConflictKind::Event, None))
        }
        (AudioEvent::Speech(left), AudioEvent::Speech(right)) => {
            speech_overlap(left.speaker.as_ref(), right.speaker.as_ref())
        }
        _ => None,
    }
}

/// 不同说话人的语音可以重叠；缺说话人或同名说话人则冲突。
fn speech_overlap(
    first: Option<&super::annotation::Speaker>,
    second: Option<&super::annotation::Speaker>,
) -> Option<(EventConflictKind, Option<String>)> {
    match (first, second) {
        (Some(first_speaker), Some(second_speaker))
            if first_speaker.name != second_speaker.name =>
        {
            None
        }
        (Some(speaker), Some(_)) => Some((EventConflictKind::Speech, Some(speaker.name.clone()))),
        _ => Some((EventConflictKind::Speech, None)),
    }
}

/// 按语音起点把各段转写全文拼成 [`Transcript`]。
///
/// 全文来自 [`Transcription::text`]，不从句子重拼。句子只进入 `segments`，
/// 没有句子的转写仍然贡献全文。
fn transcript_from_annotations<'a>(
    annotations: impl Iterator<Item = &'a AudioEvent>,
) -> Transcript {
    let mut language = None;
    let mut pieces = Vec::new();
    let mut segments = Vec::new();
    for event in annotations {
        let Some(transcription) = event.transcription() else {
            continue;
        };
        if language.is_none() && event.language().is_some() {
            language = event.language().map(str::to_owned);
        }
        // 用语音区间排序，这样只有全文、没有句子的转写也能排进正确位置。
        pieces.push((event.range(), transcription.text.clone()));
        segments.extend(transcription.sentences.iter().cloned());
    }
    pieces.sort_by_key(|(range, _)| (range.start_ms, range.end_ms));
    segments.sort_by_key(|sentence| (sentence.range.start_ms, sentence.range.end_ms));
    Transcript {
        text: join_transcript_text(pieces.iter().map(|(_, text)| text.as_str())),
        language,
        segments,
    }
}

impl Default for Timeline {
    fn default() -> Self {
        Self::new(String::new(), 0)
    }
}
