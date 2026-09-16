//! 时间轴容器：参考/预测 span，以及重叠校验。

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use uuid::Uuid;

use super::annotation::{Annotation, AudioId, TimeSpan, TimeSpanId, TimelineId};
use super::segment::{Sentence, Transcript};
use crate::utils::TimeRange;

/// 不允许重叠的标注类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeSpanConflictKind {
    /// 相同事件的活动区间重叠。
    Activity,
    /// 同一说话人的区间重叠。
    Speaker,
    /// 转写区间重叠（含说话人自带转写）。
    Transcription,
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
    /// 人工 / 参考标注，`source` 必须为空。
    pub reference: Vec<TimeSpan>,
    /// 模型预测，必须带非空 `source`。
    pub prediction: Vec<TimeSpan>,
}

/// 终端打印时使用 [`Timeline::terminal_view`] 的卡片与标注轨道布局。
impl fmt::Display for Timeline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.terminal_view().fmt(formatter)
    }
}

/// 一次非法重叠的详细信息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeSpanOverlap {
    pub kind: TimeSpanConflictKind,
    pub source: Option<String>,
    pub speaker: Option<String>,
    pub first_id: TimeSpanId,
    pub first_range: TimeRange,
    pub second_id: TimeSpanId,
    pub second_range: TimeRange,
}

impl fmt::Display for TimeSpanOverlap {
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
pub enum TimelineSpanError {
    #[error("reference annotation {annotation_id:?} must not have a source")]
    ReferenceHasSource { annotation_id: TimeSpanId },
    #[error("prediction annotation {annotation_id:?} must have a non-empty source")]
    PredictionMissingSource { annotation_id: TimeSpanId },
    #[error("prediction source must contain at least one non-whitespace character")]
    InvalidPredictionSource,
    #[error("activity event must contain at least one non-whitespace character")]
    InvalidActivityEvent,
    #[error("{0}")]
    Overlap(Box<TimeSpanOverlap>),
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

    /// 写入一条标注。
    ///
    /// 参数与 Python `Timeline.annotate_span(start_ms, end_ms, annotation)` 对齐：
    /// 默认作为 reference（`is_reference=True` 且不带 `source`）。
    /// 内容完全相同的 span 会被去重并返回已有项。
    ///
    /// 指定 `is_reference` / `source` 时请使用 [`Self::annotate_span_with`]。
    ///
    /// # Errors
    ///
    /// source 约束不满足或与已有 span 非法重叠时返回错误。
    pub fn annotate_span(
        &mut self,
        start_ms: usize,
        end_ms: usize,
        annotation: impl Into<Annotation>,
    ) -> Result<&TimeSpan, TimelineSpanError> {
        self.annotate_span_with(start_ms, end_ms, annotation, true, None)
    }

    /// 写入一条标注，并指定是否为 reference 以及 prediction source。
    ///
    /// 参数顺序与 Python
    /// `Timeline.annotate_span(start_ms, end_ms, annotation, *, is_reference, source)`
    /// 对齐。参考禁止 `source`，预测必须有非空 `source`。
    /// 内容完全相同的 span 会被去重并返回已有项。
    ///
    /// # Errors
    ///
    /// source 约束不满足或与已有 span 非法重叠时返回错误。
    pub fn annotate_span_with(
        &mut self,
        start_ms: usize,
        end_ms: usize,
        annotation: impl Into<Annotation>,
        is_reference: bool,
        source: Option<&str>,
    ) -> Result<&TimeSpan, TimelineSpanError> {
        let annotation = TimeSpan::new(
            TimeRange::new(start_ms, end_ms),
            annotation.into(),
            source.map(str::to_owned),
        );
        if is_reference {
            if annotation.source.is_some() {
                return Err(TimelineSpanError::ReferenceHasSource {
                    annotation_id: annotation.id,
                });
            }
            return push_validated(&mut self.reference, annotation, false);
        }
        if annotation
            .source
            .as_deref()
            .is_none_or(|source| source.trim().is_empty())
        {
            return Err(TimelineSpanError::PredictionMissingSource {
                annotation_id: annotation.id,
            });
        }
        push_validated(&mut self.prediction, annotation, true)
    }

    /// 先参考后预测地遍历全部 span。
    pub fn all_spans(&self) -> impl Iterator<Item = &TimeSpan> {
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
    ) -> impl Iterator<Item = &'a TimeSpan> + 'a {
        self.prediction
            .iter()
            .filter(move |annotation| annotation.source.as_deref() == Some(source))
    }

    /// 按标注类别汇总去重后的预测 source 列表。
    pub fn prediction_sources(&self) -> BTreeMap<&'static str, Vec<&str>> {
        let mut sources = [
            "activity",
            "token",
            "transcription",
            "sentence",
            "speaker",
            "language",
        ]
        .into_iter()
        .map(|kind| (kind, Vec::new()))
        .collect::<BTreeMap<_, _>>();
        for annotation in &self.prediction {
            if let Some(source) = annotation.source.as_deref() {
                sources
                    .get_mut(annotation.annotation.source_group())
                    .expect("every annotation kind has a source group")
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
            .retain(|annotation| annotation.source.as_deref() != Some(source));
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
    ) -> Result<usize, TimelineSpanError> {
        if to.trim().is_empty() {
            return Err(TimelineSpanError::InvalidPredictionSource);
        }
        let mut candidate = self.prediction.clone();
        let mut changed = 0;
        for annotation in &mut candidate {
            if annotation.source.as_deref() == Some(from) {
                annotation.source = Some(to.to_string());
                changed += 1;
            }
        }
        validate_prediction_slice(&candidate)?;
        self.prediction = candidate;
        Ok(changed)
    }

    /// 校验当前参考和预测是否满足 source 与重叠约束。
    pub fn validate_spans(&self) -> Result<(), TimelineSpanError> {
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
    annotations: &mut Vec<TimeSpan>,
    annotation: TimeSpan,
    prediction: bool,
) -> Result<&TimeSpan, TimelineSpanError> {
    validate_activity_event(&annotation)?;
    if let Some(index) = annotations
        .iter()
        .position(|existing| existing.content_eq(&annotation))
    {
        return Ok(&annotations[index]);
    }
    for existing in annotations.iter() {
        if let Some((kind, speaker)) = overlap_conflict(existing, &annotation, prediction) {
            return Err(TimelineSpanError::Overlap(Box::new(TimeSpanOverlap {
                kind,
                source: annotation.source.clone(),
                speaker,
                first_id: existing.id.clone(),
                first_range: existing.range,
                second_id: annotation.id.clone(),
                second_range: annotation.range,
            })));
        }
    }
    annotations.push(annotation);
    Ok(annotations
        .last()
        .expect("the annotation was just inserted"))
}

/// 校验参考列表：禁止 source，并检查重叠。
fn validate_reference_slice(annotations: &[TimeSpan]) -> Result<(), TimelineSpanError> {
    validate_slice(annotations, false)
}

/// 校验预测列表：必须有 source，并检查重叠。
fn validate_prediction_slice(annotations: &[TimeSpan]) -> Result<(), TimelineSpanError> {
    for annotation in annotations {
        if annotation
            .source
            .as_deref()
            .is_none_or(|source| source.trim().is_empty())
        {
            return Err(TimelineSpanError::PredictionMissingSource {
                annotation_id: annotation.id.clone(),
            });
        }
    }
    validate_slice(annotations, true)
}

/// 两两检查活动事件以及同类标注是否非法重叠。
fn validate_slice(annotations: &[TimeSpan], prediction: bool) -> Result<(), TimelineSpanError> {
    for annotation in annotations {
        validate_activity_event(annotation)?;
    }
    for (index, first) in annotations.iter().enumerate() {
        for second in &annotations[index + 1..] {
            if let Some((kind, speaker)) = overlap_conflict(first, second, prediction) {
                return Err(TimelineSpanError::Overlap(Box::new(TimeSpanOverlap {
                    kind,
                    source: second.source.clone(),
                    speaker,
                    first_id: first.id.clone(),
                    first_range: first.range,
                    second_id: second.id.clone(),
                    second_range: second.range,
                })));
            }
        }
    }
    Ok(())
}

/// 活动事件名若存在则不能是空白字符串。
fn validate_activity_event(annotation: &TimeSpan) -> Result<(), TimelineSpanError> {
    if let Annotation::Activity(activity) = &annotation.annotation
        && activity
            .event
            .as_deref()
            .is_some_and(|event| event.trim().is_empty())
    {
        return Err(TimelineSpanError::InvalidActivityEvent);
    }
    Ok(())
}

/// 判断两条 span 是否构成需要拒绝的重叠。
///
/// 预测只和相同 source 比较。活动要事件名相同，说话人要名字相同；
/// 转写之间，以及转写与带转写的说话人之间也会冲突。
fn overlap_conflict(
    first: &TimeSpan,
    second: &TimeSpan,
    prediction: bool,
) -> Option<(TimeSpanConflictKind, Option<String>)> {
    if !first.range.overlaps(&second.range)
        || prediction && first.source.as_deref() != second.source.as_deref()
    {
        return None;
    }
    match (&first.annotation, &second.annotation) {
        (Annotation::Activity(first), Annotation::Activity(second))
            if first.event == second.event =>
        {
            Some((TimeSpanConflictKind::Activity, None))
        }
        (Annotation::Speaker(first), Annotation::Speaker(second)) if first.name == second.name => {
            Some((TimeSpanConflictKind::Speaker, Some(first.name.clone())))
        }
        (Annotation::Transcription(_), Annotation::Transcription(_)) => {
            Some((TimeSpanConflictKind::Transcription, None))
        }
        (Annotation::Transcription(_), Annotation::Speaker(speaker))
        | (Annotation::Speaker(speaker), Annotation::Transcription(_))
            if speaker.transcription.is_some() =>
        {
            Some((
                TimeSpanConflictKind::Transcription,
                Some(speaker.name.clone()),
            ))
        }
        _ => None,
    }
}

/// 按时间顺序把转写 / 句子 / 说话人转写拼成 [`Transcript`]。
fn transcript_from_annotations<'a>(annotations: impl Iterator<Item = &'a TimeSpan>) -> Transcript {
    let mut segments = annotations
        .filter_map(|annotation| match &annotation.annotation {
            Annotation::Transcription(transcription) => Some((
                annotation.range.start_ms,
                Sentence {
                    text: transcription.text.clone(),
                    tokens: transcription.tokens.clone(),
                    language: transcription.language.clone(),
                },
            )),
            Annotation::Sentence(segment) => Some((annotation.range.start_ms, segment.clone())),
            Annotation::Speaker(speaker) => speaker.transcription.as_ref().map(|value| {
                (
                    annotation.range.start_ms,
                    Sentence {
                        text: value.text.clone(),
                        tokens: value.tokens.clone(),
                        language: value.language.clone(),
                    },
                )
            }),
            _ => None,
        })
        .collect::<Vec<(usize, Sentence)>>();

    segments.sort_by_key(|(start, _)| *start);
    let segments = segments
        .into_iter()
        .map(|(_, segment)| segment)
        .collect::<Vec<_>>();
    let text = segments
        .iter()
        .map(|segment| segment.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let language = segments.iter().find_map(|segment| segment.language.clone());

    Transcript {
        text,
        language,
        segments,
    }
}

impl Default for Timeline {
    fn default() -> Self {
        Self::new(String::new(), 0)
    }
}
