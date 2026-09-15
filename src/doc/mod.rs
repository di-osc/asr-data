mod display;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::audio::{
    AudioChannel, AudioChunk, AudioEncoding, AudioFormat, AudioInfo, AudioSource, Waveform,
};
use crate::timeline::{Timeline, TimelineSpanError};
use crate::utils::DurationMs;

/// An audio source together with all annotations and per-audio metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Audio {
    /// 文档 ID，会同步到各条 timeline 的 `audio_id`。
    #[serde(default)]
    pub id: String,
    /// 加载这份文档时使用的来源。
    pub source: AudioSource,
    /// 整段音频的采样率、声道和帧数。
    pub info: AudioInfo,
    /// 按声道划分的标注时间轴。
    pub(crate) timelines: BTreeMap<AudioChannel, Timeline>,
    /// 文档级 JSON metadata。
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    /// 已解码波形；仅 probe 时可能为 `None`。
    #[serde(skip)]
    pub(crate) waveform: Option<Waveform>,
}

impl Audio {
    /// Renders a compact, terminal-friendly audio summary and timeline.
    pub fn terminal_view(&self) -> impl std::fmt::Display + '_ {
        display::AudioTerminalView::auto(self)
    }

    /// 渲染带或不带 ANSI 颜色的终端摘要。
    pub fn terminal_view_with_color(&self, color: bool) -> impl std::fmt::Display + '_ {
        display::AudioTerminalView::with_color(self, color)
    }

    /// 从本地文件加载完整 [`Audio`] 文档。
    ///
    /// # Errors
    ///
    /// 打开或解码失败时返回错误。
    pub fn from_path(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
        AudioSource::from_path(path).load()
    }

    /// 从 URL 下载并加载完整文档。
    ///
    /// # Errors
    ///
    /// 下载或解码失败时返回错误。
    pub fn from_url(url: impl Into<String>) -> anyhow::Result<Self> {
        AudioSource::from_url(url).load()
    }

    /// 从带容器的编码字节加载完整文档。
    ///
    /// # Errors
    ///
    /// 解码失败时返回错误。
    pub fn from_encoded_bytes(bytes: impl Into<Vec<u8>>) -> anyhow::Result<Self> {
        AudioSource::from_encoded_bytes(bytes).load()
    }

    /// 从 base64 音频加载完整文档。
    ///
    /// # Errors
    ///
    /// 解码失败时返回错误。
    pub fn from_base64(data: impl Into<String>) -> anyhow::Result<Self> {
        AudioSource::from_base64(data).load()
    }

    /// 从 PCM S16LE 字节加载完整文档。
    ///
    /// # Errors
    ///
    /// PCM 无法对齐或解码失败时返回错误。
    pub fn from_pcm_s16le(
        bytes: impl Into<Vec<u8>>,
        sample_rate: u32,
        channels: u16,
    ) -> anyhow::Result<Self> {
        AudioSource::from_pcm_s16le(bytes, sample_rate, channels).load()
    }

    /// 从任意 [`AudioSource`] 加载文档，并生成随机 ID。
    ///
    /// # Errors
    ///
    /// 探测来源失败时返回错误。
    pub fn new(source: impl Into<AudioSource>) -> anyhow::Result<Self> {
        Self::with_id(format!("audio_{}", uuid::Uuid::new_v4().simple()), source)
    }

    /// 从任意来源加载文档并指定 ID。
    ///
    /// # Errors
    ///
    /// 探测来源失败时返回错误。
    pub fn with_id(
        audio_id: impl Into<String>,
        source: impl Into<AudioSource>,
    ) -> anyhow::Result<Self> {
        Self::with_id_from_source(audio_id, source)
    }

    /// [`Self::new`] 的别名。
    pub fn from_source(source: impl Into<AudioSource>) -> anyhow::Result<Self> {
        Self::new(source)
    }

    /// 探测 `source` 后构造带指定 ID 的文档（此时尚未解码波形）。
    ///
    /// # Errors
    ///
    /// 探测失败时返回错误。
    pub fn with_id_from_source(
        audio_id: impl Into<String>,
        source: impl Into<AudioSource>,
    ) -> anyhow::Result<Self> {
        let source = source.into();
        let info = source.probe()?;
        Ok(Self::with_id_from_info(audio_id, source, &info))
    }

    /// 异步探测来源后构造文档，自动生成 ID。
    ///
    /// # Errors
    ///
    /// 探测失败时返回错误。
    pub async fn afrom_source(source: impl Into<AudioSource>) -> anyhow::Result<Self> {
        let source = source.into();
        let info = source.aprobe().await?;
        Ok(Self::from_info(source, &info))
    }

    /// 异步探测来源后构造指定 ID 的文档。
    ///
    /// # Errors
    ///
    /// 探测失败时返回错误。
    pub async fn with_id_afrom_source(
        audio_id: impl Into<String>,
        source: impl Into<AudioSource>,
    ) -> anyhow::Result<Self> {
        let audio_id = audio_id.into();
        let source = source.into();
        let info = source.aprobe().await?;
        Ok(Self::with_id_from_info(audio_id, source, &info))
    }

    /// 用已有 [`AudioInfo`] 构造文档，自动生成 ID，不触发 I/O。
    pub fn from_info(source: impl Into<AudioSource>, info: &AudioInfo) -> Self {
        Self::with_id_from_info(
            format!("audio_{}", uuid::Uuid::new_v4().simple()),
            source,
            info,
        )
    }

    /// 用已有 [`AudioInfo`] 构造指定 ID 的文档，并按声道预建满时长 timeline。
    pub fn with_id_from_info(
        audio_id: impl Into<String>,
        source: impl Into<AudioSource>,
        info: &AudioInfo,
    ) -> Self {
        let mut doc = Self {
            id: audio_id.into(),
            source: source.into(),
            info: info.clone(),
            timelines: BTreeMap::new(),
            metadata: BTreeMap::new(),
            waveform: None,
        };
        let duration = DurationMs(info.timeline_duration_ms());
        if info.channels == 1 {
            doc.timelines
                .insert(AudioChannel::Mono, Timeline::new(doc.id.clone(), duration));
        } else {
            for index in 0..info.channels {
                doc.timelines.insert(
                    AudioChannel::from_index(index),
                    Timeline::new(doc.id.clone(), duration),
                );
            }
        }
        doc
    }

    /// Python 流式绑定：timeline 从 0 开始，波形先放空缓冲。
    ///
    /// # Errors
    ///
    /// 声道数或样本对齐非法时返回错误。
    #[cfg(feature = "python-bindings")]
    pub(crate) fn with_id_from_stream_info(
        audio_id: impl Into<String>,
        source: impl Into<AudioSource>,
        info: &AudioInfo,
    ) -> Result<Self, crate::audio::AudioError> {
        let mut audio = Self::with_id_from_info(audio_id, source, info);
        for timeline in audio.timelines.values_mut() {
            timeline.duration = DurationMs(0);
        }
        audio.waveform = Some(
            Waveform::try_new_with_channels(Vec::new(), info.sample_rate, info.channels)?
                .with_source_format(info.source_format.clone()),
        );
        Ok(audio)
    }

    /// 用已经解码的波形构造文档，`info` 从波形推导。
    pub(crate) fn with_loaded_waveform(
        audio_id: impl Into<String>,
        source: impl Into<AudioSource>,
        waveform: Waveform,
    ) -> Self {
        let info = AudioInfo {
            sample_rate: waveform.sample_rate,
            channels: waveform.channels,
            frame_count: waveform.frame_count() as u64,
            source_format: waveform.source_format.clone().unwrap_or(AudioFormat {
                encoding: AudioEncoding::Unknown,
                sample_rate: waveform.sample_rate,
                channels: waveform.channels,
            }),
        };
        let mut audio = Self::with_id_from_info(audio_id, source, &info);
        audio.waveform = Some(waveform);
        audio
    }

    /// 若尚未解码则从 `source` 加载波形，并返回引用。
    ///
    /// # Errors
    ///
    /// 解码失败时返回错误。
    fn ensure_waveform(&mut self) -> anyhow::Result<&Waveform> {
        if self.waveform.is_none() {
            self.waveform = Some(self.source.decode_waveform()?);
        }
        Ok(self.waveform.as_ref().expect("waveform was just loaded"))
    }

    /// 返回完整波形；尚未解码时会先从 source 加载。
    ///
    /// # Errors
    ///
    /// 解码失败时返回错误。
    pub fn as_waveform(&mut self) -> anyhow::Result<Waveform> {
        Ok(self.ensure_waveform()?.clone())
    }

    /// 抽出指定声道的波形；尚未解码时会先加载。
    ///
    /// # Errors
    ///
    /// 声道不规范、越界或解码失败时返回错误。
    pub fn waveform_for_channel(&mut self, channel: AudioChannel) -> anyhow::Result<Waveform> {
        validate_channel(channel)?;
        let waveform = self.ensure_waveform()?;
        match channel {
            AudioChannel::Mono => waveform.to_mono().map_err(Into::into),
            AudioChannel::Left => waveform.channel(0).map_err(Into::into),
            AudioChannel::Right => waveform.channel(1).map_err(Into::into),
            AudioChannel::Channel(index) => waveform.channel(index).map_err(Into::into),
        }
    }

    /// 放入单声道 timeline，并把文档 ID 改成该 timeline 的 `audio_id`。
    pub fn with_timeline(mut self, timeline: Timeline) -> Self {
        let audio_id = timeline.audio_id.clone();
        self.set_audio_id(audio_id);
        self.timelines.insert(AudioChannel::Mono, timeline);
        self
    }

    /// 返回指定声道的 timeline。
    ///
    /// # Errors
    ///
    /// 声道标识不规范时返回 [`AudioChannelError`]。
    pub fn timeline(&self, channel: AudioChannel) -> Result<Option<&Timeline>, AudioChannelError> {
        validate_channel(channel)?;
        Ok(self.timelines.get(&channel))
    }

    /// 可变借用指定声道的 timeline。
    ///
    /// # Errors
    ///
    /// 声道标识不规范时返回 [`AudioChannelError`]。
    pub fn timeline_mut(
        &mut self,
        channel: AudioChannel,
    ) -> Result<Option<&mut Timeline>, AudioChannelError> {
        validate_channel(channel)?;
        Ok(self.timelines.get_mut(&channel))
    }

    /// 确保指定声道存在 timeline；没有则按音频时长创建。
    ///
    /// # Errors
    ///
    /// 声道不规范，或传入时长与 `info` 不一致时返回错误。
    pub fn ensure_timeline(
        &mut self,
        channel: AudioChannel,
        duration: Option<DurationMs>,
    ) -> Result<&mut Timeline, AudioTimelineError> {
        validate_channel(channel).map_err(AudioTimelineError::InvalidChannel)?;
        let expected = Some(DurationMs(self.info.timeline_duration_ms()));
        let duration = match (expected, duration) {
            (None, None) => return Err(AudioTimelineError::MissingDuration),
            (None, Some(duration)) | (Some(duration), None) => duration,
            (Some(expected), Some(found)) if expected == found => expected,
            (Some(expected), Some(found)) => {
                return Err(AudioTimelineError::DurationMismatch { expected, found });
            }
        };
        let audio_id = self.id.clone();
        Ok(self
            .timelines
            .entry(channel)
            .or_insert_with(|| Timeline::new(audio_id, duration)))
    }

    /// 返回单声道 timeline。
    pub fn mono_timeline(&self) -> Option<&Timeline> {
        self.timelines.get(&AudioChannel::Mono)
    }

    /// 可变借用单声道 timeline。
    pub fn mono_timeline_mut(&mut self) -> Option<&mut Timeline> {
        self.timelines.get_mut(&AudioChannel::Mono)
    }

    /// 返回全部 timeline。
    pub fn timelines(&self) -> &BTreeMap<AudioChannel, Timeline> {
        &self.timelines
    }

    /// 删除指定声道的 timeline。
    ///
    /// # Errors
    ///
    /// 声道标识不规范时返回 [`AudioChannelError`]。
    pub fn remove_timeline(
        &mut self,
        channel: AudioChannel,
    ) -> Result<Option<Timeline>, AudioChannelError> {
        validate_channel(channel)?;
        Ok(self.timelines.remove(&channel))
    }

    /// 写入一条文档级 metadata。
    pub fn with_metadata_value(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// 按序号生成稳定文档 ID：`audio-{index}`。
    pub fn id_for_index(index: usize) -> String {
        format!("audio-{index}")
    }

    /// 返回清洗后的文档 ID（去掉首尾空白）。
    pub fn audio_id(&self) -> String {
        sanitize_audio_id(&self.id)
    }

    /// 同时更新文档 ID 和所有 timeline 上的 `audio_id`。
    pub fn set_audio_id(&mut self, audio_id: impl Into<String>) {
        let audio_id = audio_id.into();
        self.id.clone_from(&audio_id);
        for timeline in self.timelines.values_mut() {
            timeline.audio_id.clone_from(&audio_id);
        }
    }

    /// 文档时间轴应覆盖的整毫秒时长。
    pub fn timeline_duration(&self) -> Option<DurationMs> {
        Some(DurationMs(self.info.timeline_duration_ms()))
    }

    /// 校验 ID、info、timeline 时长以及标注边界 / source 约束。
    ///
    /// # Errors
    ///
    /// 任一约束不满足时返回对应的 [`AudioValidationError`]。
    pub fn validate(&self) -> Result<(), AudioValidationError> {
        if self.id.trim().is_empty() {
            return Err(AudioValidationError::EmptyAudioId);
        }
        if self.info.sample_rate == 0 {
            return Err(AudioValidationError::InvalidAudioInfoSampleRate);
        }
        if self.info.channels == 0 {
            return Err(AudioValidationError::InvalidAudioInfoChannels);
        }
        let expected_duration = self.timeline_duration();
        for (channel, timeline) in &self.timelines {
            if !channel.is_canonical() {
                return Err(AudioValidationError::NonCanonicalChannel { channel: *channel });
            }
            if timeline.audio_id != self.id {
                return Err(AudioValidationError::TimelineAudioIdMismatch {
                    channel: *channel,
                    expected: self.id.clone(),
                    found: timeline.audio_id.clone(),
                });
            }
            if Some(timeline.duration) != expected_duration {
                return Err(AudioValidationError::TimelineDurationMismatch {
                    channel: *channel,
                    expected: expected_duration.expect("a timeline established the duration"),
                    found: timeline.duration,
                });
            }
            for annotation in &timeline.reference {
                if annotation.source.is_some() {
                    return Err(AudioValidationError::ReferenceAnnotationHasSource {
                        channel: *channel,
                        annotation_id: annotation.id.clone(),
                    });
                }
            }
            for annotation in &timeline.prediction {
                if annotation
                    .source
                    .as_deref()
                    .is_none_or(|source| source.trim().is_empty())
                {
                    return Err(AudioValidationError::PredictionAnnotationMissingSource {
                        channel: *channel,
                        annotation_id: annotation.id.clone(),
                    });
                }
            }
            for annotation in timeline.all_spans() {
                if annotation.range.end > timeline.duration {
                    return Err(AudioValidationError::AnnotationOutOfBounds {
                        channel: *channel,
                        annotation_id: annotation.id.clone(),
                        end: annotation.range.end,
                        duration: timeline.duration,
                    });
                }
            }
            timeline.validate_spans().map_err(|error| {
                AudioValidationError::InvalidAnnotations {
                    channel: *channel,
                    error,
                }
            })?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Audio {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.terminal_view().fmt(formatter)
    }
}

/// 随 [`AudioChunk`] 迭代增长的音频文档，与 [`Audio`] 同级。
///
/// 创建时 timeline 时长为 0。每次 `next()` 从内部 PCM 生产器拉一块，追加到
/// `waveform`，并把所有 timeline 延到当前 `position_ms`。全部消费完后可用
/// [`Self::into_audio`] 转成 [`Audio`]。
///
/// 字段 `chunks` 不是已经切好的块列表，而是按需解码 / 切窗的上游 iterator。
/// 已经到来的样本在 `waveform` 里。
pub struct AudioStream {
    /// 文档 ID，会写进各条 timeline。
    pub id: String,
    /// 创建流时使用的来源。
    pub source: AudioSource,
    /// 探测得到的整段音频信息（时长在迭代完成前即可知道）。
    pub info: AudioInfo,
    timelines: BTreeMap<AudioChannel, Timeline>,
    /// 可与 chunk 共享、原地修改的文档级 metadata。
    pub metadata: BTreeMap<String, serde_json::Value>,
    /// 上游 PCM 生产器：解码、可选变换，再按目标时长吐块。
    chunks: crate::audio::stream::SourceAudioStream,
    /// 到目前为止已经接收的累积波形。
    waveform: Waveform,
    /// 当前读到的全局毫秒位置。
    position_ms: u64,
    /// 是否已经吐出最后一块。
    complete: bool,
    /// 出错或主动 [`Self::close`] 后不再继续迭代。
    closed: bool,
}

impl AudioStream {
    /// 从本地文件创建流。
    ///
    /// # Errors
    ///
    /// 探测或打开文件失败，或 `chunk_size_ms` 为 0 时返回错误。
    pub fn from_path(path: impl AsRef<Path>, chunk_size_ms: u64) -> anyhow::Result<Self> {
        AudioSource::from_path(path.as_ref().to_path_buf()).stream(chunk_size_ms)
    }

    /// 从 URL 创建流。
    ///
    /// # Errors
    ///
    /// 下载、探测失败或 `chunk_size_ms` 为 0 时返回错误。
    pub fn from_url(url: impl Into<String>, chunk_size_ms: u64) -> anyhow::Result<Self> {
        AudioSource::from_url(url).stream(chunk_size_ms)
    }

    /// 从带容器的编码字节创建流。
    ///
    /// # Errors
    ///
    /// 探测失败或 `chunk_size_ms` 为 0 时返回错误。
    pub fn from_encoded_bytes(
        bytes: impl Into<Vec<u8>>,
        chunk_size_ms: u64,
    ) -> anyhow::Result<Self> {
        AudioSource::from_encoded_bytes(bytes).stream(chunk_size_ms)
    }

    /// 从 base64 音频创建流。
    ///
    /// # Errors
    ///
    /// 解码、探测失败或 `chunk_size_ms` 为 0 时返回错误。
    pub fn from_base64(data: impl Into<String>, chunk_size_ms: u64) -> anyhow::Result<Self> {
        AudioSource::from_base64(data).stream(chunk_size_ms)
    }

    /// 从 PCM S16LE 字节创建流。
    ///
    /// # Errors
    ///
    /// PCM 无法对齐、探测失败或 `chunk_size_ms` 为 0 时返回错误。
    pub fn from_pcm_s16le(
        bytes: impl Into<Vec<u8>>,
        sample_rate: u32,
        channels: u16,
        chunk_size_ms: u64,
    ) -> anyhow::Result<Self> {
        AudioSource::from_pcm_s16le(bytes, sample_rate, channels).stream(chunk_size_ms)
    }

    /// 用已探测的 [`AudioInfo`] 构造流：timeline 从 0 时长开始增长。
    ///
    /// `chunks` 字段被初始化为源格式 PCM 生产器；`sample_rate` / `mono` 变换
    /// 由 Python 绑定在另一条路径上注入，Rust 公开构造保持源格式。
    ///
    /// # Errors
    ///
    /// 声道数非法或无法创建上游生产器时返回错误。
    pub(crate) fn new(
        audio_id: impl Into<String>,
        source: AudioSource,
        info: AudioInfo,
        chunk_size_ms: u64,
    ) -> anyhow::Result<Self> {
        let id = audio_id.into();
        let mut timelines = BTreeMap::new();
        if info.channels == 1 {
            timelines.insert(AudioChannel::Mono, Timeline::new(id.clone(), DurationMs(0)));
        } else {
            for index in 0..info.channels {
                timelines.insert(
                    AudioChannel::from_index(index),
                    Timeline::new(id.clone(), DurationMs(0)),
                );
            }
        }
        let waveform =
            Waveform::try_new_with_channels(Vec::new(), info.sample_rate, info.channels)?
                .with_source_format(info.source_format.clone());
        let chunks = crate::audio::stream::SourceAudioStream::new(
            source.clone(),
            chunk_size_ms,
            None,
            None,
        )?;
        Ok(Self {
            id,
            source,
            info,
            timelines,
            metadata: BTreeMap::new(),
            chunks,
            waveform,
            position_ms: 0,
            complete: false,
            closed: false,
        })
    }

    /// 返回指定声道的 timeline；不存在时为 `None`。
    ///
    /// # Errors
    ///
    /// 声道标识不规范时返回 [`AudioChannelError`]。
    pub fn timeline(&self, channel: AudioChannel) -> Result<Option<&Timeline>, AudioChannelError> {
        validate_channel(channel)?;
        Ok(self.timelines.get(&channel))
    }

    /// 可变借用指定声道的 timeline。
    ///
    /// # Errors
    ///
    /// 声道标识不规范时返回 [`AudioChannelError`]。
    pub fn timeline_mut(
        &mut self,
        channel: AudioChannel,
    ) -> Result<Option<&mut Timeline>, AudioChannelError> {
        validate_channel(channel)?;
        Ok(self.timelines.get_mut(&channel))
    }

    /// 返回当前全部 timeline，时长等于已经迭代到的位置。
    pub fn timelines(&self) -> &BTreeMap<AudioChannel, Timeline> {
        &self.timelines
    }

    /// 克隆到目前为止已经累积的波形。
    pub fn as_waveform(&self) -> Waveform {
        self.waveform.clone()
    }

    /// 从已累积波形中抽出指定声道。
    ///
    /// # Errors
    ///
    /// 声道不规范或下标越界时返回错误。
    pub fn waveform_for_channel(&self, channel: AudioChannel) -> anyhow::Result<Waveform> {
        validate_channel(channel)?;
        match channel {
            AudioChannel::Mono => self.waveform.to_mono().map_err(Into::into),
            AudioChannel::Left => self.waveform.channel(0).map_err(Into::into),
            AudioChannel::Right => self.waveform.channel(1).map_err(Into::into),
            AudioChannel::Channel(index) => self.waveform.channel(index).map_err(Into::into),
        }
    }

    /// 当前已经读到的全局毫秒位置。
    pub fn position_ms(&self) -> u64 {
        self.position_ms
    }

    /// 是否已经吐出最后一块。
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// 是否因错误或 [`Self::close`] 提前结束。
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// 提前结束迭代；已经 complete 的流不会被标记为 closed。
    pub fn close(&mut self) {
        if !self.complete {
            self.closed = true;
        }
    }

    /// 在完整消费后转成 [`Audio`] 文档。
    ///
    /// # Errors
    ///
    /// 尚未迭代到最后一块时返回错误。
    pub fn into_audio(self) -> anyhow::Result<Audio> {
        if !self.complete {
            anyhow::bail!("audio stream must be completely consumed before conversion");
        }
        Ok(Audio {
            id: self.id,
            source: self.source,
            info: self.info,
            timelines: self.timelines,
            metadata: self.metadata,
            waveform: Some(self.waveform),
        })
    }
}

impl Iterator for AudioStream {
    type Item = anyhow::Result<AudioChunk>;

    /// 拉下一块 PCM：追加到累积波形，并把所有 timeline 延到新的 `position_ms`。
    fn next(&mut self) -> Option<Self::Item> {
        if self.complete || self.closed {
            return None;
        }
        let chunk = match self.chunks.next()? {
            Ok(chunk) => chunk,
            Err(error) => {
                self.closed = true;
                return Some(Err(error));
            }
        };
        // 文档状态跟着块走：样本累积、时间轴延长、读到末尾则 complete。
        self.waveform.samples.extend_from_slice(&chunk.samples);
        self.position_ms = chunk
            .offset_ms
            .saturating_add(chunk.duration_ms().ceil() as u64);
        for timeline in self.timelines.values_mut() {
            timeline.extend_to(DurationMs(self.position_ms));
        }
        if chunk.is_final {
            self.complete = true;
        }
        Some(Ok(chunk))
    }
}

/// [`Audio`] 文档校验失败的原因。
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AudioValidationError {
    #[error("audio id must not be empty")]
    EmptyAudioId,
    #[error("audio info sample rate must be greater than zero")]
    InvalidAudioInfoSampleRate,
    #[error("audio info channel count must be greater than zero")]
    InvalidAudioInfoChannels,
    #[error("audio channel {channel:?} is not canonical")]
    NonCanonicalChannel { channel: AudioChannel },
    #[error("timeline {channel:?} audio id mismatch: expected {expected:?}, found {found:?}")]
    TimelineAudioIdMismatch {
        channel: AudioChannel,
        expected: String,
        found: String,
    },
    #[error("timeline {channel:?} duration mismatch: expected {expected:?}, found {found:?}")]
    TimelineDurationMismatch {
        channel: AudioChannel,
        expected: DurationMs,
        found: DurationMs,
    },
    #[error(
        "annotation {annotation_id:?} on {channel:?} ends at {end:?}, past audio duration {duration:?}"
    )]
    AnnotationOutOfBounds {
        channel: AudioChannel,
        annotation_id: String,
        end: DurationMs,
        duration: DurationMs,
    },
    #[error("reference annotation {annotation_id:?} on {channel:?} must not have a source")]
    ReferenceAnnotationHasSource {
        channel: AudioChannel,
        annotation_id: String,
    },
    #[error("prediction annotation {annotation_id:?} on {channel:?} must have a non-empty source")]
    PredictionAnnotationMissingSource {
        channel: AudioChannel,
        annotation_id: String,
    },
    #[error("invalid annotations on {channel:?}: {error}")]
    InvalidAnnotations {
        channel: AudioChannel,
        error: TimelineSpanError,
    },
}

/// 使用了 `Channel(0)` / `Channel(1)` 而不是 `Left` / `Right`。
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("channel index {index} has a named representation")]
pub struct AudioChannelError {
    pub index: u16,
}

/// 创建或补齐 timeline 时的错误。
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum AudioTimelineError {
    #[error(transparent)]
    InvalidChannel(AudioChannelError),
    #[error("duration is required when creating the first timeline")]
    MissingDuration,
    #[error("timeline duration mismatch: expected {expected:?}, found {found:?}")]
    DurationMismatch {
        expected: DurationMs,
        found: DurationMs,
    },
}

/// 拒绝把 0/1 声道写成 [`AudioChannel::Channel`]，必须用 `Left` / `Right`。
fn validate_channel(channel: AudioChannel) -> Result<(), AudioChannelError> {
    match channel {
        AudioChannel::Channel(index @ 0..=1) => Err(AudioChannelError { index }),
        _ => Ok(()),
    }
}

/// 把文档 ID 收成 ASCII 标识：字母数字和 `-_`.` 保留，其余换成 `_`。
fn sanitize_audio_id(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}
