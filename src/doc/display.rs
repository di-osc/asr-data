use std::env;
use std::fmt;
use std::io::{self, IsTerminal};

use crate::audio::{AudioChannel, AudioEncoding, AudioSource, Waveform};
use crate::timeline::{Annotation, TimeSpan, Timeline, Token, Transcription};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::Audio;

const DEFAULT_WIDTH: usize = 80;
const MIN_WIDTH: usize = 56;
const MAX_WIDTH: usize = 120;
const LABEL_WIDTH: usize = 28;
const WAVE_LEVELS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

const RESET: &str = "\x1b[0m";
const BOLD_CYAN: &str = "\x1b[1;36m";
const DIM: &str = "\x1b[2m";
const GREEN: &str = "\x1b[32m";
const BLUE: &str = "\x1b[34m";
const YELLOW: &str = "\x1b[33m";

/// 读取 `COLUMNS`，夹在最小和最大终端宽度之间。
fn terminal_width() -> usize {
    env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_WIDTH)
        .clamp(MIN_WIDTH, MAX_WIDTH)
}

/// 标准输出是 TTY 且未设置 `NO_COLOR` 时启用 ANSI 颜色。
fn auto_color() -> bool {
    io::stdout().is_terminal() && env::var_os("NO_COLOR").is_none()
}

/// 终端里的紧凑 Audio 摘要：标题、波形和标注轨道。
pub(super) struct AudioTerminalView<'a> {
    audio: &'a Audio,
    width: usize,
    color: bool,
}

impl<'a> AudioTerminalView<'a> {
    /// 按 `COLUMNS` 和是否 TTY 自动选择宽度与颜色。
    pub(super) fn auto(audio: &'a Audio) -> Self {
        Self {
            audio,
            width: terminal_width(),
            color: auto_color(),
        }
    }

    /// 强制开关颜色，宽度仍随终端。
    pub(super) fn with_color(audio: &'a Audio, color: bool) -> Self {
        Self {
            audio,
            width: terminal_width(),
            color,
        }
    }

    /// 测试用固定宽度构造器。
    #[cfg(test)]
    fn new(audio: &'a Audio, width: usize, color: bool) -> Self {
        Self {
            audio,
            width: width.clamp(MIN_WIDTH, MAX_WIDTH),
            color,
        }
    }

    /// 画顶部标题栏：顶边居中放 `Audio · {id}`，框内是格式和来源。
    fn write_header(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let top = centered_title(self.width, &format!("Audio · {}", self.audio.id));
        writeln!(formatter, "{}", paint(BOLD_CYAN, &top, self.color))?;

        let info = format!(
            "{}  ·  {}  ·  {}  ·  {:.3} s  ·  {} frames",
            encoding_name(&self.audio.info.source_format.encoding),
            sample_rate_name(self.audio.info.sample_rate),
            channel_count_name(self.audio.info.channels),
            self.audio.info.timeline_duration_ms() as f64 / 1000.0,
            grouped_number(self.audio.info.frame_count),
        );
        write_card_line(formatter, self.width, &info)?;
        write_card_line(
            formatter,
            self.width,
            &format!("source: {}", source_name(&self.audio.source)),
        )?;
        write_card_bottom(formatter, self.width, self.color)
    }

    /// 按声道画出波形和标注轨道。
    fn write_timeline(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let plot_width = self.width.saturating_sub(LABEL_WIDTH);
        let duration_ms = self.audio.info.timeline_duration_ms();
        let ticks = timeline_ticks(duration_ms, plot_width);

        writeln!(formatter)?;
        let labels = time_labels(plot_width, &ticks);
        writeln!(
            formatter,
            "{}{}",
            " ".repeat(LABEL_WIDTH),
            paint(DIM, &labels, self.color)
        )?;
        let axis = time_axis(plot_width, &ticks);
        writeln!(
            formatter,
            "{}{}",
            " ".repeat(LABEL_WIDTH),
            paint(DIM, &axis, self.color)
        )?;

        for (channel, timeline) in &self.audio.timelines {
            let waveform = self
                .audio
                .waveform
                .as_ref()
                .map(|waveform| waveform_line(waveform, *channel, plot_width))
                .unwrap_or_else(|| "·".repeat(plot_width));
            writeln!(
                formatter,
                "{:<LABEL_WIDTH$}{}",
                channel_label(*channel),
                paint(GREEN, &waveform, self.color),
            )?;

            if !timeline.reference.is_empty() {
                write_annotation_groups(
                    formatter,
                    "Reference",
                    &timeline.reference,
                    duration_ms,
                    plot_width,
                    self.width,
                    BLUE,
                    self.color,
                )?;
            }
            if !timeline.prediction.is_empty() {
                write_annotation_groups(
                    formatter,
                    "Prediction",
                    &timeline.prediction,
                    duration_ms,
                    plot_width,
                    self.width,
                    YELLOW,
                    self.color,
                )?;
            }
        }

        let footer_label = self
            .audio
            .waveform
            .as_ref()
            .map_or("waveform unavailable".to_owned(), |waveform| {
                format!("{} samples", grouped_number(waveform.samples.len() as u64))
            });
        let footer = centered_rule(&footer_label, plot_width);
        write!(
            formatter,
            "{}{}",
            " ".repeat(LABEL_WIDTH),
            paint(DIM, &footer, self.color)
        )
    }
}

impl fmt::Display for AudioTerminalView<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_header(formatter)?;
        self.write_timeline(formatter)
    }
}

/// 终端里的紧凑 Waveform 摘要：格式卡片和分声道块状波形。
pub(crate) struct WaveformTerminalView<'a> {
    waveform: &'a Waveform,
    width: usize,
    color: bool,
}

impl<'a> WaveformTerminalView<'a> {
    /// 按 `COLUMNS` 和是否 TTY 自动选择宽度与颜色。
    pub(crate) fn auto(waveform: &'a Waveform) -> Self {
        Self {
            waveform,
            width: terminal_width(),
            color: auto_color(),
        }
    }

    /// 强制开关颜色，宽度仍随终端。
    pub(crate) fn with_color(waveform: &'a Waveform, color: bool) -> Self {
        Self {
            waveform,
            width: terminal_width(),
            color,
        }
    }

    /// 测试用固定宽度构造器。
    #[cfg(test)]
    fn new(waveform: &'a Waveform, width: usize, color: bool) -> Self {
        Self {
            waveform,
            width: width.clamp(MIN_WIDTH, MAX_WIDTH),
            color,
        }
    }

    /// 画顶部标题栏：顶边居中写 `Waveform`，框内是编码、采样率和时长。
    fn write_header(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let top = centered_title(self.width, "Waveform");
        writeln!(formatter, "{}", paint(BOLD_CYAN, &top, self.color))?;
        write_card_line(formatter, self.width, &waveform_info_line(self.waveform))?;
        write_card_bottom(formatter, self.width, self.color)
    }

    /// 按声道画出时间轴和块状波形，不带标注轨道。
    fn write_plot(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let plot_width = self.width.saturating_sub(LABEL_WIDTH);
        let duration_ms = self.waveform.duration_ms() as u64;
        let ticks = timeline_ticks(duration_ms, plot_width);

        writeln!(formatter)?;
        let labels = time_labels(plot_width, &ticks);
        writeln!(
            formatter,
            "{}{}",
            " ".repeat(LABEL_WIDTH),
            paint(DIM, &labels, self.color)
        )?;
        let axis = time_axis(plot_width, &ticks);
        writeln!(
            formatter,
            "{}{}",
            " ".repeat(LABEL_WIDTH),
            paint(DIM, &axis, self.color)
        )?;

        for channel in waveform_channels(self.waveform.channels) {
            let line = waveform_line(self.waveform, channel, plot_width);
            writeln!(
                formatter,
                "{:<LABEL_WIDTH$}{}",
                channel_label(channel),
                paint(GREEN, &line, self.color),
            )?;
        }

        let footer = centered_rule(
            &format!(
                "{} samples",
                grouped_number(self.waveform.samples.len() as u64)
            ),
            plot_width,
        );
        write!(
            formatter,
            "{}{}",
            " ".repeat(LABEL_WIDTH),
            paint(DIM, &footer, self.color)
        )
    }
}

impl fmt::Display for WaveformTerminalView<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_header(formatter)?;
        self.write_plot(formatter)
    }
}

/// 终端里的紧凑 Timeline 摘要：时长、标注计数和分轨详情。
pub(crate) struct TimelineTerminalView<'a> {
    timeline: &'a Timeline,
    width: usize,
    color: bool,
}

impl<'a> TimelineTerminalView<'a> {
    /// 按 `COLUMNS` 和是否 TTY 自动选择宽度与颜色。
    pub(crate) fn auto(timeline: &'a Timeline) -> Self {
        Self {
            timeline,
            width: terminal_width(),
            color: auto_color(),
        }
    }

    /// 强制开关颜色，宽度仍随终端。
    pub(crate) fn with_color(timeline: &'a Timeline, color: bool) -> Self {
        Self {
            timeline,
            width: terminal_width(),
            color,
        }
    }

    /// 测试用固定宽度构造器。
    #[cfg(test)]
    fn new(timeline: &'a Timeline, width: usize, color: bool) -> Self {
        Self {
            timeline,
            width: width.clamp(MIN_WIDTH, MAX_WIDTH),
            color,
        }
    }

    /// 画顶部标题栏：顶边居中放 `Timeline · {id}`，框内是时长和标注计数。
    fn write_header(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let top = centered_title(self.width, &format!("Timeline · {}", self.timeline.id));
        writeln!(formatter, "{}", paint(BOLD_CYAN, &top, self.color))?;
        write_card_line(formatter, self.width, &timeline_info_line(self.timeline))?;
        write_card_line(
            formatter,
            self.width,
            &format!("audio · {}", self.timeline.audio_id),
        )?;
        write_card_bottom(formatter, self.width, self.color)
    }

    /// 画时间轴和 Reference / Prediction 标注轨道，没有波形行。
    fn write_plot(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let plot_width = self.width.saturating_sub(LABEL_WIDTH);
        let duration_ms = self.timeline.duration.0;
        let ticks = timeline_ticks(duration_ms, plot_width);

        writeln!(formatter)?;
        let labels = time_labels(plot_width, &ticks);
        writeln!(
            formatter,
            "{}{}",
            " ".repeat(LABEL_WIDTH),
            paint(DIM, &labels, self.color)
        )?;
        let axis = time_axis(plot_width, &ticks);
        writeln!(
            formatter,
            "{}{}",
            " ".repeat(LABEL_WIDTH),
            paint(DIM, &axis, self.color)
        )?;

        if !self.timeline.reference.is_empty() {
            write_annotation_groups(
                formatter,
                "Reference",
                &self.timeline.reference,
                duration_ms,
                plot_width,
                self.width,
                BLUE,
                self.color,
            )?;
        }
        if !self.timeline.prediction.is_empty() {
            write_annotation_groups(
                formatter,
                "Prediction",
                &self.timeline.prediction,
                duration_ms,
                plot_width,
                self.width,
                YELLOW,
                self.color,
            )?;
        }

        let span_count = self.timeline.reference.len() + self.timeline.prediction.len();
        let footer_label = if span_count == 0 {
            "no annotations".to_owned()
        } else {
            format!("{} annotations", grouped_number(span_count as u64))
        };
        let footer = centered_rule(&footer_label, plot_width);
        write!(
            formatter,
            "{}{}",
            " ".repeat(LABEL_WIDTH),
            paint(DIM, &footer, self.color)
        )
    }
}

impl fmt::Display for TimelineTerminalView<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_header(formatter)?;
        self.write_plot(formatter)
    }
}

/// 卡片信息行：时长和参考 / 预测条数。
fn timeline_info_line(timeline: &Timeline) -> String {
    format!(
        "{:.3} s  ·  {} reference  ·  {} prediction",
        timeline.duration.0 as f64 / 1000.0,
        grouped_number(timeline.reference.len() as u64),
        grouped_number(timeline.prediction.len() as u64),
    )
}

/// 卡片信息行：编码来自 `source_format`，没有则写 `PCM`。
fn waveform_info_line(waveform: &Waveform) -> String {
    let encoding = waveform
        .source_format
        .as_ref()
        .map(|format| encoding_name(&format.encoding))
        .unwrap_or_else(|| "PCM".to_owned());
    format!(
        "{}  ·  {}  ·  {}  ·  {:.3} s  ·  {} frames",
        encoding,
        sample_rate_name(waveform.sample_rate),
        channel_count_name(waveform.channels),
        waveform.duration_seconds(),
        grouped_number(waveform.frame_count() as u64),
    )
}

/// 按当前 PCM 声道数生成终端标签：单声道用 `Mono`，否则按下标展开。
fn waveform_channels(channels: u16) -> Vec<AudioChannel> {
    if channels <= 1 {
        vec![AudioChannel::Mono]
    } else {
        (0..channels).map(AudioChannel::from_index).collect()
    }
}

/// 编码的短名称。
fn encoding_name(encoding: &AudioEncoding) -> String {
    match encoding {
        AudioEncoding::Wav => "WAV".to_owned(),
        AudioEncoding::Flac => "FLAC".to_owned(),
        AudioEncoding::Mp3 => "MP3".to_owned(),
        AudioEncoding::Ogg => "OGG".to_owned(),
        AudioEncoding::PcmS16Le => "PCM S16LE".to_owned(),
        AudioEncoding::Other(name) => name.to_uppercase(),
        AudioEncoding::Unknown => "Unknown".to_owned(),
    }
}

/// 采样率的人类可读文本。
fn sample_rate_name(sample_rate: u32) -> String {
    if sample_rate < 1_000 {
        return format!("{sample_rate} Hz");
    }
    if sample_rate.is_multiple_of(1_000) {
        format!("{} kHz", sample_rate / 1_000)
    } else {
        format!("{:.1} kHz", f64::from(sample_rate) / 1_000.0)
    }
}

/// 声道数的展示名。
fn channel_count_name(channels: u16) -> String {
    match channels {
        1 => "Mono".to_owned(),
        2 => "Stereo".to_owned(),
        count => format!("{count} channels"),
    }
}

/// 来源路径或 URL 的截断展示。
fn source_name(source: &AudioSource) -> String {
    match source {
        AudioSource::Path(path) => path.display().to_string(),
        AudioSource::Url(url) => {
            let without_query = url.split(['?', '#']).next().unwrap_or(url);
            let filename = without_query
                .rsplit('/')
                .next()
                .filter(|value| !value.is_empty());
            match (reqwest::Url::parse(without_query).ok(), filename) {
                (Some(parsed), Some(filename)) => {
                    format!("{}…/{}", parsed.host_str().unwrap_or("url"), filename)
                }
                _ => without_query.to_owned(),
            }
        }
        AudioSource::Base64(data) => format!("[base64 audio · {} characters]", data.len()),
        AudioSource::EncodedBytes(bytes) => {
            format!(
                "[encoded audio · {} bytes]",
                grouped_number(bytes.len() as u64)
            )
        }
        AudioSource::PcmS16Le {
            bytes,
            sample_rate,
            channels,
        } => format!(
            "[PCM S16LE · {} bytes · {} · {}]",
            grouped_number(bytes.len() as u64),
            sample_rate_name(*sample_rate),
            channel_count_name(*channels),
        ),
    }
}

/// 声道在终端中的标签。
fn channel_label(channel: AudioChannel) -> String {
    match channel {
        AudioChannel::Mono => "Mono".to_owned(),
        AudioChannel::Left => "Left".to_owned(),
        AudioChannel::Right => "Right".to_owned(),
        AudioChannel::Channel(index) => format!("Channel {index}"),
    }
}

/// 把峰值能量画成字符波形。
fn waveform_line(waveform: &Waveform, channel: AudioChannel, width: usize) -> String {
    let channels = usize::from(waveform.channels.max(1));
    let channel_index = usize::from(channel.index().unwrap_or(0)).min(channels - 1);
    let frames = waveform.samples.len() / channels;
    if frames == 0 || width == 0 {
        return "▁".repeat(width);
    }

    let mut peaks = Vec::with_capacity(width);
    for column in 0..width {
        let start = column * frames / width;
        let end = ((column + 1) * frames / width).max(start + 1).min(frames);
        let peak = (start..end)
            .map(|frame| waveform.samples[frame * channels + channel_index].abs())
            .fold(0.0_f32, f32::max);
        peaks.push(peak);
    }
    let max_peak = peaks.iter().copied().fold(0.0_f32, f32::max);
    peaks
        .into_iter()
        .map(|peak| {
            let level = if max_peak > 0.0 {
                ((peak / max_peak) * (WAVE_LEVELS.len() - 1) as f32).round() as usize
            } else {
                0
            };
            WAVE_LEVELS[level.min(WAVE_LEVELS.len() - 1)]
        })
        .collect()
}

/// 时间轴刻度。
fn timeline_ticks(duration_ms: u64, width: usize) -> Vec<(usize, u64)> {
    if duration_ms == 0 {
        return vec![(0, 0), (width.saturating_sub(1), 0)];
    }

    let target_step = (duration_ms / 4).max(1);
    let magnitude = 10_u64.pow(target_step.ilog10());
    let step = [1, 2, 5, 10]
        .into_iter()
        .map(|factor| magnitude.saturating_mul(factor))
        .take_while(|candidate| *candidate <= target_step)
        .last()
        .unwrap_or(magnitude);

    let mut ticks = Vec::new();
    let mut time_ms = 0_u64;
    while time_ms.saturating_add(step / 2) < duration_ms {
        let position = (u128::from(time_ms) * width.saturating_sub(1) as u128
            / u128::from(duration_ms)) as usize;
        ticks.push((position, time_ms));
        time_ms = time_ms.saturating_add(step);
    }
    ticks.push((width.saturating_sub(1), duration_ms));
    ticks
}

/// 毫秒时间轴标尺。
fn time_axis(width: usize, ticks: &[(usize, u64)]) -> String {
    let mut axis = vec!['─'; width];
    for (index, (position, _)) in ticks.iter().copied().enumerate() {
        axis[position] = match index {
            0 => '├',
            index if index + 1 == ticks.len() => '┤',
            _ => '┼',
        };
    }
    axis.into_iter().collect()
}

/// 刻度对应的时间文字。
fn time_labels(width: usize, ticks: &[(usize, u64)]) -> String {
    let mut labels = vec![' '; width];
    for (index, (position, time_ms)) in ticks.iter().copied().enumerate() {
        let label = format_time(time_ms);
        let start = if index == 0 {
            0
        } else if index + 1 == ticks.len() {
            width.saturating_sub(label.chars().count())
        } else {
            position.saturating_sub(label.chars().count() / 2)
        };
        overlay(&mut labels, start, &label);
    }
    labels.into_iter().collect()
}

/// 把毫秒格式化成 mm:ss 或 hh:mm:ss。
fn format_time(time_ms: u64) -> String {
    if time_ms.is_multiple_of(1_000) {
        format!("{}s", time_ms / 1_000)
    } else {
        let value = format!("{:.3}", time_ms as f64 / 1_000.0);
        format!("{}s", value.trim_end_matches('0').trim_end_matches('.'))
    }
}

/// 把一类标注画成一行轨道。
fn annotation_track(
    annotations: &[&TimeSpan],
    duration_ms: u64,
    width: usize,
    reference: bool,
) -> String {
    let mut track = vec![' '; width];
    let fill = if reference { '═' } else { '─' };
    let duration = duration_ms.max(1) as u128;

    for span in annotations {
        let start = ((u128::from(span.range.start.0) * width as u128) / duration)
            .min(width.saturating_sub(1) as u128) as usize;
        let end = ((u128::from(span.range.end.0) * width as u128).div_ceil(duration))
            .max((start + 2) as u128)
            .min(width as u128) as usize;
        track[start] = '╰';
        for cell in track.iter_mut().take(end.saturating_sub(1)).skip(start + 1) {
            *cell = fill;
        }
        if end > start + 1 {
            track[end - 1] = '╯';
        }
    }
    track.into_iter().collect()
}

/// 按类型分组输出标注方块。
fn write_annotation_groups(
    formatter: &mut fmt::Formatter<'_>,
    role: &str,
    annotations: &[TimeSpan],
    duration_ms: u64,
    plot_width: usize,
    width: usize,
    color: &str,
    colored: bool,
) -> fmt::Result {
    for (title, group) in annotation_groups(role, annotations) {
        let track = annotation_track(&group, duration_ms, plot_width, role == "Reference");
        let track_label = truncate(&title, LABEL_WIDTH);
        writeln!(
            formatter,
            "{track_label:<LABEL_WIDTH$}{}",
            paint(color, &track, colored),
        )?;
        write_annotation_box(formatter, &title, &group, width)?;
    }
    Ok(())
}

/// 按标注类型分组。
fn annotation_groups<'a>(
    role: &str,
    annotations: &'a [TimeSpan],
) -> Vec<(String, Vec<&'a TimeSpan>)> {
    let mut groups: Vec<(String, Vec<&TimeSpan>)> = Vec::new();
    for annotation in annotations {
        let source = annotation.source.as_deref().unwrap_or("reference");
        let title = if role == "Reference" {
            format!("{role} · {}", annotation_type_name(&annotation.annotation))
        } else {
            format!(
                "{role} · {} · {source}",
                annotation_type_name(&annotation.annotation)
            )
        };
        if let Some((_, values)) = groups.iter_mut().find(|(key, _)| key == &title) {
            values.push(annotation);
        } else {
            groups.push((title, vec![annotation]));
        }
    }
    groups
}

/// 标注类型的短名。
fn annotation_type_name(annotation: &Annotation) -> &'static str {
    match annotation {
        Annotation::Activity(_) => "Activity",
        Annotation::Token(_) => "Token",
        Annotation::Transcription(_) => "Transcription",
        Annotation::Sentence(_) => "Sentence",
        Annotation::Speaker(_) => "Speaker",
        Annotation::Language(_) => "Language",
    }
}

/// 画一个标注详情方块。
fn write_annotation_box(
    formatter: &mut fmt::Formatter<'_>,
    name: &str,
    annotations: &[&TimeSpan],
    width: usize,
) -> fmt::Result {
    let content_width = width.saturating_sub(4);
    let title = format!(" {name} ");
    let title_width = display_width(&title);
    writeln!(
        formatter,
        "╭─{}{}╮",
        title,
        "─".repeat(width.saturating_sub(3 + title_width))
    )?;
    for span in annotations {
        for line in annotation_detail_lines(span) {
            let value = truncate(&line, content_width);
            let padding = content_width.saturating_sub(display_width(&value));
            writeln!(formatter, "│ {value}{} │", " ".repeat(padding))?;
        }
    }
    writeln!(formatter, "╰{}╯", "─".repeat(width.saturating_sub(2)))?;
    Ok(())
}

/// 标注在轨道上的短标签。
fn annotation_label(span: &TimeSpan) -> String {
    let mut label = match &span.annotation {
        Annotation::Activity(activity) => activity
            .event
            .clone()
            .unwrap_or_else(|| "activity".to_owned()),
        Annotation::Token(token) => token.text.clone(),
        Annotation::Transcription(transcription) => format!("“{}”", transcription.text),
        Annotation::Sentence(sentence) => format!("“{}”", sentence.text),
        Annotation::Speaker(speaker) => speaker.name.clone(),
        Annotation::Language(language) => language.clone(),
    };
    if let Some(confidence) = span.annotation.confidence() {
        label.push_str(&format!(" {:.0}%", confidence * 100.0));
    }
    label
}

/// 标注详情盒里的多行内容：时间范围、文本，以及嵌套 token。
fn annotation_detail_lines(span: &TimeSpan) -> Vec<String> {
    let mut lines = vec![format!(
        "{}–{}  {}",
        format_time(span.range.start.0),
        format_time(span.range.end.0),
        annotation_label(span),
    )];
    match &span.annotation {
        Annotation::Speaker(speaker) => {
            if let Some(transcription) = &speaker.transcription {
                lines.push(format!("“{}”", transcription.text));
                push_token_line(&mut lines, transcription);
            }
        }
        Annotation::Transcription(transcription) => {
            push_token_line(&mut lines, transcription);
        }
        Annotation::Sentence(sentence) => {
            if !sentence.tokens.is_empty() {
                lines.push(join_token_texts(&sentence.tokens));
            }
        }
        _ => {}
    }
    lines
}

/// 有 token 时追加一行空格拼接的词序列。
fn push_token_line(lines: &mut Vec<String>, transcription: &Transcription) {
    if !transcription.tokens.is_empty() {
        lines.push(join_token_texts(&transcription.tokens));
    }
}

/// 把 token 文本拼成一行，便于在卡片里扫读。
fn join_token_texts(tokens: &[Token]) -> String {
    tokens
        .iter()
        .map(|token| token.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// 顶边把标题放在框线正中，例如 `╭──── Waveform ────╮`。
fn centered_title(width: usize, title: &str) -> String {
    let inner = width.saturating_sub(2);
    let title = truncate(title, inner.saturating_sub(2));
    let title = format!(" {title} ");
    let leftover = inner.saturating_sub(display_width(&title));
    let left = leftover / 2;
    let right = leftover - left;
    format!("╭{}{title}{}╮", "─".repeat(left), "─".repeat(right))
}

/// 输出卡片正文一行，左右用 `│` 包住。
fn write_card_line(formatter: &mut fmt::Formatter<'_>, width: usize, value: &str) -> fmt::Result {
    let content_width = width.saturating_sub(4);
    let value = truncate(value, content_width);
    let padding = content_width.saturating_sub(display_width(&value));
    writeln!(formatter, "│ {value}{} │", " ".repeat(padding))
}

/// 画青色卡片底边。
fn write_card_bottom(formatter: &mut fmt::Formatter<'_>, width: usize, color: bool) -> fmt::Result {
    let bottom = format!("╰{}╯", "─".repeat(width.saturating_sub(2)));
    writeln!(formatter, "{}", paint(BOLD_CYAN, &bottom, color))
}

/// 居中分隔线。
fn centered_rule(label: &str, width: usize) -> String {
    let label = format!(" {label} ");
    if label.chars().count() >= width {
        return truncate(&label, width);
    }
    let remaining = width - label.chars().count();
    let left = remaining / 2;
    format!(
        "{}{}{}",
        "─".repeat(left),
        label,
        "─".repeat(remaining - left)
    )
}

/// 千分位数字。
fn grouped_number(value: u64) -> String {
    let digits = value.to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(character);
    }
    grouped
}

/// 按显示宽度截断。
fn truncate(value: &str, width: usize) -> String {
    if display_width(value) <= width {
        return value.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    if width == 1 {
        return "…".to_owned();
    }
    let mut result = String::new();
    for character in value.chars() {
        let character_width = character.width().unwrap_or(0);
        if display_width(&result) + character_width + 1 > width {
            break;
        }
        result.push(character);
    }
    result.push('…');
    result
}

/// Unicode 显示宽度。
fn display_width(value: &str) -> usize {
    UnicodeWidthStr::width(value)
}

fn paint(style: &str, value: &str, color: bool) -> String {
    if color {
        format!("{style}{value}{RESET}")
    } else {
        value.to_owned()
    }
}

/// 把短字符串叠到背景字符上。
fn overlay(target: &mut [char], start: usize, value: &str) {
    for (index, character) in value.chars().enumerate() {
        if let Some(cell) = target.get_mut(start + index) {
            *cell = character;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::audio::{AudioEncoding, AudioFormat, AudioSource, Waveform};
    use crate::timeline::{AudioActivity, SpeakerPayload, Timeline, Token, Transcription};
    use crate::utils::DurationMs;

    use super::{Audio, AudioTerminalView, TimelineTerminalView, WaveformTerminalView};

    fn test_audio() -> Audio {
        let waveform = Waveform::new(vec![0.0, 0.25, -0.5, 1.0, -0.5, 0.25, 0.0, 0.0], 8)
            .with_source_format(AudioFormat {
                encoding: AudioEncoding::Wav,
                sample_rate: 8,
                channels: 1,
            });
        Audio::with_loaded_waveform("audio_test", AudioSource::from_path("test.wav"), waveform)
    }

    #[test]
    fn terminal_view_renders_compact_audio_summary_and_waveform() {
        let output = format!("{}", AudioTerminalView::new(&test_audio(), 64, false));

        assert!(output.contains(" Audio · audio_test "));
        assert!(!output.contains("╭─ Audio "));
        assert!(output.contains("WAV  ·  8 Hz  ·  Mono  ·  1.000 s  ·  8 frames"));
        assert!(output.contains("Mono       "));
        assert!(output.contains("8 samples"));
        assert!(!output.contains("\x1b["));
        assert!(!output.contains("0.25"));
    }

    #[test]
    fn terminal_view_renders_annotation_tracks() {
        let mut audio = test_audio();
        audio
            .mono_timeline_mut()
            .expect("mono timeline")
            .annotate_span(125, 750, AudioActivity::new().with_event("speech"))
            .expect("valid annotation");

        let output = format!("{}", AudioTerminalView::new(&audio, 64, false));

        assert!(output.contains("Reference · Activity"));
        assert!(output.contains("speech"));
        assert!(output.contains('═'));
    }

    #[test]
    fn terminal_view_adds_color_only_when_enabled() {
        let output = format!("{}", AudioTerminalView::new(&test_audio(), 64, true));

        assert!(output.contains("\x1b["));
    }

    fn test_waveform() -> Waveform {
        Waveform::new(vec![0.0, 0.25, -0.5, 1.0, -0.5, 0.25, 0.0, 0.0], 8).with_source_format(
            AudioFormat {
                encoding: AudioEncoding::Wav,
                sample_rate: 8,
                channels: 1,
            },
        )
    }

    #[test]
    fn waveform_terminal_view_renders_compact_summary_and_sparkline() {
        let output = format!("{}", WaveformTerminalView::new(&test_waveform(), 64, false));

        assert!(output.contains(" Waveform "));
        assert!(output.contains("WAV  ·  8 Hz  ·  Mono  ·  1.000 s  ·  8 frames"));
        assert!(output.contains("Mono       "));
        assert!(output.contains("8 samples"));
        assert!(!output.contains("source:"));
        assert!(!output.contains("\x1b["));
        assert!(!output.contains("0.25"));
    }

    #[test]
    fn waveform_terminal_view_uses_pcm_when_source_format_is_missing() {
        let waveform = Waveform::new(vec![0.0; 8], 8);
        let output = format!("{}", WaveformTerminalView::new(&waveform, 64, false));

        assert!(output.contains("PCM  ·  8 Hz  ·  Mono  ·  1.000 s  ·  8 frames"));
    }

    #[test]
    fn waveform_terminal_view_renders_stereo_channels() {
        let waveform = Waveform::new_with_channels(vec![0.0, 1.0, 0.5, -0.5, 0.25, -0.25], 3, 2);
        let output = format!("{}", WaveformTerminalView::new(&waveform, 64, false));

        assert!(output.contains("Stereo"));
        assert!(output.contains("Left       "));
        assert!(output.contains("Right      "));
        assert!(!output.contains("Mono       "));
    }

    #[test]
    fn waveform_terminal_view_adds_color_only_when_enabled() {
        let output = format!("{}", WaveformTerminalView::new(&test_waveform(), 64, true));

        assert!(output.contains("\x1b["));
    }

    fn test_timeline() -> Timeline {
        let mut timeline = Timeline::new("audio_test", DurationMs(1_000));
        timeline
            .annotate_span(
                0,
                350,
                AudioActivity::new()
                    .with_event("speech")
                    .with_confidence(0.9),
            )
            .expect("activity");
        timeline
            .annotate_span(
                0,
                350,
                SpeakerPayload::new("female0").with_confidence(0.9).with_transcription(
                    Transcription::new("甚至出现交易几乎停滞的情况。")
                        .with_tokens(vec![Token::new("甚"), Token::new("至")]),
                ),
            )
            .expect("speaker");
        timeline
    }

    #[test]
    fn timeline_terminal_view_renders_summary_and_annotation_tracks() {
        let output = format!("{}", TimelineTerminalView::new(&test_timeline(), 64, false));

        assert!(output.contains(" Timeline · "));
        assert!(output.contains("1.000 s  ·  2 reference  ·  0 prediction"));
        assert!(output.contains("audio · audio_test"));
        assert!(output.contains("Reference · Activity"));
        assert!(output.contains("speech"));
        assert!(output.contains("female0"));
        assert!(output.contains("甚至出现交易几乎停滞的情况。"));
        assert!(output.contains("甚 至"));
        assert!(output.contains("2 annotations"));
        assert!(!output.contains("\x1b["));
        assert!(!output.contains("source:"));
    }

    #[test]
    fn timeline_terminal_view_renders_empty_state() {
        let timeline = Timeline::new("audio_test", DurationMs(1_000));
        let output = format!("{}", TimelineTerminalView::new(&timeline, 64, false));

        assert!(output.contains("1.000 s  ·  0 reference  ·  0 prediction"));
        assert!(output.contains("no annotations"));
    }

    #[test]
    fn timeline_terminal_view_adds_color_only_when_enabled() {
        let output = format!("{}", TimelineTerminalView::new(&test_timeline(), 64, true));

        assert!(output.contains("\x1b["));
    }
}
