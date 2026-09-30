use std::env;
use std::fmt;
use std::io::{self, IsTerminal};

use crate::audio::{AudioChannel, AudioEncoding, AudioSource, Waveform};
use crate::timeline::{AudioEvent, Gender, Sentence, Speaker, Timeline, Token};
use crate::utils::TimeRange;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::Audio;

const DEFAULT_WIDTH: usize = 80;
const MIN_WIDTH: usize = 56;
const MAX_WIDTH: usize = 120;
const WAVE_LEVELS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

const RESET: &str = "\x1b[0m";
const CYAN: &str = "\x1b[36m";
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
        write_card_top(
            formatter,
            self.width,
            &format!("Audio · {}", self.audio.id),
            true,
            self.color,
        )?;

        let info = format!(
            "🎵  {}  ·  {}  ·  {}  ·  {:.3} s  ·  {} frames",
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
            &format!("🔗  source: {}", source_name(&self.audio.source)),
        )?;
        write_card_bottom(formatter, self.width, self.color)
    }

    /// 每个声道单独画成一张卡片：顶边居中放声道名，框内是刻度、波形和标注。
    ///
    /// 帧数已经写在上方摘要里。交错样本数在单声道时和帧数相同，不再另起一行。
    fn write_timeline(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let layout = PlotLayout {
            card_width: self.width,
            plot_width: self.width.saturating_sub(4),
            duration_ms: self.audio.info.timeline_duration_ms() as u64,
        };

        for (channel, timeline) in &self.audio.timelines {
            writeln!(formatter)?;
            write_card_top(
                formatter,
                layout.card_width,
                &channel_label(*channel),
                false,
                self.color,
            )?;
            // 波形在时间刻度上方，和 Jupyter 卡片的排布一致。
            let waveform = self
                .audio
                .waveform
                .as_ref()
                .map(|waveform| waveform_line(waveform, *channel, layout.plot_width))
                .unwrap_or_else(|| "·".repeat(layout.plot_width));
            write_framed_line(
                formatter,
                layout.card_width,
                &waveform,
                GREEN,
                self.color,
            )?;
            write_channel_scale(formatter, layout.duration_ms, layout.card_width, self.color)?;

            if !timeline.reference.is_empty() {
                write_annotation_groups(
                    formatter,
                    "Reference",
                    &timeline.reference,
                    layout,
                    BLUE,
                    self.color,
                )?;
            }
            if !timeline.prediction.is_empty() {
                write_annotation_groups(
                    formatter,
                    "Prediction",
                    &timeline.prediction,
                    layout,
                    YELLOW,
                    self.color,
                )?;
            }
            write_card_bottom(formatter, layout.card_width, self.color)?;
        }
        Ok(())
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
        write_card_top(formatter, self.width, "Waveform", true, self.color)?;
        write_card_line(formatter, self.width, &waveform_info_line(self.waveform))?;
        write_card_bottom(formatter, self.width, self.color)
    }

    /// 每个声道单独画成一张卡片：顶边居中放声道名，框内是刻度和块状波形。
    fn write_plot(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let card_width = self.width;
        let plot_width = card_width.saturating_sub(4);
        let duration_ms = self.waveform.duration_ms() as u64;

        for channel in waveform_channels(self.waveform.channels) {
            writeln!(formatter)?;
            write_card_top(
                formatter,
                card_width,
                &channel_label(channel),
                false,
                self.color,
            )?;
            // 波形在时间刻度上方，和 Jupyter 卡片的排布一致。
            let line = waveform_line(self.waveform, channel, plot_width);
            write_framed_line(formatter, card_width, &line, GREEN, self.color)?;
            write_channel_scale(formatter, duration_ms, card_width, self.color)?;
            write_card_bottom(formatter, card_width, self.color)?;
        }
        Ok(())
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
        write_card_top(
            formatter,
            self.width,
            &format!("Timeline · {}", self.timeline.id),
            true,
            self.color,
        )?;
        write_card_line(formatter, self.width, &timeline_info_line(self.timeline))?;
        write_card_line(
            formatter,
            self.width,
            &format!("🎵  audio · {}", self.timeline.audio_id),
        )?;
        write_card_bottom(formatter, self.width, self.color)
    }

    /// 把刻度和标注轨道收进一张卡片，左右和上方摘要对齐。
    fn write_plot(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let layout = PlotLayout {
            card_width: self.width,
            plot_width: self.width.saturating_sub(4),
            duration_ms: self.timeline.duration as u64,
        };

        writeln!(formatter)?;
        write_plain_card_top(formatter, layout.card_width, self.color)?;
        write_channel_scale(formatter, layout.duration_ms, layout.card_width, self.color)?;

        if !self.timeline.reference.is_empty() {
            write_annotation_groups(
                formatter,
                "Reference",
                &self.timeline.reference,
                layout,
                BLUE,
                self.color,
            )?;
        }
        if !self.timeline.prediction.is_empty() {
            write_annotation_groups(
                formatter,
                "Prediction",
                &self.timeline.prediction,
                layout,
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
        let footer = centered_text(&footer_label, layout.plot_width);
        write_framed_line(formatter, layout.card_width, &footer, DIM, self.color)?;
        write_card_bottom(formatter, layout.card_width, self.color)
    }
}

impl fmt::Display for TimelineTerminalView<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_header(formatter)?;
        self.write_plot(formatter)
    }
}

/// 卡片信息行：时长和参考 / 预测条数，各自带图标。
fn timeline_info_line(timeline: &Timeline) -> String {
    format!(
        "⏱  {:.3} s  ·  ◆ {} reference  ·  ◇ {} prediction",
        timeline.duration as f64 / 1000.0,
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
        "🎵  {}  ·  {}  ·  {}  ·  {:.3} s  ·  {} frames",
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
/// 来源行的展示：类型 + 值，例如 `url · host…/a.wav`。
fn source_name(source: &AudioSource) -> String {
    format!("{} · {}", source_type_name(source), source_value(source))
}

/// 来源类型名，标出这条来源是路径、URL 还是别的形式。
fn source_type_name(source: &AudioSource) -> &'static str {
    match source {
        AudioSource::Path(_) => "path",
        AudioSource::Url(_) => "url",
        AudioSource::Base64(_) => "base64",
        AudioSource::EncodedBytes(_) => "encoded",
        AudioSource::PcmS16Le { .. } => "pcm s16le",
        AudioSource::ModelScope { .. } => "modelscope",
    }
}

/// 来源的值部分（不含类型），过长时会截断展示。
fn source_value(source: &AudioSource) -> String {
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
        AudioSource::Base64(data) => {
            format!("{} characters", grouped_number(data.len() as u64))
        }
        AudioSource::EncodedBytes(bytes) => {
            format!("{} bytes", grouped_number(bytes.len() as u64))
        }
        AudioSource::PcmS16Le {
            bytes,
            sample_rate,
            channels,
        } => format!(
            "{} bytes · {} · {}",
            grouped_number(bytes.len() as u64),
            sample_rate_name(*sample_rate),
            channel_count_name(*channels),
        ),
        AudioSource::ModelScope {
            repo_id,
            file_path,
            revision,
        } => format!("{repo_id}@{revision}:{file_path}"),
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
    if width == 0 {
        return String::new();
    }
    waveform_levels(waveform, channel, width)
        .into_iter()
        .map(|level| WAVE_LEVELS[level])
        .collect()
}

/// 每个显示列的峰值等级，范围是 `0..WAVE_LEVELS.len()`。
fn waveform_levels(waveform: &Waveform, channel: AudioChannel, width: usize) -> Vec<usize> {
    let channels = usize::from(waveform.channels.max(1));
    let channel_index = usize::from(channel.index().unwrap_or(0)).min(channels - 1);
    let frames = waveform.samples.len() / channels;
    if frames == 0 || width == 0 {
        return vec![0; width];
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
            level.min(WAVE_LEVELS.len() - 1)
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

/// 在卡片内画出时间文字和标尺。标尺宽度是卡片内容宽，左右由边框对齐。
fn write_channel_scale(
    formatter: &mut fmt::Formatter<'_>,
    duration_ms: u64,
    card_width: usize,
    color: bool,
) -> fmt::Result {
    let plot_width = card_width.saturating_sub(4);
    let ticks = timeline_ticks(duration_ms, plot_width);
    write_framed_line(
        formatter,
        card_width,
        &time_labels(plot_width, &ticks),
        DIM,
        color,
    )?;
    write_framed_line(
        formatter,
        card_width,
        &time_axis(plot_width, &ticks),
        DIM,
        color,
    )
}

/// 在给定宽度内居中一行文字，左右用空格补齐。
fn centered_text(value: &str, width: usize) -> String {
    let text_width = display_width(value);
    if text_width >= width {
        return truncate(value, width);
    }
    let left = (width - text_width) / 2;
    let right = width - text_width - left;
    format!("{}{value}{}", " ".repeat(left), " ".repeat(right))
}

/// 把时间范围映射到 `width` 列的半开区间 `[start, end)`。
///
/// 起点向下取整、终点向上取整，保证非空范围至少覆盖一列；同一时间点在不同轨道
/// 上落在同一列，这样多条轨道可以上下对齐。
fn time_columns(range: TimeRange, duration_ms: u64, width: usize) -> (usize, usize) {
    if width == 0 {
        return (0, 0);
    }
    let duration = duration_ms.max(1) as u128;
    let width = width as u128;
    let start = ((range.start_ms as u128 * width) / duration).min(width - 1) as usize;
    let end = ((range.end_ms as u128 * width).div_ceil(duration))
        .max((start + 1) as u128)
        .min(width) as usize;
    (start, end)
}

/// 画一条按时间定位的轨道，可选标签居中写在这段范围里。
///
/// `fill` 是轨道填充字符。标签两边补空格和轨道线分开；范围比标签窄时，标签向
/// 两侧空白延伸，仍尽量留在行内。
fn span_track(
    range: TimeRange,
    duration_ms: u64,
    width: usize,
    fill: &str,
    label: &str,
) -> String {
    if width == 0 {
        return String::new();
    }
    let mut columns = vec![" ".to_owned(); width];
    let (start, end) = time_columns(range, duration_ms, width);
    columns[start] = "╰".to_owned();
    for cell in columns.iter_mut().take(end - 1).skip(start + 1) {
        *cell = fill.to_owned();
    }
    if end > start + 1 {
        columns[end - 1] = "╯".to_owned();
    }
    let padded = format!(" {label} ");
    let fitted = if display_width(&padded) <= width {
        padded
    } else {
        truncate(&padded, width)
    };
    let fitted_width = display_width(&fitted);
    if fitted_width > 0 && fitted_width <= width {
        let center = start + end.saturating_sub(start) / 2;
        let origin = center
            .saturating_sub(fitted_width / 2)
            .min(width.saturating_sub(fitted_width));
        place_label(&mut columns, origin, &fitted);
    }
    columns.concat()
}

/// 画一条事件轨道：起止时间贴在两端，事件名居中在中间。
///
/// 事件太短、中间放不下名字时，退回只画居中的名字（不再显示两端时间），
/// 让名字仍能溢出到轨道两侧的空白里。
fn event_track(
    range: TimeRange,
    duration_ms: u64,
    width: usize,
    fill: &str,
    heading: &str,
) -> String {
    if width == 0 {
        return String::new();
    }
    let mut columns = vec![" ".to_owned(); width];
    let (start, end) = time_columns(range, duration_ms, width);
    // 端点用 `|`，和主时间轴的 `─` 区分开，多个事件时也更好分辨。
    columns[start] = "|".to_owned();
    for cell in columns.iter_mut().take(end - 1).skip(start + 1) {
        *cell = fill.to_owned();
    }
    if end > start + 1 {
        columns[end - 1] = "|".to_owned();
    }

    let padded = format!(" {heading} ");
    let fitted = if display_width(&padded) <= width {
        padded
    } else {
        truncate(&padded, width)
    };
    let fitted_width = display_width(&fitted);

    // 两端时间占用的列，剩给中间名字的区间。
    let start_label = format_time(range.start_ms as u64);
    let end_label = format_time(range.end_ms as u64);
    let center_start = start + 1 + display_width(&start_label);
    let end_origin = end
        .saturating_sub(1)
        .saturating_sub(display_width(&end_label));
    let middle = end_origin.saturating_sub(center_start);

    if fitted_width > 0 && fitted_width <= middle {
        // 放得下：起止时间贴两端，名字居中在中间。
        place_label(&mut columns, start + 1, &start_label);
        place_label(&mut columns, end_origin, &end_label);
        let origin = center_start + (middle - fitted_width) / 2;
        place_label(&mut columns, origin, &fitted);
    } else if fitted_width > 0 && fitted_width <= width {
        // 太窄：只放名字，居中在整条上，可溢出到空白。
        let center = start + end.saturating_sub(start) / 2;
        let origin = center
            .saturating_sub(fitted_width / 2)
            .min(width.saturating_sub(fitted_width));
        place_label(&mut columns, origin, &fitted);
    }
    columns.concat()
}

/// 句子的时间轨道，标签写这句的时间范围。
fn sentence_track(sentence: &Sentence, layout: PlotLayout) -> String {
    span_track(
        sentence.range,
        layout.duration_ms,
        layout.plot_width,
        "─",
        &sentence_range_label(sentence),
    )
}

/// 句子的时间范围标签，例如 `⏱  0s–4.204s`。
fn sentence_range_label(sentence: &Sentence) -> String {
    let range = sentence.range;
    format!(
        "⏱  {}–{}",
        format_time(range.start_ms as u64),
        format_time(range.end_ms as u64),
    )
}

/// token 时间轨道：每个 token 占自己的时间段，文本嵌在段里。
///
/// 返回 `None` 表示所有 token 都没有时间范围，无法定位。
fn token_track(tokens: &[Token], layout: PlotLayout) -> Option<String> {
    if tokens.iter().all(|token| token.range.is_none()) {
        return None;
    }
    let width = layout.plot_width;
    if width == 0 {
        return None;
    }
    let mut columns = vec![" ".to_owned(); width];
    let mut cursor = 0usize;
    for token in tokens {
        let Some(range) = token.range else {
            continue;
        };
        let (start, end) = time_columns(range, layout.duration_ms, width);
        // 段之间不重叠：下一段从上一段结束后开始。
        let start = start.max(cursor).min(width - 1);
        let end = end.max(start + 1).min(width);
        cursor = end;
        let span = end - start;
        if span >= 2 {
            columns[start] = "╰".to_owned();
            columns[end - 1] = "╯".to_owned();
            let inner = span - 2;
            let text_width = display_width(&token.text);
            if text_width > 0 && text_width <= inner {
                let padding = (inner - text_width) / 2;
                place_label(&mut columns, start + 1 + padding, &token.text);
            }
        } else {
            columns[start] = "╎".to_owned();
        }
    }
    Some(columns.concat())
}

/// 从 `origin` 列起写入标签。
///
/// 宽字符占两列，零宽字符跟在前一个字符后面，这样整行的显示宽度不变。
fn place_label(columns: &mut [String], origin: usize, label: &str) {
    let mut column = origin;
    let mut zero_width = String::new();
    for character in label.chars() {
        let width = character.width().unwrap_or(0);
        if width == 0 {
            zero_width.push(character);
            continue;
        }
        if column + width > columns.len() {
            break;
        }
        let mut cell = String::new();
        cell.push(character);
        cell.push_str(&zero_width);
        zero_width.clear();
        columns[column] = cell;
        for extra in 1..width {
            columns[column + extra].clear();
        }
        column += width;
    }
    if !zero_width.is_empty() && column > origin {
        columns[column - 1].push_str(&zero_width);
    }
}

/// 一张声道卡片绘图区的固定几何参数。
///
/// 事件轨道、大小和详情都用同一套宽度与时长对齐，打包成一个结构，避免在
/// 各绘制函数之间反复传相同的一组参数。
#[derive(Clone, Copy)]
struct PlotLayout {
    /// 卡片内容宽度，时间刻度和事件条都按它对齐。
    card_width: usize,
    /// 事件轨道宽度，等于 `card_width` 减去左右留白。
    plot_width: usize,
    /// 时间轴总时长，单位毫秒。
    duration_ms: u64,
}

/// 把同一角色的标注画成一棵树。角色写在事件条里，详情挂在事件条下面。
fn write_annotation_groups(
    formatter: &mut fmt::Formatter<'_>,
    role: &str,
    annotations: &[AudioEvent],
    layout: PlotLayout,
    color: &str,
    colored: bool,
) -> fmt::Result {
    for (title, group) in annotation_tracks(role, annotations) {
        for (index, event) in group.iter().enumerate() {
            let last = index + 1 == group.len();
            write_event_row(
                formatter,
                event,
                layout,
                role == "Reference",
                &title,
                color,
                colored,
            )?;
            // 非最后一条事件的子树要留出竖线，接到下一个兄弟节点。
            let stem = if last { "   " } else { "│  " };
            let rows = event_detail_rows(event, stem, layout);
            write_detail_rows(formatter, &rows, layout.card_width, color, colored)?;
        }
    }
    Ok(())
}

/// 参考合成一条轨道；预测按 source 拆开，source 写在轨道名后面。
fn annotation_tracks<'a>(
    role: &str,
    annotations: &'a [AudioEvent],
) -> Vec<(String, Vec<&'a AudioEvent>)> {
    if role == "Reference" {
        return vec![("Reference".to_owned(), annotations.iter().collect())];
    }

    let mut groups: Vec<(String, Vec<&AudioEvent>)> = Vec::new();
    for event in annotations {
        let title = match event.source() {
            Some(source) if !source.is_empty() => format!("Prediction · {source}"),
            _ => "Prediction".to_owned(),
        };
        if let Some((_, values)) = groups.iter_mut().find(|(key, _)| key == &title) {
            values.push(event);
        } else {
            groups.push((title, vec![event]));
        }
    }
    groups
}

/// 画出事件的时间轨道，事件名和角色居中写在这段范围里。
///
/// `role` 是轨道所属的角色，例如 `Reference` 或 `Prediction · whisper`，和
/// Jupyter 卡片上的 `🎙 Speech · Reference` 写法一致。
fn write_event_row(
    formatter: &mut fmt::Formatter<'_>,
    event: &AudioEvent,
    layout: PlotLayout,
    reference: bool,
    role: &str,
    color: &str,
    colored: bool,
) -> fmt::Result {
    let heading = format!("{role} · {} {}", event_icon(event), event_heading(event));
    let fill = if reference { "═" } else { "─" };
    let track = event_track(
        event.range(),
        layout.duration_ms,
        layout.plot_width,
        fill,
        &heading,
    );
    write_framed_line(formatter, layout.card_width, &track, color, colored)
}

/// 按事件类型选图标。语音固定用麦克风，常见基类事件用各自的符号。
fn event_icon(event: &AudioEvent) -> &'static str {
    if event.is_speech() {
        return "🎙";
    }
    match event.name() {
        "music" => "🎵",
        "noise" => "🔊",
        "silence" => "🔇",
        _ => "●",
    }
}

/// 事件标题。语音显示派生类名，基类事件保留原始名字。
fn event_heading(event: &AudioEvent) -> String {
    if event.is_speech() {
        "Speech".to_owned()
    } else {
        event.name().to_owned()
    }
}

/// 事件详情里的一行：树形文本，或一条按时间定位的轨道。
enum DetailRow {
    /// 带树形前缀（含 `├─` / `└─`）的文本行。
    Text { prefix: String, text: String },
    /// 已经按卡片内容宽度排好的时间轨道。
    Track(String),
}

/// 一条句子轨道最多逐个展开的 token 数；超过就只给摘要，避免糊成一片。
const MAX_TOKEN_TRACK: usize = 12;

/// 事件的详情行：说话人、语种，以及每句转写的文本和一条时间轨道。
///
/// 每句在文本下面画一条轨道，轨道和事件条共用同一套列坐标，所以上下能按时间
/// 对齐；层级由文本行的缩进体现。句内 token 少时逐个嵌字，多时收敛成摘要。
fn event_detail_rows(event: &AudioEvent, stem: &str, layout: PlotLayout) -> Vec<DetailRow> {
    let speaker = event.speaker();
    let language = event.language().map(|language| format!("🌐  {language}"));
    let transcription = event.transcription();
    let sentences: Vec<&Sentence> = transcription
        .map(|transcription| {
            transcription
                .sentences
                .iter()
                .filter(|sentence| !sentence.text.is_empty() || !sentence.tokens.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let full_text = transcription
        .filter(|transcription| transcription.sentences.is_empty())
        .map(|transcription| transcription.text.clone())
        .filter(|text| !text.trim().is_empty());

    let top_count = usize::from(speaker.is_some())
        + usize::from(language.is_some())
        + usize::from(full_text.is_some())
        + sentences.len();

    let mut rows = Vec::new();
    let mut position = 0usize;
    let push_top = |rows: &mut Vec<DetailRow>, position: &mut usize, text: String| {
        *position += 1;
        rows.push(detail_text_row(stem, *position == top_count, text));
    };

    if let Some(speaker) = speaker {
        position += 1;
        let last = position == top_count;
        rows.push(detail_text_row(stem, last, speaker_name_line(speaker)));
    }
    if let Some(text) = language {
        push_top(&mut rows, &mut position, text);
    }
    if let Some(text) = full_text {
        push_top(&mut rows, &mut position, format!("💬  {text}"));
    }
    let sentence_count = sentences.len();
    for sentence in sentences {
        position += 1;
        let last = position == top_count;
        let child_stem = format!("{stem}{}", if last { "   " } else { "│  " });
        let kind = sentence_track_kind(sentence, sentence_count, layout);
        let text = match &kind {
            SentenceTrack::Counts { tokens, sentences } => format!(
                "💬  {} ({tokens} tokens, {sentences} sentences)",
                sentence.text
            ),
            _ => format!("💬  {}", sentence.text),
        };
        rows.push(detail_text_row(stem, last, text));
        match kind {
            SentenceTrack::Tokens(track) | SentenceTrack::TimeTrack(track) => {
                rows.push(DetailRow::Track(track));
            }
            // token 没有时间范围时无法定位，退回一行文本，避免信息丢失。
            SentenceTrack::Text(text) => rows.push(DetailRow::Text {
                prefix: format!("{child_stem}└─ "),
                text,
            }),
            // 计数已经并进句子文本，不再单独画轨道。
            SentenceTrack::Counts { .. } => {}
        }
    }
    rows
}

/// 一句转写下面那条轨道的形态。
enum SentenceTrack {
    /// 逐个 token 嵌字的时间轨道。
    Tokens(String),
    /// 只画句子时间范围的轨道。
    TimeTrack(String),
    /// 退化到一行文本（token 没有时间范围时）。
    Text(String),
    /// token 太多，不画轨道，只把计数并入句子文本。
    Counts { tokens: usize, sentences: usize },
}

/// 选出句子轨道：没有 token 时画句子范围条，token 少时逐个嵌字，太多时只报计数。
fn sentence_track_kind(
    sentence: &Sentence,
    sentence_count: usize,
    layout: PlotLayout,
) -> SentenceTrack {
    if sentence.tokens.is_empty() {
        return SentenceTrack::TimeTrack(sentence_track(sentence, layout));
    }
    if sentence.tokens.len() <= MAX_TOKEN_TRACK {
        if let Some(track) = token_track(&sentence.tokens, layout) {
            return SentenceTrack::Tokens(track);
        }
        return SentenceTrack::Text(format!("🔤  {}", join_token_texts(&sentence.tokens)));
    }
    SentenceTrack::Counts {
        tokens: sentence.tokens.len(),
        sentences: sentence_count,
    }
}

/// 一行树形文本详情，`last` 决定用 `└─` 还是 `├─`。
fn detail_text_row(stem: &str, last: bool, text: String) -> DetailRow {
    let branch = if last { "└─ " } else { "├─ " };
    DetailRow::Text {
        prefix: format!("{stem}{branch}"),
        text,
    }
}

/// 把 token 文本拼成一行，用间隔号分开，用于没有时间范围的退化展示。
fn join_token_texts(tokens: &[Token]) -> String {
    tokens
        .iter()
        .map(|token| token.text.as_str())
        .collect::<Vec<_>>()
        .join(" · ")
}

/// 写事件详情：文本行按树形缩进，轨道行整行铺开、和事件条共用时间坐标。
fn write_detail_rows(
    formatter: &mut fmt::Formatter<'_>,
    rows: &[DetailRow],
    card_width: usize,
    color: &str,
    colored: bool,
) -> fmt::Result {
    for row in rows {
        match row {
            DetailRow::Text { prefix, text } => {
                write_framed_segments(formatter, card_width, &[(DIM, prefix), ("", text)], colored)?;
            }
            DetailRow::Track(track) => {
                write_framed_line(formatter, card_width, track, color, colored)?;
            }
        }
    }
    Ok(())
}

/// 说话人名字行。女性、男性和未知各用不同头像；性别放在名字后面的括号里。
fn speaker_name_line(speaker: &Speaker) -> String {
    let icon = match speaker.gender {
        Some(Gender::Female) => "👩",
        Some(Gender::Male) => "👨",
        Some(Gender::Unknown) | None => "🧑",
    };
    match speaker.gender {
        Some(gender) => format!("{icon}  {} ({})", speaker.name, gender_text(gender)),
        None => format!("{icon}  {}", speaker.name),
    }
}

/// 性别的展示文本，单独一行展示时用。
fn gender_text(gender: Gender) -> &'static str {
    match gender {
        Gender::Female => "female",
        Gender::Male => "male",
        Gender::Unknown => "unknown",
    }
}

/// 顶边把标题放在框线正中，例如 `╭──── Waveform ────╮`。
///
/// `bold_title` 只加粗标题文字。整行加粗时，笔记本里的粗体字宽会大于正文，底边会伸出去。
/// 声道名不加粗，避免短标题把这一行撑宽。
fn write_card_top(
    formatter: &mut fmt::Formatter<'_>,
    width: usize,
    title: &str,
    bold_title: bool,
    color: bool,
) -> fmt::Result {
    let inner = width.saturating_sub(2);
    let title = truncate(title, inner.saturating_sub(2));
    let title = format!(" {title} ");
    let leftover = inner.saturating_sub(display_width(&title));
    let left = leftover / 2;
    let right = leftover - left;
    let title_style = if bold_title { BOLD_CYAN } else { CYAN };
    writeln!(
        formatter,
        "{}{}{}",
        paint(CYAN, &format!("╭{}", "─".repeat(left)), color),
        paint(title_style, &title, color),
        paint(CYAN, &format!("{}╮", "─".repeat(right)), color),
    )
}

/// 没有标题的顶边，给时间轴卡片用。
fn write_plain_card_top(
    formatter: &mut fmt::Formatter<'_>,
    width: usize,
    color: bool,
) -> fmt::Result {
    let top = format!("╭{}╮", "─".repeat(width.saturating_sub(2)));
    writeln!(formatter, "{}", paint(CYAN, &top, color))
}

/// 在卡片内写一行已经按内容宽度排好的文字，左右用竖线包住。
fn write_framed_line(
    formatter: &mut fmt::Formatter<'_>,
    card_width: usize,
    value: &str,
    style: &str,
    color: bool,
) -> fmt::Result {
    write_framed_segments(formatter, card_width, &[(style, value)], color)
}

/// 在卡片内写一行分段上色的文字。超出内容宽度时截断并加省略号。
fn write_framed_segments(
    formatter: &mut fmt::Formatter<'_>,
    card_width: usize,
    segments: &[(&str, &str)],
    color: bool,
) -> fmt::Result {
    let content_width = card_width.saturating_sub(4);
    let mut rendered = String::new();
    let mut used = 0usize;
    for (style, text) in segments {
        if used >= content_width {
            break;
        }
        let room = content_width - used;
        let piece = if display_width(text) <= room {
            (*text).to_owned()
        } else {
            truncate(text, room)
        };
        let piece_width = display_width(&piece);
        used += piece_width;
        rendered.push_str(&paint(style, &piece, color));
        if piece_width < display_width(text) {
            break;
        }
    }
    let padding = content_width.saturating_sub(used);
    let border = paint(CYAN, "│", color);
    writeln!(
        formatter,
        "{border} {rendered}{} {border}",
        " ".repeat(padding)
    )
}

/// 输出卡片正文一行，左右用 `│` 包住。
fn write_card_line(formatter: &mut fmt::Formatter<'_>, width: usize, value: &str) -> fmt::Result {
    let content_width = width.saturating_sub(4);
    let value = truncate(value, content_width);
    let padding = content_width.saturating_sub(display_width(&value));
    writeln!(formatter, "│ {value}{} │", " ".repeat(padding))
}

/// 画青色卡片底边。不加粗，避免和正文的字宽不一致。
fn write_card_bottom(formatter: &mut fmt::Formatter<'_>, width: usize, color: bool) -> fmt::Result {
    let bottom = format!("╰{}╯", "─".repeat(width.saturating_sub(2)));
    writeln!(formatter, "{}", paint(CYAN, &bottom, color))
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
    if color && !style.is_empty() {
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

#[cfg(feature = "python-bindings")]
const NOTEBOOK_BARS: usize = 64;
#[cfg(feature = "python-bindings")]
const NOTEBOOK_BORDER: &str = "#67e8f9";

#[cfg(feature = "python-bindings")]
impl Audio {
    /// Jupyter 卡片。边框用 CSS 画，波形用色条，不依赖等宽字符对齐。
    pub(crate) fn notebook_html(&self) -> String {
        let header = notebook_card(
            &format!("Audio · {}", self.id),
            true,
            &format!(
                "<div>{}</div><div style=\"overflow-wrap:anywhere;\">🔗  source: {}</div>",
                html_escape(&format!(
                    "🎵  {}  ·  {}  ·  {}  ·  {:.3} s  ·  {} frames",
                    encoding_name(&self.info.source_format.encoding),
                    sample_rate_name(self.info.sample_rate),
                    channel_count_name(self.info.channels),
                    self.info.timeline_duration_ms() as f64 / 1000.0,
                    grouped_number(self.info.frame_count),
                )),
                html_escape(&source_name(&self.source)),
            ),
        );
        let duration_ms = self.info.timeline_duration_ms() as u64;
        let mut channels = String::new();
        for (channel, timeline) in &self.timelines {
            // 波形在主时间轴上面，标注画在刻度下面。
            let mut body = String::new();
            if let Some(waveform) = &self.waveform {
                body.push_str(&waveform_bars_html(waveform, *channel));
            }
            body.push_str(&time_scale_html(duration_ms));
            body.push_str(&annotation_groups_html(
                "Reference",
                &timeline.reference,
                duration_ms,
                "#93c5fd",
            ));
            body.push_str(&annotation_groups_html(
                "Prediction",
                &timeline.prediction,
                duration_ms,
                "#fde68a",
            ));
            channels.push_str(&notebook_card(&channel_label(*channel), false, &body));
        }
        notebook_page(&format!("{header}{channels}"))
    }
}

#[cfg(feature = "python-bindings")]
impl Waveform {
    /// Jupyter 卡片。每个声道一条色条波形，边框不靠字符宽度对齐。
    pub(crate) fn notebook_html(&self) -> String {
        let header = notebook_card(
            "Waveform",
            true,
            &format!("<div>{}</div>", html_escape(&waveform_info_line(self))),
        );
        let mut channels = String::new();
        let duration_ms = self.duration_ms() as u64;
        for channel in waveform_channels(self.channels) {
            let body = format!(
                "{}{}",
                waveform_bars_html(self, channel),
                time_scale_html(duration_ms)
            );
            channels.push_str(&notebook_card(&channel_label(channel), false, &body));
        }
        notebook_page(&format!("{header}{channels}"))
    }
}

#[cfg(feature = "python-bindings")]
impl Timeline {
    /// Jupyter 卡片。标注和时间范围用 CSS 定位，右边框不会随字宽错位。
    pub(crate) fn notebook_html(&self) -> String {
        let header = notebook_card(
            &format!("Timeline · {}", self.id),
            true,
            &format!(
                "<div>{}</div><div>🎵  audio · {}</div>",
                html_escape(&timeline_info_line(self)),
                html_escape(&self.audio_id),
            ),
        );
        let duration_ms = self.duration as u64;
        let mut body = time_scale_html(duration_ms);
        body.push_str(&annotation_groups_html(
            "Reference",
            &self.reference,
            duration_ms,
            "#93c5fd",
        ));
        body.push_str(&annotation_groups_html(
            "Prediction",
            &self.prediction,
            duration_ms,
            "#fde68a",
        ));
        let span_count = self.reference.len() + self.prediction.len();
        let footer = if span_count == 0 {
            "no annotations".to_owned()
        } else {
            format!("{} annotations", grouped_number(span_count as u64))
        };
        body.push_str(&format!(
            "<div style=\"text-align:center;color:#94a3b8;margin-top:6px;\">{}</div>",
            html_escape(&footer),
        ));
        notebook_page(&format!("{header}{}", notebook_card("", false, &body)))
    }
}

/// 笔记本外壳：深色底，内容在里面换行，不使用字符边框。
#[cfg(feature = "python-bindings")]
fn notebook_page(body: &str) -> String {
    format!(
        "<div style=\"font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#d4d4d4;background:#1e1e1e;padding:12px 12px 4px;border-radius:8px;line-height:1.45;\">{body}</div>"
    )
}

/// 一张圆角卡片。标题压在顶边上，边框是一条 CSS 线。
#[cfg(feature = "python-bindings")]
fn notebook_card(title: &str, bold: bool, body: &str) -> String {
    let weight = if bold { "700" } else { "400" };
    let title_html = if title.is_empty() {
        String::new()
    } else {
        format!(
            "<div style=\"color:{NOTEBOOK_BORDER};font-weight:{weight};text-align:center;margin:-18px auto 8px;background:#1e1e1e;width:fit-content;padding:0 8px;\">{}</div>",
            html_escape(title)
        )
    };
    format!(
        "<div style=\"border:1px solid {NOTEBOOK_BORDER};border-radius:8px;padding:10px 12px 8px;margin-bottom:12px;\">{title_html}{body}</div>"
    )
}

/// 时间刻度：两端和中间的时间文字，下面一条横线。
#[cfg(feature = "python-bindings")]
fn time_scale_html(duration_ms: u64) -> String {
    let ticks = timeline_ticks(duration_ms, 100);
    let mut labels = String::from(
        "<div style=\"display:flex;justify-content:space-between;color:#94a3b8;font-size:12px;\">",
    );
    for (_, time_ms) in ticks {
        labels.push_str(&format!(
            "<span>{}</span>",
            html_escape(&format_time(time_ms))
        ));
    }
    labels.push_str("</div><div style=\"height:1px;background:#64748b;margin:2px 0 8px;\"></div>");
    labels
}

/// 用一排绿色竖条画波形。高度用像素，避免百分比在 flex 里被算成 0。
#[cfg(feature = "python-bindings")]
fn waveform_bars_html(waveform: &Waveform, channel: AudioChannel) -> String {
    let mut html = String::from(
        "<div style=\"display:flex;align-items:flex-end;height:36px;gap:1px;margin:4px 0 8px;\">",
    );
    for level in waveform_levels(waveform, channel, NOTEBOOK_BARS) {
        let height = 4 + level * 32 / (WAVE_LEVELS.len() - 1);
        html.push_str(&format!(
            "<span style=\"flex:1 1 0;min-width:1px;height:{height}px;background:#86efac;border-radius:1px;\"></span>"
        ));
    }
    html.push_str("</div>");
    html
}

/// 一类标注：事件条 + 事件详情（说话人、转写及其句子 / token 时间条）。
#[cfg(feature = "python-bindings")]
fn annotation_groups_html(
    role: &str,
    annotations: &[AudioEvent],
    duration_ms: u64,
    color: &str,
) -> String {
    if annotations.is_empty() {
        return String::new();
    }
    let mut html = String::new();
    for (title, group) in annotation_tracks(role, annotations) {
        for (index, event) in group.iter().enumerate() {
            let last = index + 1 == group.len();
            let heading = format!("{title} · {} {}", event_icon(event), event_heading(event));
            html.push_str(&event_bar_html(event.range(), &heading, duration_ms, color));
            // 非最后一条事件的子树留出竖线，接到下一个兄弟节点。
            let stem = if last { "   " } else { "│  " };
            html.push_str(&event_detail_html(event, stem, duration_ms, color));
        }
    }
    html
}

/// 事件条：中间是事件名（`Reference · 🎙 Speech`），起止时间贴在两端。
///
/// 事件太窄时不再显示两端时间，避免和名字挤在一起。
#[cfg(feature = "python-bindings")]
fn event_bar_html(range: TimeRange, heading: &str, duration_ms: u64, color: &str) -> String {
    let duration = duration_ms.max(1) as f64;
    let start = (range.start_ms as f64 / duration * 100.0).clamp(0.0, 100.0);
    let end = (range.end_ms as f64 / duration * 100.0).clamp(start, 100.0);
    let width = (end - start).max(0.8);
    let middle = (start + width / 2.0).clamp(0.0, 100.0);
    let heading = html_escape(heading);
    let mut ends = String::new();
    if width >= 30.0 {
        let start_label = html_escape(&format_time(range.start_ms as u64));
        let end_label = html_escape(&format_time(range.end_ms as u64));
        ends.push_str(&format!(
            "<div style=\"position:absolute;left:{start:.2}%;top:8px;transform:translateY(-50%);background:#1e1e1e;padding:0 4px;color:{color};white-space:nowrap;line-height:16px;\">{start_label}</div>\
             <div style=\"position:absolute;left:{end:.2}%;top:8px;transform:translate(-100%,-50%);background:#1e1e1e;padding:0 4px;color:{color};white-space:nowrap;line-height:16px;\">{end_label}</div>"
        ));
    }
    format!(
        "<div style=\"position:relative;height:18px;margin:6px 0 2px;\">\
         <div style=\"position:absolute;left:{start:.2}%;top:1px;width:1px;height:8px;background:{color};\"></div>\
         <div style=\"position:absolute;left:{start:.2}%;width:{width:.2}%;top:8px;height:1px;background:{color};\"></div>\
         <div style=\"position:absolute;left:{end:.2}%;top:1px;width:1px;height:8px;margin-left:-1px;background:{color};\"></div>\
         {ends}\
         <div style=\"position:absolute;left:{middle:.2}%;top:8px;transform:translate(-50%,-50%);z-index:1;background:#1e1e1e;padding:0 6px;color:{color};white-space:nowrap;line-height:16px;\">{heading}</div>\
         </div>"
    )
}

/// 事件详情：树形文本行 + 整宽的时间轨道，和终端保持一致的层级。
#[cfg(feature = "python-bindings")]
fn event_detail_html(event: &AudioEvent, stem: &str, duration_ms: u64, color: &str) -> String {
    let speaker = event.speaker();
    let language = event.language().map(|language| format!("🌐  {language}"));
    let transcription = event.transcription();
    let sentences: Vec<&Sentence> = transcription
        .map(|transcription| {
            transcription
                .sentences
                .iter()
                .filter(|sentence| !sentence.text.is_empty() || !sentence.tokens.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let full_text = transcription
        .filter(|transcription| transcription.sentences.is_empty())
        .map(|transcription| transcription.text.clone())
        .filter(|text| !text.trim().is_empty());

    let top_count = usize::from(speaker.is_some())
        + usize::from(language.is_some())
        + usize::from(full_text.is_some())
        + sentences.len();

    let mut html = String::new();
    let mut position = 0usize;
    let push_text = |html: &mut String, position: &mut usize, text: String| {
        *position += 1;
        html.push_str(&detail_text_html(stem, *position == top_count, &text));
    };

    if let Some(speaker) = speaker {
        push_text(&mut html, &mut position, speaker_name_line(speaker));
    }
    if let Some(text) = language {
        push_text(&mut html, &mut position, text);
    }
    if let Some(text) = full_text {
        push_text(&mut html, &mut position, format!("💬  {text}"));
    }
    let sentence_count = sentences.len();
    for sentence in sentences {
        position += 1;
        let last = position == top_count;
        if sentence.tokens.len() > MAX_TOKEN_TRACK {
            // token 太多，不画轨道，只把计数并入句子文本。
            html.push_str(&detail_text_html(
                stem,
                last,
                &format!(
                    "💬  {} ({} tokens, {} sentences)",
                    sentence.text,
                    sentence.tokens.len(),
                    sentence_count,
                ),
            ));
            continue;
        }
        html.push_str(&detail_text_html(stem, last, &format!("💬  {}", sentence.text)));
        if sentence.tokens.is_empty() {
            html.push_str(&span_bar_html(
                sentence.range,
                &sentence_range_label(sentence),
                duration_ms,
                color,
            ));
        } else {
            match token_bar_html(&sentence.tokens, duration_ms, color) {
                Some(tokens) => html.push_str(&tokens),
                // token 没有时间范围时退回文本，避免信息丢失。
                None => {
                    let child_stem = format!("{stem}{}", if last { "   " } else { "│  " });
                    html.push_str(&detail_text_html(
                        &child_stem,
                        true,
                        &format!("🔤  {}", join_token_texts(&sentence.tokens)),
                    ));
                }
            }
        }
    }
    html
}

/// 一行树形详情文本：`stem` 是上级缩进，`last` 决定用 `└─` 还是 `├─`。
#[cfg(feature = "python-bindings")]
fn detail_text_html(stem: &str, last: bool, text: &str) -> String {
    let branch = if last { "└─ " } else { "├─ " };
    format!(
        "<div style=\"padding:1px 0;white-space:pre;\"><span style=\"color:#64748b;\">{}</span>{}</div>",
        html_escape(&format!("{stem}{branch}")),
        html_escape(text)
    )
}

/// 用细线画出主时间线上的一段，两端向上勾，名字写在线的正中。
#[cfg(feature = "python-bindings")]
fn span_bar_html(range: TimeRange, label: &str, duration_ms: u64, color: &str) -> String {
    let duration = duration_ms.max(1) as f64;
    let start = (range.start_ms as f64 / duration * 100.0).clamp(0.0, 100.0);
    let end = (range.end_ms as f64 / duration * 100.0).clamp(start, 100.0);
    let width = (end - start).max(0.8);
    let middle = (start + width / 2.0).clamp(0.0, 100.0);
    let label = html_escape(label);
    // 竖勾从细线向上，表示这段是主时间线里的一个区间。名字垫底色，避免被线切开。
    format!(
        "<div style=\"position:relative;height:18px;margin:6px 0 2px;\">\
         <div style=\"position:absolute;left:{start:.2}%;top:1px;width:1px;height:8px;background:{color};\"></div>\
         <div style=\"position:absolute;left:{start:.2}%;width:{width:.2}%;top:8px;height:1px;background:{color};\"></div>\
         <div style=\"position:absolute;left:{end:.2}%;top:1px;width:1px;height:8px;margin-left:-1px;background:{color};\"></div>\
         <div style=\"position:absolute;left:{middle:.2}%;top:8px;transform:translate(-50%,-50%);z-index:1;background:#1e1e1e;padding:0 6px;color:{color};white-space:nowrap;line-height:16px;\">{label}</div>\
         </div>"
    )
}

/// token 时间条：每个 token 一段细线，文本写在这段的中点。
///
/// 返回 `None` 表示所有 token 都没有时间范围，无法定位。
#[cfg(feature = "python-bindings")]
fn token_bar_html(tokens: &[Token], duration_ms: u64, color: &str) -> Option<String> {
    if tokens.iter().all(|token| token.range.is_none()) {
        return None;
    }
    let duration = duration_ms.max(1) as f64;
    let mut html = String::from("<div style=\"position:relative;height:18px;margin:2px 0;\">");
    for token in tokens {
        let Some(range) = token.range else {
            continue;
        };
        let start = (range.start_ms as f64 / duration * 100.0).clamp(0.0, 100.0);
        let end = (range.end_ms as f64 / duration * 100.0).clamp(start, 100.0);
        let width = (end - start).max(0.8);
        let middle = (start + width / 2.0).clamp(0.0, 100.0);
        html.push_str(&format!(
            "<div style=\"position:absolute;left:{start:.3}%;width:{width:.3}%;top:9px;height:1px;background:{color};\"></div>\
             <div style=\"position:absolute;left:{start:.3}%;top:2px;width:1px;height:8px;background:{color};\"></div>\
             <div style=\"position:absolute;left:{end:.3}%;top:2px;width:1px;height:8px;margin-left:-1px;background:{color};\"></div>\
             <div style=\"position:absolute;left:{middle:.3}%;top:9px;transform:translate(-50%,-50%);background:#1e1e1e;padding:0 3px;color:{color};white-space:nowrap;font-size:12px;line-height:14px;\">{}</div>",
            html_escape(&token.text),
        ));
    }
    html.push_str("</div>");
    Some(html)
}

/// 转义会破坏 HTML 的字符。
#[cfg(feature = "python-bindings")]
fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use crate::audio::{AudioEncoding, AudioFormat, AudioSource, Waveform};
    use crate::timeline::{
        AudioEvent, Gender, Sentence, Speaker, Speech, Timeline, Token, Transcription,
    };

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
        assert!(channel_box_wraps_plot(&output, "Mono", 64));
        assert!(!output.contains("samples"));
        assert!(!output.contains("\x1b["));
        assert!(!output.contains("0.25"));
    }

    #[test]
    fn terminal_view_renders_annotation_tracks() {
        let mut audio = test_audio();
        audio
            .mono_timeline_mut()
            .expect("mono timeline")
            .annotate(
                0,
                1_000,
                Speech::new()
                    .with_speaker(Speaker::new("female0"))
                    .with_transcription(Transcription::new("你好")),
            )
            .expect("valid annotation");

        let output = format!("{}", AudioTerminalView::new(&audio, 64, false));

        assert!(output.contains("🎙 Speech"));
        assert!(output.contains("Reference"));
        assert!(output.contains("└─"));
        assert!(output.contains("🧑  female0"));
        assert!(output.contains("💬  你好"));
        assert!(output.contains('═'));
        assert!(channel_box_wraps_plot(&output, "Mono", 64));
    }

    #[test]
    fn terminal_view_adds_color_only_when_enabled() {
        let output = format!("{}", AudioTerminalView::new(&test_audio(), 64, true));

        assert!(output.contains("\x1b["));
    }

    #[test]
    fn colored_card_frame_matches_body_and_keeps_rules_regular() {
        let output = format!("{}", AudioTerminalView::new(&test_audio(), 72, true));
        let lines: Vec<&str> = output.lines().take(4).collect();
        let widths: Vec<usize> = lines
            .iter()
            .map(|line| super::display_width(&strip_ansi(line)))
            .collect();

        assert_eq!(widths, vec![72, 72, 72, 72]);
        assert!(lines[0].contains("\x1b[1;36m"));
        assert!(lines[3].contains("\x1b[36m"));
        assert!(!lines[3].contains("\x1b[1;36m"));
    }

    /// 声道卡片内：第一行是波形，第二行是时间文字，第三行是时间标尺，整框每行同宽。
    fn channel_box_wraps_plot(output: &str, name: &str, width: usize) -> bool {
        let lines: Vec<&str> = output.lines().collect();
        let Some(start) = lines.iter().position(|line| {
            line.contains(&format!(" {name} ")) && line.starts_with('╭') && line.ends_with('╮')
        }) else {
            return false;
        };
        let Some(end) = lines
            .iter()
            .skip(start)
            .position(|line| line.starts_with('╰') && line.ends_with('╯'))
        else {
            return false;
        };
        let end = start + end;
        let has_waveform = lines.get(start + 1).is_some_and(|waveform| {
            waveform.starts_with('│') && waveform.chars().any(|c| super::WAVE_LEVELS.contains(&c))
        });
        let has_labels = lines
            .get(start + 2)
            .is_some_and(|labels| labels.contains('s') && labels.starts_with('│'));
        let has_axis = lines.get(start + 3).is_some_and(|axis| {
            axis.contains('├') && axis.contains('┤') && axis.starts_with('│')
        });
        lines[start..=end]
            .iter()
            .all(|line| super::display_width(line) == width)
            && has_waveform
            && has_labels
            && has_axis
    }

    /// 去掉 ANSI 颜色码，便于比较可见宽度。
    fn strip_ansi(value: &str) -> String {
        let mut output = String::new();
        let mut chars = value.chars();
        while let Some(character) = chars.next() {
            if character == '\u{1b}' {
                for next in chars.by_ref() {
                    if next == 'm' {
                        break;
                    }
                }
            } else {
                output.push(character);
            }
        }
        output
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
        assert!(channel_box_wraps_plot(&output, "Mono", 64));
        assert!(!output.contains("samples"));
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
        assert!(channel_box_wraps_plot(&output, "Left", 64));
        assert!(channel_box_wraps_plot(&output, "Right", 64));
        assert!(!output.contains(" Mono "));
    }

    #[test]
    fn waveform_terminal_view_adds_color_only_when_enabled() {
        let output = format!("{}", WaveformTerminalView::new(&test_waveform(), 64, true));

        assert!(output.contains("\x1b["));
    }

    fn test_timeline() -> Timeline {
        let mut timeline = Timeline::new("audio_test", 1_000);
        timeline
            .annotate(
                0,
                350,
                Speech::new()
                    .with_confidence(0.9)
                    .with_speaker(Speaker::new("female0").with_gender(Gender::Female))
                    .with_transcription(
                        Transcription::new("甚至出现交易几乎停滞的情况。").with_sentences(vec![
                            Sentence::new("甚至出现交易几乎停滞的情况。", 0, 350)
                                .with_tokens(vec![Token::new("甚"), Token::new("至")]),
                        ]),
                    ),
            )
            .expect("speech");
        timeline
            .annotate(400, 700, AudioEvent::new("music"))
            .expect("music");
        timeline
    }

    #[test]
    fn timeline_terminal_view_renders_summary_and_annotation_tracks() {
        let output = format!("{}", TimelineTerminalView::new(&test_timeline(), 64, false));

        assert!(output.contains(" Timeline · "));
        assert!(output.contains("⏱  1.000 s  ·  ◆ 2 reference  ·  ◇ 0 prediction"));
        assert!(output.contains("🎵  audio · audio_test"));
        assert!(output.contains("🎙 Speech"));
        assert!(output.contains("🎵 music"));
        assert!(output.contains("Reference"));
        assert!(output.contains("👩  female0 (female)"));
        assert!(output.contains("💬  甚至出现交易几乎停滞的情况。"));
        assert!(output.contains("🔤  甚 · 至"));
        assert!(output.contains("├─"));
        assert!(output.contains("└─"));
        assert!(output.contains("│"));
        assert!(output.contains("2 annotations"));
        assert!(!output.contains("\x1b["));
        assert!(!output.contains("source:"));
    }

    #[test]
    fn timeline_terminal_view_renders_speaker_icons_and_prediction_source() {
        let mut timeline = Timeline::new("audio_test", 1_000);
        timeline
            .annotate(
                0,
                150,
                Speech::new().with_speaker(Speaker::new("male0").with_gender(Gender::Male)),
            )
            .expect("male");
        timeline
            .annotate(
                200,
                350,
                Speech::new().with_speaker(Speaker::new("unknown0").with_gender(Gender::Unknown)),
            )
            .expect("unknown gender");
        timeline
            .annotate(400, 550, Speech::new().with_speaker(Speaker::new("anon")))
            .expect("unspecified gender");
        timeline
            .annotate(560, 590, AudioEvent::new("beep"))
            .expect("generic event");
        timeline
            .annotate_with(
                600,
                750,
                Speech::new().with_language("zh"),
                false,
                Some("whisper"),
            )
            .expect("prediction");
        timeline
            .annotate_with(800, 900, AudioEvent::new("noise"), false, Some("whisper"))
            .expect("noise");
        timeline
            .annotate_with(920, 1_000, AudioEvent::new("silence"), false, Some("vad"))
            .expect("silence");

        let output = format!("{}", TimelineTerminalView::new(&timeline, 80, false));

        assert!(output.contains("👨  male0 (male)"));
        assert!(output.contains("🧑  unknown0 (unknown)"));
        assert!(output.contains("🧑  anon"));
        assert!(!output.contains("anon ·"));
        assert!(output.contains("🎙 Speech"));
        assert!(output.contains("🌐  zh"));
        assert!(output.contains("🔊 noise"));
        assert!(output.contains("🔇 silence"));
        assert!(output.contains("● beep"));
    }

    /// 事件条上 `Reference` / `Prediction` 写在事件名前面，居中显示。
    #[test]
    fn event_role_is_shown_before_the_name() {
        let mut timeline = Timeline::new("audio_test", 1_000);
        timeline
            .annotate_with(0, 1_000, Speech::new(), false, Some("whisper"))
            .expect("prediction");

        let output = format!("{}", TimelineTerminalView::new(&timeline, 80, false));

        assert!(output.contains("Prediction · whisper · 🎙 Speech"));
    }

    /// speaker 的性别放在名字后面的括号里，不另起一行。
    #[test]
    fn speaker_gender_is_shown_in_parentheses() {
        let mut timeline = Timeline::new("audio_test", 1_000);
        timeline
            .annotate(
                0,
                1_000,
                Speech::new().with_speaker(Speaker::new("female0").with_gender(Gender::Female)),
            )
            .expect("speech");

        let output = format!("{}", TimelineTerminalView::new(&timeline, 80, false));

        assert!(output.contains("👩  female0 (female)"));
        assert!(!output.contains("female0 · female"));
    }

    #[test]
    fn timeline_terminal_view_shows_text_when_sentences_are_absent() {
        let mut timeline = Timeline::new("audio_test", 1_000);
        timeline
            .annotate(
                0,
                1_000,
                Speech::new().with_transcription(Transcription::new("只有全文")),
            )
            .expect("text only");

        let output = format!("{}", TimelineTerminalView::new(&timeline, 64, false));

        assert!(output.contains("💬  只有全文"));
        assert!(!output.contains("🔤"));
    }

    #[test]
    fn timeline_terminal_view_renders_empty_state() {
        let timeline = Timeline::new("audio_test", 1_000);
        let output = format!("{}", TimelineTerminalView::new(&timeline, 64, false));

        assert!(output.contains("⏱  1.000 s  ·  ◆ 0 reference  ·  ◇ 0 prediction"));
        assert!(output.contains("no annotations"));
    }

    #[test]
    fn timeline_terminal_view_adds_color_only_when_enabled() {
        let output = format!("{}", TimelineTerminalView::new(&test_timeline(), 64, true));

        assert!(output.contains("\x1b["));
    }

    #[test]
    #[cfg(feature = "python-bindings")]
    fn notebook_html_draws_css_border_and_waveform_bars() {
        let mut audio = test_audio();
        audio
            .mono_timeline_mut()
            .expect("mono timeline")
            .annotate(
                0,
                1_000,
                Speech::new().with_transcription(Transcription::new("你好<")),
            )
            .expect("speech");

        let html = audio.notebook_html();

        assert!(html.contains("border:1px solid #67e8f9"));
        assert!(html.contains("height:36px"));
        assert!(html.contains("background:#86efac"));
        let waveform_at = html.find("height:36px").expect("waveform");
        let axis_at = html.find(">0s<").expect("main axis");
        assert!(waveform_at < axis_at);
        assert!(html.contains("🎙 Speech"));
        assert!(html.contains("Reference"));
        assert!(html.contains("height:1px"));
        assert!(html.contains("width:1px;height:8px"));
        assert!(html.contains("white-space:pre"));
        assert!(!html.contains("90%"));
        assert!(!html.contains("◆ Reference"));
        assert!(html.contains("你好&lt;"));
        assert!(!html.contains('│'));
        assert!(!html.contains('▁'));

        let waveform = test_waveform().notebook_html();
        assert!(waveform.contains("border:1px solid #67e8f9"));
        assert!(waveform.contains("background:#86efac"));

        let timeline = test_timeline().notebook_html();
        assert!(timeline.contains("border:1px solid #67e8f9"));
        assert!(timeline.contains("Speech"));
        assert!(timeline.contains("└─"));
    }

    /// 句子和 token 各自按时间画轨道，和事件条共用同一套列坐标。
    #[test]
    fn transcription_draws_nested_time_tracks() {
        let mut timeline = Timeline::new("audio_test", 1_000);
        timeline
            .annotate(
                0,
                1_000,
                Speech::new().with_transcription(Transcription::new("甲乙").with_sentences(vec![
                    Sentence::new("甲乙", 0, 1_000).with_tokens(vec![
                        Token::new("甲").with_range(0, 500),
                        Token::new("乙").with_range(500, 1_000),
                    ]),
                ])),
            )
            .expect("speech");

        let output = format!("{}", TimelineTerminalView::new(&timeline, 64, false));

        // token 少时句子文本下面直接逐个 token 嵌字，不再单独写句子时间标签。
        assert!(!output.contains("⏱  0s–1s"));
        assert!(output.contains('甲'));
        assert!(output.contains('乙'));
        assert!(output.contains('╰'));
        // 有时间范围时不再退化到一行 token 文本。
        assert!(!output.contains("🔤"));
    }

    /// token 超过阈值时收敛成摘要，不再逐个展开。
    #[test]
    fn transcription_summarizes_when_tokens_exceed_the_track_cap() {
        let mut timeline = Timeline::new("audio_test", 1_000);
        let tokens: Vec<Token> = (0..20)
            .map(|index| Token::new("甲").with_range(index * 50, index * 50 + 50))
            .collect();
        timeline
            .annotate(
                0,
                1_000,
                Speech::new().with_transcription(Transcription::new("甲").with_sentences(vec![
                    Sentence::new("甲", 0, 1_000).with_tokens(tokens),
                ])),
            )
            .expect("speech");

        let output = format!("{}", TimelineTerminalView::new(&timeline, 80, false));

        // token 太多时不画轨道，只把计数并入句子文本。
        assert!(output.contains("(20 tokens, 1 sentences)"));
        assert!(!output.contains("╰甲╯"));
    }

    /// token 没有时间范围时退回一行文本，不丢信息。
    #[test]
    fn transcription_falls_back_to_token_text_without_ranges() {
        let mut timeline = Timeline::new("audio_test", 1_000);
        timeline
            .annotate(
                0,
                1_000,
                Speech::new().with_transcription(Transcription::new("甲乙").with_sentences(vec![
                    Sentence::new("甲乙", 0, 1_000)
                        .with_tokens(vec![Token::new("甲"), Token::new("乙")]),
                ])),
            )
            .expect("speech");

        let output = format!("{}", TimelineTerminalView::new(&timeline, 64, false));

        assert!(output.contains("🔤  甲 · 乙"));
    }

    /// Jupyter 卡片同样为句子和 token 画时间条。
    #[test]
    #[cfg(feature = "python-bindings")]
    fn notebook_html_draws_sentence_and_token_bars() {
        let mut audio = test_audio();
        audio
            .mono_timeline_mut()
            .expect("mono timeline")
            .annotate(
                0,
                1_000,
                Speech::new().with_transcription(Transcription::new("甲乙").with_sentences(vec![
                    Sentence::new("甲乙", 0, 1_000).with_tokens(vec![
                        Token::new("甲").with_range(0, 500),
                        Token::new("乙").with_range(500, 1_000),
                    ]),
                ])),
            )
            .expect("speech");

        let html = audio.notebook_html();

        // 事件条两端有起止时间，token 逐个画段。
        assert!(html.contains(">0s<"));
        assert!(html.contains('甲'));
        assert!(html.contains('乙'));
        assert!(!html.contains("🔤"));
    }
}
