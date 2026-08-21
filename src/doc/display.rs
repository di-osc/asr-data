use std::env;
use std::fmt;
use std::io::{self, IsTerminal};

use crate::audio::{AudioChannel, AudioEncoding, AudioSource};
use crate::timeline::{Annotation, TimeSpan};
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

pub(super) struct AudioTerminalView<'a> {
    audio: &'a Audio,
    width: usize,
    color: bool,
}

impl<'a> AudioTerminalView<'a> {
    pub(super) fn auto(audio: &'a Audio) -> Self {
        let width = env::var("COLUMNS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_WIDTH)
            .clamp(MIN_WIDTH, MAX_WIDTH);
        let color = io::stdout().is_terminal() && env::var_os("NO_COLOR").is_none();
        Self {
            audio,
            width,
            color,
        }
    }

    pub(super) fn with_color(audio: &'a Audio, color: bool) -> Self {
        let width = env::var("COLUMNS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_WIDTH)
            .clamp(MIN_WIDTH, MAX_WIDTH);
        Self {
            audio,
            width,
            color,
        }
    }

    #[cfg(test)]
    fn new(audio: &'a Audio, width: usize, color: bool) -> Self {
        Self {
            audio,
            width: width.clamp(MIN_WIDTH, MAX_WIDTH),
            color,
        }
    }

    fn paint(&self, style: &str, value: &str) -> String {
        if self.color {
            format!("{style}{value}{RESET}")
        } else {
            value.to_owned()
        }
    }

    fn write_header(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let title = " Audio ";
        let rule_width = self.width.saturating_sub(2 + title.chars().count());
        let top = format!("╭─{title}{}╮", "─".repeat(rule_width.saturating_sub(1)));
        writeln!(formatter, "{}", self.paint(BOLD_CYAN, &top))?;

        let info = format!(
            "{}  ·  {}  ·  {}  ·  {:.3} s  ·  {} frames",
            encoding_name(&self.audio.info.source_format.encoding),
            sample_rate_name(self.audio.info.sample_rate),
            channel_count_name(self.audio.info.channels),
            self.audio.info.timeline_duration_ms() as f64 / 1000.0,
            grouped_number(self.audio.info.frame_count),
        );
        self.write_card_line(formatter, &self.audio.id)?;
        self.write_card_line(formatter, &info)?;
        self.write_card_line(
            formatter,
            &format!("source: {}", source_name(&self.audio.source)),
        )?;

        let bottom = format!("╰{}╯", "─".repeat(self.width.saturating_sub(2)));
        writeln!(formatter, "{}", self.paint(BOLD_CYAN, &bottom))
    }

    fn write_card_line(&self, formatter: &mut fmt::Formatter<'_>, value: &str) -> fmt::Result {
        let content_width = self.width.saturating_sub(4);
        let value = truncate(value, content_width);
        let padding = content_width.saturating_sub(display_width(&value));
        writeln!(formatter, "│ {value}{} │", " ".repeat(padding))
    }

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
            self.paint(DIM, &labels)
        )?;
        let axis = time_axis(plot_width, &ticks);
        writeln!(
            formatter,
            "{}{}",
            " ".repeat(LABEL_WIDTH),
            self.paint(DIM, &axis)
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
                self.paint(GREEN, &waveform),
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
            self.paint(DIM, &footer)
        )
    }
}

impl fmt::Display for AudioTerminalView<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_header(formatter)?;
        self.write_timeline(formatter)
    }
}

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

fn channel_count_name(channels: u16) -> String {
    match channels {
        1 => "Mono".to_owned(),
        2 => "Stereo".to_owned(),
        count => format!("{count} channels"),
    }
}

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

fn channel_label(channel: AudioChannel) -> String {
    match channel {
        AudioChannel::Mono => "Mono".to_owned(),
        AudioChannel::Left => "Left".to_owned(),
        AudioChannel::Right => "Right".to_owned(),
        AudioChannel::Channel(index) => format!("Channel {index}"),
    }
}

fn waveform_line(waveform: &crate::audio::Waveform, channel: AudioChannel, width: usize) -> String {
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

fn format_time(time_ms: u64) -> String {
    if time_ms.is_multiple_of(1_000) {
        format!("{}s", time_ms / 1_000)
    } else {
        let value = format!("{:.3}", time_ms as f64 / 1_000.0);
        format!("{}s", value.trim_end_matches('0').trim_end_matches('.'))
    }
}

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
        let label = format!(
            "{}–{}  {}",
            format_time(span.range.start.0),
            format_time(span.range.end.0),
            annotation_label(span),
        );
        let value = truncate(&label, content_width);
        let padding = content_width.saturating_sub(display_width(&value));
        writeln!(formatter, "│ {value}{} │", " ".repeat(padding))?;
    }
    writeln!(formatter, "╰{}╯", "─".repeat(width.saturating_sub(2)))?;
    Ok(())
}

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
    use crate::timeline::{Annotation, AudioActivity, TimeSpan};
    use crate::utils::{DurationMs, TimeRange};

    use super::{Audio, AudioTerminalView};

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

        assert!(output.contains("╭─ Audio "));
        assert!(output.contains("audio_test"));
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
            .annotate_span(
                true,
                TimeSpan::new(
                    TimeRange::new(DurationMs(125), DurationMs(750)),
                    Annotation::Activity(AudioActivity::new().with_event("speech")),
                    None,
                ),
            )
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
}
