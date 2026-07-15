# Channel-scoped timelines

## Goal

Represent a dual-channel call as one `Audio` record while storing independent
annotations and transcripts for its left and right channels. Annotation code
must continue to operate on a mono `Waveform` and must not need to know which
source channel produced that waveform.

This design does not add implicit channel extraction or downmixing to `Audio`.
Callers continue to use `Waveform::channel` and `Waveform::to_mono` explicitly.

## Data model

Replace the single `Audio::timeline` value with a channel-keyed collection:

```rust
pub struct Audio {
    pub source: AudioSource,
    timelines: BTreeMap<AudioChannel, Timeline>,
    pub metadata: BTreeMap<String, serde_json::Value>,
}
```

`AudioChannel` remains the association between a signal derived from the audio
source and its timeline:

```rust
pub enum AudioChannel {
    Mono,
    Left,
    Right,
    Channel(u16),
}
```

The variants have these meanings:

- `Mono` contains annotations produced from a mono source or an explicit
  `Waveform::to_mono()` result.
- `Left` identifies source channel index 0.
- `Right` identifies source channel index 1.
- `Channel(n)` identifies source channel index `n` for `n >= 2`.

`Channel(0)` and `Channel(1)` are rejected by public constructors and accessors
to prevent duplicate keys for the left and right channels.

Every newly constructed `Audio` contains an empty `Mono` timeline for backward
compatibility. Left, right, and higher-channel timelines are created lazily
when requested. All timelines belonging to an `Audio` share its `audio_id`, but
each retains a unique `TimelineId`.

## Processing boundary

Channel selection remains an orchestration concern:

```rust
let waveform = audio.load()?;

let left_waveform = waveform.channel(0)?;
let left_annotations = annotator.annotate(&left_waveform)?;
audio
    .ensure_timeline(AudioChannel::Left)?
    .extend(left_annotations);

let right_waveform = waveform.channel(1)?;
let right_annotations = annotator.annotate(&right_waveform)?;
audio
    .ensure_timeline(AudioChannel::Right)?
    .extend(right_annotations);
```

The annotator remains channel-agnostic:

```rust
fn annotate(&self, waveform: &Waveform) -> Result<Vec<Annotation>, Error>;
```

Downmixing is also explicit:

```rust
let mono = waveform.to_mono()?;
let annotations = annotator.annotate(&mono)?;
audio
    .ensure_timeline(AudioChannel::Mono)?
    .extend(annotations);
```

Calling `channel` or `to_mono` does not mutate `Audio` and does not
automatically create or select a timeline.

## Rust API

`Audio` provides non-panicking lookup and explicit lazy creation:

```rust
pub fn timeline(&self, channel: AudioChannel) -> Result<Option<&Timeline>, AudioChannelError>;
pub fn timeline_mut(
    &mut self,
    channel: AudioChannel,
) -> Result<Option<&mut Timeline>, AudioChannelError>;
pub fn ensure_timeline(
    &mut self,
    channel: AudioChannel,
) -> Result<&mut Timeline, AudioChannelError>;
```

The error type reports non-canonical channel values such as `Channel(0)` and
`Channel(1)`. Timeline lookup does not decode or inspect the audio source, so it
does not attempt to validate the source's physical channel count. Actual
channel extraction remains validated by `Waveform::channel`.

The existing single-timeline convenience operations use `AudioChannel::Mono`.
Transcript generation remains a property of one `Timeline`; the library does
not implicitly merge left and right transcripts.

## Python API

Expose one uniform timeline accessor. Backward compatibility with the old
`audio.timeline` property is intentionally not retained:

```python
audio.timeline("mono")                 # Mono timeline, created on demand
audio.timeline("left")                 # Left timeline, created on demand
audio.timeline("right")                # Right timeline, created on demand
audio.timeline(2)                      # Channel(2), created on demand
audio.timelines                        # Read-only view keyed by channel
```

`PyTimeline` stores the selected `AudioChannel` alongside its shared `Audio`
handle. Its methods resolve that key before reading or mutating annotations.
There is no separate `channel_timeline` alias. The Python annotator-facing code
still receives only a `Waveform`.

Python integer channels are normalized so that `0` selects `Left`, `1` selects
`Right`, and `n >= 2` selects `Channel(n)`. Invalid channel names and negative
indices raise `ValueError`. The `timelines` view does not permit callers to
bypass timeline creation and `audio_id` validation.

## Persistence and migration

The database continues to store one `Audio` record per call. The v3 timelines
blob contains the complete `BTreeMap<AudioChannel, Timeline>` instead of one
`Timeline`. This keeps the existing one-to-one `audios`/`timelines` table shape
and avoids multiplying query rows.

When a v1 or v2 database is opened for writing, its single timeline is migrated
to:

```rust
BTreeMap::from([(AudioChannel::Mono, old_timeline)])
```

Read-only compatibility decodes the old single-timeline blob into the same
in-memory map without rewriting the database. Existing MessagePack imports use
the same mapping. No migration assumes that an old timeline belongs to the
left channel, even when its source happens to be stereo, because the old format
did not preserve that fact.

Updates continue to compare and write the timelines blob independently from
the audio source and metadata blobs.

## Invariants and errors

- An `Audio` has at most one timeline for each canonical `AudioChannel`.
- Every contained timeline has the same `audio_id` as its parent `Audio`.
- Replacing or inserting a timeline with a different `audio_id` returns an
  error instead of silently rewriting it.
- `Left`, `Right`, and `Channel(n)` do not imply speaker identity. Caller and
  agent labels remain ordinary speaker annotations or audio metadata.
- The model permits a mono timeline and channel timelines to coexist because
  they may come from different processing runs.

## Testing

Rust and Python tests cover:

- mono construction and access through `audio.timeline("mono")`;
- independent left and right annotations on one `Audio`;
- explicit `Waveform::channel` and `Waveform::to_mono` processing;
- rejection of non-canonical `Channel(0)` and `Channel(1)` keys;
- timeline `audio_id` invariant enforcement;
- database round trips containing mono, left, and right timelines;
- read-only decoding and read-write migration of v1/v2 databases;
- Python mutation through `timeline(channel)` updating only the selected
  timeline;
- rejection of the removed property-style and `channel_timeline` APIs.

## Out of scope

- Automatic speaker assignment from left/right channel identity.
- Automatic transcript merging across channels.
- Implicit downmixing or channel extraction during `Audio::load`.
- Arbitrary channel mixtures or derived signal-processing graphs.
