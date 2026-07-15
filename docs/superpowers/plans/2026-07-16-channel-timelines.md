# Channel Timelines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store independent mono, left, and right timelines in one `Audio` record without exposing channel identity to annotation code.

**Architecture:** Replace `Audio::timeline` with a `BTreeMap<AudioChannel, Timeline>`. Python uses one `audio.timeline(channel)` method for mono and physical channels, with no property-style compatibility API. `Waveform::channel` and `Waveform::to_mono` remain explicit processing operations. Database schema v3 stores the complete map in the existing timelines blob and decodes v1/v2 single timelines as the mono entry.

**Tech Stack:** Rust, serde, MessagePack via `rmp-serde`, SQLite via `rusqlite`, PyO3, pytest.

---

### Task 1: Channel-keyed Rust model

**Files:**
- Modify: `src/media.rs`
- Modify: `src/record.rs`
- Modify: `src/timeline.rs`
- Modify: `src/fasr.rs`
- Test: `tests/data_behaviour.rs`

- [ ] **Step 1: Write a failing model test**

Add a test that creates one `Audio`, writes different transcription annotations to `AudioChannel::Left` and `AudioChannel::Right`, and asserts that each timeline returns only its own transcript while the mono timeline remains empty. Also assert that `Channel(0)` and `Channel(1)` are rejected.

- [ ] **Step 2: Run the model test and verify RED**

Run: `cargo test --test data_behaviour audio_keeps_independent_channel_timelines -- --exact`

Expected: compilation fails because `Audio::ensure_timeline` and the `timelines` collection do not exist.

- [ ] **Step 3: Implement the minimal model API**

Derive ordering for `AudioChannel`, add `AudioChannelError`, and add canonical validation. Change `Audio` to hold `BTreeMap<AudioChannel, Timeline>`. Make construction insert a mono timeline, make `with_timeline` replace the mono entry, and add:

```rust
pub fn timeline(&self, channel: AudioChannel) -> Result<Option<&Timeline>, AudioChannelError>;
pub fn timeline_mut(&mut self, channel: AudioChannel) -> Result<Option<&mut Timeline>, AudioChannelError>;
pub fn ensure_timeline(&mut self, channel: AudioChannel) -> Result<&mut Timeline, AudioChannelError>;
pub fn mono_timeline(&self) -> &Timeline;
pub fn mono_timeline_mut(&mut self) -> &mut Timeline;
```

Add `Timeline::extend` and refactor existing Rust callers to use the mono helpers.

- [ ] **Step 4: Run the focused and full Rust model tests**

Run: `cargo test --test data_behaviour audio_keeps_independent_channel_timelines -- --exact && cargo test --workspace`

Expected: all tests pass.

### Task 2: SQLite schema v3 and compatibility

**Files:**
- Modify: `src/db.rs`
- Test: `tests/data_behaviour.rs`

- [ ] **Step 1: Write failing persistence tests**

Extend the CRUD round-trip test with left and right timelines. Update the migration fixture expectation to schema 3 and add a read-only schema-v2 fixture assertion that its old single timeline appears as `AudioChannel::Mono`.

- [ ] **Step 2: Run persistence tests and verify RED**

Run: `cargo test --test data_behaviour audio_db -- --nocapture`

Expected: at least one assertion fails because the database still encodes one `Timeline` and reports schema version 2.

- [ ] **Step 3: Implement schema v3**

Bump `SCHEMA_VERSION` to 3. Encode `audio.timelines` in insert/update paths. Decode schema 1/2 blobs as one mono timeline and schema 3 blobs as the map. Migrate v1 to v2 with the existing table split, then migrate every v2 timeline blob to a map and set `user_version = 3` atomically.

- [ ] **Step 4: Run persistence and workspace tests**

Run: `cargo test --test data_behaviour audio_db -- --nocapture && cargo test --workspace`

Expected: all tests pass, including schema migration and multi-timeline round trips.

### Task 3: Python channel timeline API

**Files:**
- Modify: `src/python.rs`
- Modify: `asr_data/_native.pyi`
- Test: `tests/test_bindings.py`

- [ ] **Step 1: Write a failing Python API test**

Create stereo PCM audio, add different transcriptions through `timeline("left")` and `timeline("right")`, assert independent transcripts, assert `timeline(0/1)` selects the same timelines, and verify a database round trip. Assert that mono is selected through `timeline("mono")` and that `channel_timeline` is absent.

- [ ] **Step 2: Build bindings and verify RED**

Run: `maturin develop && pytest tests/test_bindings.py::test_audio_channel_timelines_round_trip -q`

Expected: failure with `AttributeError` because `channel_timeline` does not exist.

- [ ] **Step 3: Implement Python bindings**

Make `PyTimeline` carry an `AudioChannel`, route every read and mutation through that selected timeline, add `timeline(channel)` with string/integer normalization, remove the old property and `channel_timeline`, and expose a read-only snapshot through `audio.timelines`. Update type stubs accordingly.

- [ ] **Step 4: Run Python and Rust tests**

Run: `maturin develop && pytest tests/test_bindings.py -q && cargo test --workspace`

Expected: all tests pass.

### Task 4: Final verification

**Files:**
- Verify all modified source, test, stub, spec, and plan files.

- [ ] **Step 1: Format and inspect**

Run: `cargo fmt --all -- --check && git diff --check && git status --short`

Expected: formatting and diff checks pass; status lists only intended files.

- [ ] **Step 2: Run the full suites**

Run: `cargo test --workspace && maturin develop && pytest tests/test_bindings.py -q`

Expected: all Rust and Python tests pass.

- [ ] **Step 3: Review the final diff against the design**

Confirm that annotation types contain no channel field, `Audio::load` performs no implicit conversion, old timelines map to `Mono`, and left/right transcripts remain independent.
