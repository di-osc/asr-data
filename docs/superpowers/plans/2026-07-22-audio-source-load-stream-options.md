# AudioSource Load and Stream Options Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add consistent `sample_rate` and `mono` options to synchronous and asynchronous source loading/streaming, guarantee finite `float32` samples in `[-1, 1]`, and remove all peak-normalization APIs and state.

**Architecture:** Rust owns decoding, continuous stream transformation, range sanitation, and background chunk production. The Python bindings expose identical options on all five source classes; the pure-Python package wrapper turns the background stream receiver into a non-blocking `AsyncIterator`.

**Tech Stack:** Rust 2024, PyO3, Tokio, Rubato, NumPy, Python asyncio, pytest

---

## File Structure

- Modify `src/audio/data.rs`: remove normalization state/methods and add reusable finite-range sanitation.
- Modify `src/audio/mod.rs`: redefine `AudioLoadOptions` with optional sample rate and mono conversion and apply the shared transform pipeline.
- Modify `src/audio/decode.rs`: sanitize decoded samples and expose metadata needed by continuous stream transformation.
- Modify `src/media.rs`: make synchronous and asynchronous source loads use load options.
- Modify `src/python.rs`: expose four source APIs, continuous sync streaming, background async streaming, and remove normalization bindings.
- Modify `asr_data/__init__.py`: provide awaitable load and true async iterator wrappers.
- Modify `asr_data/_native.pyi`: publish final signatures and remove normalization members.
- Modify `tests/data_behaviour.rs`: cover sanitation, options, and removed Rust state.
- Modify `tests/test_bindings.py`: cover all Python source modes, async behavior, and removed members.

### Task 1: Remove normalization state and guarantee sample range

**Files:**
- Modify: `src/audio/data.rs`
- Modify: `src/audio/decode.rs`
- Modify: `tests/data_behaviour.rs`
- Modify: `tests/test_bindings.py`

- [ ] **Step 1: Replace normalization tests with failing sanitation and API-removal tests**

Add Rust assertions that sanitation converts `[NaN, -Inf, -1.5, 0.5, 1.5, Inf]` to `[0.0, 0.0, -1.0, 0.5, 1.0, 0.0]`. Update Python tests to assert `not hasattr(audio, "normalize")`, `not hasattr(audio, "is_normalized")`, and the same for `AudioChunk`.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
cargo test waveform_samples_are_sanitized
uv run pytest tests/test_bindings.py -k 'normalization_api_is_removed' -q
```

Expected: the Rust test fails because sanitation is absent and the Python test fails because normalization members still exist.

- [ ] **Step 3: Remove normalization and implement sanitation**

Delete `is_normalized` from `Audio`, `AudioChunk`, and `AudioChunks`; delete both `normalize` methods and their Python getters/methods. Add one shared function with this exact rule:

```rust
pub(crate) fn sanitize_samples(samples: &mut [f32]) {
    for sample in samples {
        *sample = if sample.is_finite() {
            sample.clamp(-1.0, 1.0)
        } else {
            0.0
        };
    }
}
```

Call it after decoder conversion and after any later resampling. Keep serde compatible with old map records by relying on the existing default unknown-field behavior.

- [ ] **Step 4: Run focused tests and verify pass**

Run:

```bash
cargo test waveform_samples_are_sanitized
uv run maturin develop
uv run pytest tests/test_bindings.py -k 'normalization_api_is_removed' -q
```

Expected: both focused test groups pass.

### Task 2: Add complete-load options

**Files:**
- Modify: `src/audio/mod.rs`
- Modify: `src/media.rs`
- Modify: `src/python.rs`
- Modify: `tests/data_behaviour.rs`
- Modify: `tests/test_bindings.py`

- [ ] **Step 1: Add failing option tests**

Test `AudioPcm(..., sample_rate=8000, channels=2).load(sample_rate=16000, mono=True)` and require a one-channel 16 kHz `Audio`. Test `sample_rate=0` raises an error and `mono=False` preserves two channels. Add the equivalent Rust `AudioLoader::load` test.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
cargo test audio_loader_applies_optional_sample_rate_and_mono
uv run pytest tests/test_bindings.py -k 'source_load_options' -q
```

Expected: compile/signature failures because the options do not exist.

- [ ] **Step 3: Implement load options**

Use this Rust shape:

```rust
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AudioLoadOptions {
    pub sample_rate: Option<u32>,
    pub mono: Option<bool>,
}
```

Apply mono conversion only for `Some(true)`, resample only when `Some(rate)` differs, reject zero sample rate, and sanitize last. Add `#[pyo3(signature = (*, sample_rate=None, mono=None))]` to every source `load` method and route them through one helper.

- [ ] **Step 4: Run focused tests and verify pass**

Run:

```bash
cargo test audio_loader_applies_optional_sample_rate_and_mono
uv run maturin develop
uv run pytest tests/test_bindings.py -k 'source_load_options' -q
```

Expected: both focused test groups pass.

### Task 3: Add continuous transformed synchronous streaming

**Files:**
- Modify: `src/audio/decode.rs`
- Modify: `src/python.rs`
- Modify: `tests/test_bindings.py`

- [ ] **Step 1: Add failing stream tests**

For all five source classes, compare `source.load(sample_rate=16000, mono=True)` with concatenated `source.stream(chunk_size_ms=2, sample_rate=16000, mono=True)`. Assert default `source.stream()` uses 100ms chunks, output offsets are monotonic, only the final chunk has `is_final=True`, and zero chunk/sample rates fail.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
uv run pytest tests/test_bindings.py -k 'source_stream_options or source_stream_default_chunk' -q
```

Expected: signature failures.

- [ ] **Step 3: Implement the transformed iterator**

Wrap the raw decoded/PCM iterator in a stateful iterator that performs mono conversion before resampling, keeps one Rubato resampler across input chunks, buffers transformed samples, emits `ceil(chunk_size_ms * target_rate / 1000)` frames per output chunk, sanitizes every emitted sample, and calculates offsets from emitted output frames. Preserve `source_format` from the original source.

Expose this signature on every source class:

```rust
#[pyo3(signature = (chunk_size_ms=100, *, sample_rate=None, mono=None))]
fn stream(
    &self,
    py: Python<'_>,
    chunk_size_ms: u64,
    sample_rate: Option<u32>,
    mono: Option<bool>,
) -> PyResult<PyAudioIterator>
```

- [ ] **Step 4: Run focused tests and verify pass**

Run:

```bash
uv run maturin develop
uv run pytest tests/test_bindings.py -k 'source_stream_options or source_stream_default_chunk' -q
```

Expected: all selected tests pass.

### Task 4: Add asynchronous load and true asynchronous streaming

**Files:**
- Modify: `src/media.rs`
- Modify: `src/python.rs`
- Modify: `asr_data/__init__.py`
- Modify: `tests/test_bindings.py`

- [ ] **Step 1: Add failing async tests**

Add tests that `await source.aload(sample_rate=16000, mono=True)` equals sync `load`, and that the following consumes the same samples as sync stream while a delayed HTTP response does not block an independent event-loop probe:

```python
chunks = []
async for chunk in source.astream(
    chunk_size_ms=100,
    sample_rate=16000,
    mono=True,
):
    chunks.append(chunk)
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
uv run pytest tests/test_bindings.py -k 'aload_options or astream' -q
```

Expected: missing parameter and missing method failures.

- [ ] **Step 3: Implement async APIs**

Pass `AudioLoadOptions` into the existing background `aload` task. Add a bounded Rust channel with capacity two and a background blocking producer that owns the transformed sync iterator. Expose a private task with non-blocking `try_recv`; in `asr_data/__init__.py`, implement `__aiter__` and `__anext__` that poll without blocking the event loop and raise `StopAsyncIteration` at channel completion. Dropping the receiver must make the producer exit on its next failed send.

- [ ] **Step 4: Run focused tests and verify pass**

Run:

```bash
uv run maturin develop
uv run pytest tests/test_bindings.py -k 'aload_options or astream' -q
```

Expected: all selected tests pass.

### Task 5: Update declarations and verify the complete change

**Files:**
- Modify: `asr_data/_native.pyi`
- Modify: `README.md` if its minimal example or wording is affected

- [ ] **Step 1: Update type declarations**

Import `AsyncIterator`; give all five source types identical `load`, `stream`, `aload`, and `astream` signatures; remove `normalize` and `is_normalized` from `Audio` and `AudioChunk`.

- [ ] **Step 2: Rebuild the Python extension**

Run:

```bash
uv run maturin develop
```

Expected: release or development extension installs successfully in the project environment.

- [ ] **Step 3: Run complete verification**

Run:

```bash
cargo fmt --check
cargo test
uv run pytest tests/test_bindings.py -q
git diff --check
rg -n 'is_normalized|fn normalize|def normalize' src asr_data
```

Expected: formatting succeeds, all Rust and Python tests pass, `git diff --check` is clean, and the final `rg` returns no matches.

- [ ] **Step 4: Review and commit implementation**

Run:

```bash
git status --short
git diff --stat
git add src/audio/data.rs src/audio/mod.rs src/audio/decode.rs src/media.rs src/python.rs asr_data/__init__.py asr_data/_native.pyi tests/data_behaviour.rs tests/test_bindings.py README.md
git commit -m "feat: add audio source loading options"
```

Expected: only task-owned files are staged; the pre-existing untracked `test.ipynb` remains untouched.
