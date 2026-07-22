# Project Module Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the Rust core and Python bindings by domain while preserving all existing public APIs and runtime behavior.

**Architecture:** Keep `lib.rs` as the stable public facade, assemble each Rust domain through a small `mod.rs`, and split the PyO3 extension into matching domain files with explicit registration functions. Perform the migration in independently verified stages so structural failures remain local and no data, SQL, or API changes are mixed into the refactor.

**Tech Stack:** Rust 2024, Serde, SQLite/rusqlite, PyO3, Tokio, Maturin, pytest

---

## File Structure

- Modify `src/lib.rs`: declare internal domain modules and preserve all crate-root re-exports.
- Rename `src/time.rs` to `src/utils.rs`: shared duration/range types only.
- Move `src/media.rs` to `src/audio/source.rs`: audio source and format types.
- Keep `src/audio/data.rs`, `decode.rs`, and `stream.rs`: existing audio responsibilities.
- Modify `src/audio/mod.rs`: assemble and re-export audio types while keeping `audio::decode` public.
- Move `src/cer.rs` to `src/metrics/cer.rs`; create `src/metrics/mod.rs`.
- Split `src/timeline.rs`, `src/segment.rs`, and `src/token.rs` into `src/timeline/`.
- Split `src/doc.rs` into `src/doc/mod.rs` and `src/doc/legacy.rs`.
- Split `src/db.rs` into `src/db/mod.rs`, `src/db/schema.rs`, and `src/db/query.rs`.
- Split `src/python.rs` into `src/python/mod.rs`, `common.rs`, `audio.rs`, `timeline.rs`, `doc.rs`, and `db.rs`.
- Create `tests/public_api.rs`: compile-time coverage of stable root and audio module paths.

### Task 1: Add public API compatibility coverage

**Files:**
- Create: `tests/public_api.rs`

- [x] **Step 1: Add a compile-time public API test**

Create `tests/public_api.rs` with representative imports from the crate root and the existing public audio module:

```rust
use std::path::Path;

use asr_data::audio::{self, decode};
use asr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, Audio, AudioChannel,
    AudioChunk, AudioChunks, AudioDb, AudioDbError, AudioDbInfo, AudioDbMode, AudioDoc,
    AudioEncoding, AudioError, AudioFormat, AudioLoadOptions, AudioLoader, AudioQuery, AudioSource,
    CerStats, DEFAULT_QUERY_LIMIT, DurationMs, MAX_QUERY_LIMIT, SampleIndex, TextSpan, TimeRange,
    Timeline, Token, Transcript, compute_cer, import_legacy_msgpack_to_db, normalize_for_cer,
    read_audio_db_info, read_legacy_msgpack,
};

#[test]
fn stable_public_paths_compile() {
    let _: Option<Audio> = None;
    let _: Option<audio::Audio> = None;
    let _: Option<AudioDb> = None;
    let _: Option<AudioChunk> = None;
    let _: Option<AudioChunks> = None;
    let _: Option<AudioDoc> = None;
    let _: Option<Annotation> = None;
    let _: Option<AnnotationPayload> = None;
    let _: Option<AnnotationSource> = None;
    let _: Option<AnnotationStatus> = None;
    let _: Option<AudioChannel> = None;
    let _: Option<AudioDbError> = None;
    let _: Option<AudioDbInfo> = None;
    let _: Option<AudioDbMode> = None;
    let _: Option<AudioEncoding> = None;
    let _: Option<AudioError> = None;
    let _: Option<AudioFormat> = None;
    let _: Option<AudioLoadOptions> = None;
    let _: Option<AudioLoader> = None;
    let _: Option<AudioQuery> = None;
    let _: Option<AudioSource> = None;
    let _: Option<CerStats> = None;
    let _: Option<DurationMs> = None;
    let _: Option<SampleIndex> = None;
    let _: Option<TextSpan> = None;
    let _: Option<TimeRange> = None;
    let _: Option<Timeline> = None;
    let _: Option<Token> = None;
    let _: Option<Transcript> = None;
    let _: fn(&str, &str) -> CerStats = compute_cer;
    let _: fn(&str, bool) -> String = normalize_for_cer;
    let _: fn(&Path) -> anyhow::Result<Audio> = decode::decode_path_audio;
    let _: usize = DEFAULT_QUERY_LIMIT;
    let _: usize = MAX_QUERY_LIMIT;
    let _ = import_legacy_msgpack_to_db::<&Path, &Path>;
    let _ = read_audio_db_info::<&Path>;
    let _ = read_legacy_msgpack::<&Path>;
}
```

- [x] **Step 2: Run the compatibility test before moving files**

Run:

```bash
cargo test --test public_api
```

Expected: one test passes against the current layout, establishing the public compatibility baseline.

- [x] **Step 3: Commit the compatibility guard**

```bash
git add tests/public_api.rs
git commit -m "test: guard public module paths"
```

### Task 2: Move utilities, metrics, and audio sources

**Files:**
- Rename: `src/time.rs` to `src/utils.rs`
- Rename: `src/cer.rs` to `src/metrics/cer.rs`
- Create: `src/metrics/mod.rs`
- Rename: `src/media.rs` to `src/audio/source.rs`
- Modify: `src/audio/mod.rs`
- Modify: `src/lib.rs`

- [x] **Step 1: Verify the target layout is absent**

Run:

```bash
test ! -f src/utils.rs && test ! -f src/metrics/mod.rs && test ! -f src/audio/source.rs
```

Expected: exit code 0 before migration. This records that the layout change has not already been partially applied.

- [x] **Step 2: Perform Git-aware file moves**

```bash
mkdir -p src/metrics
git mv src/time.rs src/utils.rs
git mv src/cer.rs src/metrics/cer.rs
git mv src/media.rs src/audio/source.rs
```

- [x] **Step 3: Assemble metrics and audio source modules**

Create `src/metrics/mod.rs`:

```rust
mod cer;

pub use cer::{CerStats, compute_cer, normalize_for_cer};
```

Add to `src/audio/mod.rs` while leaving `pub mod decode;` unchanged:

```rust
mod source;

pub use source::{AudioChannel, AudioEncoding, AudioFormat, AudioSource};
```

Update `src/lib.rs` declarations:

```rust
pub mod audio;
mod db;
mod doc;
mod metrics;
#[cfg(feature = "python-bindings")]
mod python;
mod segment;
mod timeline;
mod token;
mod utils;
```

Replace the old `cer`, `media`, and `time` re-exports with:

```rust
pub use audio::{AudioChannel, AudioEncoding, AudioFormat, AudioSource};
pub use metrics::{CerStats, compute_cer, normalize_for_cer};
pub use utils::{DurationMs, SampleIndex, TimeRange};
```

- [x] **Step 4: Replace facade-based imports inside the audio domain**

Use sibling or defining-module imports in the moved audio code:

```rust
// src/audio/data.rs
use super::{AudioEncoding, AudioFormat, AudioSource};
use crate::utils::DurationMs;

// src/audio/stream.rs, under the existing python-bindings cfg
use super::{AudioChunk, AudioChunks, AudioLoadOptions, AudioSource};
```

Remove `use crate::AudioSource` from `src/audio/mod.rs`; the local `source` re-export supplies the type. In `src/audio/source.rs`, replace crate-root audio facade references with `super::Audio`, `super::AudioLoader`, `super::AudioLoadOptions`, `super::decode`, and `super::transform_loaded_audio` without changing method signatures.

- [x] **Step 5: Verify the first migration phase**

Run:

```bash
cargo fmt --check
cargo test
cargo check --all-features --all-targets
cargo clippy --all-features --all-targets -- -D warnings
```

Expected: all commands pass, including `tests/public_api.rs`.

- [x] **Step 6: Commit the first migration phase**

```bash
git add src/lib.rs src/utils.rs src/metrics src/audio/mod.rs src/audio/source.rs src/time.rs src/cer.rs src/media.rs
git commit -m "refactor: group utilities metrics and audio sources"
```

### Task 3: Assemble the timeline domain

**Files:**
- Create: `src/timeline/mod.rs`
- Create: `src/timeline/annotation.rs`
- Create: `src/timeline/data.rs`
- Rename: `src/segment.rs` to `src/timeline/segment.rs`
- Rename: `src/token.rs` to `src/timeline/token.rs`
- Delete: `src/timeline.rs`
- Modify: `src/lib.rs`

- [x] **Step 1: Verify the directory module does not yet exist**

Run:

```bash
test ! -d src/timeline
```

Expected: exit code 0.

- [x] **Step 2: Move existing timeline support files**

```bash
mkdir -p src/timeline
git mv src/segment.rs src/timeline/segment.rs
git mv src/token.rs src/timeline/token.rs
```

- [x] **Step 3: Split annotations from timeline behavior**

Move these items from `src/timeline.rs` into `src/timeline/annotation.rs` without changing derives, Serde attributes, fields, or method bodies:

```text
AudioId, TimelineId, AnnotationId, SpeakerId, LanguageTag,
AnnotationStatus, AnnotationSource, HotwordMatch, AcousticEvent,
Diagnostic, AnnotationPayload, Annotation and impl Annotation
```

Move `Timeline`, `deserialize_duration`, `impl Timeline`, `transcript_from_annotations`, and `impl Default for Timeline` into `src/timeline/data.rs`. Its imports must be:

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, AudioId, TextSpan,
    TimelineId, Transcript,
};
use crate::utils::DurationMs;
```

Replace every `crate::DurationMs` in the moved timeline implementation with `DurationMs`. In `annotation.rs`, import `TextSpan`, `Token`, and `TimeRange` from `super` rather than from the crate-root facade. Update `segment.rs` to import `Token` from `super`, and update `token.rs` to import `TimeRange` from `crate::utils`.

Create `src/timeline/mod.rs`:

```rust
mod annotation;
mod data;
mod segment;
mod token;

pub use annotation::{
    AcousticEvent, Annotation, AnnotationId, AnnotationPayload, AnnotationSource,
    AnnotationStatus, AudioId, Diagnostic, HotwordMatch, LanguageTag, SpeakerId, TimelineId,
};
pub use data::Timeline;
pub use segment::{TextSpan, Transcript};
pub use token::Token;
```

- [x] **Step 4: Point the crate facade at the timeline domain**

Remove `mod segment;` and `mod token;` from `src/lib.rs`. Replace their separate re-exports and the current timeline re-export with one block:

```rust
pub use timeline::{
    AcousticEvent, Annotation, AnnotationId, AnnotationPayload, AnnotationSource,
    AnnotationStatus, AudioId, Diagnostic, HotwordMatch, LanguageTag, SpeakerId, TextSpan,
    Timeline, TimelineId, Token, Transcript,
};
```

- [x] **Step 5: Verify and commit the timeline phase**

Run:

```bash
cargo fmt --check
cargo test
cargo check --all-features --all-targets
cargo clippy --all-features --all-targets -- -D warnings
```

Expected: all commands pass and serialized timeline tests remain unchanged.

Then commit:

```bash
git add src/lib.rs src/timeline src/timeline.rs src/segment.rs src/token.rs
git commit -m "refactor: assemble timeline domain"
```

### Task 4: Split document core and legacy migration

**Files:**
- Create: `src/doc/mod.rs`
- Create: `src/doc/legacy.rs`
- Delete: `src/doc.rs`

- [x] **Step 1: Verify the directory module does not yet exist**

Run:

```bash
test ! -d src/doc
```

Expected: exit code 0.

- [x] **Step 2: Move document core types into `doc/mod.rs`**

Move the imports and definitions from the start of `src/doc.rs` through the `From<String> for AudioDoc` implementation into `src/doc/mod.rs`. Keep `AudioDoc`, its fields, all validation errors, `validate_channel`, and conversion implementations byte-for-byte equivalent apart from module-relative imports.

Append:

```rust
mod legacy;

pub use legacy::{LegacyImportError, read_legacy_msgpack};
```

- [x] **Step 3: Move legacy wire formats into `doc/legacy.rs`**

Move `LegacyImportError`, `read_legacy_msgpack`, all `Legacy*` wire types and deserializers, `MigratedAsset`, `migrate_legacy_asset`, and `legacy_format_value` into `src/doc/legacy.rs`. Use:

```rust
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde::{Deserialize, Deserializer};
use thiserror::Error;

use super::AudioDoc;
use crate::audio::{AudioChannel, AudioEncoding, AudioSource};
use crate::timeline::{Annotation, Timeline};
use crate::utils::DurationMs;
```

Keep `sanitize_audio_id` in `doc/mod.rs`, because `AudioDoc::audio_id` is its caller.
In `doc/mod.rs`, import audio types from `crate::audio`, timeline types from `crate::timeline`, and `DurationMs` from `crate::utils`; do not import them through the crate-root facade.

- [x] **Step 4: Verify and commit the document phase**

Run:

```bash
cargo fmt --check
cargo test
cargo check --all-features --all-targets
cargo clippy --all-features --all-targets -- -D warnings
```

Expected: all commands pass, including all legacy v1-v3 migration tests.

Then commit:

```bash
git add src/doc src/doc.rs
git commit -m "refactor: split document legacy migration"
```

### Task 5: Split database schema and queries

**Files:**
- Create: `src/db/mod.rs`
- Create: `src/db/schema.rs`
- Create: `src/db/query.rs`
- Delete: `src/db.rs`

- [x] **Step 1: Verify the directory module does not yet exist**

Run:

```bash
test ! -d src/db
```

Expected: exit code 0.

- [x] **Step 2: Define the database module facade**

Move these constants and types into `src/db/mod.rs` without changing their public fields, derives, errors, or defaults:

```text
SCHEMA_VERSION, CHANNEL_TIMELINE_SCHEMA_VERSION, SPLIT_TABLE_SCHEMA_VERSION,
LEGACY_SCHEMA_VERSION, APPLICATION_ID, DEFAULT_QUERY_LIMIT, MAX_QUERY_LIMIT,
AudioDbError, AudioDb, AudioQuery, impl Default for AudioQuery,
AudioDbMode, AudioDbTransaction, AudioDbInfo
```

Declare and re-export implementation entry points:

```rust
mod query;
mod schema;

pub use query::{import_legacy_msgpack_to_db, read_audio_db_info};
```

Keep `AudioDb.connection`, `AudioDb.schema_version`, and `AudioDbTransaction.transaction` private; child modules can access parent-private fields.

- [x] **Step 3: Move schema behavior**

Move `initialize`, `configure`, `migrate`, `migrate_v1_to_v2`, `migrate_v2_to_v3`, and `validate` into `src/db/schema.rs`. Mark `initialize`, `configure`, and `validate` as `pub(super)` because `query.rs` calls them. Import constants and errors from `super`:

```rust
use rusqlite::Connection;

use super::{
    APPLICATION_ID, AudioDbError, CHANNEL_TIMELINE_SCHEMA_VERSION, LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION, SPLIT_TABLE_SCHEMA_VERSION,
};
```

- [x] **Step 4: Move database operations and codecs**

Move both database `impl` blocks, `import_legacy_msgpack_to_db`, `read_audio_db_info`, CRUD helpers, query construction, row decoding, and MessagePack/SQL error helpers into `src/db/query.rs`. Import `initialize`, `configure`, and `validate` from `super::schema`; import database types and version constants from `super`. Do not alter any SQL string, query ordering, schema-version branch, or encoding call.

In `db/mod.rs`, import `LegacyImportError` from `crate::doc` and `DurationMs` from `crate::utils`. In `db/query.rs`, import `AudioDoc` from `crate::doc` and `DurationMs` from `crate::utils`. Call `crate::doc::read_legacy_msgpack` from the legacy database importer. Do not use crate-root compatibility re-exports inside the database domain.

- [x] **Step 5: Verify and commit the database phase**

Run:

```bash
cargo fmt --check
cargo test
cargo check --all-features --all-targets
cargo clippy --all-features --all-targets -- -D warnings
```

Expected: all commands pass, including database CRUD and legacy schema tests.

Then commit:

```bash
git add src/db src/db.rs
git commit -m "refactor: split database schema and queries"
```

### Task 6: Split Python bindings by domain

**Files:**
- Create: `src/python/mod.rs`
- Create: `src/python/common.rs`
- Create: `src/python/audio.rs`
- Create: `src/python/timeline.rs`
- Create: `src/python/doc.rs`
- Create: `src/python/db.rs`
- Delete: `src/python.rs`
- Test: `tests/test_bindings.py`

- [x] **Step 1: Record the current Python API surface**

Run:

```bash
env -u CONDA_PREFIX uv run maturin develop
uv run --with pytest python -m pytest tests/test_bindings.py -q
```

Expected: 28 tests pass and one test is skipped before the split.

- [x] **Step 2: Create shared Python helpers**

Move `py_error`, `py_db_error`, `poisoned`, annotation status/source conversion, audio channel conversion, source/status naming, encoding naming, truncation, duration formatting, and source formatting into `src/python/common.rs`. Also define the shared owner there:

```rust
pub(super) type SharedAudio = Arc<RwLock<crate::doc::AudioDoc>>;
```

Mark helpers used by sibling modules `pub(super)` and import the shared Rust/PyO3 types from their defining Rust domain modules rather than from crate-root re-exports.

- [x] **Step 3: Move audio bindings**

Move these bindings and their helper state into `src/python/audio.rs`:

```text
PyAudioFormat, PyAudio, AsyncLoadResult, async_runtime, PyAudioLoadTask,
spawn_source_aload, AsyncStreamState, PyAudioStreamTask, spawn_source_astream,
PyAudioIterator, PyAudioChunk, stream_source, load_source,
PyAudioPath, PyAudioUrl, PyAudioBytes, PyAudioBase64, PyAudioPcm,
rust_source_from_py, py_source_from_rust
```

Make cross-domain types/functions `pub(super)`: `PyAudio`, `PyAudioFormat`, `PyAudioLoadTask`, `async_runtime`, `rust_source_from_py`, and `py_source_from_rust`. Add:

```rust
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAudioFormat>()?;
    module.add_class::<PyAudio>()?;
    module.add_class::<PyAudioChunk>()?;
    module.add_class::<PyAudioPath>()?;
    module.add_class::<PyAudioUrl>()?;
    module.add_class::<PyAudioBytes>()?;
    module.add_class::<PyAudioBase64>()?;
    module.add_class::<PyAudioPcm>()?;
    module.add_class::<PyAudioLoadTask>()?;
    module.add_class::<PyAudioStreamTask>()?;
    module.add_class::<PyAudioIterator>()?;
    Ok(())
}
```

- [x] **Step 4: Move timeline and document bindings**

Move `PyAnnotation`, `PyTranscript`, `PyTimeline`, and their implementations into `src/python/timeline.rs`. Make `PyTimeline` and its constructor helpers `pub(super)` where `doc.rs` needs them. Add a `register` function for `PyAnnotation`, `PyTranscript`, and `PyTimeline`.

Move `PyAudioDoc` and its implementations into `src/python/doc.rs`. Import `SharedAudio` and conversions from `super::common`, source conversions from `super::audio`, and timeline wrappers from `super::timeline`. Make `PyAudioDoc` `pub(super)`. Add a `register` function for `PyAudioDoc`.

- [x] **Step 5: Move database bindings**

Move `PyAudioDb`, `PyAudioDbIterator`, and their implementations into `src/python/db.rs`. Import `PyAudioDoc` from `super::doc` and error/format helpers from `super::common`. Add:

```rust
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAudioDb>()?;
    module.add_class::<PyAudioDbIterator>()?;
    Ok(())
}
```

- [x] **Step 6: Assemble the PyO3 module**

Create `src/python/mod.rs` with child declarations, the exception, and module registration:

```rust
mod audio;
mod common;
mod db;
mod doc;
mod timeline;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(_native, AsrDataError, PyException);

#[pymodule]
fn _native(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = audio::async_runtime();
    module.add("AsrDataError", py.get_type::<AsrDataError>())?;
    audio::register(module)?;
    timeline::register(module)?;
    doc::register(module)?;
    db::register(module)?;
    Ok(())
}
```

- [x] **Step 7: Compile and test the split bindings**

Run:

```bash
cargo fmt --check
cargo check --all-features --all-targets
cargo clippy --all-features --all-targets -- -D warnings
env -u CONDA_PREFIX uv run maturin develop
uv run --with pytest python -m pytest tests/test_bindings.py -q
```

Expected: the Rust checks pass and the Python result remains 28 passed, 1 skipped. Do not change `asr_data/__init__.py`, `asr_data/__init__.pyi`, or `asr_data/_native.pyi` unless a compile error proves an internal import is required; no public declarations should change.

- [x] **Step 8: Commit the Python phase**

```bash
git add src/python src/python.rs
git commit -m "refactor: split Python bindings by domain"
```

### Task 7: Final structure and compatibility verification

**Files:**
- Modify: `docs/superpowers/plans/2026-07-22-project-module-reorganization.md`

- [x] **Step 1: Confirm the target layout and retired files**

Run:

```bash
test -f src/utils.rs
test -f src/audio/source.rs
test -f src/metrics/mod.rs
test -f src/timeline/mod.rs
test -f src/doc/mod.rs
test -f src/db/mod.rs
test -f src/python/mod.rs
test ! -f src/time.rs
test ! -f src/media.rs
test ! -f src/cer.rs
test ! -f src/timeline.rs
test ! -f src/segment.rs
test ! -f src/token.rs
test ! -f src/doc.rs
test ! -f src/db.rs
test ! -f src/python.rs
```

Expected: every command exits successfully.

- [x] **Step 2: Run complete verification**

```bash
cargo fmt --check
cargo test
cargo check --all-features --all-targets
cargo clippy --all-features --all-targets -- -D warnings
env -u CONDA_PREFIX uv run maturin develop
uv run --with pytest python -m pytest tests/test_bindings.py -q
uv run python -m compileall -q asr_data
git diff --check
```

Expected: all commands pass; Rust public-path tests pass; Python reports 28 passed and 1 skipped.

- [x] **Step 3: Review the final diff for behavioral changes**

Run:

```bash
git status --short
git diff --stat f42c0ee..HEAD
git diff f42c0ee..HEAD -- Cargo.toml pyproject.toml asr_data tests/test_bindings.py tests/data_behaviour.rs
```

Expected: no dependency, Python API, or behavior-test changes beyond the new `tests/public_api.rs`; `test.ipynb` remains untracked.

- [x] **Step 4: Mark this plan complete and commit**

Change every checkbox in this plan from `[ ]` to `[x]`, then run:

```bash
git add -f docs/superpowers/plans/2026-07-22-project-module-reorganization.md
git commit -m "docs: complete module reorganization plan"
```

Expected: the completed execution record is committed on `main`.
