# Annotation Overlap and Schema v6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove annotation status and legacy storage support, enforce reference and per-source prediction overlap rules, and publish database schema version 6.

**Architecture:** Centralize fallible, idempotent annotation insertion and collection validation in `Timeline`. `AudioDoc` and Python mutations delegate to the same validation, while the database accepts only the new v6 wire format and removes every migration/import branch.

**Tech Stack:** Rust, Serde, SQLite/rusqlite, PyO3, Python type stubs, pytest, MDX/Next.js

---

### Task 1: Remove annotation status from the Rust model

**Files:**
- Modify: `src/timeline/annotation.rs`
- Modify: `src/timeline/data.rs`
- Modify: `src/timeline/evaluation.rs`
- Modify: `src/timeline/mod.rs`
- Modify: `src/lib.rs`
- Modify: `tests/data_behaviour.rs`
- Modify: `tests/public_api.rs`

- [ ] **Step 1: Change Rust tests to construct status-free annotations**

Change test constructors to the new three-argument API and add a transcript test proving every stored transcription participates:

```rust
let annotation = Annotation::new(
    TimeRange::new(DurationMs(0), DurationMs(100)),
    AnnotationPayload::Transcription(Transcription::new("hello")),
    None,
);
timeline.push_reference(annotation).unwrap();
assert_eq!(timeline.reference_transcript().text, "hello");
```

Remove `AnnotationStatus` from `tests/public_api.rs` imports and public-path assertions.

- [ ] **Step 2: Run the Rust tests and verify compilation fails**

Run: `cargo test`

Expected: compilation fails because `Annotation::new` still requires status and insertion is not fallible.

- [ ] **Step 3: Remove status from the model and consumers**

Make `Annotation` status-free:

```rust
pub struct Annotation {
    pub id: AnnotationId,
    pub range: TimeRange,
    pub source: Option<String>,
    pub confidence: Option<f32>,
    pub payload: AnnotationPayload,
}

pub fn new(
    range: TimeRange,
    payload: AnnotationPayload,
    source: Option<String>,
) -> Self
```

Remove `AnnotationStatus`, `Timeline::by_status`, status comparisons in `content_eq`, transcript filtering, evaluation filtering, and all Rust exports/imports.

- [ ] **Step 4: Remove legacy annotation wire adapters**

Derive normal `Deserialize` for `SpeakerPayload`, `Annotation`, and `Timeline`. Remove the legacy speaker-string, legacy source, flat `annotations`, and `Segment` alias adapters and their tests.

- [ ] **Step 5: Run Rust formatting and tests**

Run: `cargo fmt --check && cargo test`

Expected: all current Rust tests pass with the status-free model.

- [ ] **Step 6: Commit the status removal**

```bash
git add src/timeline src/lib.rs tests/data_behaviour.rs tests/public_api.rs
git commit -m "refactor: remove annotation status"
```

### Task 2: Enforce annotation overlap rules in Rust

**Files:**
- Modify: `src/timeline/data.rs`
- Modify: `src/timeline/mod.rs`
- Modify: `src/doc/mod.rs`
- Modify: `src/python/timeline.rs`
- Modify: `tests/data_behaviour.rs`

- [ ] **Step 1: Add failing reference overlap tests**

Add tests covering speech overlap rejection and adjacency, same-name speaker rejection, different-name speaker overlap, transcription overlap, speaker-transcription versus top-level transcription, and duplicate idempotency:

```rust
let first = timeline.push_reference(speech(0, 100, None)).unwrap().id.clone();
assert_eq!(
    timeline.push_reference(speech(0, 100, None)).unwrap().id,
    first
);
assert!(timeline.push_reference(speech(50, 150, None)).is_err());
assert!(timeline.push_reference(speech(100, 150, None)).is_ok());
```

- [ ] **Step 2: Add failing prediction and relabel tests**

Test same-source conflicts, different-source overlap, same/different speaker names, transcription-bearing speakers, missing/blank source, and atomic relabel:

```rust
timeline.push_prediction(speech(0, 100, Some("a"))).unwrap();
timeline.push_prediction(speech(0, 100, Some("b"))).unwrap();
let before = timeline.prediction.clone();
assert!(timeline.relabel_prediction_source("b", "a").is_err());
assert_eq!(timeline.prediction, before);
```

- [ ] **Step 3: Run focused tests and verify the rules fail**

Run: `cargo test annotation_overlap`

Expected: tests fail because insertion accepts conflicts and relabel is infallible.

- [ ] **Step 4: Implement centralized validation**

Add public structured types in `src/timeline/data.rs`:

```rust
pub enum AnnotationConflictKind {
    Speech,
    Speaker,
    Transcription,
}

pub enum TimelineAnnotationError {
    PredictionMissingSource { annotation_id: AnnotationId },
    Overlap {
        kind: AnnotationConflictKind,
        source: Option<String>,
        speaker: Option<String>,
        first_id: AnnotationId,
        first_range: TimeRange,
        second_id: AnnotationId,
        second_range: TimeRange,
    },
}
```

Implement `validate_reference_annotations` and `validate_prediction_annotations`. Consolidate insertion into:

```rust
pub fn push_reference(
    &mut self,
    annotation: Annotation,
) -> Result<&Annotation, TimelineAnnotationError>;

pub fn push_prediction(
    &mut self,
    annotation: Annotation,
) -> Result<&Annotation, TimelineAnnotationError>;
```

Check content equality before overlap and remove `push_reference_unique`/`push_prediction_unique`.

- [ ] **Step 5: Make relabel and document validation atomic**

Change relabel to return `Result<usize, TimelineAnnotationError>`, validate a cloned candidate collection, and only assign it after success. Add an `AudioValidationError` variant carrying channel and `TimelineAnnotationError`, and invoke timeline validation from `AudioDoc::validate()`.

- [ ] **Step 6: Make Python payload replacement use candidate validation**

In `src/python/timeline.rs`, clone the selected collection, replace payload in the candidate, validate reference or prediction rules, then commit the candidate only when validation succeeds. Map failures with `py_error`.

- [ ] **Step 7: Run Rust tests and Clippy**

Run: `cargo fmt --check && cargo test && cargo clippy --all-features --all-targets -- -D warnings`

Expected: all tests pass and Clippy emits no warnings.

- [ ] **Step 8: Commit overlap validation**

```bash
git add src/timeline src/doc/mod.rs src/python/timeline.rs tests/data_behaviour.rs
git commit -m "feat: validate annotation overlaps"
```

### Task 3: Replace storage with schema v6 only

**Files:**
- Delete: `src/doc/legacy.rs`
- Modify: `src/doc/mod.rs`
- Modify: `src/db/mod.rs`
- Modify: `src/db/schema.rs`
- Modify: `src/db/query.rs`
- Modify: `src/lib.rs`
- Modify: `tests/data_behaviour.rs`
- Modify: `tests/public_api.rs`

- [ ] **Step 1: Replace migration tests with schema v6 tests**

Add a test that a new database reports version 6 and a loop that creates valid application-ID databases with versions 1 through 5 and expects `UnsupportedSchema`:

```rust
assert_eq!(AudioDb::SCHEMA_VERSION, 6);
for version in 1..=5 {
    let error = AudioDb::open(&path, AudioDbMode::ReadWrite).unwrap_err();
    assert!(matches!(
        error,
        AudioDbError::UnsupportedSchema { found, expected: 6 } if found == version
    ));
}
```

Remove tests for migration, old read-only decoding, and legacy MessagePack import.

- [ ] **Step 2: Run database tests and verify they fail**

Run: `cargo test audio_db_schema_v6`

Expected: failure because the current schema is 5 and old versions migrate or open.

- [ ] **Step 3: Simplify schema initialization and validation**

Set `SCHEMA_VERSION` to 6. For nonzero `user_version`, call `validate()` directly. Accept only application ID `VASR` and exact version 6. Delete migration constants and all migration functions.

- [ ] **Step 4: Remove legacy query and import paths**

Delete `import_legacy_msgpack_to_db`, `AudioDbTransaction`, legacy error variants, version-dependent SQL, mono timeline wrapping, and duration repair. Decode only the current `BTreeMap<AudioChannel, Timeline>` shape.

- [ ] **Step 5: Delete legacy document support and exports**

Delete `src/doc/legacy.rs`, its module declaration and exports, and the public crate exports/imports. Remove all remaining `legacy` API references from current source and tests.

- [ ] **Step 6: Run database and full Rust tests**

Run: `rg -n "import_legacy|read_legacy|LegacyImport|migrate_v[1-5]|LEGACY_SCHEMA" src tests`

Expected: no matches.

Run: `cargo fmt --check && cargo test && cargo check --all-features --all-targets`

Expected: all commands pass.

- [ ] **Step 7: Commit schema v6**

```bash
git add src/db src/doc src/lib.rs tests/data_behaviour.rs tests/public_api.rs
git commit -m "refactor: replace legacy storage with schema v6"
```

### Task 4: Update the Python API and observable validation

**Files:**
- Modify: `src/python/common.rs`
- Modify: `src/python/timeline.rs`
- Modify: `asr_data/_types.py`
- Modify: `asr_data/annotation.py`
- Modify: `asr_data/annotation.pyi`
- Modify: `asr_data/__init__.py`
- Modify: `asr_data/__init__.pyi`
- Modify: `asr_data/_native.pyi`
- Modify: `tests/test_bindings.py`

- [ ] **Step 1: Change Python tests to the new API**

Remove `AnnotationStatus` imports/assertions, remove `status=` arguments, and assert `Annotation` has no status property. Add tests for reference overlap rules, prediction per-source rules, duplicate idempotency, atomic payload replacement, and atomic relabel:

```python
first = timeline.reference.add_speech(0, 100)
assert timeline.reference.add_speech(0, 100).id == first.id
with pytest.raises(AsrDataError, match="overlap"):
    timeline.reference.add_speech(50, 150)
```

- [ ] **Step 2: Run focused Python tests and verify failure**

Run: `env -u CONDA_PREFIX uv run maturin develop`

Run: `uv run --with pytest python -m pytest tests/test_bindings.py -q`

Expected: failures from stale status exports/signatures and missing overlap error propagation.

- [ ] **Step 3: Remove Python status bindings and types**

Delete status parsing/rendering helpers, the `Annotation.status` getter and repr field, `AnnotationStatus` aliases/exports, and every `status` PyO3 argument. Route insertion through the new fallible Rust methods.

- [ ] **Step 4: Propagate atomic mutation errors**

Map Rust insertion/relabel/validation errors through `AsrDataError`. Ensure payload replacement and relabel leave the original collection byte-for-byte unchanged on error.

- [ ] **Step 5: Rebuild and run Python tests**

Run: `env -u CONDA_PREFIX uv run maturin develop`

Run: `uv run --with pytest python -m pytest tests/test_bindings.py -q`

Expected: all Python binding tests pass.

- [ ] **Step 6: Commit the Python API change**

```bash
git add src/python asr_data tests/test_bindings.py
git commit -m "feat: expose annotation overlap validation"
```

### Task 5: Update documentation and run final verification

**Files:**
- Modify: `README.md`
- Modify in `../libraries`: `docs/asr-data/api.mdx`
- Modify in `../libraries`: `docs/asr-data/annotations.mdx`
- Modify in `../libraries`: `docs/asr-data/database.mdx`
- Modify in `../libraries`: `docs/asr-data/rust.mdx`
- Modify in `../libraries`: `meta/type-annotations.json`

- [ ] **Step 1: Update project documentation**

Remove status and legacy-storage references. Document half-open ranges, reference overlap rules, per-source prediction rules, atomic relabel, and schema v6 incompatibility.

- [ ] **Step 2: Verify the asr-data repository**

Run: `cargo fmt --check && cargo test && cargo check --all-features --all-targets && cargo clippy --all-features --all-targets -- -D warnings`

Run: `env -u CONDA_PREFIX uv run maturin develop && uv run --with pytest python -m pytest tests/test_bindings.py -q && uv run python -m compileall -q asr_data && git diff --check`

Expected: every command exits successfully.

- [ ] **Step 3: Commit project documentation**

```bash
git add README.md
git commit -m "docs: describe annotation validation"
```

- [ ] **Step 4: Verify the libraries site**

Run from `../libraries`: `./node_modules/.bin/prettier --check docs/asr-data meta/type-annotations.json`

Run from `../libraries`: `npm test -- --runInBand && npm run build`

Expected: formatter, 38 tests, and static export pass.

- [ ] **Step 5: Commit libraries documentation**

```bash
git add docs/asr-data meta/type-annotations.json
git commit -m "docs: describe annotation validation and schema v6"
```
