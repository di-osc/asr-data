# Project Module Reorganization Design

## Goal

Reorganize the Rust core and Python bindings by domain without changing runtime behavior, serialized data, database schemas, or existing public APIs.

## Compatibility Requirements

- Preserve all existing crate-root imports, including `asr_data::{Audio, AudioDb, AudioDoc}`.
- Preserve existing `asr_data::audio::*` paths, including the public `audio::decode` module.
- Preserve Python class names, method signatures, exceptions, imports, and type declarations.
- Preserve Serde field names and defaults, legacy MessagePack migration, SQLite schemas, and migration behavior.
- Treat the work as a structural refactor only. Do not combine feature or behavior changes with it.

## Target Layout

```text
src/
├── lib.rs
├── utils.rs
├── audio/
│   ├── mod.rs
│   ├── data.rs
│   ├── source.rs
│   ├── decode.rs
│   └── stream.rs
├── db/
│   ├── mod.rs
│   ├── schema.rs
│   └── query.rs
├── doc/
│   ├── mod.rs
│   └── legacy.rs
├── timeline/
│   ├── mod.rs
│   ├── annotation.rs
│   ├── data.rs
│   ├── segment.rs
│   └── token.rs
├── metrics/
│   ├── mod.rs
│   └── cer.rs
└── python/
    ├── mod.rs
    ├── common.rs
    ├── audio.rs
    ├── timeline.rs
    ├── doc.rs
    └── db.rs
```

## Rust Module Responsibilities

### Utilities

`utils.rs` contains only the shared `DurationMs`, `SampleIndex`, and `TimeRange` types. It is not a general dumping ground for unrelated helpers.

### Audio

- `audio/data.rs`: `Audio`, `AudioChunk`, chunk iteration, slicing, channel operations, and waveform resampling.
- `audio/source.rs`: `AudioSource`, `AudioFormat`, `AudioEncoding`, `AudioChannel`, and synchronous/asynchronous source loading.
- `audio/decode.rs`: format probing and decoding from paths, URLs, encoded bytes, and Base64.
- `audio/stream.rs`: transformed source streams, chunk buffering, and continuous streaming resampling.
- `audio/mod.rs`: module declarations, audio load options, shared transforms, and compatibility re-exports.

### Timeline

- `timeline/annotation.rs`: annotation identifiers, statuses, sources, payloads, events, diagnostics, and `Annotation`.
- `timeline/data.rs`: `Timeline`, duration deserialization, and transcript derivation.
- `timeline/segment.rs`: `TextSpan` and `Transcript`.
- `timeline/token.rs`: `Token`.
- `timeline/mod.rs`: private assembly and crate-facing re-exports.

### Document

- `doc/mod.rs`: `AudioDoc`, channel/timeline operations, validation, errors, and conversions.
- `doc/legacy.rs`: legacy wire structures, MessagePack reading, asset migration, and legacy identifier sanitation.

### Database

- `db/mod.rs`: public database types, errors, modes, transactions, and stable entry points.
- `db/schema.rs`: connection configuration, schema initialization, validation, and migrations.
- `db/query.rs`: CRUD implementation, SQL construction, row encoding/decoding, and MessagePack helpers.

### Metrics

- `metrics/cer.rs`: CER normalization, computation, and statistics.
- `metrics/mod.rs`: metrics assembly and crate-facing re-exports.

## Python Binding Responsibilities

- `python/mod.rs`: exception creation, submodule declarations, and the single `_native` registration entry point.
- `python/common.rs`: shared error conversion, string formatting, and annotation enum conversion helpers.
- `python/audio.rs`: audio data, formats, source conversion, sources, loading tasks, streaming tasks, and iterators.
- `python/timeline.rs`: annotations, transcripts, and timeline bindings.
- `python/doc.rs`: `AudioDoc` binding and document-specific synchronization.
- `python/db.rs`: database and database iterator bindings.

Each Python domain module exposes an internal `register(module)` function. `python/mod.rs` calls these functions in dependency order while keeping every exported Python class and exception name unchanged.

## Dependency Direction

```text
utils
  ↑
audio / timeline / metrics
  ↑
doc
  ↑
db
  ↑
python
```

Internal modules should import from their defining module or a sibling through `super`, not through crate-root compatibility re-exports. `lib.rs` remains a public facade and should contain only module declarations and stable `pub use` statements.

## Migration Sequence

1. Move shared time types, CER metrics, and audio source types while preserving re-exports.
2. Split timeline and document models without changing Serde representations.
3. Split database schema and query responsibilities without changing SQL or schema versions.
4. Split Python bindings and introduce domain registration functions without changing the Python API.

Each phase must compile and pass its relevant tests before the next phase starts. File movement should use Git-aware moves where practical so history remains understandable.

## Verification

- Add an external Rust integration test that imports representative crate-root and `audio` module paths.
- Run `cargo fmt --check` and the complete Rust test suite after every phase.
- Run `cargo check --all-features --all-targets` and strict Clippy after structural moves.
- Rebuild the Python extension with Maturin and run the complete Python binding suite.
- Run repository searches to confirm retired top-level files and stale module paths are gone.
- Run `git diff --check` and inspect the final diff for behavior changes.
- Leave the pre-existing untracked `test.ipynb` untouched.
