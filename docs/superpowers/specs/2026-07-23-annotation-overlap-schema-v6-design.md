# Annotation Overlap and Schema v6 Design

## Goal

Make stored annotations represent only the current valid state, enforce deterministic overlap rules for speech, speaker, and transcription annotations, and remove all historical database and MessagePack compatibility.

## Annotation Model

- Remove `AnnotationStatus` and `Annotation.status` from Rust and Python.
- Remove status parameters from every annotation constructor and Python `add_*` method.
- Existing annotations are always active. Revision replaces content and deletion physically removes an annotation.
- Transcript composition and evaluation use every matching annotation instead of filtering for `Final`.
- Keep the existing half-open time-range rule: `[start, end)` may touch another range at an endpoint without overlapping.
- Keep exact duplicate insertion idempotent. Content equality ignores only the generated annotation ID.

## Reference Validation

Validation is local to one channel timeline's reference collection.

- Speech annotations may not overlap other speech annotations.
- Speaker annotations with the same exact, case-sensitive `name` may not overlap.
- Speaker annotations with different names may overlap.
- Top-level transcription annotations may not overlap other top-level transcription annotations.
- A top-level transcription may not overlap a speaker annotation that contains a transcription.
- Speaker annotations with different names may contain overlapping transcriptions.
- Speech may overlap speaker and transcription annotations.
- Payload types outside speech, speaker, and transcription receive no new overlap rule.

## Prediction Validation

- Every prediction annotation must have a source containing at least one non-whitespace character.
- Sources are compared exactly and remain case-sensitive; they are not trimmed or normalized.
- Different sources are independent and may overlap freely.
- Within one source, apply the same speech, speaker-name, transcription, and speaker-transcription rules as reference annotations.
- Relabeling a source validates the complete candidate prediction collection first. Any conflict aborts the entire relabel without mutation.

## Mutation and Errors

- Consolidate Rust insertion into idempotent, fallible `push_reference` and `push_prediction` methods; remove the separate `push_*_unique` methods.
- Return a structured timeline annotation error for missing prediction sources and overlap conflicts. Conflict data includes annotation IDs, ranges, kind, source when present, and speaker name when relevant.
- Validate duplicate content before overlap so exact repeated inserts return the existing annotation.
- Python maps insertion, payload replacement, and relabel errors to `AsrDataError`.
- Replacing `Annotation.payload` uses a candidate copy and commits only after collection validation succeeds.
- `AudioDoc.validate()` repeats source and overlap checks so direct Rust vector mutation cannot write invalid data.
- Database insert and update continue calling `AudioDoc.validate()` before serialization.
- Do not merge, trim, split, overwrite, or otherwise repair conflicting annotations automatically.

## Database and Compatibility

- Set the database schema version to `6`.
- A new database is created directly as version 6.
- Opening any existing nonzero version other than 6 returns `UnsupportedSchema` in both read-write and read-only modes.
- Remove all v1-v5 schema migration functions, version-specific SQL, and version-specific timeline decoding.
- Remove `read_legacy_msgpack`, `import_legacy_msgpack_to_db`, `LegacyImportError`, and the transaction wrapper used only by import.
- Remove legacy annotation, speaker, timeline, and flat-annotation deserialization adapters that only supported historical MessagePack records.
- Do not provide a migration CLI, deprecated alias, or compatibility mode.

## Python API

- Remove `AnnotationStatus` from `_types.py`, `annotation.py`, package exports, and stubs.
- Remove `Annotation.status`.
- Remove `status=` from reference and prediction `add_transcription` and `add_speaker`.
- Preserve required prediction `source=` parameters and exact-duplicate idempotency.
- Conflict operations raise `AsrDataError` without partially modifying collections.

## Documentation and Verification

- Update README examples and the libraries asr-data Python, Rust, annotation, and database documentation.
- Add Rust tests for every reference and per-source prediction overlap rule, endpoint adjacency, duplicate idempotency, payload replacement validation, atomic source relabel, document validation, schema v6 creation, and rejection of versions 1-5.
- Add Python tests for the public API removals and the same observable overlap and atomicity behavior.
- Remove migration and legacy import tests.
- Run formatting, all Rust tests, all-feature checks, Clippy, Python extension build/tests, and the libraries formatter/tests/static build.
