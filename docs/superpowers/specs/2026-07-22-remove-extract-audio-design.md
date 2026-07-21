# Remove `extract_audio` Design

## Goal

Remove the unused embedded-audio extraction utility and its public Rust API from the crate.

## Scope

- Delete `src/extract_audio.rs` in full.
- Remove the `extract_audio` module declaration from `src/lib.rs`.
- Remove the public exports `ExtractAudioSummary`, `extract_embedded_audio`, and
  `extract_embedded_audio_from_db`.
- Remove the private `AudioDb::load_all` helper because this module is its only caller.
- Do not replace the utility or add a Python equivalent.
- Do not change audio loading, streaming, database storage, or decoding behavior.

## Compatibility

This intentionally breaks Rust callers that import the three removed public API items. No internal
Rust code, Python bindings, README examples, or other repository tests currently use them.

## Verification

- Search the repository to confirm no references remain.
- Run formatting, Rust tests, all-feature checks, Clippy, Python binding tests, and diff checks.
- Leave the pre-existing untracked `test.ipynb` untouched.
