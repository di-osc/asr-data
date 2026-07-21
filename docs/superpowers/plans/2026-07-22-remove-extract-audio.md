# Remove `extract_audio` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the unused embedded-audio extraction module and its public Rust API.

**Architecture:** Delete the self-contained module and remove its declaration and re-exports from the crate root. No replacement API or unrelated behavior changes are included.

**Tech Stack:** Rust 2024, Cargo, PyO3, pytest

---

## File Structure

- Delete `src/extract_audio.rs`: remove the extraction implementation and its local unit test.
- Modify `src/lib.rs`: remove the private module declaration and three public exports.
- Modify `src/db.rs`: remove the private `load_all` helper left unused by the module deletion.

### Task 1: Remove the module and exports

**Files:**
- Delete: `src/extract_audio.rs`
- Modify: `src/lib.rs`
- Modify: `src/db.rs`

- [x] **Step 1: Verify the removal check fails before implementation**

Run:

```bash
test -z "$(rg -n 'extract_audio|ExtractAudioSummary|extract_embedded_audio' src)"
```

Expected: exit code 1 because the module and exports still exist.

- [x] **Step 2: Delete the implementation and crate-root API**

Delete `src/extract_audio.rs`. In `src/lib.rs`, delete:

```rust
mod extract_audio;
```

and:

```rust
pub use extract_audio::{
    ExtractAudioSummary, extract_embedded_audio, extract_embedded_audio_from_db,
};
```

Delete the now-unused private `AudioDb::load_all` method from `src/db.rs`.

- [x] **Step 3: Verify no references remain**

Run:

```bash
test -z "$(rg -n 'extract_audio|ExtractAudioSummary|extract_embedded_audio' src)"
```

Expected: exit code 0 with no output.

- [x] **Step 4: Run complete verification**

Run:

```bash
cargo fmt --check
cargo test
cargo check --all-features --all-targets
cargo clippy --all-features --all-targets -- -D warnings
env -u CONDA_PREFIX uv run maturin develop
uv run --with pytest python -m pytest tests/test_bindings.py -q
git diff --check
```

Expected: all commands exit successfully; Rust and Python test suites have no failures.

- [x] **Step 5: Commit the removal**

Run:

```bash
git add src/lib.rs src/db.rs src/extract_audio.rs docs/superpowers/specs/2026-07-22-remove-extract-audio-design.md docs/superpowers/plans/2026-07-22-remove-extract-audio.md
git commit -m "refactor: remove unused audio extraction module"
```

Expected: the implementation deletion, crate-root cleanup, and completed plan are committed while `test.ipynb` remains untracked.
