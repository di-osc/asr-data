# `normalize_zh` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the Rust Chinese text normalizer to `normalize_zh` and expose the same function at the Python package root.

**Architecture:** Keep the existing embedded WeText FST implementation and cache unchanged. Rename its Rust entry point and add a small PyO3 metrics module that delegates to it and maps errors through the existing `AsrDataError` helper.

**Tech Stack:** Rust, PyO3, Python type stubs, pytest, Cargo tests

---

### Task 1: Rename the Rust public API

**Files:**
- Modify: `tests/public_api.rs`
- Modify: `src/metrics/normalization.rs`
- Modify: `src/metrics/mod.rs`
- Modify: `src/lib.rs`
- Modify: `src/timeline/evaluation.rs`

- [ ] **Step 1: Change the public API test to require the new name**

Replace the `normalize_zh_tn` import and call in `tests/public_api.rs`:

```rust
// In the existing grouped `use asr_data::{ ... }` declaration:
compute_cer, import_legacy_msgpack_to_db, normalize_for_cer, normalize_zh,
read_audio_db_info, read_legacy_msgpack,

// In `stable_public_paths_compile`:
let _ = normalize_zh("2026");
```

- [ ] **Step 2: Run the public API test and verify it fails**

Run: `cargo test --test public_api`

Expected: compilation fails because `asr_data::normalize_zh` is not exported.

- [ ] **Step 3: Rename the Rust function and every internal export/call**

In `src/metrics/normalization.rs`:

```rust
pub fn normalize_zh(text: &str) -> Result<String, TextNormalizationError> {
    static NORMALIZER: OnceLock<Result<ChineseTn, TextNormalizationError>> = OnceLock::new();
    match NORMALIZER.get_or_init(ChineseTn::embedded) {
        Ok(normalizer) => normalizer.normalize(text),
        Err(error) => Err(error.clone()),
    }
}
```

Export `normalize_zh` from `src/metrics/mod.rs` and `src/lib.rs`. In `src/timeline/evaluation.rs`, import `normalize_zh` and invoke it as follows:

```rust
normalize_zh(text).map(|text| normalize_for_cer(&text, true))
```

In the normalization tests, use the renamed import and call:

```rust
use super::{normalize_zh, reorder_zh_tn_tokens};

assert_eq!(normalize_zh("2024年"), Ok("二零二四年".to_owned()));
```

Do not retain `normalize_zh_tn`.

- [ ] **Step 4: Run Rust tests and verify the rename**

Run: `cargo test`

Expected: all Rust tests pass.

- [ ] **Step 5: Commit the Rust rename**

```bash
git add tests/public_api.rs src/metrics/normalization.rs src/metrics/mod.rs src/lib.rs src/timeline/evaluation.rs
git commit -m "refactor: rename Chinese text normalizer"
```

### Task 2: Expose `normalize_zh` to Python

**Files:**
- Create: `src/python/metrics.rs`
- Modify: `src/python/mod.rs`
- Modify: `asr_data/__init__.py`
- Modify: `asr_data/__init__.pyi`
- Modify: `asr_data/_native.pyi`
- Modify: `tests/test_bindings.py`

- [ ] **Step 1: Add a failing Python public-API test**

Add to `tests/test_bindings.py`:

```python
def test_normalize_zh_is_public():
    from asr_data import normalize_zh

    assert normalize_zh("2024年") == "二零二四年"
    assert normalize_zh("") == ""
    with pytest.raises(TypeError):
        normalize_zh(2024)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `uv run --with pytest python -m pytest tests/test_bindings.py::test_normalize_zh_is_public -q`

Expected: failure because `normalize_zh` cannot be imported from `asr_data`.

- [ ] **Step 3: Add the minimal PyO3 binding**

Create `src/python/metrics.rs`:

```rust
use pyo3::prelude::*;

use crate::normalize_zh as rust_normalize_zh;

use super::common::py_error;

#[pyfunction]
fn normalize_zh(text: &str) -> PyResult<String> {
    rust_normalize_zh(text).map_err(py_error)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(normalize_zh, module)?)?;
    Ok(())
}
```

Declare `mod metrics;` and call `metrics::register(module)?;` in `src/python/mod.rs`.

- [ ] **Step 4: Export and type the Python function**

Import `normalize_zh` from `._native` and include it in `__all__` in `asr_data/__init__.py`. Add these declarations:

```python
# asr_data/_native.pyi
def normalize_zh(text: str) -> str: ...

# asr_data/__init__.pyi
from ._native import normalize_zh as normalize_zh
```

- [ ] **Step 5: Rebuild the extension and verify the focused test passes**

Run: `env -u CONDA_PREFIX uv run maturin develop`

Run: `uv run --with pytest python -m pytest tests/test_bindings.py::test_normalize_zh_is_public -q`

Expected: one test passes.

- [ ] **Step 6: Verify old naming is absent and run the full suite**

Run: `rg -n "normalize_zh_tn" src tests asr_data README.md`

Expected: no matches.

Run: `cargo fmt --check && cargo test && cargo check --all-features --all-targets && cargo clippy --all-features --all-targets -- -D warnings`

Run: `env -u CONDA_PREFIX uv run maturin develop && uv run --with pytest python -m pytest tests/test_bindings.py -q && uv run python -m compileall -q asr_data && git diff --check`

Expected: every command exits successfully.

- [ ] **Step 7: Commit the Python binding**

```bash
git add src/python/metrics.rs src/python/mod.rs asr_data/__init__.py asr_data/__init__.pyi asr_data/_native.pyi tests/test_bindings.py
git commit -m "feat: expose Chinese text normalization to Python"
```
