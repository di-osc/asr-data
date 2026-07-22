# `normalize_zh` Design

## Goal

Expose the existing Chinese text-normalization pipeline through one consistently named Rust and Python function.

## Public API

Rust exposes:

```rust
pub fn normalize_zh(text: &str) -> Result<String, TextNormalizationError>
```

Python exposes:

```python
def normalize_zh(text: str) -> str: ...
```

The Python function is available directly from `asr_data`:

```python
from asr_data import normalize_zh

assert normalize_zh("2024年") == "二零二四年"
```

## Behavior

- Reuse the existing embedded WeText FST resources and process-wide cached normalizer.
- Preserve the current handling of empty input and surrounding whitespace.
- Preserve `Timeline.eval(normalize=True)` behavior; only its internal Rust call changes name.
- Rust errors remain `TextNormalizationError`.
- Python converts normalization failures to `AsrDataError` through the existing Python error mapping.

## Compatibility

- Remove the Rust name `normalize_zh_tn` directly, without a deprecated alias.
- Do not add a `Normalizer` class.
- Do not add language, mode, batch, or configuration parameters.
- Do not change the public `Timeline.eval` signature.

## Code Organization

- Rename the Rust function and all internal/public references in the metrics and timeline modules.
- Add a focused Python metrics binding module containing `normalize_zh`.
- Register and re-export the function from the native and top-level Python modules.
- Update Rust public-API tests, Python binding tests, and Python type stubs.
- Keep the README unchanged for this small utility addition.

## Verification

- Rust tests cover the renamed function and public export.
- Python tests cover the top-level import, representative Chinese normalization, empty input, and type errors.
- Full Rust tests, all-feature checks, Clippy, Python extension build, and Python binding tests must pass.
