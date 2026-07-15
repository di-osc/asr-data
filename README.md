# asr-data

`asr-data` provides a Rust audio/transcript data model, a SQLite-backed
`AudioDb`, audio loading utilities, and Python bindings powered by PyO3. All
Rust code, including the binding implementation, lives in `src`; the Python
package lives in the root-level `asr_data` directory.

## Rust

```bash
cargo test --workspace
```

The command-line utility is available as `asr-data`:

```bash
cargo run --bin asr-data -- --help
```

## Python

Build and install the Python extension from the project root:

```bash
maturin develop
pytest tests/test_bindings.py
```

The public Python package is imported as `asr_data`.

`AudioDB` exposes four explicit data operations; Python container protocols
provide lookup, membership, length, and iteration:

```python
from asr_data import AudioDB

db = AudioDB("data.vasr")
db.insert(audio)
batch = db.query(limit=100, metadata={"split": "train"})
audio = db["audio-id"]
changed = db.update(audio)
deleted = db.delete("audio-id")
```

Queries are ordered by `audio_id`. Pass the last ID from one batch as the
exclusive cursor for the next batch:

```python
next_batch = db.query(limit=100, after=batch[-1].id)
```

`query` also accepts inclusive `min_duration_ms` and `max_duration_ms` filters.
Iteration uses the same cursor internally, so `for audio in db` reads the
database lazily in bounded batches.

Existing `.vasr` SQLite databases remain supported. The file extension and
SQLite application ID are retained as a stable on-disk format identifier.
