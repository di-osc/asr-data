import asyncio
import ast
import base64
import doctest
import io
import inspect
from pathlib import Path
import struct
import threading
import time
import typing
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import asr_data
import numpy as np
import pytest

from asr_data import (
    AudioDB,
    AudioDataset,
    Audio,
    AudioSource,
    Waveform,
)
from asr_data.annotation import (
    Annotation,
    AudioActivity,
    Speaker,
    Token,
    Transcription,
)


def test_all_public_python_callables_have_standard_runtime_docstrings():
    public_classes = [
        getattr(asr_data, name)
        for name in asr_data.__all__
        if inspect.isclass(getattr(asr_data, name, None)) and name != "AsrDataError"
    ]
    issues = []
    for cls in public_classes:
        if not (cls.__doc__ or "").strip():
            issues.append(f"{cls.__name__}: missing class docstring")
        for name in dir(cls):
            if name.startswith("_"):
                continue
            member = getattr(cls, name)
            if not callable(member):
                continue
            doc = inspect.getdoc(member) or ""
            try:
                signature = inspect.signature(member)
            except (TypeError, ValueError):
                signature = None
            required = ["Returns:", "Examples:"]
            if signature and any(
                parameter.name not in {"self", "cls"}
                for parameter in signature.parameters.values()
            ):
                required.append("Args:")
            for section in required:
                if section not in doc:
                    issues.append(f"{cls.__name__}.{name}: missing {section}")

    public_functions = [
        getattr(asr_data, name)
        for name in asr_data.__all__
        if inspect.isroutine(getattr(asr_data, name, None))
    ]
    for function in public_functions:
        doc = inspect.getdoc(function) or ""
        required = ["Returns:", "Examples:"]
        if any(
            parameter.name not in {"self", "cls"}
            for parameter in inspect.signature(function).parameters.values()
        ):
            required.append("Args:")
        for section in required:
            if section not in doc:
                issues.append(f"{function.__name__}: missing {section}")

    assert issues == []


def test_public_stub_callables_have_standard_docstrings():
    """Keep editor-visible stub docs aligned with the runtime API standard."""

    issues = []
    for relative_path in ("asr_data/_native.pyi", "asr_data/__init__.pyi"):
        path = Path(__file__).parents[1] / relative_path
        tree = ast.parse(path.read_text())

        for parent in ast.walk(tree):
            if not isinstance(parent, (ast.Module, ast.ClassDef)):
                continue
            if isinstance(parent, ast.ClassDef) and parent.name.startswith("_"):
                continue
            for node in parent.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name.startswith("_"):
                    continue
                decorators = {
                    decorator.id
                    for decorator in node.decorator_list
                    if isinstance(decorator, ast.Name)
                }
                has_attribute_decorator = any(
                    isinstance(decorator, ast.Attribute)
                    for decorator in node.decorator_list
                )
                if "property" in decorators or has_attribute_decorator:
                    continue

                doc = ast.get_docstring(node) or ""
                parameters = [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
                required = ["Returns:", "Examples:"]
                if any(
                    parameter.arg not in {"self", "cls"} for parameter in parameters
                ):
                    required.append("Args:")
                for section in required:
                    if section not in doc:
                        issues.append(
                            f"{relative_path}:{node.lineno} "
                            f"{node.name}: missing {section}"
                        )

    assert issues == []


def test_local_public_runtime_docstring_examples_are_executable():
    """Execute every self-contained example that does not require network or Jupyter."""

    finder = doctest.DocTestFinder(exclude_empty=True)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    skipped_name_parts = {
        "from_path",
        "from_url",
        "from_bytes",
        "from_base64",
        "display",
        "aload_from_path",
        "from_modelscope",
    }
    seen = set()

    for export in asr_data.__all__:
        obj = getattr(asr_data, export, None)
        if not (inspect.isclass(obj) or inspect.isroutine(obj)):
            continue
        for test in finder.find(obj, f"asr_data.{export}"):
            if (
                test.name in seen
                or not test.examples
                or any(part in test.name for part in skipped_name_parts)
            ):
                continue
            seen.add(test.name)
            runner.run(test)

    assert runner.failures == 0
    assert runner.tries >= 180


def test_normalize_zh_is_public():
    from asr_data import normalize_zh

    assert normalize_zh("2024年") == "二零二四年"
    assert normalize_zh("") == ""
    with pytest.raises(TypeError):
        normalize_zh(2024)


def test_audio_source_factories_and_variant_properties():
    source_type = asr_data.AudioSource

    path = source_type.from_path("audio.wav")
    assert path.kind == "path"
    assert path.path == "audio.wav"
    assert path.url is None
    assert path.bytes is None
    assert path.base64 is None
    assert path.pcm is None
    assert path.sample_rate is None
    assert path.channels is None

    url = source_type.from_url("https://example.com/audio.wav")
    assert url.kind == "url"
    assert url.url == "https://example.com/audio.wav"

    encoded = source_type.from_bytes(b"encoded")
    assert encoded.kind == "bytes"
    assert encoded.bytes == b"encoded"

    base64_source = source_type.from_base64("ZW5jb2RlZA==")
    assert base64_source.kind == "base64"
    assert base64_source.base64 == "ZW5jb2RlZA=="

    pcm = source_type.from_pcm(b"\0\0", sample_rate=16000)
    assert pcm.kind == "pcm"
    assert pcm.pcm == b"\0\0"
    assert pcm.sample_rate == 16000
    assert pcm.channels == 1

    with pytest.raises(TypeError):
        source_type()

    for old_name in ("AudioPath", "AudioUrl", "AudioBytes", "AudioBase64", "AudioPcm"):
        assert not hasattr(asr_data, old_name)


def test_annotation_payload_types_are_public_and_complete():
    assert asr_data.annotation.Speaker is Speaker
    assert asr_data.annotation.AudioActivity is AudioActivity
    assert asr_data.annotation.Token is Token
    assert asr_data.annotation.Transcription is Transcription
    assert AudioActivity.__module__ == "asr_data.annotation"
    assert Speaker.__module__ == "asr_data.annotation"
    assert Token.__module__ == "asr_data.annotation"
    assert Transcription.__module__ == "asr_data.annotation"
    assert set(typing.get_args(Annotation)) == {
        AudioActivity,
        Speaker,
        Token,
        Transcription,
    }
    assert not hasattr(asr_data.annotation, "AnnotationStatus")
    assert not hasattr(asr_data, "AnnotationStatus")


def test_audio_waveform_timeline_and_db(tmp_path):
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    audio = Audio(AudioSource.from_pcm(pcm, sample_rate=8000, channels=2), id="call-1")
    assert isinstance(audio.source, AudioSource)
    assert set(audio.timelines) == {"left", "right"}
    assert audio.timeline("mono") is None
    audio.metadata["speaker"] = {"name": "alice", "age": 30}
    audio.timeline("left").annotate_span(
        0, 1, AudioActivity(confidence=0.9), is_reference=True
    )
    audio.timeline("left").annotate_span(
        0, 1, Transcription("hello", language="en"), is_reference=True
    )

    waveform = audio.source.load().as_waveform()
    assert waveform.sample_rate == 8000
    assert waveform.channels == 2
    assert waveform.source_format.encoding == "pcm_s16le"
    assert waveform.samples.dtype == np.float32
    assert waveform.samples.shape == (4,)
    left = waveform.channel(0)
    right = waveform.channel(1)
    assert left.channels == 1
    assert right.channels == 1
    np.testing.assert_allclose(
        left.samples,
        np.array([0.0, -1000 / 32768], dtype=np.float32),
    )
    np.testing.assert_allclose(
        right.samples,
        np.array([1000 / 32768, 2000 / 32768], dtype=np.float32),
    )

    mono = waveform.to_mono().resample(16000)
    assert mono.channels == 1
    assert mono.sample_rate == 16000
    assert mono.source_format.sample_rate == 8000
    assert audio.timeline("left").reference.transcript().text == "hello"
    assert len(audio.timeline("left").reference.spans) == 2

    db_path = tmp_path / "test.vasr"
    db = AudioDB.create(str(db_path))
    for removed in (
        "upsert",
        "insert_many",
        "get",
        "list",
        "all",
        "remove",
        "contains",
        "relabel_annotation_sources",
    ):
        assert not hasattr(db, removed)
    db.insert(audio)
    db.set_metadata("dataset", {"name": "calls"})
    assert db.metadata_value("dataset") == {"name": "calls"}
    assert db.metadata["dataset"]["name"] == "calls"
    assert len(db) == 1
    assert [item.id for item in db] == ["call-1"]
    assert "call-1" in db

    loaded = db["call-1"]
    assert loaded.id == "call-1"
    assert isinstance(loaded, Audio)
    assert isinstance(loaded.source, AudioSource)
    assert loaded.metadata["speaker"]["name"] == "alice"
    assert loaded.timeline("left").reference.transcript().text == "hello"
    loaded.metadata["speaker"]["name"] = "bob"
    assert db.update(loaded) is True
    assert db["call-1"].metadata["speaker"]["name"] == "bob"
    assert not hasattr(loaded, "set_metadata")
    with pytest.raises(KeyError):
        _ = db["missing"]
    assert db.delete("call-1") is True
    assert db.delete("call-1") is False


def test_audio_db_create_and_open_are_explicit(tmp_path):
    path = tmp_path / "explicit.db"

    with pytest.raises(TypeError):
        AudioDB(str(path))
    with pytest.raises(FileNotFoundError):
        AudioDB.open(str(path))

    created = AudioDB.create(str(path))
    assert len(created) == 0
    with pytest.raises(FileExistsError):
        AudioDB.create(str(path))

    opened = AudioDB.open(str(path))
    readonly = AudioDB.open(str(path), read_only=True)
    assert len(opened) == len(readonly) == 0


def test_audio_dataset_wraps_optional_audio_databases(tmp_path):
    train = AudioDB.create(str(tmp_path / "train.db"))
    test = AudioDB.create(str(tmp_path / "test.db"))
    audio = Audio(
        AudioSource.from_pcm(b"\0\0" * 10, sample_rate=16000),
        id="sample-1",
    )
    train.insert(audio)

    dataset = AudioDataset(train=train, test=test)

    assert dataset.name == ""
    assert dataset.version == ""
    assert dataset.license == ""
    assert [item.id for item in dataset.train] == ["sample-1"]
    assert dataset.val is None
    assert len(dataset.test) == 0
    assert not hasattr(dataset, "split")
    assert not hasattr(dataset, "db")
    assert "AudioDataset" in repr(dataset)
    assert "val=None" in repr(dataset)
    assert str(dataset) == ""

    with pytest.raises(AttributeError):
        dataset.name = "other"

    empty = AudioDataset()
    assert empty.train is empty.val is empty.test is None


def test_audio_dataset_from_modelscope_validates_identity_before_download():
    with pytest.raises(ValueError, match="repository id"):
        AudioDataset.from_modelscope("")
    with pytest.raises(ValueError, match="revision"):
        AudioDataset.from_modelscope("di-osc/calls", revision="")


def test_ensure_timeline_accepts_fractional_audio_duration():
    waveform = Waveform([0.0, 0.0], sample_rate=3)
    doc = Audio(AudioSource.from_pcm(b"\0\0" * 2, sample_rate=3), id="fractional")

    timeline = doc.ensure_timeline("mono", duration_ms=waveform.duration_ms)

    assert waveform.duration_ms == pytest.approx(1000 * 2 / 3)
    assert timeline.duration_ms == 667

    for invalid in (-1.0, float("nan"), float("inf")):
        invalid_doc = Audio(
            AudioSource.from_pcm(b"\0\0", sample_rate=3),
            id=f"invalid-{invalid}",
        )
        with pytest.raises(ValueError, match="finite non-negative"):
            invalid_doc.ensure_timeline("mono", duration_ms=invalid)


def test_audio_source_probe_and_audiodoc_initialize_timelines():
    source = AudioSource.from_pcm(b"\0\0" * 6, sample_rate=1000, channels=2)
    info = source.probe()
    doc = Audio(source, id="stereo")

    assert info.sample_rate == 1000
    assert info.channels == 2
    assert info.frame_count == 3
    assert info.duration_ms == pytest.approx(3)
    assert info.source_format.encoding == "pcm_s16le"
    assert doc.info.sample_rate == info.sample_rate
    assert doc.info.channels == info.channels
    assert doc.info.frame_count == info.frame_count
    assert doc.info.duration_ms == info.duration_ms
    assert list(doc.timelines) == ["left", "right"]
    assert doc.timeline("left").duration_ms == 3
    assert doc.timeline("right").duration_ms == 3
    assert doc.timeline("mono") is None


def test_audiodoc_uses_mono_and_indexed_multichannel_timelines():
    mono_doc = Audio(AudioSource.from_pcm(b"\0\0" * 2, sample_rate=3))

    assert list(mono_doc.timelines) == ["mono"]
    assert mono_doc.timeline("mono").duration_ms == 667
    assert mono_doc.timeline().id == mono_doc.timeline("mono").id

    multi_doc = Audio(AudioSource.from_pcm(b"\0\0" * 8, sample_rate=1000, channels=4))
    assert list(multi_doc.timelines) == ["left", "right", "2", "3"]

    with pytest.raises(asr_data.AsrDataError, match="whole number"):
        AudioSource.from_pcm(b"\0\0\0", sample_rate=16000, channels=2).probe()


def test_audio_source_aprobe_and_aload():
    source = AudioSource.from_pcm(b"\0\0" * 4, sample_rate=1000, channels=2)

    async def run():
        return await source.aprobe(), await source.aload(id="async")

    info, doc = asyncio.run(run())
    assert (info.frame_count, info.channels) == (2, 2)
    assert doc.id == "async"
    assert list(doc.timelines) == ["left", "right"]


def test_audio_db_query_filters_cursor_and_lazy_iteration(tmp_path):
    db = AudioDB.create(str(tmp_path / "query.vasr"))
    for index in range(105):
        audio = Audio(
            AudioSource.from_pcm(b"\0\0" * (index * 10), sample_rate=1000),
            id=f"audio-{index:03}",
        )
        audio.metadata["split"] = "train" if index % 2 == 0 else "test"
        db.insert(audio)

    first = db.query(
        5,
        min_duration_ms=200,
        max_duration_ms=400,
        metadata={"split": "train"},
    )
    assert [audio.id for audio in first] == [
        "audio-020",
        "audio-022",
        "audio-024",
        "audio-026",
        "audio-028",
    ]

    second = db.query(
        limit=5,
        after=first[-1].id,
        min_duration_ms=200,
        max_duration_ms=400,
        metadata={"split": "train"},
    )
    assert [audio.id for audio in second] == [
        "audio-030",
        "audio-032",
        "audio-034",
        "audio-036",
        "audio-038",
    ]

    assert db["audio-104"].id == "audio-104"
    assert [audio.id for audio in db] == [f"audio-{index:03}" for index in range(105)]


def test_audio_channel_timelines_round_trip(tmp_path):
    audio = Audio(
        AudioSource.from_pcm(b"\0\0" * 200, sample_rate=1000, channels=2),
        id="call-stereo",
    )

    audio.timeline("left").annotate_span(
        0, 100, Transcription("caller"), is_reference=True
    )
    audio.timeline("right").annotate_span(
        0, 100, Transcription("agent"), is_reference=True
    )

    assert audio.timeline("mono") is None
    assert audio.timeline("left").reference.transcript().text == "caller"
    assert audio.timeline(0).reference.transcript().text == "caller"
    assert audio.timeline("right").reference.transcript().text == "agent"
    assert audio.timeline(1).reference.transcript().text == "agent"
    assert set(audio.timelines) == {"left", "right"}
    assert not hasattr(audio, "channel_timeline")
    assert audio.timeline(2) is None
    created = audio.ensure_timeline(2)
    assert created.id == audio.timeline(2).id
    assert audio.remove_timeline(2) is True
    assert audio.timeline(2) is None

    db = AudioDB.create(str(tmp_path / "stereo.sqlite"))
    db.insert(audio)
    loaded = db["call-stereo"]

    assert loaded.timeline("left").reference.transcript().text == "caller"
    assert loaded.timeline("right").reference.transcript().text == "agent"


def test_setting_timeline_audio_id_updates_the_whole_audio():
    audio = Audio(
        AudioSource.from_pcm(b"\0\0" * 200, sample_rate=1000, channels=2),
        id="old-id",
    )

    audio.timeline("left").audio_id = "new-id"

    assert audio.id == "new-id"
    assert audio.timeline("left").audio_id == "new-id"
    assert audio.timeline("right").audio_id == "new-id"


def test_timeline_duration_is_required_shared_and_read_only():
    audio = Audio(AudioSource.from_pcm(b"\0\0" * 4000, sample_rate=8000), id="duration")
    assert not hasattr(audio, "duration_ms")

    mono = audio.timeline("mono")
    right = audio.ensure_timeline("right")
    assert mono.duration_ms == 500
    assert right.duration_ms == 500
    with pytest.raises(Exception, match="duration mismatch"):
        audio.ensure_timeline("left", duration_ms=600)
    with pytest.raises(AttributeError):
        mono.duration_ms = 600

    audio.validate()


def test_audio_from_numpy_shares_input():
    samples = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    waveform = Waveform(samples, 16000)
    view = waveform.samples
    assert np.shares_memory(samples, view)
    samples[:] = 1.0
    np.testing.assert_array_equal(view, samples)


def test_waveform_from_pcm_matches_source_load():
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    source = AudioSource.from_pcm(pcm, sample_rate=8000, channels=2)
    via_source = source.load().as_waveform()
    via_waveform = Waveform.from_pcm(pcm, sample_rate=8000, channels=2)
    assert via_waveform.sample_rate == via_source.sample_rate
    assert via_waveform.channels == via_source.channels
    np.testing.assert_allclose(via_waveform.samples, via_source.samples)


def test_source_load_options():
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    source = AudioSource.from_pcm(pcm, sample_rate=8000, channels=2)

    transformed = source.load().as_waveform().to_mono().resample(16000)
    assert transformed.sample_rate == 16000
    assert transformed.channels == 1
    assert transformed.samples.dtype == np.float32
    assert np.isfinite(transformed.samples).all()
    assert np.abs(transformed.samples).max(initial=0.0) <= 1.0

    preserved = source.load().as_waveform()
    assert preserved.sample_rate == 8000
    assert preserved.channels == 2

    with pytest.raises(Exception, match="sample rate"):
        preserved.resample(0)


def test_all_audio_sources_stream_waveforms_without_padding(tmp_path):
    samples = (100, 1000, 200, 2000, 300, 3000, 400, 4000, 500, 5000)
    pcm = struct.pack("<" + "h" * len(samples), *samples)
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(pcm)
    wav_bytes = encoded.getvalue()
    path = tmp_path / "stream.wav"
    path.write_bytes(wav_bytes)

    sources = [
        AudioSource.from_path(str(path)),
        AudioSource.from_url(path.as_uri()),
        AudioSource.from_bytes(wav_bytes),
        AudioSource.from_base64(base64.b64encode(wav_bytes).decode()),
        AudioSource.from_pcm(pcm, sample_rate=1000, channels=2),
    ]

    for source in sources:
        full = source.load().as_waveform()
        chunks = list(source.stream(chunk_size_ms=2))

        assert all(type(chunk).__name__ == "AudioChunk" for chunk in chunks)
        assert [chunk.info.frame_count for chunk in chunks] == [2, 2, 1]
        assert [chunk.offset_ms for chunk in chunks] == [0, 2, 4]
        assert [chunk.is_final for chunk in chunks] == [False, False, True]
        assert all(chunk.info.sample_rate == 1000 for chunk in chunks)
        assert all(chunk.info.channels == 2 for chunk in chunks)
        np.testing.assert_array_equal(
            np.concatenate([chunk.as_waveform().samples for chunk in chunks]),
            full.samples,
        )

        chunk_waveform = chunks[0].as_waveform()
        first_view = chunk_waveform.samples
        second_view = chunk_waveform.samples
        assert np.shares_memory(first_view, second_view)
        assert first_view.flags.writeable is False
        with pytest.raises(ValueError):
            first_view[0] = 0

    with pytest.raises(Exception, match="chunk_size_ms must be greater than zero"):
        list(sources[0].stream(chunk_size_ms=0))


def test_stream_chunks_can_be_transformed_as_waveforms(tmp_path):
    samples = (100, 1000, 200, 2000, 300, 3000, 400, 4000, 500, 5000)
    pcm = struct.pack("<" + "h" * len(samples), *samples)
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(pcm)
    wav_bytes = encoded.getvalue()
    path = tmp_path / "stream-options.wav"
    path.write_bytes(wav_bytes)
    sources = [
        AudioSource.from_path(str(path)),
        AudioSource.from_url(path.as_uri()),
        AudioSource.from_bytes(wav_bytes),
        AudioSource.from_base64(base64.b64encode(wav_bytes).decode()),
        AudioSource.from_pcm(pcm, sample_rate=1000, channels=2),
    ]

    for source in sources:
        full = source.load().as_waveform().to_mono()
        chunks = list(source.stream(chunk_size_ms=2))
        transformed = [chunk.as_waveform("mono").resample(2000) for chunk in chunks]

        assert chunks
        assert all(chunk.info.sample_rate == 1000 for chunk in chunks)
        assert all(chunk.info.channels == 2 for chunk in chunks)
        assert [chunk.offset_ms for chunk in chunks] == sorted(
            chunk.offset_ms for chunk in chunks
        )
        assert [chunk.is_final for chunk in chunks[:-1]] == [False] * (len(chunks) - 1)
        assert chunks[-1].is_final is True
        streamed = np.concatenate([waveform.samples for waveform in transformed])
        original = np.concatenate(
            [chunk.as_waveform("mono").samples for chunk in chunks]
        )
        np.testing.assert_allclose(original, full.samples, atol=1e-6)
        assert all(waveform.sample_rate == 2000 for waveform in transformed)
        assert np.isfinite(streamed).all()
        assert np.abs(streamed).max(initial=0.0) <= 1.0

    with pytest.raises(Exception, match="sample rate"):
        sources[0].load().as_waveform().resample(0)


def test_source_stream_default_chunk():
    pcm = struct.pack("<" + "h" * 250, *range(250))

    source = AudioSource.from_pcm(pcm, sample_rate=1000)
    stream = source.stream()
    chunks = list(stream)

    assert [chunk.info.frame_count for chunk in chunks] == [100, 100, 50]
    assert [chunk.offset_ms for chunk in chunks] == [0, 100, 200]
    assert [chunk.is_final for chunk in chunks] == [False, False, True]
    assert type(stream).__name__ == "AudioStream"
    assert hasattr(source, "stream")
    assert not hasattr(source, "astream")
    assert not hasattr(source, "open")
    assert not hasattr(source, "aopen")


def test_audio_stream_lifecycle_growing_timeline_and_conversion():
    source = AudioSource.from_pcm(b"\0\0" * 250, sample_rate=1000)
    stream = source.stream(chunk_size_ms=100, id="stream")
    stream.metadata["model"] = "streaming-vad"
    assert stream.id == "stream"
    assert stream.position_ms == 0
    assert stream.timeline("mono").duration_ms == 0
    assert stream.as_waveform().frame_count == 0

    first = next(stream)
    assert first.id == stream.id
    assert first.index == 0
    assert first.source.kind == stream.source.kind
    assert first.info.frame_count == 100
    assert first.info.duration_ms == 100
    assert first.offset_ms == 0
    assert first.end_ms == 100
    assert first.to_timeline_range(25, 75) == (25, 75)
    assert first.timeline("mono").id == stream.timeline("mono").id
    assert first.metadata is stream.metadata
    assert first.timelines["mono"].id == stream.timelines["mono"].id
    assert stream.timeline("mono").duration_ms == 100
    assert stream.as_waveform().frame_count == 100
    span = stream.timeline("mono").annotate_span(
        0,
        100,
        AudioActivity(event="speech"),
        source="vad",
        is_reference=False,
    )

    chunks = [first, *list(stream)]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert [chunk.end_ms for chunk in chunks] == [100, 200, 250]
    assert first.timeline("mono").duration_ms == 250
    assert first.metadata["model"] == "streaming-vad"
    assert stream.is_complete is True
    assert stream.is_closed is False
    audio = stream.to_audio()
    np.testing.assert_array_equal(
        audio.as_waveform().samples,
        np.concatenate([chunk.as_waveform().samples for chunk in chunks]),
    )
    assert audio.metadata == {"model": "streaming-vad"}
    assert audio.timeline("mono").prediction.spans[0].id == span.id
    assert not hasattr(audio, "is_loaded")
    assert not hasattr(audio, "unload")
    assert not hasattr(audio, "stream")
    for name in (
        "samples",
        "sample_rate",
        "channels",
        "frame_count",
        "source_format",
        "resample",
        "to_mono",
        "channel",
        "slice_ms",
        "to_global_span",
    ):
        assert not hasattr(first, name)


def test_audio_stream_close_prevents_conversion():
    stream = AudioSource.from_pcm(b"\0\0" * 250, sample_rate=1000).stream(100)
    next(stream)
    stream.close()

    assert stream.is_complete is False
    assert stream.is_closed is True
    assert list(stream) == []
    with pytest.raises(RuntimeError, match="completely consumed"):
        stream.to_audio()


def test_audio_stream_annotations_are_limited_to_received_audio():
    stream = AudioSource.from_pcm(b"\0\0" * 500, sample_rate=1000).stream(100)
    chunk = next(stream)
    timeline = stream.timeline("mono")
    span = timeline.annotate_span(
        10, 90, AudioActivity(event="speech"), is_reference=True
    )

    assert timeline.duration_ms == chunk.end_ms == 100
    assert span.as_waveform().frame_count == 80
    with pytest.raises(ValueError, match="must not exceed timeline"):
        timeline.annotate_span(
            100, 200, AudioActivity(event="speech"), is_reference=True
        )


def test_audio_stream_sync_and_async_iteration_cannot_be_mixed():
    stream = AudioSource.from_pcm(b"\0\0" * 500, sample_rate=1000).stream(100)
    next(stream)

    async def consume():
        return [chunk async for chunk in stream]

    with pytest.raises(RuntimeError, match="cannot mix"):
        asyncio.run(consume())


def test_audio_db_accepts_completed_audio_not_audio_stream(tmp_path):
    db = AudioDB.create(str(tmp_path / "stream.db"))
    stream = AudioSource.from_pcm(b"\0\0" * 100, sample_rate=1000).stream(
        50, id="stream"
    )

    with pytest.raises(TypeError):
        db.insert(stream)

    list(stream)
    audio = stream.to_audio()
    db.insert(audio)
    assert db["stream"].timeline("mono").duration_ms == 100


def test_timeline_and_timespan_waveform_views():
    source = AudioSource.from_pcm(
        struct.pack("<" + "h" * 1000, *range(1000)),
        sample_rate=1000,
    )
    audio = source.load()
    timeline = audio.timeline("mono")
    span = timeline.annotate_span(
        100, 300, AudioActivity(event="speech"), is_reference=True
    )

    assert not hasattr(span, "kind")
    assert isinstance(span.annotation, AudioActivity)
    assert timeline.as_waveform().frame_count == 1000
    assert span.as_waveform().frame_count == 200


def test_waveform_split_at_low_energy_is_lossless_and_frame_aligned():
    samples = np.ones(62, dtype=np.float32)
    samples[50:56] = 0.0
    waveform = Waveform(samples, sample_rate=10, channels=2)

    chunks = waveform.split_at_low_energy(max_duration_ms=3000)

    assert len(chunks) == 2
    assert all(chunk.frame_count <= 30 for chunk in chunks)
    assert all(chunk.channels == 2 for chunk in chunks)
    np.testing.assert_array_equal(
        np.concatenate([chunk.samples for chunk in chunks]),
        samples,
    )

    with pytest.raises(Exception, match="chunk size must be greater than zero"):
        waveform.split_at_low_energy(max_duration_ms=0)


def test_audio_numpy_constructor_and_output_share_memory():
    samples = np.arange(8, dtype=np.float32)
    audio = Waveform(samples, sample_rate=8000)
    view = audio.samples

    assert np.shares_memory(samples, view)
    assert view.ctypes.data == samples.ctypes.data
    assert view.flags.writeable is False
    samples[0] = 42.0
    assert view[0] == 42.0


def test_normalization_api_is_removed():
    audio = Waveform(np.array([0.1, -0.25, 0.5], dtype=np.float32), sample_rate=16000)
    chunk = next(
        AudioSource.from_pcm(b"\0\0", sample_rate=16000).stream(chunk_size_ms=100)
    )

    assert not hasattr(audio, "normalize")
    assert not hasattr(audio, "is_normalized")
    assert not hasattr(chunk, "normalize")
    assert not hasattr(chunk, "is_normalized")


def test_waveform_from_base64_decodes_wav_bytes():
    wav = io.BytesIO()
    with wave.open(wav, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(struct.pack("<hh", 0, 1000))
    encoded = base64.b64encode(wav.getvalue()).decode("ascii")

    waveform = Waveform.from_base64(encoded)

    assert waveform.sample_rate == 8000
    assert waveform.channels == 1
    assert waveform.source_format.encoding == "wav"


def test_source_aload_returns_loaded_audio():
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    source = AudioSource.from_pcm(pcm, sample_rate=8000, channels=2)

    async def load():
        return await source.aload()

    waveform = asyncio.run(load()).as_waveform()

    assert waveform.sample_rate == 8000
    assert waveform.channels == 2


def test_source_aload_from_path_returns_loaded_audio(tmp_path):
    wav_path = tmp_path / "audio.wav"
    with wave.open(str(wav_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(struct.pack("<hh", 0, 1000))

    async def load():
        return await AudioSource.from_path(str(wav_path)).aload()

    waveform = asyncio.run(load()).as_waveform()

    assert waveform.sample_rate == 8000
    assert waveform.channels == 1
    assert waveform.source_format.encoding == "wav"


def test_audio_aload_returns_waveform_without_blocking_api_changes():
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    source = AudioSource.from_pcm(pcm, sample_rate=8000, channels=2)

    async def load():
        return await source.aload()

    waveform = asyncio.run(load()).as_waveform()

    assert waveform.sample_rate == 8000
    assert waveform.channels == 2


def test_source_aload_supports_explicit_waveform_transformations():
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    source = AudioSource.from_pcm(pcm, sample_rate=8000, channels=2)

    async def load():
        return (await source.aload()).as_waveform().to_mono().resample(16000)

    asynchronous = asyncio.run(load())
    synchronous = source.load().as_waveform().to_mono().resample(16000)

    assert asynchronous.sample_rate == 16000
    assert asynchronous.channels == 1
    np.testing.assert_array_equal(asynchronous.samples, synchronous.samples)


def test_audio_stream_async_iteration_matches_sync_stream():
    pcm = struct.pack("<" + "h" * 800, *[index % 1000 for index in range(800)])
    source = AudioSource.from_pcm(pcm, sample_rate=8000)

    async def collect():
        stream = source.stream(chunk_size_ms=20)
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert stream.is_complete is True
        return chunks

    asynchronous = asyncio.run(collect())
    synchronous = list(source.stream(chunk_size_ms=20))

    assert [chunk.offset_ms for chunk in asynchronous] == [
        chunk.offset_ms for chunk in synchronous
    ]
    assert [chunk.is_final for chunk in asynchronous] == [
        chunk.is_final for chunk in synchronous
    ]
    np.testing.assert_array_equal(
        np.concatenate([chunk.as_waveform().samples for chunk in asynchronous]),
        np.concatenate([chunk.as_waveform().samples for chunk in synchronous]),
    )


def test_url_audio_stream_async_iteration_does_not_block_the_event_loop():
    wav = io.BytesIO()
    with wave.open(wav, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(struct.pack("<" + "h" * 800, *([100] * 800)))
    payload = wav.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            time.sleep(0.2)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = AudioSource.from_url(f"http://127.0.0.1:{server.server_port}/audio.wav")

    async def collect_and_probe():
        async def collect():
            stream = await asyncio.to_thread(source.stream, 20)
            return [chunk async for chunk in stream]

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.03)
        assert not task.done()
        return await task

    try:
        chunks = asyncio.run(collect_and_probe())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert chunks
    assert chunks[-1].is_final is True


def test_url_aload_uses_async_download_and_blocking_decode():
    wav = io.BytesIO()
    with wave.open(wav, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(struct.pack("<hhhh", 0, 1000, -1000, 2000))
    payload = wav.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            time.sleep(0.2)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = AudioSource.from_url(f"http://127.0.0.1:{server.server_port}/audio.wav")

    async def load_and_probe_loop():
        task = asyncio.create_task(source.aload())
        await asyncio.sleep(0.03)
        assert not task.done(), (
            "download should still be waiting on the delayed HTTP response"
        )
        return await task

    try:
        audio = asyncio.run(load_and_probe_loop())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    waveform = audio.as_waveform()
    assert waveform.sample_rate == 8000
    assert waveform.channels == 1
    assert waveform.source_format.encoding == "wav"


def test_audio_display_builds_ipython_player_for_selected_range(monkeypatch):
    import IPython.display

    displayed = []
    players = []

    def make_player(**kwargs):
        players.append(kwargs)
        return object()

    monkeypatch.setattr(IPython.display, "Audio", make_player)
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    audio = Waveform(np.arange(10, dtype=np.float32), sample_rate=10)

    audio.display(start_ms=200, end_ms=500, autoplay=True)

    assert len(displayed) == 1
    assert len(players) == 1
    np.testing.assert_array_equal(players[0]["data"], np.array([2.0, 3.0, 4.0]))
    assert players[0]["rate"] == 10
    assert players[0]["autoplay"] is True

    with pytest.raises(ValueError, match="end_ms must be greater"):
        audio.display(start_ms=500, end_ms=200)


def test_all_waveform_views_can_display(monkeypatch):
    import IPython.display

    players = []

    def make_player(**kwargs):
        players.append(kwargs)
        return object()

    monkeypatch.setattr(IPython.display, "Audio", make_player)
    monkeypatch.setattr(IPython.display, "display", lambda _player: None)

    source = AudioSource.from_pcm(b"\0\0" * 100, sample_rate=1000)
    audio = source.load(id="complete")
    timeline = audio.timeline("mono")
    span = timeline.annotate_span(
        10,
        30,
        AudioActivity(event="speech"),
        is_reference=True,
    )
    stream = source.stream(chunk_size_ms=50, id="stream")
    chunk = next(stream)

    audio.display()
    timeline.display()
    span.display()
    stream.display()
    chunk.display()

    assert [len(player["data"]) for player in players] == [100, 100, 20, 50, 50]
    assert all(player["rate"] == 1000 for player in players)


def test_public_types_have_informative_repr(tmp_path):
    pcm = b"\0\0" * (3250 * 8 * 2)
    audio = Audio(AudioSource.from_pcm(pcm, sample_rate=8000, channels=2), id="call-1")
    annotation = audio.timeline("left").annotate_span(
        100,
        800,
        Transcription("hello world", confidence=0.95),
        is_reference=True,
    )
    audio.metadata["speaker"] = "alice"
    waveform = audio.source.load().as_waveform()
    db = AudioDB.create(str(tmp_path / "repr.vasr"))
    db.insert(audio)

    assert repr(audio) == (
        'Audio(id="call-1", pcm_bytes=104000, sample_rate=8000, channels=2, '
        'duration="3.25s", annotations=1)'
    )
    assert str(audio) == 'Audio "call-1" (3.25s)'
    assert "duration=3.25s" in repr(waveform)
    assert 'text="hello world"' in repr(annotation)
    assert str(annotation) == 'transcription [100..800ms]: "hello world"'
    assert 'duration="3.25s"' in repr(audio.timeline("left"))
    assert repr(db).endswith('mode="read-write", audios=1, duration="3.25s")')


def test_audio_source_url_repr_keeps_filename_and_hides_query():
    source = AudioSource.from_url(
        "https://audio.example.com/a/very/long/path/to/session_123456789.wav"
        "?token=secret&expires=never"
    )

    rendered = repr(source)
    assert rendered.startswith('AudioSource(url="https://audio.example.com/')
    assert rendered.endswith('session_123456789.wav?…")')
    assert "secret" not in rendered


def test_model_annotations_can_be_written_queried_and_removed():
    audio = Audio(AudioSource.from_pcm(b"\0\0" * 8000, sample_rate=8000), id="sources")
    audio.ensure_timeline("mono", duration_ms=1000)
    timeline = audio.timeline("mono")
    reference = timeline.annotate_span(
        0, 1000, Transcription("reference", language="zh"), is_reference=True
    )
    prediction = timeline.annotate_span(
        0,
        1000,
        Transcription(
            "prediction",
            language="zh",
            confidence=0.88,
            tokens=[Token("prediction", start_ms=0, end_ms=1000)],
        ),
        source="tegasr",
        is_reference=False,
    )
    transcription = Transcription(
        "speaker text",
        language="zh",
        confidence=0.9,
        tokens=[
            Token("speaker", start_ms=0, end_ms=500, confidence=0.95),
            Token("text", start_ms=500, end_ms=1000, confidence=0.85),
        ],
    )
    speaker = timeline.annotate_span(
        0,
        1000,
        Speaker("user", transcription=transcription, confidence=0.92),
        source="channel_mapping",
        is_reference=False,
    )

    assert reference.source is None
    assert prediction.source == "tegasr"
    assert prediction.annotation.language == "zh"
    assert not hasattr(prediction, "confidence")
    assert prediction.annotation.confidence == pytest.approx(0.88)
    assert prediction.annotation.tokens[0].text == "prediction"
    assert isinstance(speaker.annotation, Speaker)
    assert speaker.annotation.name == "user"
    assert speaker.annotation.confidence == pytest.approx(0.92)
    assert speaker.annotation.transcription.text == "speaker text"
    assert speaker.annotation.transcription.language == "zh"
    assert speaker.annotation.transcription.confidence == pytest.approx(0.9)
    assert [token.text for token in speaker.annotation.transcription.tokens] == [
        "speaker",
        "text",
    ]
    assert speaker.annotation.transcription.tokens[0].start_ms == 0
    assert not hasattr(speaker, "text")
    assert not hasattr(speaker, "name")
    assert not hasattr(speaker, "language")
    assert not hasattr(speaker, "transcription")
    assert [item.id for item in timeline.prediction.by_source("tegasr")] == [
        prediction.id
    ]
    assert timeline.reference.transcript().text == "reference"
    assert timeline.prediction.transcript("tegasr").text == "prediction"
    assert timeline.prediction.sources == {
        "activity": [],
        "language": [],
        "sentence": [],
        "speaker": ["channel_mapping"],
        "token": [],
        "transcription": ["tegasr"],
    }
    original_id = prediction.id
    assert timeline.prediction.relabel_source("tegasr", "tegasr-v2") == 1
    relabeled = timeline.prediction.by_source("tegasr-v2")
    assert [item.id for item in relabeled] == [original_id]
    assert timeline.prediction.remove_by_source("tegasr-v2") == 1
    assert timeline.prediction.by_source("tegasr-v2") == []


def test_speaker_transcription_round_trips_through_database(tmp_path):
    audio = Audio(AudioSource.from_pcm(b"\0\0" * 8000, sample_rate=8000), id="speaker")
    timeline = audio.ensure_timeline("mono", duration_ms=1000)
    timeline.annotate_span(
        0,
        1000,
        Speaker(
            "agent",
            confidence=0.87,
            transcription=Transcription(
                "hello",
                language="en",
                confidence=0.91,
                tokens=[Token("hello", start_ms=100, end_ms=900, confidence=0.93)],
            ),
        ),
        is_reference=True,
    )

    db = AudioDB.create(str(tmp_path / "speaker.vasr"))
    db.insert(audio)
    loaded = db["speaker"].timeline("mono")
    speaker = loaded.reference.spans[0]

    assert speaker.annotation.name == "agent"
    assert speaker.annotation.confidence == pytest.approx(0.87)
    assert speaker.annotation.transcription.text == "hello"
    assert speaker.annotation.transcription.tokens[0].end_ms == 900
    assert loaded.reference.transcript().text == "hello"


def test_speaker_rejects_transcription_token_outside_its_range():
    audio = Audio(AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000), id="speaker")
    timeline = audio.ensure_timeline("mono", duration_ms=1000)
    transcription = Transcription(
        "outside",
        tokens=[Token("outside", start_ms=0, end_ms=900)],
    )

    with pytest.raises(ValueError, match="token range must be within"):
        timeline.annotate_span(
            100, 800, Speaker("agent", transcription=transcription), is_reference=True
        )


def test_annotation_add_methods_are_idempotent():
    audio = Audio(AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000), id="dedupe")
    timeline = audio.ensure_timeline("mono", duration_ms=1000)
    activity_payload = AudioActivity(event="speech", confidence=0.9)
    first_activity = timeline.annotate_span(
        0, 1000, activity_payload, is_reference=True
    )
    duplicate_activity = timeline.annotate_span(
        0, 1000, activity_payload, is_reference=True
    )
    speaker_payload = Speaker("agent")
    first_speaker = timeline.annotate_span(0, 1000, speaker_payload, is_reference=True)
    duplicate_speaker = timeline.annotate_span(
        0, 1000, speaker_payload, is_reference=True
    )
    first_text = timeline.annotate_span(
        0, 1000, Transcription("text"), is_reference=True
    )
    duplicate_text = timeline.annotate_span(
        0, 1000, Transcription("text"), is_reference=True
    )

    assert duplicate_activity.id == first_activity.id
    assert first_activity.annotation.confidence == pytest.approx(0.9)
    assert duplicate_speaker.id == first_speaker.id
    assert duplicate_text.id == first_text.id
    assert len(timeline.reference.spans) == 3

    with pytest.raises(asr_data.AsrDataError, match="overlaps"):
        timeline.annotate_span(
            0,
            1000,
            Speaker("agent", transcription=Transcription("updated")),
            is_reference=True,
        )
    assert len(timeline.reference.spans) == 3


def test_annotation_overlap_rules_and_atomic_mutations():
    audio = Audio(AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000), id="lanes")
    timeline = audio.ensure_timeline("mono", duration_ms=1000)

    speech = AudioActivity(event="speech")
    music = AudioActivity(event="music")
    timeline.annotate_span(0, 500, speech, is_reference=True)
    timeline.annotate_span(500, 1000, speech, is_reference=True)
    with pytest.raises(asr_data.AsrDataError, match="Activity.*overlaps"):
        timeline.annotate_span(400, 600, speech, is_reference=True)
    timeline.annotate_span(400, 600, music, is_reference=True)

    alice = timeline.annotate_span(0, 400, Speaker("alice"), is_reference=True)
    timeline.annotate_span(200, 600, Speaker("bob"), is_reference=True)
    with pytest.raises(asr_data.AsrDataError, match="Speaker.*overlaps"):
        timeline.annotate_span(300, 500, Speaker("alice"), is_reference=True)

    timeline.annotate_span(300, 500, Transcription("top-level"), is_reference=True)
    with pytest.raises(asr_data.AsrDataError, match="Transcription.*overlaps"):
        alice.annotation = Speaker("alice", transcription=Transcription("conflict"))
    assert alice.annotation.transcription is None

    timeline.annotate_span(0, 600, speech, source="vad-a", is_reference=False)
    timeline.annotate_span(300, 900, speech, source="vad-b", is_reference=False)
    candidate = timeline.annotate_span(
        500, 1000, speech, source="vad-c", is_reference=False
    )
    with pytest.raises(asr_data.AsrDataError, match="Activity.*overlaps"):
        timeline.prediction.relabel_source("vad-c", "vad-a")
    assert candidate.source == "vad-c"


def test_annotation_status_api_is_removed():
    audio = Audio(AudioSource.from_pcm(b"\0\0", sample_rate=1000), id="status-free")
    annotation = audio.ensure_timeline("mono", duration_ms=1).annotate_span(
        0, 1, AudioActivity(), is_reference=True
    )

    assert not hasattr(annotation, "status")
    with pytest.raises(TypeError):
        audio.timeline("mono").annotate_span(
            0, 1, Transcription("text"), status="final", is_reference=True
        )


def test_annotation_add_rejects_ranges_past_timeline_duration():
    audio = Audio(AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000), id="bounds")
    timeline = audio.ensure_timeline("mono", duration_ms=1000)
    assert not hasattr(timeline.reference, "add_silence")
    assert not hasattr(timeline.prediction, "add_silence")

    # The inclusive endpoint may equal the timeline duration.
    timeline.annotate_span(0, 1000, AudioActivity(), is_reference=True)

    invalid_adds = [
        lambda: timeline.annotate_span(0, 1001, AudioActivity(), is_reference=True),
        lambda: timeline.annotate_span(
            0, 1001, Transcription("outside"), is_reference=True
        ),
        lambda: timeline.annotate_span(0, 1001, Speaker("outside"), is_reference=True),
        lambda: timeline.annotate_span(
            0, 1001, AudioActivity(), source="vad", is_reference=False
        ),
    ]

    for add in invalid_adds:
        with pytest.raises(ValueError, match="must not exceed timeline duration_ms"):
            add()

    assert len(timeline.reference.spans) == 1
    assert timeline.prediction.spans == []


def test_prediction_source_is_required_preserved_and_queryable(tmp_path):
    audio = Audio(AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000), id="sources")
    timeline = audio.ensure_timeline("mono", duration_ms=1000)
    whisper = timeline.annotate_span(
        0,
        500,
        Speaker("caller"),
        source="whisper",
        is_reference=False,
    )
    qwen = timeline.annotate_span(
        500,
        1000,
        Speaker("agent"),
        source="qwen-asr",
        is_reference=False,
    )

    assert whisper.source == "whisper"
    assert qwen.source == "qwen-asr"
    assert [
        annotation.id for annotation in timeline.prediction.by_source("whisper")
    ] == [whisper.id]
    assert timeline.prediction.remove_by_source("qwen-asr") == 1
    assert [annotation.id for annotation in timeline.prediction.spans] == [whisper.id]
    with pytest.raises(ValueError, match="non-empty"):
        timeline.annotate_span(0, 1, AudioActivity(), source="", is_reference=False)

    db = AudioDB.create(str(tmp_path / "prediction-source.vasr"))
    db.insert(audio)
    loaded = db["sources"].timeline("mono").prediction.spans[0]
    assert loaded.source == "whisper"


def test_timeline_exposes_all_top_level_annotations_by_type():
    audio = Audio(AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000), id="types")
    timeline = audio.ensure_timeline("mono", duration_ms=1000)

    timeline.annotate_span(
        0, 500, Transcription("reference"), is_reference=True
    )
    timeline.annotate_span(
        0,
        500,
        Transcription("prediction"),
        is_reference=False,
        source="asr",
    )
    timeline.annotate_span(
        0, 500, AudioActivity(event="speech"), is_reference=True
    )
    timeline.annotate_span(
        0,
        500,
        AudioActivity(event="speech"),
        is_reference=False,
        source="vad",
    )
    timeline.annotate_span(
        0, 500, Speaker("caller"), is_reference=True
    )
    timeline.annotate_span(
        0, 500, Speaker("caller"), is_reference=False, source="diarization"
    )

    assert [annotation.text for annotation in timeline.transcriptions] == [
        "reference",
        "prediction",
    ]
    assert [annotation.event for annotation in timeline.activities] == [
        "speech",
        "speech",
    ]
    assert [annotation.name for annotation in timeline.speakers] == [
        "caller",
        "caller",
    ]
    assert not hasattr(timeline, "tokens")
    assert not hasattr(timeline, "sentences")
    assert not hasattr(timeline, "languages")


def test_speaker_transcription_can_be_attached_after_creation():
    audio = Audio(AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000), id="attach")
    timeline = audio.ensure_timeline("mono", duration_ms=1000)
    speaker = timeline.annotate_span(100, 900, Speaker("agent"), is_reference=True)
    original_id = speaker.id

    speaker.annotation = Speaker(
        "agent",
        transcription=Transcription(
            "hello",
            tokens=[Token("hello", start_ms=100, end_ms=900)],
        ),
    )

    assert speaker.id == original_id
    assert speaker.annotation.transcription.text == "hello"
    assert timeline.reference.spans[0].id == original_id
    assert timeline.reference.spans[0].annotation.transcription.text == "hello"
    assert timeline.reference.transcript().text == "hello"

    speaker.annotation = Speaker("agent")
    assert speaker.annotation.transcription is None
    assert timeline.reference.spans[0].annotation.transcription is None


def test_payload_assignment_validates_target_and_token_range():
    audio = Audio(AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000), id="invalid")
    timeline = audio.ensure_timeline("mono", duration_ms=1000)
    speaker = timeline.annotate_span(100, 900, Speaker("agent"), is_reference=True)
    activity = timeline.annotate_span(
        0, 1000, AudioActivity(event="speech"), is_reference=True
    )
    assert activity.annotation.event == "speech"
    outside = Transcription(
        "outside",
        tokens=[Token("outside", start_ms=0, end_ms=1000)],
    )

    with pytest.raises(ValueError, match="token range must be within"):
        speaker.annotation = Speaker("agent", transcription=outside)
    with pytest.raises(ValueError, match="activity annotation must be AudioActivity"):
        activity.annotation = Transcription("not allowed")

    activity.annotation = AudioActivity(event="music")
    assert activity.annotation.event == "music"


def test_audio_activity_rejects_blank_event():
    with pytest.raises(ValueError, match="non-whitespace"):
        AudioActivity(event="   ")


def test_timeline_eval_reports_transcription_metrics_without_normalization():
    audio = Audio(
        AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000), id="eval-text"
    )
    timeline = audio.ensure_timeline("mono", duration_ms=1000)
    timeline.annotate_span(
        0, 1000, Transcription("交易停滞"), is_reference=True
    )
    timeline.annotate_span(
        0,
        1000,
        Transcription("交易停止"),
        source="qwen-asr",
        is_reference=False,
    )

    evaluation = timeline.eval(transcription="qwen-asr", normalize=False)
    result = evaluation.transcription["qwen-asr"]

    assert evaluation.activity == {}
    assert result.source == "qwen-asr"
    assert result.normalization == "none"
    assert result.reference == "交易停滞"
    assert result.hypothesis == "交易停止"
    assert result.normalized_reference == "交易停滞"
    assert result.normalized_hypothesis == "交易停止"
    assert result.matches == 3
    assert result.substitutions == 1
    assert result.deletions == 0
    assert result.insertions == 0
    assert result.reference_chars == 4
    assert result.hypothesis_chars == 4
    assert result.cer == 0.25
    assert result.precision == 0.75
    assert result.recall == 0.75
    assert result.f1 == 0.75
    assert result.exact_match is False


def test_timeline_eval_uses_embedded_chinese_tn_by_default():
    audio = Audio(AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000), id="eval-tn")
    timeline = audio.ensure_timeline("mono", duration_ms=1000)
    timeline.annotate_span(0, 1000, Transcription("2024年交易"), is_reference=True)
    timeline.annotate_span(
        0,
        1000,
        Transcription("二零二四年交易"),
        source="qwen-asr",
        is_reference=False,
    )

    result = timeline.eval(transcription="qwen-asr").transcription["qwen-asr"]

    assert result.normalization == "zh_tn"
    assert result.normalized_reference == "二零二四年交易"
    assert result.normalized_hypothesis == "二零二四年交易"
    assert result.cer == 0.0
    assert result.exact_match is True


def test_timeline_eval_reports_activity_duration_confusion_matrix():
    audio = Audio(AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000), id="eval-vad")
    timeline = audio.ensure_timeline("mono", duration_ms=1000)
    timeline.annotate_span(100, 400, AudioActivity(event="speech"), is_reference=True)
    timeline.annotate_span(400, 600, AudioActivity(event="speech"), is_reference=True)
    timeline.annotate_span(
        200, 700, AudioActivity(event="speech"), source="silero-vad", is_reference=False
    )

    evaluation = timeline.eval(activity="silero-vad")
    result = evaluation.activity["silero-vad"]

    assert evaluation.transcription == {}
    assert result.reference_ms == 500
    assert result.predicted_ms == 500
    assert result.true_positive_ms == 400
    assert result.true_negative_ms == 400
    assert result.false_positive_ms == 100
    assert result.false_negative_ms == 100
    assert result.precision == 0.8
    assert result.recall == 0.8
    assert result.f1 == pytest.approx(0.8)
    assert result.iou == pytest.approx(2 / 3)
    assert result.events["speech"].f1 == pytest.approx(0.8)


def test_timeline_eval_separates_activity_detection_from_event_classification():
    audio = Audio(
        AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000),
        id="eval-events",
    )
    timeline = audio.ensure_timeline("mono", duration_ms=1000)
    timeline.annotate_span(0, 400, AudioActivity(event="speech"), is_reference=True)
    timeline.annotate_span(400, 600, AudioActivity(), is_reference=True)
    timeline.annotate_span(
        0, 600, AudioActivity(event="music"), source="classifier", is_reference=False
    )

    result = timeline.eval(activity="classifier").activity["classifier"]

    assert result.f1 == 1.0
    assert result.events["speech"].true_positive_ms == 0
    assert result.events["speech"].false_negative_ms == 400
    assert result.events["music"].false_positive_ms == 400
    assert result.events["music"].false_positive_ms != 600


def test_timeline_eval_auto_discovers_sources_and_validates_explicit_selection():
    audio = Audio(
        AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000), id="eval-errors"
    )
    timeline = audio.ensure_timeline("mono", duration_ms=1000)

    with pytest.raises(asr_data.AsrDataError, match="no reference annotations"):
        timeline.eval()
    with pytest.raises(
        asr_data.AsrDataError, match="reference annotations are missing"
    ):
        timeline.eval(transcription="qwen-asr", normalize=False)

    timeline.annotate_span(0, 1000, Transcription("参考"), is_reference=True)
    with pytest.raises(asr_data.AsrDataError, match="qwen-asr"):
        timeline.eval(transcription="qwen-asr", normalize=False)

    timeline.annotate_span(
        0, 1000, Transcription("参考"), source="qwen-asr", is_reference=False
    )
    timeline.annotate_span(
        0, 1000, Transcription("参照"), source="whisper", is_reference=False
    )
    result = timeline.eval(normalize=False)
    assert list(result.transcription) == ["qwen-asr", "whisper"]
    assert result.transcription["qwen-asr"].cer == 0.0
    assert result.transcription["whisper"].cer == 0.5


def test_dataset_eval_aggregates_corpus_metrics_and_coverage(tmp_path):
    docs = []
    for audio_id, reference, predictions in [
        ("first", "aaaa", {"qwen": "aaab", "whisper": "aaaa"}),
        ("second", "a", {"qwen": ""}),
    ]:
        doc = Audio(
            AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000),
            id=audio_id,
        )
        timeline = doc.timeline("mono")
        timeline.annotate_span(0, 1000, Transcription(reference), is_reference=True)
        for source, text in predictions.items():
            timeline.annotate_span(
                0, 1000, Transcription(text), source=source, is_reference=False
            )
        docs.append(doc)

    unannotated = Audio(
        AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000),
        id="third",
    )
    docs.append(unannotated)

    result = asr_data.evaluate_dataset(docs, normalize=False)
    assert result.documents == 3
    assert result.timelines == 3
    assert result.transcription["qwen"].cer == 0.4
    assert result.transcription["qwen"].unannotated_timelines == 1
    assert result.transcription["whisper"].coverage == 0.5
    assert result.transcription["whisper"].missing_prediction_ids == ["second:mono"]

    db = AudioDB.create(str(tmp_path / "dataset-eval.db"))
    for doc in docs:
        doc.metadata["split"] = "test"
        db.insert(doc)
    db_result = db.eval(
        transcription=["qwen", "whisper"],
        normalize=False,
        batch_size=1,
        metadata={"split": "test"},
    )
    assert db_result.documents == 3
    assert db_result.transcription["qwen"].cer == 0.4


def test_dataset_eval_aggregates_activity_and_event_metrics():
    docs = []
    for audio_id, event in [("first", "speech"), ("second", "music")]:
        doc = Audio(
            AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000),
            id=audio_id,
        )
        timeline = doc.timeline("mono")
        timeline.annotate_span(0, 400, AudioActivity(event="speech"), is_reference=True)
        timeline.annotate_span(
            0, 400, AudioActivity(event=event), source="classifier", is_reference=False
        )
        docs.append(doc)

    result = asr_data.evaluate_dataset(docs, activity="classifier")
    activity = result.activity["classifier"]

    assert activity.evaluated_documents == 2
    assert activity.evaluated_timelines == 2
    assert activity.f1 == 1.0
    assert activity.events["speech"].true_positive_ms == 400
    assert activity.events["speech"].false_negative_ms == 400
    assert activity.events["music"].false_positive_ms == 400


def test_database_update_detects_changes(tmp_path):
    path = tmp_path / "timeline-only.vasr"
    audio = Audio(
        AudioSource.from_pcm(b"\0\0" * 8000, sample_rate=8000), id="timeline-only"
    )
    audio.ensure_timeline("mono", duration_ms=1000)
    audio.metadata["preserved"] = True
    db = AudioDB.create(str(path))
    db.insert(audio)

    audio.timeline("mono").annotate_span(
        0, 1000, Transcription("prediction"), source="old-model", is_reference=False
    )
    assert db.update(audio) is True
    assert db.update(audio) is False
    missing = Audio(AudioSource.from_pcm(b"\0\0", sample_rate=8000), id="missing")
    with pytest.raises(KeyError, match="missing"):
        db.update(missing)
    loaded = db["timeline-only"]
    assert loaded.metadata["preserved"] is True
    assert (
        loaded.timeline("mono").prediction.transcript("old-model").text == "prediction"
    )
    assert (
        loaded.timeline("mono").prediction.relabel_source("old-model", "new-model") == 1
    )
    assert db.update(loaded) is True
    loaded = db["timeline-only"]
    assert loaded.timeline("mono").prediction.by_source("old-model") == []
    assert (
        loaded.timeline("mono").prediction.transcript("new-model").text == "prediction"
    )

    loaded.timeline("mono").annotate_span(
        0, 1000, Transcription("second"), source="second-model", is_reference=False
    )
    assert db.update(loaded) is True
    assert (
        db["timeline-only"].timeline("mono").prediction.transcript("second-model").text
        == "second"
    )

    another = db["timeline-only"]
    another.metadata["batch"] = True
    assert db.update_many([another]) == 1


def test_database_query_filters_automatic_creation_and_update_times(tmp_path):
    import time
    from datetime import datetime, timedelta, timezone

    db = AudioDB.create(str(tmp_path / "timestamps.db"))
    audio = Audio(
        AudioSource.from_pcm(b"\0\0" * 1000, sample_rate=1000),
        id="timestamped",
    )
    before_insert = datetime.now(timezone.utc) - timedelta(seconds=1)
    db.insert(audio)
    after_insert = datetime.now(timezone.utc)

    assert [
        doc.id
        for doc in db.query(
            created_from=before_insert,
            created_until=after_insert,
        )
    ] == ["timestamped"]
    assert db.query(created_until=before_insert) == []

    time.sleep(0.002)
    update_boundary = datetime.now(timezone.utc)
    assert db.update(audio) is False
    assert db.query(updated_from=update_boundary) == []

    time.sleep(0.002)
    audio.metadata["changed"] = True
    assert db.update(audio) is True
    assert [doc.id for doc in db.query(updated_from=update_boundary)] == ["timestamped"]
    assert db.query(created_from=update_boundary) == []


def test_database_query_validates_datetime_filters(tmp_path):
    from datetime import datetime, timedelta, timezone

    db = AudioDB.create(str(tmp_path / "timestamp-validation.db"))
    now = datetime.now(timezone.utc)

    with pytest.raises(ValueError, match="timezone-aware"):
        db.query(created_from=datetime.now())
    with pytest.raises(ValueError, match="created_from must not exceed created_until"):
        db.query(created_from=now, created_until=now - timedelta(seconds=1))
    with pytest.raises(ValueError, match="updated_from must not exceed updated_until"):
        db.query(updated_from=now, updated_until=now - timedelta(seconds=1))


def test_audio_and_audio_stream_convenience_factories(tmp_path):
    wav_path = tmp_path / "audio.wav"
    with wave.open(str(wav_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(struct.pack("<hh", 0, 1000))

    encoded = wav_path.read_bytes()
    encoded_base64 = base64.b64encode(encoded).decode()
    from_file = Audio.from_path(str(wav_path), id="file")
    from_url = Audio.from_url(wav_path.as_uri(), id="url")
    from_bytes = Audio.from_bytes(encoded, id="bytes")
    from_base64 = Audio.from_base64(encoded_base64, id="base64")
    from_pcm = Audio.from_pcm(struct.pack("<hh", 0, 1000), 8000, id="pcm")
    assert isinstance(from_file.source, AudioSource)
    assert isinstance(from_bytes.source, AudioSource)
    assert {
        audio.id
        for audio in (from_file, from_url, from_bytes, from_base64, from_pcm)
    } == {"file", "url", "bytes", "base64", "pcm"}
    assert all(
        audio.as_waveform().channels == 1
        for audio in (from_file, from_url, from_bytes, from_base64, from_pcm)
    )

    streams = [
        asr_data.AudioStream.from_path(str(wav_path), id="file-stream"),
        asr_data.AudioStream.from_url(wav_path.as_uri(), id="url-stream"),
        asr_data.AudioStream.from_bytes(encoded, id="bytes-stream"),
        asr_data.AudioStream.from_base64(encoded_base64, id="base64-stream"),
        asr_data.AudioStream.from_pcm(
            struct.pack("<hh", 0, 1000), 8000, id="pcm-stream"
        ),
    ]
    assert [stream.id for stream in streams] == [
        "file-stream",
        "url-stream",
        "bytes-stream",
        "base64-stream",
        "pcm-stream",
    ]
    assert all(len(list(stream)) == 1 for stream in streams)

    assert Waveform.from_path(str(wav_path)).channels == 1
    assert Waveform.from_bytes(wav_path.read_bytes()).channels == 1
    assert Waveform.from_source(from_file.source).channels == 1
    assert not hasattr(Audio, "from_file")
    assert not hasattr(from_file, "load")
    assert not hasattr(from_file.source.load().as_waveform(), "num_channels")
    assert repr(from_file).startswith('Audio(id="file", file="')
    assert 'duration="1ms"' in repr(from_file)
    assert str(from_file) == 'Audio "file" (1ms)'


def test_timeline_has_one_annotation_write_api():
    timeline = Audio.from_pcm(b"\0\0" * 100, 1000).timeline("mono")

    reference = timeline.annotate_span(
        0, 100, AudioActivity(event="speech"), is_reference=True
    )
    prediction = timeline.annotate_span(
        0,
        100,
        Transcription("hello"),
        is_reference=False,
        source="asr",
    )

    assert reference.source is None
    assert prediction.source == "asr"
    assert not hasattr(timeline.reference, "annotate_span")
    assert not hasattr(timeline.prediction, "annotate_span")
    with pytest.raises(ValueError, match="must be omitted"):
        timeline.annotate_span(
            0,
            100,
            AudioActivity(event="music"),
            is_reference=True,
            source="manual",
        )
    with pytest.raises(ValueError, match="is required"):
        timeline.annotate_span(
            0, 100, AudioActivity(event="music"), is_reference=False
        )


def test_audiodb_restores_audio_info_without_reopening_source(tmp_path):
    wav_path = tmp_path / "ephemeral.wav"
    with wave.open(str(wav_path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(b"\0\0" * 16)

    doc = Audio(AudioSource.from_path(str(wav_path)), id="ephemeral")
    db = AudioDB.create(str(tmp_path / "audio-info.db"))
    db.insert(doc)
    wav_path.unlink()

    loaded = db["ephemeral"]
    assert not hasattr(loaded, "is_loaded")
    assert loaded.info.sample_rate == 8000
    assert loaded.info.channels == 2
    assert loaded.info.frame_count == 8
    assert loaded.info.duration_ms == pytest.approx(1)
    assert list(loaded.timelines) == ["left", "right"]
