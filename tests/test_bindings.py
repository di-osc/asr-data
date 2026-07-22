import asyncio
import base64
import io
import os
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
    AnnotationKind,
    AnnotationSourceKind,
    AnnotationStatus,
    AudioDB,
    AudioDoc,
    AudioSource,
    Audio,
)


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


def test_annotation_literal_types_are_public_and_complete():
    assert set(typing.get_args(AnnotationKind)) == {
        "speech",
        "silence",
        "token",
        "transcription",
        "sentence",
        "speaker",
        "language",
        "hotword",
        "acoustic_event",
        "diagnostic",
    }
    assert set(typing.get_args(AnnotationStatus)) == {
        "partial",
        "final",
        "revised",
        "deleted",
    }
    assert set(typing.get_args(AnnotationSourceKind)) == {
        "user",
        "model",
        "stage",
        "system",
    }


def test_audio_waveform_timeline_and_db(tmp_path):
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    audio = AudioDoc(AudioSource.from_pcm(pcm, sample_rate=8000, channels=2), id="call-1")
    assert isinstance(audio.source, AudioSource)
    assert audio.timelines == {}
    assert audio.timeline("mono") is None
    audio.metadata["speaker"] = {"name": "alice", "age": 30}
    audio.ensure_timeline("mono", duration_ms=1)
    audio.timeline("mono").add_speech(0, 1, confidence=0.9)
    audio.timeline("mono").add_transcription(0, 1, "hello", language="en")

    waveform = audio.source.load()
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
    assert audio.timeline("mono").transcript().text == "hello"
    assert len(audio.timeline("mono").annotations) == 2

    db_path = tmp_path / "test.vasr"
    db = AudioDB(str(db_path))
    for removed in (
        "create",
        "open",
        "upsert",
        "insert_many",
        "get",
        "list",
        "all",
        "remove",
        "contains",
        "import_legacy_msgpack",
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
    assert isinstance(loaded, AudioDoc)
    assert isinstance(loaded.source, AudioSource)
    assert loaded.metadata["speaker"]["name"] == "alice"
    assert loaded.timeline("mono").transcript().text == "hello"
    loaded.metadata["speaker"]["name"] = "bob"
    assert db.update(loaded) is True
    assert db["call-1"].metadata["speaker"]["name"] == "bob"
    assert not hasattr(loaded, "set_metadata")
    with pytest.raises(KeyError):
        _ = db["missing"]
    assert db.delete("call-1") is True
    assert db.delete("call-1") is False


def test_audio_db_query_filters_cursor_and_lazy_iteration(tmp_path):
    db = AudioDB(str(tmp_path / "query.vasr"))
    for index in range(105):
        audio = AudioDoc(AudioSource.from_pcm(b"\0\0", sample_rate=8000), id=f"audio-{index:03}")
        audio.ensure_timeline("mono", duration_ms=index * 10)
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
    audio = AudioDoc(
        AudioSource.from_pcm(b"\0\0" * 4, sample_rate=8000, channels=2), id="call-stereo"
    )

    audio.ensure_timeline("left", duration_ms=100).add_transcription(0, 100, "caller")
    audio.ensure_timeline("right").add_transcription(0, 100, "agent")

    assert audio.timeline("mono") is None
    assert audio.timeline("left").transcript().text == "caller"
    assert audio.timeline(0).transcript().text == "caller"
    assert audio.timeline("right").transcript().text == "agent"
    assert audio.timeline(1).transcript().text == "agent"
    assert set(audio.timelines) == {"left", "right"}
    assert not hasattr(audio, "channel_timeline")
    assert audio.timeline(2) is None
    created = audio.ensure_timeline(2)
    assert created.id == audio.timeline(2).id
    assert audio.remove_timeline(2) is True
    assert audio.timeline(2) is None

    db = AudioDB(str(tmp_path / "stereo.sqlite"))
    db.insert(audio)
    loaded = db["call-stereo"]

    assert loaded.timeline("left").transcript().text == "caller"
    assert loaded.timeline("right").transcript().text == "agent"


def test_setting_timeline_audio_id_updates_the_whole_audio():
    audio = AudioDoc(AudioSource.from_pcm(b"\0\0" * 4, sample_rate=8000, channels=2), id="old-id")
    audio.ensure_timeline("left", duration_ms=100)
    audio.ensure_timeline("right")

    audio.timeline("left").audio_id = "new-id"

    assert audio.id == "new-id"
    assert audio.timeline("left").audio_id == "new-id"
    assert audio.timeline("right").audio_id == "new-id"


def test_timeline_duration_is_required_shared_and_read_only():
    audio = AudioDoc(AudioSource.from_pcm(b"\0\0" * 4000, sample_rate=8000), id="duration")
    assert not hasattr(audio, "duration_ms")
    with pytest.raises(Exception, match="duration is required"):
        audio.ensure_timeline("right")

    mono = audio.ensure_timeline("mono", duration_ms=500)
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
    waveform = Audio(samples, 16000)
    view = waveform.samples
    assert np.shares_memory(samples, view)
    samples[:] = 1.0
    np.testing.assert_array_equal(view, samples)


def test_waveform_from_pcm_matches_source_load():
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    source = AudioSource.from_pcm(pcm, sample_rate=8000, channels=2)
    via_source = source.load()
    via_waveform = Audio.from_pcm(pcm, sample_rate=8000, channels=2)
    assert via_waveform.sample_rate == via_source.sample_rate
    assert via_waveform.channels == via_source.channels
    np.testing.assert_allclose(via_waveform.samples, via_source.samples)


def test_source_load_options():
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    source = AudioSource.from_pcm(pcm, sample_rate=8000, channels=2)

    transformed = source.load(sample_rate=16000, mono=True)
    assert transformed.sample_rate == 16000
    assert transformed.channels == 1
    assert transformed.samples.dtype == np.float32
    assert np.isfinite(transformed.samples).all()
    assert np.abs(transformed.samples).max(initial=0.0) <= 1.0

    preserved = source.load(mono=False)
    assert preserved.sample_rate == 8000
    assert preserved.channels == 2

    with pytest.raises(Exception, match="sample rate"):
        source.load(sample_rate=0)


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
        full = source.load()
        chunks = list(source.stream(chunk_size_ms=2))

        assert all(type(chunk).__name__ == "AudioChunk" for chunk in chunks)
        assert [chunk.frame_count for chunk in chunks] == [2, 2, 1]
        assert [chunk.offset_ms for chunk in chunks] == [0, 2, 4]
        assert [chunk.is_final for chunk in chunks] == [False, False, True]
        assert all(chunk.sample_rate == 1000 for chunk in chunks)
        assert all(chunk.channels == 2 for chunk in chunks)
        np.testing.assert_array_equal(
            np.concatenate([chunk.samples for chunk in chunks]),
            full.samples,
        )

        first_view = chunks[0].samples
        second_view = chunks[0].samples
        assert np.shares_memory(first_view, second_view)
        assert first_view.flags.writeable is False
        with pytest.raises(ValueError):
            first_view[0] = 0

    with pytest.raises(Exception, match="chunk size must be greater than zero"):
        list(sources[0].stream(chunk_size_ms=0))


def test_source_stream_options(tmp_path):
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
        full = source.load(sample_rate=2000, mono=True)
        chunks = list(
            source.stream(chunk_size_ms=2, sample_rate=2000, mono=True)
        )

        assert chunks
        assert all(chunk.sample_rate == 2000 for chunk in chunks)
        assert all(chunk.channels == 1 for chunk in chunks)
        assert [chunk.offset_ms for chunk in chunks] == sorted(
            chunk.offset_ms for chunk in chunks
        )
        assert [chunk.is_final for chunk in chunks[:-1]] == [False] * (
            len(chunks) - 1
        )
        assert chunks[-1].is_final is True
        streamed = np.concatenate([chunk.samples for chunk in chunks])
        np.testing.assert_allclose(streamed, full.samples, atol=1e-6)
        assert np.isfinite(streamed).all()
        assert np.abs(streamed).max(initial=0.0) <= 1.0

    with pytest.raises(Exception, match="sample rate"):
        list(sources[0].stream(sample_rate=0))


def test_source_stream_default_chunk():
    pcm = struct.pack("<" + "h" * 250, *range(250))

    chunks = list(AudioSource.from_pcm(pcm, sample_rate=1000).stream())

    assert [chunk.frame_count for chunk in chunks] == [100, 100, 50]
    assert [chunk.offset_ms for chunk in chunks] == [0, 100, 200]
    assert [chunk.is_final for chunk in chunks] == [False, False, True]


def test_waveform_split_at_low_energy_is_lossless_and_frame_aligned():
    samples = np.ones(62, dtype=np.float32)
    samples[50:56] = 0.0
    waveform = Audio(samples, sample_rate=10, channels=2)

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
    audio = Audio(samples, sample_rate=8000)
    view = audio.samples

    assert np.shares_memory(samples, view)
    assert view.ctypes.data == samples.ctypes.data
    assert view.flags.writeable is False
    samples[0] = 42.0
    assert view[0] == 42.0


def test_normalization_api_is_removed():
    audio = Audio(np.array([0.1, -0.25, 0.5], dtype=np.float32), sample_rate=16000)
    chunk = next(AudioSource.from_pcm(b"\0\0", sample_rate=16000).stream(chunk_size_ms=100))

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

    waveform = Audio.from_base64(encoded)

    assert waveform.sample_rate == 8000
    assert waveform.channels == 1
    assert waveform.source_format.encoding == "wav"


def test_waveform_aload_from_source_returns_pcm_waveform():
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    source = AudioSource.from_pcm(pcm, sample_rate=8000, channels=2)

    async def load():
        return await Audio.aload_from_source(source)

    waveform = asyncio.run(load())

    assert waveform.sample_rate == 8000
    assert waveform.channels == 2


def test_waveform_aload_from_path_returns_waveform(tmp_path):
    wav_path = tmp_path / "audio.wav"
    with wave.open(str(wav_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(struct.pack("<hh", 0, 1000))

    async def load():
        return await Audio.aload_from_path(str(wav_path))

    waveform = asyncio.run(load())

    assert waveform.sample_rate == 8000
    assert waveform.channels == 1
    assert waveform.source_format.encoding == "wav"


def test_audio_aload_returns_waveform_without_blocking_api_changes():
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    source = AudioSource.from_pcm(pcm, sample_rate=8000, channels=2)

    async def load():
        return await source.aload()

    waveform = asyncio.run(load())

    assert waveform.sample_rate == 8000
    assert waveform.channels == 2


def test_source_aload_options_match_sync_load():
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    source = AudioSource.from_pcm(pcm, sample_rate=8000, channels=2)

    async def load():
        return await source.aload(sample_rate=16000, mono=True)

    asynchronous = asyncio.run(load())
    synchronous = source.load(sample_rate=16000, mono=True)

    assert asynchronous.sample_rate == 16000
    assert asynchronous.channels == 1
    np.testing.assert_array_equal(asynchronous.samples, synchronous.samples)


def test_source_astream_is_async_and_matches_sync_stream():
    pcm = struct.pack("<" + "h" * 800, *[index % 1000 for index in range(800)])
    source = AudioSource.from_pcm(pcm, sample_rate=8000)

    async def collect():
        chunks = []
        async for chunk in source.astream(
            chunk_size_ms=20,
            sample_rate=16000,
            mono=True,
        ):
            chunks.append(chunk)
        return chunks

    asynchronous = asyncio.run(collect())
    synchronous = list(
        source.stream(chunk_size_ms=20, sample_rate=16000, mono=True)
    )

    assert [chunk.offset_ms for chunk in asynchronous] == [
        chunk.offset_ms for chunk in synchronous
    ]
    assert [chunk.is_final for chunk in asynchronous] == [
        chunk.is_final for chunk in synchronous
    ]
    np.testing.assert_array_equal(
        np.concatenate([chunk.samples for chunk in asynchronous]),
        np.concatenate([chunk.samples for chunk in synchronous]),
    )


def test_url_astream_does_not_block_the_event_loop():
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
            return [chunk async for chunk in source.astream(chunk_size_ms=20)]

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
        waveform = asyncio.run(load_and_probe_loop())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

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
    audio = Audio(np.arange(10, dtype=np.float32), sample_rate=10)

    audio.display(start_ms=200, end_ms=500, autoplay=True)

    assert len(displayed) == 1
    assert len(players) == 1
    np.testing.assert_array_equal(players[0]["data"], np.array([2.0, 3.0, 4.0]))
    assert players[0]["rate"] == 10
    assert players[0]["autoplay"] is True

    with pytest.raises(ValueError, match="end_ms must be greater"):
        audio.display(start_ms=500, end_ms=200)


def test_existing_db_can_be_opened_read_only():
    fixture = "tests/fixtures/lbg_call-100.vasr"
    if not os.path.exists(fixture):
        pytest.skip("optional legacy database fixture is not available")
    db = AudioDB(fixture, read_only=True)
    assert len(db) == 99
    assert len(list(db)[:2]) == 2


def test_public_types_have_informative_repr(tmp_path):
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    audio = AudioDoc(AudioSource.from_pcm(pcm, sample_rate=8000, channels=2), id="call-1")
    audio.ensure_timeline("mono", duration_ms=3250)
    annotation = audio.timeline("mono").add_transcription(
        100,
        800,
        "hello world",
        confidence=0.95,
    )
    audio.metadata["speaker"] = "alice"
    waveform = audio.source.load()
    db = AudioDB(str(tmp_path / "repr.vasr"))
    db.insert(audio)

    assert repr(audio) == (
        'AudioDoc(id="call-1", pcm_bytes=8, sample_rate=8000, channels=2, '
        'duration="3.25s", annotations=1)'
    )
    assert str(audio) == 'AudioDoc "call-1" (3.25s)'
    assert "duration=0ms" in repr(waveform)
    assert 'text="hello world"' in repr(annotation)
    assert str(annotation) == 'transcription [100..800ms]: "hello world"'
    assert 'duration="3.25s"' in repr(audio.timeline("mono"))
    assert repr(db).endswith('mode="read-write", audios=1, duration="3.25s")')


def test_model_annotations_can_be_written_queried_and_removed():
    audio = AudioDoc(AudioSource.from_pcm(b"\0\0" * 8000, sample_rate=8000), id="sources")
    audio.ensure_timeline("mono", duration_ms=1000)
    reference = audio.timeline("mono").add_transcription(
        0, 1000, "reference", source="xlsx_import", language="zh"
    )
    prediction = audio.timeline("mono").add_transcription(
        0,
        1000,
        "prediction",
        source="tegasr",
        source_kind="model",
        language="zh",
        confidence=0.8,
    )
    speaker = audio.timeline("mono").add_speaker(
        0, 1000, "user", source="channel_mapping", source_kind="stage"
    )

    assert reference.source_kind == "stage"
    assert reference.source == "xlsx_import"
    assert prediction.source_kind == "model"
    assert prediction.source == "tegasr"
    assert prediction.language == "zh"
    assert speaker.kind == "speaker"
    assert speaker.speaker == "user"
    assert speaker.source_kind == "stage"
    assert [
        item.id for item in audio.timeline("mono").annotations_by_source("tegasr")
    ] == [prediction.id]
    assert audio.timeline("mono").transcript_by_source("tegasr").text == "prediction"
    assert (
        audio.timeline("mono")
        .transcript_by_source("xlsx_import", source_kind="stage")
        .text
        == "reference"
    )
    original_id = prediction.id
    assert (
        audio.timeline("mono").relabel_annotations_source(
            "tegasr", "tegasr-v2", from_source_kind="model"
        )
        == 1
    )
    relabeled = audio.timeline("mono").annotations_by_source("tegasr-v2")
    assert [item.id for item in relabeled] == [original_id]
    assert relabeled[0].source_kind == "model"
    assert audio.timeline("mono").remove_annotations_by_source("tegasr-v2") == 1
    assert audio.timeline("mono").annotations_by_source("tegasr-v2") == []


def test_database_update_detects_changes(tmp_path):
    path = tmp_path / "timeline-only.vasr"
    audio = AudioDoc(AudioSource.from_pcm(b"\0\0" * 8000, sample_rate=8000), id="timeline-only")
    audio.ensure_timeline("mono", duration_ms=1000)
    audio.metadata["preserved"] = True
    db = AudioDB(str(path))
    db.insert(audio)

    audio.timeline("mono").add_transcription(
        0, 1000, "prediction", source="old-model", source_kind="model"
    )
    assert db.update(audio) is True
    assert db.update(audio) is False
    missing = AudioDoc(AudioSource.from_pcm(b"\0\0", sample_rate=8000), id="missing")
    with pytest.raises(KeyError, match="missing"):
        db.update(missing)
    loaded = db["timeline-only"]
    assert loaded.metadata["preserved"] is True
    assert (
        loaded.timeline("mono").transcript_by_source("old-model").text == "prediction"
    )

    assert (
        loaded.timeline("mono").relabel_annotations_source(
            "old-model",
            "new-model",
            from_source_kind="model",
            to_source_kind="model",
        )
        == 1
    )
    assert db.update(loaded) is True
    loaded = db["timeline-only"]
    assert loaded.timeline("mono").annotations_by_source("old-model") == []
    assert (
        loaded.timeline("mono").transcript_by_source("new-model").text == "prediction"
    )

    loaded.timeline("mono").add_transcription(
        0, 1000, "second", source="second-model", source_kind="model"
    )
    assert db.update(loaded) is True
    assert (
        db["timeline-only"].timeline("mono").transcript_by_source("second-model").text
        == "second"
    )

    another = db["timeline-only"]
    another.metadata["batch"] = True
    assert db.update_many([another]) == 1


def test_audiodoc_has_no_legacy_load_factories(tmp_path):
    wav_path = tmp_path / "audio.wav"
    with wave.open(str(wav_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(struct.pack("<hh", 0, 1000))

    from_file = AudioDoc(AudioSource.from_path(str(wav_path)), id="file")
    from_bytes = AudioDoc(AudioSource.from_bytes(wav_path.read_bytes()), id="bytes")
    assert isinstance(from_file.source, AudioSource)
    assert isinstance(from_bytes.source, AudioSource)
    assert from_file.source.load().channels == 1
    assert from_bytes.source.load().channels == 1
    assert Audio.from_path(str(wav_path)).channels == 1
    assert Audio.from_bytes(wav_path.read_bytes()).channels == 1
    assert Audio.from_source(from_file.source).channels == 1
    assert not hasattr(AudioDoc, "from_file")
    assert not hasattr(AudioDoc, "from_url")
    assert not hasattr(AudioDoc, "from_bytes")
    assert not hasattr(AudioDoc, "from_pcm")
    assert not hasattr(from_file, "load")
    assert not hasattr(from_file.source.load(), "num_channels")
    assert repr(from_file).startswith('AudioDoc(id="file", file="')
    assert repr(from_file).endswith('")')
    assert str(from_file) == 'AudioDoc "file"'
