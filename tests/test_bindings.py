import asyncio
import io
import os
import struct
import threading
import time
import typing
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from asr_data import (
    AnnotationKind,
    AnnotationSourceKind,
    AnnotationStatus,
    Audio,
    AudioDB,
    Waveform,
)


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
    audio = Audio.from_pcm(pcm, sample_rate=8000, channels=2, id="call-1")
    audio.metadata["speaker"] = {"name": "alice", "age": 30}
    audio.timeline("mono").duration_ms = 1
    audio.timeline("mono").add_speech(0, 1, confidence=0.9)
    audio.timeline("mono").add_transcription(0, 1, "hello", language="en")

    waveform = audio.load()
    assert waveform.sample_rate == 8000
    assert waveform.channels == 2
    assert waveform.is_normalized is True
    assert waveform.source_format.encoding == "pcm_s16le"
    assert waveform.numpy().dtype == np.float32
    assert waveform.numpy().shape == (4,)
    left = waveform.channel(0)
    right = waveform.channel(1)
    assert left.channels == 1
    assert right.channels == 1
    np.testing.assert_allclose(
        left.numpy(),
        np.array([0.0, -1000 / 32768], dtype=np.float32),
    )
    np.testing.assert_allclose(
        right.numpy(),
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
        audio = Audio.from_pcm(b"\0\0", sample_rate=8000, id=f"audio-{index:03}")
        audio.timeline("mono").duration_ms = index * 10
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
    audio = Audio.from_pcm(b"\0\0" * 4, sample_rate=8000, channels=2, id="call-stereo")

    audio.ensure_timeline("left").add_transcription(0, 100, "caller")
    audio.ensure_timeline("right").add_transcription(0, 100, "agent")

    assert audio.timeline("mono").transcript().text == ""
    assert audio.timeline("left").transcript().text == "caller"
    assert audio.timeline(0).transcript().text == "caller"
    assert audio.timeline("right").transcript().text == "agent"
    assert audio.timeline(1).transcript().text == "agent"
    assert set(audio.timelines) == {"mono", "left", "right"}
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
    audio = Audio.from_pcm(b"\0\0" * 4, sample_rate=8000, channels=2, id="old-id")
    audio.ensure_timeline("left")
    audio.ensure_timeline("right")

    audio.timeline("left").audio_id = "new-id"

    assert audio.id == "new-id"
    assert audio.timeline("mono").audio_id == "new-id"
    assert audio.timeline("right").audio_id == "new-id"


def test_audio_duration_and_validation_are_audio_scoped():
    audio = Audio.from_pcm(b"\0\0" * 4000, sample_rate=8000, id="duration")
    audio.ensure_timeline("right")
    audio.duration_ms = 500

    assert audio.duration_ms == 500
    assert audio.timeline("mono").duration_ms == 500
    assert audio.timeline("right").duration_ms == 500
    audio.validate()


def test_waveform_from_numpy_copies_input():
    samples = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    waveform = Waveform(samples, 16000)
    samples[:] = 1.0
    np.testing.assert_array_equal(waveform.numpy(), np.array([0.0, 0.5, -0.5], np.float32))


def test_audio_aload_returns_waveform_without_blocking_api_changes():
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    audio = Audio.from_pcm(pcm, sample_rate=8000, channels=2, id="async-call")

    async def load():
        return await audio.aload()

    waveform = asyncio.run(load())

    assert waveform.sample_rate == 8000
    assert waveform.channels == 2


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
    audio = Audio.from_url(
        f"http://127.0.0.1:{server.server_port}/audio.wav", id="url-call"
    )

    async def load_and_probe_loop():
        task = asyncio.create_task(audio.aload())
        await asyncio.sleep(0.03)
        assert not task.done(), "download should still be waiting on the delayed HTTP response"
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


def test_waveform_display_builds_ipython_player(monkeypatch):
    import IPython.display

    displayed = []
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    waveform = Waveform(np.array([0.0, 0.25, -0.25], dtype=np.float32), 16000)

    waveform.display(autoplay=True)

    assert len(displayed) == 1
    assert isinstance(displayed[0], IPython.display.Audio)
    assert displayed[0].autoplay is True


def test_existing_db_can_be_opened_read_only():
    fixture = "tests/fixtures/lbg_call-100.vasr"
    if not os.path.exists(fixture):
        pytest.skip("optional legacy database fixture is not available")
    db = AudioDB(fixture, read_only=True)
    assert len(db) == 99
    assert len(list(db)[:2]) == 2


def test_public_types_have_informative_repr(tmp_path):
    pcm = struct.pack("<hhhh", 0, 1000, -1000, 2000)
    audio = Audio.from_pcm(pcm, sample_rate=8000, channels=2, id="call-1")
    audio.timeline("mono").duration_ms = 3250
    annotation = audio.timeline("mono").add_transcription(
        100,
        800,
        "hello world",
        confidence=0.95,
    )
    audio.metadata["speaker"] = "alice"
    waveform = audio.load()
    db = AudioDB(str(tmp_path / "repr.vasr"))
    db.insert(audio)

    assert repr(audio) == (
        'Audio(id="call-1", pcm_bytes=8, sample_rate=8000, channels=2, '
        'duration="3.25s", annotations=1)'
    )
    assert str(audio) == 'Audio "call-1" (3.25s)'
    assert "duration=0ms" in repr(waveform)
    assert "text=\"hello world\"" in repr(annotation)
    assert str(annotation) == 'transcription [100..800ms]: "hello world"'
    assert "duration=\"3.25s\"" in repr(audio.timeline("mono"))
    assert repr(db).endswith('mode="read-write", audios=1, duration="3.25s")')


def test_model_annotations_can_be_written_queried_and_removed():
    audio = Audio.from_pcm(b"\0\0" * 8000, sample_rate=8000, id="sources")
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
    assert [item.id for item in audio.timeline("mono").annotations_by_source("tegasr")] == [
        prediction.id
    ]
    assert audio.timeline("mono").transcript_by_source("tegasr").text == "prediction"
    assert audio.timeline("mono").transcript_by_source(
        "xlsx_import", source_kind="stage"
    ).text == "reference"
    original_id = prediction.id
    assert audio.timeline("mono").relabel_annotations_source(
        "tegasr", "tegasr-v2", from_source_kind="model"
    ) == 1
    relabeled = audio.timeline("mono").annotations_by_source("tegasr-v2")
    assert [item.id for item in relabeled] == [original_id]
    assert relabeled[0].source_kind == "model"
    assert audio.timeline("mono").remove_annotations_by_source("tegasr-v2") == 1
    assert audio.timeline("mono").annotations_by_source("tegasr-v2") == []


def test_database_update_detects_changes(tmp_path):
    path = tmp_path / "timeline-only.vasr"
    audio = Audio.from_pcm(
        b"\0\0" * 8000, sample_rate=8000, id="timeline-only"
    )
    audio.metadata["preserved"] = True
    db = AudioDB(str(path))
    db.insert(audio)

    audio.timeline("mono").add_transcription(
        0, 1000, "prediction", source="old-model", source_kind="model"
    )
    assert db.update(audio) is True
    assert db.update(audio) is False
    missing = Audio.from_pcm(b"\0\0", sample_rate=8000, id="missing")
    with pytest.raises(KeyError, match="missing"):
        db.update(missing)
    loaded = db["timeline-only"]
    assert loaded.metadata["preserved"] is True
    assert loaded.timeline("mono").transcript_by_source("old-model").text == "prediction"

    assert loaded.timeline("mono").relabel_annotations_source(
        "old-model",
        "new-model",
        from_source_kind="model",
        to_source_kind="model",
    ) == 1
    assert db.update(loaded) is True
    loaded = db["timeline-only"]
    assert loaded.timeline("mono").annotations_by_source("old-model") == []
    assert loaded.timeline("mono").transcript_by_source("new-model").text == "prediction"

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


def test_audio_has_only_explicit_public_constructors(tmp_path):
    wav_path = tmp_path / "audio.wav"
    with wave.open(str(wav_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(struct.pack("<hh", 0, 1000))

    from_file = Audio.from_file(str(wav_path), id="file")
    from_bytes = Audio.from_bytes(wav_path.read_bytes(), id="bytes")
    assert from_file.load().channels == 1
    assert from_bytes.load().channels == 1
    assert not hasattr(Audio, "from_source")
    assert not hasattr(Audio, "from_base64")
    assert not hasattr(Audio, "from_pcm_s16le")
    assert not hasattr(from_file, "source")
    assert not hasattr(from_file.load(), "num_channels")
    assert repr(from_file).startswith('Audio(id="file", file="')
    assert repr(from_file).endswith('")')
    assert str(from_file) == 'Audio "file"'
