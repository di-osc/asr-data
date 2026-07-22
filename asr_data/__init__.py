import asyncio as _asyncio

from ._types import AnnotationKind, AnnotationStatus
from ._native import (
    Annotation,
    AudioDB,
    AudioDoc,
    AudioFormat,
    AudioSource,
    PredictionAnnotations,
    ReferenceAnnotations,
    Speaker,
    Timeline,
    Token,
    Transcript,
    Transcription,
    AsrDataError,
    Audio,
    AudioChunk,
)


async def _source_aload(self, *, sample_rate=None, mono=None):
    task = self._start_aload(sample_rate=sample_rate, mono=mono)
    while not task.done():
        await _asyncio.sleep(0.005)
    return task.result()


class _SourceAsyncIterator:
    def __init__(self, task):
        self._task = task

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            chunk = self._task.next_result()
            if chunk is not None:
                return chunk
            if self._task.done():
                raise StopAsyncIteration
            await _asyncio.sleep(0.005)


def _source_astream(
    self,
    chunk_size_ms=100,
    *,
    sample_rate=None,
    mono=None,
):
    return _SourceAsyncIterator(
        self._start_astream(
            chunk_size_ms=chunk_size_ms,
            sample_rate=sample_rate,
            mono=mono,
        )
    )


async def _wf_aload_from_path(cls, path: str):
    task = cls._start_aload_from_path(path)
    while not task.done():
        await _asyncio.sleep(0.005)
    return task.result()


async def _wf_aload_from_source(cls, source):
    task = cls._start_aload_from_source(source)
    while not task.done():
        await _asyncio.sleep(0.005)
    return task.result()


AudioSource.aload = _source_aload
AudioSource.astream = _source_astream

Audio.aload_from_path = classmethod(_wf_aload_from_path)
Audio.aload_from_source = classmethod(_wf_aload_from_source)

__all__ = [
    "Annotation",
    "AnnotationKind",
    "AnnotationStatus",
    "AudioDB",
    "AudioDoc",
    "AudioFormat",
    "AudioSource",
    "PredictionAnnotations",
    "ReferenceAnnotations",
    "Speaker",
    "Timeline",
    "Token",
    "Transcript",
    "Transcription",
    "AsrDataError",
    "Audio",
    "AudioChunk",
]
