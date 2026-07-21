import asyncio as _asyncio

from ._types import AnnotationKind, AnnotationSourceKind, AnnotationStatus
from ._native import (
    Annotation,
    AudioBase64,
    AudioBytes,
    AudioDB,
    AudioDoc,
    AudioFormat,
    AudioPath,
    AudioPcm,
    AudioUrl,
    Timeline,
    Transcript,
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


for _cls in (AudioPath, AudioUrl, AudioBytes, AudioBase64, AudioPcm):
    _cls.aload = _source_aload
    _cls.astream = _source_astream

Audio.aload_from_path = classmethod(_wf_aload_from_path)
Audio.aload_from_source = classmethod(_wf_aload_from_source)

__all__ = [
    "Annotation",
    "AnnotationKind",
    "AnnotationSourceKind",
    "AnnotationStatus",
    "AudioBase64",
    "AudioBytes",
    "AudioDB",
    "AudioDoc",
    "AudioFormat",
    "AudioPath",
    "AudioPcm",
    "AudioUrl",
    "Timeline",
    "Transcript",
    "AsrDataError",
    "Audio",
    "AudioChunk",
]
