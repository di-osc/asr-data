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
    Waveform,
)


async def _source_aload(self):
    task = self._start_aload()
    while not task.done():
        await _asyncio.sleep(0.005)
    return task.result()


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

Waveform.aload_from_path = classmethod(_wf_aload_from_path)
Waveform.aload_from_source = classmethod(_wf_aload_from_source)

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
    "Waveform",
]
