import asyncio as _asyncio

from ._types import AnnotationKind, AnnotationSourceKind, AnnotationStatus
from ._native import (
    Annotation,
    Audio,
    AudioBase64,
    AudioBytes,
    AudioDB,
    AudioFormat,
    AudioPath,
    AudioPcm,
    AudioUrl,
    Timeline,
    Transcript,
    AsrDataError,
    Waveform,
)


async def _audio_aload(self: Audio) -> Waveform:
    """Download asynchronously and decode on a Rust blocking worker."""
    task = self._start_aload()
    while not task.done():
        await _asyncio.sleep(0.005)
    return task.result()


async def _source_aload(self):
    task = self._start_aload()
    while not task.done():
        await _asyncio.sleep(0.005)
    return task.result()


Audio.aload = _audio_aload

for _cls in (AudioPath, AudioUrl, AudioBytes, AudioBase64, AudioPcm):
    _cls.aload = _source_aload

__all__ = [
    "Annotation",
    "AnnotationKind",
    "AnnotationSourceKind",
    "AnnotationStatus",
    "Audio",
    "AudioBase64",
    "AudioBytes",
    "AudioDB",
    "AudioFormat",
    "AudioPath",
    "AudioPcm",
    "AudioUrl",
    "Timeline",
    "Transcript",
    "AsrDataError",
    "Waveform",
]
