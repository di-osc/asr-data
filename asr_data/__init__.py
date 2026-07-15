import asyncio as _asyncio

from ._types import AnnotationKind, AnnotationSourceKind, AnnotationStatus
from ._native import (
    Annotation,
    Audio,
    AudioDB,
    AudioFormat,
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


Audio.aload = _audio_aload

__all__ = [
    "Annotation",
    "AnnotationKind",
    "AnnotationSourceKind",
    "AnnotationStatus",
    "Audio",
    "AudioDB",
    "AudioFormat",
    "Timeline",
    "Transcript",
    "AsrDataError",
    "Waveform",
]
