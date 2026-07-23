import asyncio as _asyncio

from .annotation import Annotation, AudioActivity, Speaker, Token, Transcription
from ._native import (
    ActivityEvaluation,
    ActivityEventEvaluation,
    AsrDataError,
    Audio,
    AudioChunk,
    AudioDB,
    AudioFormat,
    AudioInfo,
    AudioSource,
    DatasetActivityEvaluation,
    DatasetActivityEventEvaluation,
    DatasetEvaluation,
    DatasetTranscriptionEvaluation,
    PredictionSpans,
    ReferenceSpans,
    TimeSpan,
    Timeline,
    TimelineEvaluation,
    Transcript,
    TranscriptionEvaluation,
    Waveform,
    evaluate_dataset,
    normalize_zh,
)


async def _source_aopen(self, *, id=None):
    """异步探测来源并返回尚未加载波形的 Audio。

    Args:
        id: 可选稳定 Audio ID。

    Returns:
        已包含 info 和 timelines 的 Audio。

    Examples:
        >>> audio = await source.aopen(id="sample")
    """
    return await _asyncio.to_thread(self.open, id=id)


async def _source_aload(self, *, id=None):
    """异步解码来源并返回已加载的 Audio。

    Args:
        id: 可选稳定 Audio ID。

    Returns:
        已携带完整波形的 Audio。

    Examples:
        >>> audio = await source.aload(id="sample")
    """
    return await _asyncio.to_thread(self.load, id=id)


async def _source_aprobe(self):
    """异步读取来源的 AudioInfo。

    Returns:
        不包含解码采样的 AudioInfo。

    Examples:
        >>> info = await source.aprobe()
    """
    return await _asyncio.to_thread(self.probe)


class _AudioAsyncIterator:
    def __init__(self, iterator):
        self._iterator = iterator

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await _asyncio.to_thread(_next_or_none, self._iterator)
        if item is None:
            raise StopAsyncIteration
        return item

    async def aclose(self):
        await _asyncio.to_thread(self._iterator.close)


def _next_or_none(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


def _audio_astream(self, chunk_size_ms=100):
    """异步产生与 ``stream`` 相同的 AudioChunk。

    Args:
        chunk_size_ms: 每个 chunk 的目标时长，单位为毫秒。

    Returns:
        AudioChunk 异步迭代器。

    Examples:
        >>> async for chunk in audio.astream(100):
        ...     process(chunk)
    """
    return _AudioAsyncIterator(self.stream(chunk_size_ms))


AudioSource.aopen = _source_aopen
AudioSource.aload = _source_aload
AudioSource.aprobe = _source_aprobe
Audio.astream = _audio_astream

__all__ = [
    "ActivityEvaluation",
    "ActivityEventEvaluation",
    "Annotation",
    "AsrDataError",
    "Audio",
    "AudioActivity",
    "AudioChunk",
    "AudioDB",
    "AudioFormat",
    "AudioInfo",
    "AudioSource",
    "DatasetActivityEvaluation",
    "DatasetActivityEventEvaluation",
    "DatasetEvaluation",
    "DatasetTranscriptionEvaluation",
    "PredictionSpans",
    "ReferenceSpans",
    "Speaker",
    "TimeSpan",
    "Timeline",
    "TimelineEvaluation",
    "Token",
    "Transcript",
    "Transcription",
    "TranscriptionEvaluation",
    "Waveform",
    "evaluate_dataset",
    "normalize_zh",
]
