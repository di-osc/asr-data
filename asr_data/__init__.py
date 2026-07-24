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
    AudioStream,
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


async def _source_aload(self, *, id=None):
    """异步解码来源并返回已加载的 Audio。

    Args:
        id: 可选的文档 ID。

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


class _AudioStreamAsyncIterator:
    def __init__(self, stream):
        self._stream = stream

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await _asyncio.to_thread(self._stream._next_async)
        if item is None:
            raise StopAsyncIteration
        return item

    async def aclose(self):
        await _asyncio.to_thread(self._stream.close)


def _stream_aiter(self):
    """异步迭代当前 AudioStream，不能与同步迭代混用。"""
    return _AudioStreamAsyncIterator(self)


AudioSource.aload = _source_aload
AudioSource.aprobe = _source_aprobe
AudioStream.__aiter__ = _stream_aiter

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
    "AudioStream",
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
