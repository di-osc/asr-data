import asyncio as _asyncio

from .annotation import (
    Annotation,
    AnnotationKind,
    AnnotationPayload,
    Speaker,
    Token,
    Transcription,
)
from ._native import (
    AudioDB,
    AudioDoc,
    AudioFormat,
    AudioInfo,
    AudioSource,
    DatasetEvaluation,
    DatasetSpeechEvaluation,
    DatasetTranscriptionEvaluation,
    PredictionAnnotations,
    ReferenceAnnotations,
    Timeline,
    TimelineEvaluation,
    TranscriptionEvaluation,
    SpeechEvaluation,
    Transcript,
    AsrDataError,
    Audio,
    AudioChunk,
    evaluate_dataset,
    normalize_zh,
)


async def _source_aload(self, *, sample_rate=None, mono=None):
    """异步加载并解码完整音频。

    Args:
        sample_rate: 可选目标采样率。
        mono: 设为 ``True`` 时混合为单声道。

    Returns:
        解码后的完整 :class:`Audio`。

    Raises:
        AsrDataError: 来源无法读取、解码或转换。

    Examples:
        >>> import asyncio
        >>> from asr_data import AudioSource
        >>> source = AudioSource.from_pcm(b"\\0\\0" * 16000, 16000)
        >>> audio = asyncio.run(source.aload())
        >>> audio.duration_ms
        1000.0
    """
    task = self._start_aload(sample_rate=sample_rate, mono=mono)
    while not task.done():
        await _asyncio.sleep(0.005)
    return task.result()


async def _source_aprobe(self):
    """异步读取音频格式和时长信息，但不解码浮点采样。

    Returns:
        不包含采样数据的 :class:`AudioInfo`。

    Raises:
        AsrDataError: 来源无法读取或探测。

    Examples:
        >>> import asyncio
        >>> from asr_data import AudioSource
        >>> source = AudioSource.from_pcm(b"\\0\\0" * 16000, 16000)
        >>> asyncio.run(source.aprobe()).duration_ms
        1000.0
    """
    task = self._start_aprobe()
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
    """异步流式解码音频。

    Args:
        chunk_size_ms: 每个片段的目标时长，单位为毫秒。
        sample_rate: 可选目标采样率。
        mono: 设为 ``True`` 时混合为单声道。

    Returns:
        产生 :class:`AudioChunk` 的异步迭代器。

    Raises:
        ValueError: ``chunk_size_ms`` 为零。
        AsrDataError: 来源无法读取或解码。

    Examples:
        >>> import asyncio
        >>> from asr_data import AudioSource
        >>> async def collect():
        ...     source = AudioSource.from_pcm(b"\\0\\0" * 16000, 16000)
        ...     return [chunk async for chunk in source.astream(500)]
        >>> len(asyncio.run(collect()))
        2
    """
    return _SourceAsyncIterator(
        self._start_astream(
            chunk_size_ms=chunk_size_ms,
            sample_rate=sample_rate,
            mono=mono,
        )
    )


async def _wf_aload_from_path(cls, path: str):
    """异步读取本地文件并返回完整 Audio。

    Args:
        path: 本地音频文件路径。

    Returns:
        解码后的完整 :class:`Audio`。

    Raises:
        AsrDataError: 文件无法读取或解码。

    Examples:
        >>> import asyncio
        >>> from tempfile import NamedTemporaryFile
        >>> from urllib.request import urlretrieve
        >>> from asr_data import Audio
        >>> url = "https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav"
        >>> with NamedTemporaryFile(suffix=".wav") as file:
        ...     _ = urlretrieve(url, file.name)
        ...     audio = asyncio.run(Audio.aload_from_path(file.name))
    """
    task = cls._start_aload_from_path(path)
    while not task.done():
        await _asyncio.sleep(0.005)
    return task.result()


async def _wf_aload_from_source(cls, source):
    """异步加载任意 AudioSource 并返回完整 Audio。

    Args:
        source: 要加载的 :class:`AudioSource`。

    Returns:
        解码后的完整 :class:`Audio`。

    Raises:
        AsrDataError: 来源无法读取或解码。

    Examples:
        >>> import asyncio
        >>> from asr_data import Audio, AudioSource
        >>> source = AudioSource.from_pcm(b"\\0\\0" * 10, 16000)
        >>> asyncio.run(Audio.aload_from_source(source)).frame_count
        10
    """
    task = cls._start_aload_from_source(source)
    while not task.done():
        await _asyncio.sleep(0.005)
    return task.result()


async def _doc_afrom_source(cls, source, id=None):
    """异步探测来源并创建 AudioDoc。

    Args:
        source: 音频来源。
        id: 可选稳定文档 ID。

    Returns:
        已初始化 AudioInfo 和 timelines 的 :class:`AudioDoc`。

    Raises:
        AsrDataError: 来源无法读取或探测。

    Examples:
        >>> import asyncio
        >>> from asr_data import AudioDoc, AudioSource
        >>> source = AudioSource.from_pcm(b"\\0\\0" * 10, 16000)
        >>> asyncio.run(AudioDoc.afrom_source(source)).audio_info.frame_count
        10
    """
    task = cls._start_afrom_source(source, id=id)
    while not task.done():
        await _asyncio.sleep(0.005)
    return task.result()


AudioSource.aload = _source_aload
AudioSource.aprobe = _source_aprobe
AudioSource.astream = _source_astream
AudioDoc.afrom_source = classmethod(_doc_afrom_source)

Audio.aload_from_path = classmethod(_wf_aload_from_path)
Audio.aload_from_source = classmethod(_wf_aload_from_source)

__all__ = [
    "Annotation",
    "AnnotationKind",
    "AnnotationPayload",
    "AudioDB",
    "AudioDoc",
    "AudioFormat",
    "AudioInfo",
    "AudioSource",
    "DatasetEvaluation",
    "DatasetSpeechEvaluation",
    "DatasetTranscriptionEvaluation",
    "PredictionAnnotations",
    "ReferenceAnnotations",
    "Speaker",
    "Timeline",
    "TimelineEvaluation",
    "TranscriptionEvaluation",
    "SpeechEvaluation",
    "Token",
    "Transcript",
    "Transcription",
    "AsrDataError",
    "Audio",
    "AudioChunk",
    "evaluate_dataset",
    "normalize_zh",
]
