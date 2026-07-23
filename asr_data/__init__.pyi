from collections.abc import Awaitable

from .annotation import Annotation as Annotation
from .annotation import AnnotationKind as AnnotationKind
from .annotation import AnnotationPayload as AnnotationPayload
from .annotation import Speaker as Speaker
from .annotation import Token as Token
from .annotation import Transcription as Transcription
from ._native import AudioDB as AudioDB
from ._native import AudioDoc as _AudioDoc
from ._native import AudioFormat as AudioFormat
from ._native import AudioInfo as AudioInfo
from ._native import AudioSource as AudioSource
from ._native import DatasetEvaluation as DatasetEvaluation
from ._native import DatasetSpeechEvaluation as DatasetSpeechEvaluation
from ._native import DatasetTranscriptionEvaluation as DatasetTranscriptionEvaluation
from ._native import PredictionAnnotations as PredictionAnnotations
from ._native import ReferenceAnnotations as ReferenceAnnotations
from ._native import Timeline as Timeline
from ._native import TimelineEvaluation as TimelineEvaluation
from ._native import TranscriptionEvaluation as TranscriptionEvaluation
from ._native import SpeechEvaluation as SpeechEvaluation
from ._native import Transcript as Transcript
from ._native import AsrDataError as AsrDataError
from ._native import Audio as _Audio
from ._native import AudioChunk as AudioChunk
from ._native import normalize_zh as normalize_zh
from ._native import evaluate_dataset as evaluate_dataset

class Audio(_Audio):
    """已解码到内存中的音频波形。"""

    @staticmethod
    def aload_from_path(path: str) -> Awaitable[Audio]:
        """异步读取本地文件并返回完整 Audio。

        Args:
            path: 本地音频文件路径。

        Returns:
            可等待的完整 Audio。

        Raises:
            AsrDataError: 文件无法读取或解码。

        Examples:
            >>> import asyncio
            >>> from asr_data import Audio
            >>> audio = asyncio.run(Audio.aload_from_path("audio.wav"))
        """
    @staticmethod
    def aload_from_source(
        source: AudioSource,
    ) -> Awaitable[Audio]:
        """异步加载任意 AudioSource 并返回完整 Audio。

        Args:
            source: 要加载的 AudioSource。

        Returns:
            可等待的完整 Audio。

        Raises:
            AsrDataError: 来源无法读取或解码。

        Examples:
            >>> import asyncio
            >>> from asr_data import Audio, AudioSource
            >>> source = AudioSource.from_pcm(b"\0\0" * 10, 16000)
            >>> asyncio.run(Audio.aload_from_source(source)).frame_count
            10
        """

class AudioDoc(_AudioDoc):
    """音频来源、元信息、时间轴、标注和 metadata 的集合。"""

    @staticmethod
    def afrom_source(source: AudioSource, id: str | None = None) -> Awaitable[AudioDoc]:
        """异步探测来源并创建 AudioDoc。

        Args:
            source: 音频来源。
            id: 可选稳定文档 ID。

        Returns:
            可等待的 AudioDoc。

        Raises:
            AsrDataError: 来源无法读取或探测。

        Examples:
            >>> import asyncio
            >>> from asr_data import AudioDoc, AudioSource
            >>> source = AudioSource.from_pcm(b"\0\0" * 10, 16000)
            >>> asyncio.run(AudioDoc.afrom_source(source)).audio_info.frame_count
            10
        """

__all__: list[str]
