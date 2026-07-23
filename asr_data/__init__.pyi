from collections.abc import AsyncIterator

from .annotation import Annotation as Annotation
from .annotation import AudioActivity as AudioActivity
from .annotation import Speaker as Speaker
from .annotation import Token as Token
from .annotation import Transcription as Transcription
from ._native import ActivityEvaluation as ActivityEvaluation
from ._native import ActivityEventEvaluation as ActivityEventEvaluation
from ._native import AsrDataError as AsrDataError
from ._native import Audio as _Audio
from ._native import AudioChunk as AudioChunk
from ._native import AudioDB as AudioDB
from ._native import AudioFormat as AudioFormat
from ._native import AudioInfo as AudioInfo
from ._native import AudioSource as AudioSource
from ._native import DatasetActivityEvaluation as DatasetActivityEvaluation
from ._native import DatasetActivityEventEvaluation as DatasetActivityEventEvaluation
from ._native import DatasetEvaluation as DatasetEvaluation
from ._native import DatasetTranscriptionEvaluation as DatasetTranscriptionEvaluation
from ._native import PredictionSpans as PredictionSpans
from ._native import ReferenceSpans as ReferenceSpans
from ._native import TimeSpan as TimeSpan
from ._native import Timeline as Timeline
from ._native import TimelineEvaluation as TimelineEvaluation
from ._native import Transcript as Transcript
from ._native import TranscriptionEvaluation as TranscriptionEvaluation
from ._native import Waveform as Waveform
from ._native import evaluate_dataset as evaluate_dataset
from ._native import normalize_zh as normalize_zh

class Audio(_Audio):
    def astream(self, chunk_size_ms: int = 100) -> AsyncIterator[AudioChunk]:
        """异步产生与 ``stream`` 相同的 AudioChunk。

        Args:
            chunk_size_ms: 每个 chunk 的目标时长。

        Returns:
            AudioChunk 异步迭代器。

        Examples:
            >>> async for chunk in audio.astream(100):
            ...     process(chunk)
        """

__all__: list[str]
