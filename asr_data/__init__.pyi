from collections.abc import AsyncIterator

from .annotation import Annotation as Annotation
from .annotation import AudioActivity as AudioActivity
from .annotation import Speaker as Speaker
from .annotation import Token as Token
from .annotation import Transcription as Transcription
from ._native import ActivityEvaluation as ActivityEvaluation
from ._native import ActivityEventEvaluation as ActivityEventEvaluation
from ._native import AsrDataError as AsrDataError
from ._native import Audio as Audio
from ._native import AudioChunk as AudioChunk
from ._native import AudioDB as AudioDB
from ._native import AudioFormat as AudioFormat
from ._native import AudioInfo as AudioInfo
from ._native import AudioSource as AudioSource
from ._native import AudioStream as _AudioStream
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

class AudioStream(_AudioStream):
    def __aiter__(self) -> AsyncIterator[AudioChunk]:
        """异步迭代当前流，不能与同步迭代混用。"""

__all__: list[str]
