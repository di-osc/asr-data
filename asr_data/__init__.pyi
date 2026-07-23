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

class Audio(_Audio):
    @staticmethod
    def aload_from_path(path: str) -> Awaitable[Audio]: ...
    @staticmethod
    def aload_from_source(
        source: AudioSource,
    ) -> Awaitable[Audio]: ...

class AudioDoc(_AudioDoc):
    @staticmethod
    def afrom_source(
        source: AudioSource, id: str | None = None
    ) -> Awaitable[AudioDoc]: ...

__all__: list[str]
