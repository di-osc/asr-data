from collections.abc import Awaitable

from ._types import AnnotationKind as AnnotationKind
from ._types import AnnotationSourceKind as AnnotationSourceKind
from ._types import AnnotationStatus as AnnotationStatus
from ._native import Annotation as Annotation
from ._native import AudioDB as AudioDB
from ._native import AudioDoc as AudioDoc
from ._native import AudioFormat as AudioFormat
from ._native import AudioSource as AudioSource
from ._native import Timeline as Timeline
from ._native import Transcript as Transcript
from ._native import AsrDataError as AsrDataError
from ._native import Audio as _Audio
from ._native import AudioChunk as AudioChunk

class Audio(_Audio):
    @staticmethod
    def aload_from_path(path: str) -> Awaitable[Audio]: ...
    @staticmethod
    def aload_from_source(
        source: AudioSource,
    ) -> Awaitable[Audio]: ...

__all__: list[str]
