from typing import TypeAlias

from ._native import Annotation as Annotation
from ._native import Speaker as Speaker
from ._native import Token as Token
from ._native import Transcription as Transcription
from ._types import AnnotationKind as AnnotationKind
from ._types import AnnotationStatus as AnnotationStatus

AnnotationPayload: TypeAlias = Speaker | Token | Transcription | dict[str, object] | None

__all__: list[str]
