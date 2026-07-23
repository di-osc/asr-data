from typing import TypeAlias

from ._native import Annotation, Speaker, Token, Transcription
from ._types import AnnotationKind

AnnotationPayload: TypeAlias = Speaker | Token | Transcription | dict[str, object] | None

__all__ = [
    "Annotation",
    "AnnotationKind",
    "AnnotationPayload",
    "Speaker",
    "Token",
    "Transcription",
]
