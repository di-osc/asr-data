from typing import Literal, TypeAlias

AnnotationKind: TypeAlias = Literal[
    "speech",
    "token",
    "transcription",
    "sentence",
    "speaker",
    "language",
    "acoustic_event",
]

AnnotationStatus: TypeAlias = Literal[
    "partial",
    "final",
    "revised",
    "deleted",
]
