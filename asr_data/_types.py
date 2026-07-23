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
