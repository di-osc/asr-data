from typing import Literal, TypeAlias

AnnotationKind: TypeAlias = Literal[
    "speech",
    "silence",
    "token",
    "transcription",
    "sentence",
    "speaker",
    "language",
    "hotword",
    "acoustic_event",
    "diagnostic",
]

AnnotationStatus: TypeAlias = Literal[
    "partial",
    "final",
    "revised",
    "deleted",
]

AnnotationSourceKind: TypeAlias = Literal[
    "user",
    "model",
    "stage",
    "system",
]
