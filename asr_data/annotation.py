from typing import TypeAlias

from ._native import AudioActivity, Speaker, Token, Transcription

Annotation: TypeAlias = AudioActivity | Speaker | Token | Transcription

__all__ = [
    "Annotation",
    "AudioActivity",
    "Speaker",
    "Token",
    "Transcription",
]
