from typing import TypeAlias

from ._native import AudioEvent, Sentence, Speaker, Speech, Token, Transcription

Annotation: TypeAlias = AudioEvent

__all__ = [
    "Annotation",
    "AudioEvent",
    "Sentence",
    "Speaker",
    "Speech",
    "Token",
    "Transcription",
]
