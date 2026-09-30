from typing import TypeAlias

from ._native import AudioEvent as AudioEvent
from ._native import Sentence as Sentence
from ._native import Speaker as Speaker
from ._native import Speech as Speech
from ._native import Token as Token
from ._native import Transcription as Transcription

Annotation: TypeAlias = AudioEvent

__all__: list[str]
