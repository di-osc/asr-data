from typing import TypeAlias

from ._native import AudioActivity as AudioActivity
from ._native import Speaker as Speaker
from ._native import Token as Token
from ._native import Transcription as Transcription

Annotation: TypeAlias = AudioActivity | Speaker | Token | Transcription

__all__: list[str]
