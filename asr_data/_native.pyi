from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import Any, Awaitable, Literal
import numpy as np
import numpy.typing as npt

class AsrDataError(Exception): ...

def normalize_zh(text: str) -> str:
    """使用内嵌中文 TN 资源把书写形式转换为口语形式。

    Args:
        text: 要标准化的原始文本。

    Returns:
        转换为口语形式的文本。

    Raises:
        AsrDataError: 内嵌 FST 无法执行。

    Examples:
        >>> from asr_data import normalize_zh
        >>> normalize_zh("2024年")
        '二零二四年'
    """

class AudioFormat:
    """音频编码、采样率和声道数组成的格式信息。"""
    @property
    def encoding(self) -> str: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def channels(self) -> int: ...

class AudioInfo:
    """不包含解码采样的音频元信息。"""
    @property
    def sample_rate(self) -> int: ...
    @property
    def channels(self) -> int: ...
    @property
    def frame_count(self) -> int: ...
    @property
    def duration_ms(self) -> float: ...
    @property
    def source_format(self) -> AudioFormat: ...

class _AudioLoadTask:
    def done(self) -> bool: ...
    def result(self) -> Waveform: ...

class AudioSource:
    """尚未解码的音频来源描述。"""
    @staticmethod
    def from_path(path: str) -> AudioSource:
        """从本地路径创建来源，不立即读取文件。

        Args:
            path: 相对或绝对文件路径。

        Returns:
            尚未加载的 AudioSource。

        Examples:
            >>> from asr_data import AudioSource
            >>> AudioSource.from_path("audio.wav").path
            'audio.wav'
        """
    @staticmethod
    def from_url(url: str) -> AudioSource:
        """从 URL 创建来源，不立即发起请求。

        Args:
            url: HTTP 或 HTTPS 音频地址。

        Returns:
            尚未发起请求的 AudioSource。

        Examples:
            >>> from asr_data import AudioSource
            >>> AudioSource.from_url("https://example.com/a.wav").kind
            'url'
        """
    @staticmethod
    def from_bytes(data: bytes) -> AudioSource:
        """从带容器或编码信息的音频字节创建来源。

        Args:
            data: WAV、MP3 等带格式信息的编码字节。

        Returns:
            保存编码字节的 AudioSource。

        Examples:
            >>> from asr_data import AudioSource
            >>> AudioSource.from_bytes(b"RIFF").kind
            'bytes'
        """
    @staticmethod
    def from_base64(data: str) -> AudioSource:
        """从 base64 字符串或 data URL 创建来源。

        Args:
            data: base64 内容。

        Returns:
            保存原字符串的 AudioSource。

        Examples:
            >>> from asr_data import AudioSource
            >>> AudioSource.from_base64("UklGRg==").kind
            'base64'
        """
    @staticmethod
    def from_pcm(data: bytes, sample_rate: int, channels: int = 1) -> AudioSource:
        """从 PCM S16LE 原始字节创建来源。

        Args:
            data: 按帧交错的有符号 16 位小端字节。
            sample_rate: 采样率。
            channels: 声道数，默认为 1。

        Returns:
            保存 PCM 数据和格式参数的 AudioSource。

        Raises:
            ValueError: 采样率、声道数或 PCM 帧长度无效。

        Examples:
            >>> from asr_data import AudioSource
            >>> AudioSource.from_pcm(b"\0\0" * 10, 16000).channels
            1
        """
    @property
    def kind(self) -> Literal["path", "url", "bytes", "base64", "pcm"]: ...
    @property
    def path(self) -> str | None: ...
    @property
    def url(self) -> str | None: ...
    @property
    def bytes(self) -> bytes | None: ...
    @property
    def base64(self) -> str | None: ...
    @property
    def pcm(self) -> bytes | None: ...
    @property
    def sample_rate(self) -> int | None: ...
    @property
    def channels(self) -> int | None: ...
    def load(self, *, id: str | None = None) -> Audio:
        """创建并完整解码 Audio。

        Args:
            id: 可选的文档 ID。

        Returns:
            已在内存中保留完整 Waveform 的 Audio。

        Examples:
            >>> from asr_data import AudioSource
            >>> audio = AudioSource.from_pcm(b"\0\0", 16000).load(id="sample")
        """
    def stream(
        self, chunk_size_ms: int = 100, *, id: str | None = None
    ) -> AudioStream:
        """创建 timeline 随 AudioChunk 迭代增长的 AudioStream。

        Args:
            chunk_size_ms: 每个 chunk 的目标时长。
            id: 可选的文档 ID。

        Returns:
            可同步或异步迭代的 AudioStream。

        Raises:
            ValueError: chunk_size_ms 为零。
            AsrDataError: 来源无法探测或初始化流。

        Examples:
            >>> stream = source.stream(100, id="sample")
        """
    def probe(self) -> AudioInfo:
        """读取格式和时长信息，但不解码浮点采样。

        Returns:
            不包含采样数据的 AudioInfo。

        Raises:
            AsrDataError: 来源无法读取或探测。

        Examples:
            >>> from asr_data import AudioSource
            >>> AudioSource.from_pcm(b"\0\0" * 16000, 16000).probe().duration_ms
            1000.0
        """
    def aprobe(self) -> Awaitable[AudioInfo]:
        """异步读取格式和时长信息，但不解码浮点采样。

        Returns:
            可等待的 AudioInfo。

        Raises:
            AsrDataError: 来源无法读取或探测。

        Examples:
            >>> import asyncio
            >>> from asr_data import AudioSource
            >>> source = AudioSource.from_pcm(b"\0\0" * 16000, 16000)
            >>> asyncio.run(source.aprobe()).duration_ms
            1000.0
        """
    def aload(self, *, id: str | None = None) -> Awaitable[Audio]:
        """异步创建并完整解码 Audio。

        Args:
            id: 可选的文档 ID。

        Returns:
            可等待的已加载 Audio。

        Examples:
            >>> audio = await source.aload(id="sample")
        """

class Waveform:
    """已解码到内存中的音频波形。

    Args:
        samples: 一维 float32 兼容数组；多声道样本按帧交错排列。
        sample_rate: 每秒每个声道的采样帧数。
        channels: 声道数，默认为 1。

    Raises:
        ValueError: 格式参数无效，或样本数不能整除声道数。

    Examples:
        >>> import numpy as np
        >>> from asr_data import Waveform
        >>> Waveform(np.zeros(16000), 16000).duration_ms
        1000.0
    """
    def __init__(
        self, samples: npt.ArrayLike, sample_rate: int, channels: int = 1
    ) -> None: ...
    @staticmethod
    def from_path(path: str) -> Waveform:
        """从本地文件加载并解码音频。

        Args:
            path: 本地音频文件路径。

        Returns:
            解码后的完整 Waveform。

        Raises:
            AsrDataError: 文件无法读取或音频无法解码。

        Examples:
            >>> from asr_data import Waveform
            >>> audio = Waveform.from_path("audio.wav")
        """
    @staticmethod
    def from_url(url: str) -> Waveform:
        """从 HTTP 或 HTTPS URL 下载并解码音频。

        Args:
            url: 音频 URL。

        Returns:
            解码后的完整 Waveform。

        Raises:
            AsrDataError: 请求失败或音频无法解码。

        Examples:
            >>> from asr_data import Waveform
            >>> audio = Waveform.from_url(
            ...     "https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav"
            ... )
        """
    @staticmethod
    def from_bytes(data: bytes) -> Waveform:
        """从 WAV、MP3 等编码字节解码音频。

        Args:
            data: 包含音频容器或编码信息的字节。

        Returns:
            解码后的完整 Waveform。

        Raises:
            AsrDataError: 字节不是受支持的音频。

        Examples:
            >>> from urllib.request import urlopen
            >>> from asr_data import Waveform
            >>> url = "https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav"
            >>> audio = Waveform.from_bytes(urlopen(url).read())
        """
    @staticmethod
    def from_base64(data: str) -> Waveform:
        """从 base64 编码的音频字符串解码音频。

        Args:
            data: base64 字符串或 data URL。

        Returns:
            解码后的完整 Waveform。

        Raises:
            AsrDataError: base64 或音频编码无效。

        Examples:
            >>> import base64
            >>> from urllib.request import urlopen
            >>> from asr_data import Waveform
            >>> url = "https://deepasset.oss-cn-beijing.aliyuncs.com/example.wav"
            >>> data = base64.b64encode(urlopen(url).read()).decode()
            >>> audio = Waveform.from_base64(data)
        """
    @staticmethod
    def from_pcm(data: bytes, sample_rate: int, channels: int = 1) -> Waveform:
        """从 PCM S16LE 原始字节创建音频。

        Args:
            data: 按帧交错的有符号 16 位小端 PCM 字节。
            sample_rate: 采样率。
            channels: 声道数，默认为 1。

        Returns:
            转换为 float32 样本的 Waveform。

        Raises:
            ValueError: PCM 参数或帧长度无效。

        Examples:
            >>> from asr_data import Waveform
            >>> Waveform.from_pcm(b"\0\0" * 16000, 16000).duration_ms
            1000.0
        """
    @staticmethod
    def from_source(
        source: AudioSource,
    ) -> Waveform:
        """加载任意 AudioSource 并解码完整音频。

        Args:
            source: 要加载的 AudioSource。

        Returns:
            解码后的完整 Waveform。

        Raises:
            AsrDataError: 来源无法读取或解码。

        Examples:
            >>> from asr_data import Waveform, AudioSource
            >>> source = AudioSource.from_pcm(b"\0\0" * 10, 16000)
            >>> Waveform.from_source(source).frame_count
            10
        """
    @staticmethod
    def _start_aload_from_path(path: str) -> _AudioLoadTask: ...
    @staticmethod
    def _start_aload_from_source(
        source: AudioSource,
    ) -> _AudioLoadTask: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def channels(self) -> int: ...
    @property
    def frame_count(self) -> int: ...
    @property
    def duration_ms(self) -> float: ...
    @property
    def source_format(self) -> AudioFormat | None: ...
    @property
    def samples(self) -> npt.NDArray[np.float32]: ...
    def display(
        self,
        start_ms: int | None = None,
        end_ms: int | None = None,
        autoplay: bool = False,
    ) -> None:
        """在 Jupyter 中显示音频播放器。

        Args:
            start_ms: 可选播放起始时间。
            end_ms: 可选播放结束时间。
            autoplay: 是否自动播放。

        Returns:
            None；播放器直接发送到当前 Jupyter 输出。

        Raises:
            ValueError: 结束时间早于起始时间。
            AsrDataError: IPython 不可用。

        Examples:
            >>> import numpy as np
            >>> from asr_data import Waveform
            >>> Waveform(np.zeros(16000), 16000).display(end_ms=500)
        """
    def to_mono(self) -> Waveform:
        """混合所有声道并返回新的单声道 Waveform。

        Returns:
            不修改原对象的新 Waveform。

        Examples:
            >>> import numpy as np
            >>> from asr_data import Waveform
            >>> Waveform(np.zeros(20), 16000, 2).to_mono().channels
            1
        """
    def channel(self, index: int) -> Waveform:
        """提取指定声道。

        Args:
            index: 从 0 开始的声道索引。

        Returns:
            提取出的单声道 Waveform。

        Raises:
            AsrDataError: 索引超出范围。

        Examples:
            >>> import numpy as np
            >>> from asr_data import Waveform
            >>> Waveform(np.zeros(20), 16000, 2).channel(0).channels
            1
        """
    def resample(self, sample_rate: int) -> Waveform:
        """重采样并返回新的 Waveform。

        Args:
            sample_rate: 目标采样率。

        Returns:
            不修改原对象的新 Waveform。

        Raises:
            ValueError: 目标采样率为零。

        Examples:
            >>> import numpy as np
            >>> from asr_data import Waveform
            >>> Waveform(np.zeros(160), 16000).resample(8000).sample_rate
            8000
        """
    def slice_ms(self, start_ms: int, end_ms: int) -> Waveform:
        """按半开毫秒范围截取音频。

        Args:
            start_ms: 起始时间，包含。
            end_ms: 结束时间，不包含。

        Returns:
            截取后的新 Waveform。

        Examples:
            >>> import numpy as np
            >>> from asr_data import Waveform
            >>> Waveform(np.zeros(16000), 16000).slice_ms(0, 500).duration_ms
            500.0
        """
    def split_at_low_energy(self, max_duration_ms: int) -> list[Waveform]:
        """在低能量位置拆分音频。

        Args:
            max_duration_ms: 每段的最大目标时长。

        Returns:
            保持原顺序的 Waveform 列表。

        Raises:
            ValueError: ``max_duration_ms`` 为零。

        Examples:
            >>> import numpy as np
            >>> from asr_data import Waveform
            >>> len(Waveform(np.zeros(32000), 16000).split_at_low_energy(1000))
            3
        """

class AudioChunk:
    """AudioStream 当前迭代出的局部音频块，共享父流的文档上下文。"""
    @property
    def id(self) -> str: ...
    @property
    def source(self) -> AudioSource: ...
    @property
    def info(self) -> AudioInfo: ...
    @property
    def metadata(self) -> dict[str, JsonValue]: ...
    @property
    def timelines(self) -> dict[str, Timeline]: ...
    @property
    def index(self) -> int: ...
    @property
    def offset_ms(self) -> int: ...
    @property
    def end_ms(self) -> int: ...
    @property
    def is_final(self) -> bool: ...
    @property
    def duration_ms(self) -> float: ...
    def to_timeline_range(self, start_ms: int, end_ms: int) -> tuple[int, int]:
        """把 chunk 内的局部范围转换为共享 Timeline 的全局范围。

        Args:
            start_ms: chunk 内相对起始时间。
            end_ms: chunk 内相对结束时间。

        Returns:
            ``(start_ms, end_ms)`` Timeline 全局毫秒范围。

        Examples:
            >>> chunk.to_timeline_range(0, 100)
            (1000, 1100)
        """
    def as_waveform(self, channel: str | int | None = None) -> Waveform:
        """返回 chunk 的波形视图。

        Args:
            channel: 可选声道名称或索引。

        Returns:
            当前 chunk 对应的 Waveform。

        Examples:
            >>> waveform = chunk.as_waveform()
        """
    def display(
        self,
        start_ms: int | None = None,
        end_ms: int | None = None,
        autoplay: bool = False,
    ) -> None:
        """在 Jupyter 中显示当前 chunk。

        Args:
            start_ms: chunk 内可选播放起始时间。
            end_ms: chunk 内可选播放结束时间。
            autoplay: 是否自动播放。

        Returns:
            None；播放器直接发送到当前 Jupyter 输出。

        Raises:
            ValueError: 结束时间早于起始时间。
            AsrDataError: IPython 不可用。

        Examples:
            >>> chunk.display(end_ms=50)
        """
    def timeline(self, channel: str | int) -> Timeline:
        """返回父 AudioStream 当前的全局 Timeline。

        Args:
            channel: 声道名称或索引。

        Returns:
            父 AudioStream 上对应声道的 Timeline。

        Examples:
            >>> timeline = chunk.timeline("mono")
        """

class AudioStream:
    """与 Audio 平级、随 chunk 迭代持续增长的流式音频文档。"""
    @staticmethod
    def from_path(
        path: str, chunk_size_ms: int = 100, *, id: str | None = None
    ) -> AudioStream:
        """从本地文件创建流。

        Args:
            path: 文件路径。
            chunk_size_ms: chunk 目标时长。
            id: 可选的文档 ID。

        Returns:
            新的 AudioStream。

        Examples:
            >>> stream = AudioStream.from_path("audio.wav")
        """
    @staticmethod
    def from_url(
        url: str, chunk_size_ms: int = 100, *, id: str | None = None
    ) -> AudioStream:
        """从 URL 创建流。

        Args:
            url: 音频 URL。
            chunk_size_ms: chunk 目标时长。
            id: 可选的文档 ID。

        Returns:
            新的 AudioStream。

        Examples:
            >>> stream = AudioStream.from_url("https://example.com/audio.wav")
        """
    @staticmethod
    def from_bytes(
        data: bytes, chunk_size_ms: int = 100, *, id: str | None = None
    ) -> AudioStream:
        """从编码音频字节创建流。

        Args:
            data: 编码音频字节。
            chunk_size_ms: chunk 目标时长。
            id: 可选的文档 ID。

        Returns:
            新的 AudioStream。

        Examples:
            >>> stream = AudioStream.from_bytes(encoded_audio)
        """
    @staticmethod
    def from_base64(
        data: str, chunk_size_ms: int = 100, *, id: str | None = None
    ) -> AudioStream:
        """从 base64 编码音频创建流。

        Args:
            data: base64 字符串。
            chunk_size_ms: chunk 目标时长。
            id: 可选的文档 ID。

        Returns:
            新的 AudioStream。

        Examples:
            >>> stream = AudioStream.from_base64(encoded)
        """
    @staticmethod
    def from_pcm(
        data: bytes,
        sample_rate: int,
        channels: int = 1,
        chunk_size_ms: int = 100,
        *,
        id: str | None = None,
    ) -> AudioStream:
        """从 PCM S16LE 字节创建流。

        Args:
            data: PCM S16LE 字节。
            sample_rate: 采样率。
            channels: 声道数。
            chunk_size_ms: chunk 目标时长。
            id: 可选的文档 ID。

        Returns:
            新的 AudioStream。

        Examples:
            >>> stream = AudioStream.from_pcm(b"\0\0", 16000)
        """
    @property
    def position_ms(self) -> int: ...
    @property
    def is_complete(self) -> bool: ...
    @property
    def is_closed(self) -> bool: ...
    @property
    def id(self) -> str: ...
    @property
    def source(self) -> AudioSource: ...
    @property
    def info(self) -> AudioInfo: ...
    @property
    def timelines(self) -> dict[str, Timeline]: ...
    @property
    def metadata(self) -> dict[str, Any]: ...
    def as_waveform(self) -> Waveform:
        """返回目前已经接收的全部波形。

        Returns:
            当前已接收范围对应的 Waveform。

        Examples:
            >>> waveform = stream.as_waveform()
        """
    def display(
        self,
        start_ms: int | None = None,
        end_ms: int | None = None,
        autoplay: bool = False,
    ) -> None:
        """在 Jupyter 中显示当前累计音频。

        Args:
            start_ms: 可选播放起始时间。
            end_ms: 可选播放结束时间。
            autoplay: 是否自动播放。

        Returns:
            None；播放器直接发送到当前 Jupyter 输出。

        Raises:
            ValueError: 结束时间早于起始时间。
            AsrDataError: IPython 不可用。

        Examples:
            >>> stream.display()
        """
    def timeline(self, channel: str | int) -> Timeline | None:
        """查询正在增长的全局 timeline。

        Args:
            channel: 声道名称或索引。

        Returns:
            对应 Timeline；不存在时为 None。

        Examples:
            >>> timeline = stream.timeline("mono")
        """
    def close(self) -> None:
        """提前关闭流。

        Returns:
            None。

        Examples:
            >>> stream.close()
        """
    def to_audio(self) -> Audio:
        """完整消费后转换为 Audio，不重新解码来源。

        Returns:
            完整 Audio。

        Raises:
            RuntimeError: 流尚未完整消费。

        Examples:
            >>> audio = stream.to_audio()
        """
    def _next_async(self) -> AudioChunk | None: ...
    def __iter__(self) -> AudioStream: ...
    def __next__(self) -> AudioChunk: ...
    def __enter__(self) -> AudioStream: ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...

class AudioActivity:
    """音频中的一个活动事件 payload。

    Args:
        event: 可选事件名称；省略时只表示存在活动。
        confidence: 可选活动检测置信度。

    Raises:
        ValueError: event 仅包含空白字符。

    Examples:
        >>> from asr_data.annotation import AudioActivity
        >>> AudioActivity(event="speech", confidence=0.98).event
        'speech'
    """
    def __init__(
        self,
        *,
        event: str | None = None,
        confidence: float | None = None,
    ) -> None: ...
    @property
    def event(self) -> str | None: ...
    @property
    def confidence(self) -> float | None: ...

class Token:
    """转写中的细粒度文本单元。

    Args:
        text: Token 文本。
        start_ms: 可选起始时间，单位为毫秒。
        end_ms: 可选结束时间，单位为毫秒。
        confidence: 可选置信度。

    Raises:
        ValueError: 时间参数未成对提供，或结束时间早于起始时间。

    Examples:
        >>> from asr_data.annotation import Token
        >>> Token("你好", start_ms=0, end_ms=300).text
        '你好'
    """
    def __init__(
        self,
        text: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        confidence: float | None = None,
    ) -> None: ...
    @property
    def text(self) -> str: ...
    @property
    def start_ms(self) -> int | None: ...
    @property
    def end_ms(self) -> int | None: ...
    @property
    def confidence(self) -> float | None: ...

class Transcription:
    """完整转写文本及其 token、语言和置信度。

    Args:
        text: 完整转写文本。
        tokens: 可选 Token 列表。
        language: 可选语言标签。
        confidence: 可选转写级置信度。

    Examples:
        >>> from asr_data.annotation import Transcription
        >>> Transcription("你好", language="zh").language
        'zh'
    """
    def __init__(
        self,
        text: str,
        *,
        tokens: list[Token] | None = None,
        language: str | None = None,
        confidence: float | None = None,
    ) -> None: ...
    @property
    def text(self) -> str: ...
    @property
    def tokens(self) -> list[Token]: ...
    @property
    def language(self) -> str | None: ...
    @property
    def confidence(self) -> float | None: ...

class Speaker:
    """一次说话人发话的 payload。

    Args:
        name: 说话人名称或业务标识。
        transcription: 该次发话携带的可选完整转写。
        confidence: 可选说话人识别置信度。

    Examples:
        >>> from asr_data.annotation import Speaker, Transcription
        >>> Speaker("agent", transcription=Transcription("你好")).name
        'agent'
    """
    def __init__(
        self,
        name: str,
        *,
        transcription: Transcription | None = None,
        confidence: float | None = None,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def transcription(self) -> Transcription | None: ...
    @property
    def confidence(self) -> float | None: ...

class TimeSpan:
    """Timeline 上一条带时间范围的标注记录。"""
    @property
    def id(self) -> str: ...
    @property
    def start_ms(self) -> int: ...
    @property
    def end_ms(self) -> int: ...
    @property
    def source(self) -> str | None: ...
    @property
    def annotation(
        self,
    ) -> AudioActivity | Token | Transcription | Speaker | dict[str, Any]: ...
    @annotation.setter
    def annotation(
        self, value: AudioActivity | Token | Transcription | Speaker
    ) -> None: ...
    def as_waveform(self) -> Waveform:
        """返回当前时间范围对应的波形。

        Returns:
            从父 Audio 截取出的 Waveform。

        Examples:
            >>> waveform = span.as_waveform()
        """
    def display(
        self,
        start_ms: int | None = None,
        end_ms: int | None = None,
        autoplay: bool = False,
    ) -> None:
        """在 Jupyter 中显示当前时间范围。

        Args:
            start_ms: TimeSpan 内可选播放起始时间。
            end_ms: TimeSpan 内可选播放结束时间。
            autoplay: 是否自动播放。

        Returns:
            None；播放器直接发送到当前 Jupyter 输出。

        Raises:
            ValueError: 结束时间早于起始时间。
            AsrDataError: IPython 不可用。

        Examples:
            >>> span.display()
        """

class Transcript:
    """按时间顺序组合得到的转写视图。"""
    @property
    def text(self) -> str: ...
    @property
    def language(self) -> str | None: ...

class TranscriptionEvaluation:
    """单个 source 的 timeline 转写评测结果。"""
    @property
    def source(self) -> str: ...
    @property
    def reference(self) -> str: ...
    @property
    def hypothesis(self) -> str: ...
    @property
    def normalized_reference(self) -> str: ...
    @property
    def normalized_hypothesis(self) -> str: ...
    @property
    def normalization(self) -> Literal["none", "zh_tn"]: ...
    @property
    def matches(self) -> int: ...
    @property
    def substitutions(self) -> int: ...
    @property
    def deletions(self) -> int: ...
    @property
    def insertions(self) -> int: ...
    @property
    def reference_chars(self) -> int: ...
    @property
    def hypothesis_chars(self) -> int: ...
    @property
    def cer(self) -> float: ...
    @property
    def precision(self) -> float: ...
    @property
    def recall(self) -> float: ...
    @property
    def f1(self) -> float: ...
    @property
    def exact_match(self) -> bool: ...

class ActivityEventEvaluation:
    """单个 event 的 timeline 区间评测结果。"""
    @property
    def event(self) -> str: ...
    @property
    def reference_ms(self) -> int: ...
    @property
    def predicted_ms(self) -> int: ...
    @property
    def true_positive_ms(self) -> int: ...
    @property
    def true_negative_ms(self) -> int: ...
    @property
    def false_positive_ms(self) -> int: ...
    @property
    def false_negative_ms(self) -> int: ...
    @property
    def precision(self) -> float: ...
    @property
    def recall(self) -> float: ...
    @property
    def f1(self) -> float: ...
    @property
    def iou(self) -> float: ...

class ActivityEvaluation:
    """单个 source 的 timeline Activity 评测结果。"""
    @property
    def source(self) -> str: ...
    @property
    def reference_ms(self) -> int: ...
    @property
    def predicted_ms(self) -> int: ...
    @property
    def true_positive_ms(self) -> int: ...
    @property
    def true_negative_ms(self) -> int: ...
    @property
    def false_positive_ms(self) -> int: ...
    @property
    def false_negative_ms(self) -> int: ...
    @property
    def precision(self) -> float: ...
    @property
    def recall(self) -> float: ...
    @property
    def f1(self) -> float: ...
    @property
    def iou(self) -> float: ...
    @property
    def events(self) -> dict[str, ActivityEventEvaluation]: ...

class TimelineEvaluation:
    """按任务和 prediction source 分组的 timeline 评测结果。"""
    @property
    def transcription(self) -> dict[str, TranscriptionEvaluation]: ...
    @property
    def activity(self) -> dict[str, ActivityEvaluation]: ...

class ReferenceSpans:
    """Timeline 的参考真值标注集合。"""
    @property
    def spans(self) -> list[TimeSpan]: ...
    def transcript(self) -> Transcript:
        """按时间顺序组合全部 reference 文本。

        Returns:
            组合后的 Transcript。

        Examples:
            >>> from asr_data import Audio, AudioSource
            >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
            >>> doc.timeline("mono").reference.transcript().text
            ''
        """
    def __len__(self) -> int: ...

class PredictionSpans:
    """Timeline 的模型 prediction 标注集合。"""
    @property
    def spans(self) -> list[TimeSpan]: ...
    @property
    def sources(self) -> dict[str, list[str]]: ...
    def by_source(self, source: str) -> list[TimeSpan]:
        """返回指定 source 的全部 prediction annotation。

        Args:
            source: 要查询的来源。

        Returns:
            保持存储顺序的 TimeSpan 列表。

        Examples:
            >>> from asr_data import Audio, AudioSource
            >>> timeline = Audio(
            ...     AudioSource.from_pcm(b"\0\0" * 10, 16000)
            ... ).timeline("mono")
            >>> timeline.prediction.by_source("asr")
            []
        """
    def transcript(self, source: str) -> Transcript:
        """按时间顺序组合指定 source 的预测文本。

        Args:
            source: 要组合的来源。

        Returns:
            组合后的 Transcript。

        Examples:
            >>> from asr_data import Audio, AudioSource
            >>> timeline = Audio(
            ...     AudioSource.from_pcm(b"\0\0" * 10, 16000)
            ... ).timeline("mono")
            >>> timeline.prediction.transcript("asr").text
            ''
        """
    def remove_by_source(self, source: str) -> int:
        """删除指定 source 的全部 prediction。

        Args:
            source: 要删除的来源。

        Returns:
            删除的 annotation 数量。

        Examples:
            >>> from asr_data import Audio, AudioSource
            >>> timeline = Audio(
            ...     AudioSource.from_pcm(b"\0\0" * 10, 16000)
            ... ).timeline("mono")
            >>> timeline.prediction.remove_by_source("asr")
            0
        """
    def relabel_source(self, from_source: str, to_source: str) -> int:
        """原子重命名 prediction source。

        Args:
            from_source: 原来源。
            to_source: 新来源。

        Returns:
            修改的 annotation 数量。

        Raises:
            ValueError: 新来源为空。
            AsrDataError: 重命名后会产生重叠冲突。

        Examples:
            >>> from asr_data import Audio, AudioSource
            >>> timeline = Audio(
            ...     AudioSource.from_pcm(b"\0\0" * 10, 16000)
            ... ).timeline("mono")
            >>> timeline.prediction.relabel_source("asr", "asr-v2")
            0
        """
    def __len__(self) -> int: ...

class Timeline:
    """一个声道上的参考真值和模型预测时间轴。"""
    @property
    def id(self) -> str: ...
    @property
    def audio_id(self) -> str: ...
    @audio_id.setter
    def audio_id(self, value: str) -> None: ...
    @property
    def duration_ms(self) -> int: ...
    @property
    def reference(self) -> ReferenceSpans: ...
    @property
    def prediction(self) -> PredictionSpans: ...
    def annotate_span(
        self,
        start_ms: int,
        end_ms: int,
        annotation: AudioActivity | Token | Transcription | Speaker,
        *,
        is_reference: bool,
        source: str | None = None,
    ) -> TimeSpan:
        """添加 reference 或 prediction 标注。

        Args:
            start_ms: 全局起始时间。
            end_ms: 全局结束时间。
            annotation: AudioActivity、Token、Transcription 或 Speaker。
            is_reference: 是否为参考答案。
            source: prediction 来源；reference 必须省略。

        Returns:
            新建或去重后已有的 TimeSpan。

        Raises:
            ValueError: 时间范围、is_reference 与 source 组合无效。
            AsrDataError: 标注与已有内容冲突。

        Examples:
            >>> span = timeline.annotate_span(
            ...     0, timeline.duration_ms, transcription, is_reference=True
            ... )
        """
    def as_waveform(self) -> Waveform:
        """返回当前声道的完整波形。

        Returns:
            当前 Timeline 对应的 Waveform。

        Examples:
            >>> waveform = timeline.as_waveform()
        """
    def display(
        self,
        start_ms: int | None = None,
        end_ms: int | None = None,
        autoplay: bool = False,
    ) -> None:
        """在 Jupyter 中显示当前声道。

        Args:
            start_ms: 可选播放起始时间。
            end_ms: 可选播放结束时间。
            autoplay: 是否自动播放。

        Returns:
            None；播放器直接发送到当前 Jupyter 输出。

        Raises:
            ValueError: 结束时间早于起始时间。
            AsrDataError: IPython 不可用。

        Examples:
            >>> timeline.display(end_ms=500)
        """
    def eval(
        self,
        *,
        transcription: str | list[str] | None = None,
        activity: str | list[str] | None = None,
        normalize: bool = True,
    ) -> TimelineEvaluation:
        """评测一个或多个 prediction source。

        Args:
            transcription: 转写来源或来源名称列表。
            activity: Activity 来源或来源名称列表。
            normalize: 是否在计算 CER 前执行中文文本标准化。

        Returns:
            按任务和 source 分组的 TimelineEvaluation。

        Raises:
            AsrDataError: reference 缺失、source 不存在或没有可评测内容。
            TypeError: source 参数不是字符串或字符串序列。
            ValueError: source 是空字符串。

        Examples:
            >>> from asr_data import Audio, AudioSource
            >>> from asr_data.annotation import Transcription
            >>> timeline = Audio(
            ...     AudioSource.from_pcm(b"\0\0" * 10, 16000)
            ... ).timeline("mono")
            >>> _ = timeline.annotate_span(
            ...     0, timeline.duration_ms, Transcription("你好"), is_reference=True
            ... )
            >>> _ = timeline.annotate_span(
            ...     0, timeline.duration_ms, Transcription("你好"),
            ...     is_reference=False, source="asr"
            ... )
            >>> timeline.eval().transcription["asr"].cer
            0.0
        """

class DatasetTranscriptionEvaluation:
    """单个 source 的数据集 corpus 转写评测结果。"""
    @property
    def source(self) -> str: ...
    @property
    def evaluated_documents(self) -> int: ...
    @property
    def evaluated_timelines(self) -> int: ...
    @property
    def unannotated_timelines(self) -> int: ...
    @property
    def missing_predictions(self) -> int: ...
    @property
    def unannotated_ids(self) -> list[str]: ...
    @property
    def missing_prediction_ids(self) -> list[str]: ...
    @property
    def normalization(self) -> Literal["none", "zh_tn"]: ...
    @property
    def substitutions(self) -> int: ...
    @property
    def deletions(self) -> int: ...
    @property
    def insertions(self) -> int: ...
    @property
    def reference_chars(self) -> int: ...
    @property
    def hypothesis_chars(self) -> int: ...
    @property
    def matches(self) -> int: ...
    @property
    def exact_matches(self) -> int: ...
    @property
    def cer(self) -> float: ...
    @property
    def precision(self) -> float: ...
    @property
    def recall(self) -> float: ...
    @property
    def f1(self) -> float: ...
    @property
    def exact_match_rate(self) -> float: ...
    @property
    def coverage(self) -> float: ...

class DatasetActivityEventEvaluation:
    """单个 event 的数据集区间聚合结果。"""
    @property
    def event(self) -> str: ...
    @property
    def evaluated_documents(self) -> int: ...
    @property
    def evaluated_timelines(self) -> int: ...
    @property
    def reference_ms(self) -> int: ...
    @property
    def predicted_ms(self) -> int: ...
    @property
    def true_positive_ms(self) -> int: ...
    @property
    def true_negative_ms(self) -> int: ...
    @property
    def false_positive_ms(self) -> int: ...
    @property
    def false_negative_ms(self) -> int: ...
    @property
    def precision(self) -> float: ...
    @property
    def recall(self) -> float: ...
    @property
    def f1(self) -> float: ...
    @property
    def iou(self) -> float: ...

class DatasetActivityEvaluation:
    """单个 source 的数据集 Activity 聚合结果。"""
    @property
    def source(self) -> str: ...
    @property
    def evaluated_documents(self) -> int: ...
    @property
    def evaluated_timelines(self) -> int: ...
    @property
    def unannotated_timelines(self) -> int: ...
    @property
    def missing_predictions(self) -> int: ...
    @property
    def unannotated_ids(self) -> list[str]: ...
    @property
    def missing_prediction_ids(self) -> list[str]: ...
    @property
    def reference_ms(self) -> int: ...
    @property
    def predicted_ms(self) -> int: ...
    @property
    def true_positive_ms(self) -> int: ...
    @property
    def true_negative_ms(self) -> int: ...
    @property
    def false_positive_ms(self) -> int: ...
    @property
    def false_negative_ms(self) -> int: ...
    @property
    def precision(self) -> float: ...
    @property
    def recall(self) -> float: ...
    @property
    def f1(self) -> float: ...
    @property
    def iou(self) -> float: ...
    @property
    def coverage(self) -> float: ...
    @property
    def events(self) -> dict[str, DatasetActivityEventEvaluation]: ...

class DatasetEvaluation:
    """按任务和 source 分组的数据集级评测结果。"""
    @property
    def documents(self) -> int: ...
    @property
    def timelines(self) -> int: ...
    @property
    def transcription(self) -> dict[str, DatasetTranscriptionEvaluation]: ...
    @property
    def activity(self) -> dict[str, DatasetActivityEvaluation]: ...

def evaluate_dataset(
    docs: list[Audio],
    *,
    transcription: str | list[str] | None = None,
    activity: str | list[str] | None = None,
    normalize: bool = True,
) -> DatasetEvaluation:
    """聚合内存中多个 Audio 的评测统计量。

    Args:
        docs: 要评测的 Audio 列表。
        transcription: 转写来源或来源列表；省略时自动发现。
        activity: Activity 来源或来源列表；省略时自动发现。
        normalize: 是否在计算 CER 前执行中文文本标准化。

    Returns:
        按任务和 source 分组的数据集级结果。

    Raises:
        AsrDataError: 没有可评测内容或显式 source 不存在。
        TypeError: source 参数类型无效。
        ValueError: source 为空字符串。

    Notes:
        每条 timeline 独立对齐后再累计统计量，不会跨文档拼接文本。

    Examples:
        >>> from asr_data import Audio, AudioSource, evaluate_dataset
        >>> from asr_data.annotation import Transcription
        >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
        >>> timeline = doc.timeline("mono")
        >>> _ = timeline.annotate_span(
        ...     0, timeline.duration_ms, Transcription("你好"), is_reference=True
        ... )
        >>> _ = timeline.annotate_span(
        ...     0, timeline.duration_ms, Transcription("你好"),
        ...     is_reference=False, source="asr"
        ... )
        >>> evaluate_dataset([doc]).transcription["asr"].cer
        0.0
    """

class Audio:
    """音频来源、元信息、时间轴、标注和 metadata 的集合。

    构造时会完整解码音频并按声道自动创建最终时长的 timeline。

    Args:
        source: AudioSource、路径或 URL。
        id: 可选的文档 ID；省略时自动生成。

    Raises:
        AsrDataError: 来源无法探测。

    Examples:
        >>> from asr_data import Audio, AudioSource
        >>> source = AudioSource.from_pcm(b"\0\0" * 16000, 16000)
        >>> Audio(source, id="sample-1").timeline("mono").duration_ms
        1000
    """
    def __init__(
        self,
        source: AudioSource,
        id: str | None = None,
    ) -> None: ...
    @staticmethod
    def from_path(path: str, *, id: str | None = None) -> Audio:
        """从本地文件加载音频。

        Args:
            path: 文件路径。
            id: 可选的文档 ID。

        Returns:
            完整 Audio。

        Examples:
            >>> audio = Audio.from_path("audio.wav")
        """
    @staticmethod
    def from_url(url: str, *, id: str | None = None) -> Audio:
        """从 URL 加载音频。

        Args:
            url: 音频 URL。
            id: 可选的文档 ID。

        Returns:
            完整 Audio。

        Examples:
            >>> audio = Audio.from_url("https://example.com/audio.wav")
        """
    @staticmethod
    def from_bytes(data: bytes, *, id: str | None = None) -> Audio:
        """从编码音频字节加载音频。

        Args:
            data: 编码音频字节。
            id: 可选的文档 ID。

        Returns:
            完整 Audio。

        Examples:
            >>> audio = Audio.from_bytes(encoded_audio)
        """
    @staticmethod
    def from_base64(data: str, *, id: str | None = None) -> Audio:
        """从 base64 编码音频加载音频。

        Args:
            data: base64 字符串。
            id: 可选的文档 ID。

        Returns:
            完整 Audio。

        Examples:
            >>> audio = Audio.from_base64(encoded)
        """
    @staticmethod
    def from_pcm(
        data: bytes,
        sample_rate: int,
        channels: int = 1,
        *,
        id: str | None = None,
    ) -> Audio:
        """从 PCM S16LE 字节加载音频。

        Args:
            data: PCM S16LE 字节。
            sample_rate: 采样率。
            channels: 声道数。
            id: 可选的文档 ID。

        Returns:
            完整 Audio。

        Examples:
            >>> audio = Audio.from_pcm(b"\0\0", 16000)
        """
    @property
    def id(self) -> str: ...
    @property
    def source(self) -> AudioSource: ...
    @property
    def info(self) -> AudioInfo: ...
    def as_waveform(self) -> Waveform:
        """返回完整波形。

        Returns:
            完整的 Waveform。

        Examples:
            >>> waveform = audio.as_waveform()
        """
    def display(
        self,
        start_ms: int | None = None,
        end_ms: int | None = None,
        autoplay: bool = False,
    ) -> None:
        """在 Jupyter 中显示完整音频。

        Args:
            start_ms: 可选播放起始时间。
            end_ms: 可选播放结束时间。
            autoplay: 是否自动播放。

        Returns:
            None；播放器直接发送到当前 Jupyter 输出。

        Raises:
            ValueError: 结束时间早于起始时间。
            AsrDataError: IPython 不可用。

        Examples:
            >>> audio.display(end_ms=500)
        """
    def timeline(self, channel: str | int) -> Timeline | None:
        """查询指定声道的 timeline。

        Args:
            channel: 声道名称或索引。

        Returns:
            对应 Timeline；不存在时为 None。

        Raises:
            ValueError: 声道名称或索引无效。

        Examples:
            >>> from asr_data import Audio, AudioSource
            >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
            >>> doc.timeline("mono").duration_ms
            1
        """
    def ensure_timeline(
        self, channel: str | int, duration_ms: int | float | None = None
    ) -> Timeline:
        """取得或创建指定声道的 timeline。

        Args:
            channel: 声道名称或索引。
            duration_ms: 可选时长；必须与文档音频时长一致。

        Returns:
            已有或新建的 Timeline。

        Raises:
            ValueError: 时长无效或与文档不一致。

        Examples:
            >>> from asr_data import Audio, AudioSource
            >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
            >>> doc.ensure_timeline("mono") is not None
            True
        """
    def remove_timeline(self, channel: str | int) -> bool:
        """删除指定声道的 timeline。

        Args:
            channel: 声道名称或索引。

        Returns:
            确实删除时为 True。

        Raises:
            ValueError: 声道无效。

        Examples:
            >>> from asr_data import Audio, AudioSource
            >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
            >>> doc.remove_timeline("mono")
            True
        """
    @property
    def timelines(self) -> dict[str, Timeline]: ...
    @property
    def metadata(self) -> dict[str, Any]: ...
    def validate(self) -> None:
        """校验文档、timeline、annotation 和 source 约束。

        Returns:
            None。

        Raises:
            AsrDataError: 文档包含无效数据。

        Examples:
            >>> from asr_data import Audio, AudioSource
            >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
            >>> doc.validate() is None
            True
        """

class AudioDB:
    """持久化 Audio 的 SQLite 数据库。"""
    @staticmethod
    def create(path: str) -> AudioDB:
        """创建新数据库。

        Args:
            path: 新数据库文件路径。

        Returns:
            可读写的 AudioDB。

        Raises:
            FileExistsError: 目标路径已经存在。

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> from asr_data import AudioDB
            >>> with TemporaryDirectory() as directory:
            ...     db = AudioDB.create(f"{directory}/dataset.db")
        """
    @staticmethod
    def open(path: str, read_only: bool = False) -> AudioDB:
        """打开并校验已有数据库。

        Args:
            path: 已有数据库文件路径。
            read_only: 是否以只读模式打开。

        Returns:
            已打开的 AudioDB。

        Raises:
            FileNotFoundError: 数据库不存在。
            AsrDataError: 文件不是受支持的 asr-data 数据库。

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> from asr_data import AudioDB
            >>> with TemporaryDirectory() as directory:
            ...     path = f"{directory}/dataset.db"
            ...     _ = AudioDB.create(path)
            ...     db = AudioDB.open(path)
        """
    def insert(self, audio: Audio) -> None:
        """插入一条新 Audio。

        Args:
            audio: 要插入的完整 Audio。

        Returns:
            None。

        Raises:
            AsrDataError: ID 已存在或文档校验失败。

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> from asr_data import AudioDB, Audio, AudioSource
            >>> directory = TemporaryDirectory()
            >>> db = AudioDB.create(f"{directory.name}/dataset.db")
            >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
            >>> db.insert(doc)
        """
    def query(
        self,
        limit: int = 100,
        *,
        after: str | None = None,
        min_duration_ms: int | None = None,
        max_duration_ms: int | None = None,
        created_from: datetime | None = None,
        created_until: datetime | None = None,
        updated_from: datetime | None = None,
        updated_until: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Audio]:
        """按游标、时长、时间和 metadata 查询文档。

        Args:
            limit: 最大返回数量。
            after: 上一页最后一个 Audio ID。
            min_duration_ms: 可选最短时长。
            max_duration_ms: 可选最长时长。
            created_from: 带时区的创建时间下界。
            created_until: 带时区的创建时间上界，不包含。
            updated_from: 带时区的修改时间下界。
            updated_until: 带时区的修改时间上界，不包含。
            metadata: 要精确匹配的 JSON metadata。

        Returns:
            按 Audio ID 排序的文档列表。

        Raises:
            ValueError: 范围反向、datetime 无时区或 limit 无效。

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> from asr_data import AudioDB, Audio, AudioSource
            >>> directory = TemporaryDirectory()
            >>> db = AudioDB.create(f"{directory.name}/dataset.db")
            >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
            >>> doc.metadata["split"] = "test"
            >>> db.insert(doc)
            >>> page = db.query(limit=10, metadata={"split": "test"})
        """
    def eval(
        self,
        *,
        transcription: str | list[str] | None = None,
        activity: str | list[str] | None = None,
        normalize: bool = True,
        batch_size: int = 100,
        after: str | None = None,
        min_duration_ms: int | None = None,
        max_duration_ms: int | None = None,
        created_from: datetime | None = None,
        created_until: datetime | None = None,
        updated_from: datetime | None = None,
        updated_until: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetEvaluation:
        """自动分页评测全部匹配文档。

        Args:
            transcription: 转写来源或来源列表。
            activity: Activity 来源或来源列表。
            normalize: 是否执行中文文本标准化。
            batch_size: 每批读取的文档数。
            after: 可选起始 Audio ID 游标。
            min_duration_ms: 可选最短时长。
            max_duration_ms: 可选最长时长。
            created_from: 创建时间下界。
            created_until: 创建时间上界，不包含。
            updated_from: 修改时间下界。
            updated_until: 修改时间上界，不包含。
            metadata: 要精确匹配的 JSON metadata。

        Returns:
            按任务和 source 分组的数据集级评测结果。

        Raises:
            ValueError: batch_size 为零或筛选范围无效。
            AsrDataError: 没有可评测内容或显式 source 不存在。

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> from asr_data import AudioDB, Audio, AudioSource
            >>> from asr_data.annotation import Transcription
            >>> directory = TemporaryDirectory()
            >>> db = AudioDB.create(f"{directory.name}/dataset.db")
            >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
            >>> timeline = doc.timeline("mono")
            >>> _ = timeline.annotate_span(
            ...     0, timeline.duration_ms, Transcription("你好"), is_reference=True
            ... )
            >>> _ = timeline.annotate_span(
            ...     0, timeline.duration_ms, Transcription("你好"),
            ...     is_reference=False, source="asr"
            ... )
            >>> db.insert(doc)
            >>> result = db.eval(transcription="asr")
        """
    def update(self, audio: Audio) -> bool:
        """更新已有 Audio。

        Args:
            audio: 包含新内容的完整 Audio。

        Returns:
            实际发生更新时为 True，否则为 False。

        Raises:
            KeyError: 文档 ID 不存在。

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> from asr_data import AudioDB, Audio, AudioSource
            >>> directory = TemporaryDirectory()
            >>> db = AudioDB.create(f"{directory.name}/dataset.db")
            >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
            >>> db.insert(doc)
            >>> doc.metadata["checked"] = True
            >>> changed = db.update(doc)
        """
    def update_many(self, audios: list[Audio]) -> int:
        """在单个事务中批量更新文档。

        Args:
            audios: 要更新的完整 Audio 列表。

        Returns:
            实际发生变化的文档数量。

        Raises:
            KeyError: 任一文档 ID 不存在。

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> from asr_data import AudioDB
            >>> directory = TemporaryDirectory()
            >>> db = AudioDB.create(f"{directory.name}/dataset.db")
            >>> changed = db.update_many([])
            >>> changed
            0
        """
    def delete(self, audio_id: str) -> bool:
        """删除指定 ID 的文档。

        Args:
            audio_id: 文档 ID。

        Returns:
            文档存在并被删除时为 True。

        Raises:
            AsrDataError: 数据库写入失败。

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> from asr_data import AudioDB
            >>> directory = TemporaryDirectory()
            >>> db = AudioDB.create(f"{directory.name}/dataset.db")
            >>> db.delete("missing")
            False
        """
    @property
    def metadata(self) -> dict[str, Any]: ...
    def set_metadata(self, key: str, value: Any) -> None:
        """设置数据库级 JSON metadata。

        Args:
            key: metadata 键。
            value: 可序列化为 JSON 的值。

        Returns:
            None。

        Raises:
            TypeError: value 不能序列化为 JSON。

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> from asr_data import AudioDB
            >>> directory = TemporaryDirectory()
            >>> db = AudioDB.create(f"{directory.name}/dataset.db")
            >>> db.set_metadata("version", "2026-07")
        """
    def metadata_value(self, key: str) -> Any | None:
        """读取一个数据库级 metadata 值。

        Args:
            key: metadata 键。

        Returns:
            解码后的值；不存在时为 None。

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> from asr_data import AudioDB
            >>> directory = TemporaryDirectory()
            >>> db = AudioDB.create(f"{directory.name}/dataset.db")
            >>> db.metadata_value("missing") is None
            True
        """
    def delete_metadata(self, key: str) -> bool:
        """删除数据库级 metadata。

        Args:
            key: metadata 键。

        Returns:
            键存在并被删除时为 True。

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> from asr_data import AudioDB
            >>> directory = TemporaryDirectory()
            >>> db = AudioDB.create(f"{directory.name}/dataset.db")
            >>> db.delete_metadata("missing")
            False
        """
    def __getitem__(self, audio_id: str) -> Audio: ...
    def __contains__(self, audio_id: str) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Audio]: ...
