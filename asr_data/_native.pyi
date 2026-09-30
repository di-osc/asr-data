from collections.abc import Iterator
from datetime import datetime
from os import PathLike
from typing import Any, Awaitable, Literal

import numpy as np
import numpy.typing as npt

class AsrDataError(Exception): ...

def normalize_zh(
    text: str,
    *,
    traditional_to_simple: bool = True,
    full_to_half: bool = True,
    remove_erhua: bool = True,
    remove_interjections: bool = True,
    remove_puncts: bool = True,
) -> str:
    """使用内嵌中文 TN 资源把书写形式转换为口语形式。

    Args:
        text: 要标准化的原始文本。
        traditional_to_simple: 是否将繁体中文转换为简体中文。
        full_to_half: 是否将全角字符转换为半角字符。
        remove_erhua: 是否去除儿化音“儿”。
        remove_interjections: 是否去除“嗯”“啊”“呃”等语气词。
        remove_puncts: 是否去除标点符号。

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
    def from_path(path: str | PathLike[str]) -> AudioSource:
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
    @staticmethod
    def from_modelscope(
        repo_id: str, file_path: str, *, revision: str | None = None
    ) -> AudioSource:
        """从 ModelScope 数据集中的单个音频文件创建来源，不立即下载。

        Args:
            repo_id: ModelScope 数据集仓库 ID。
            file_path: 仓库内相对路径。
            revision: 可选仓库 revision，默认 master。

        Returns:
            尚未下载的 AudioSource。

        Raises:
            ValueError: repo_id、file_path 或 revision 为空。

        Examples:
            >>> from asr_data import AudioSource
            >>> AudioSource.from_modelscope("org/name", "wav/a.wav").kind
            'modelscope'
        """
    @property
    def kind(
        self,
    ) -> Literal["path", "url", "bytes", "base64", "pcm", "modelscope"]: ...
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
    @property
    def repo_id(self) -> str | None: ...
    @property
    def file_path(self) -> str | None: ...
    @property
    def revision(self) -> str | None: ...
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
    def stream(self, chunk_size_ms: int = 100, *, id: str | None = None) -> AudioStream:
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
    def from_path(path: str | PathLike[str]) -> Waveform:
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
    def from_modelscope(
        repo_id: str, file_path: str, *, revision: str | None = None
    ) -> Waveform:
        """下载 ModelScope 数据集中的单个音频文件并解码。

        Args:
            repo_id: ModelScope 数据集仓库 ID。
            file_path: 仓库内相对路径。
            revision: 可选仓库 revision，默认 master。

        Returns:
            解码后的完整 Waveform。

        Raises:
            ValueError: repo_id、file_path 或 revision 为空。
            AsrDataError: 下载失败或音频无法解码。

        Examples:
            >>> from asr_data import Waveform
            >>> audio = Waveform.from_modelscope("org/name", "wav/a.wav")
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
    def _start_aload_from_path(path: str | PathLike[str]) -> _AudioLoadTask: ...
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
    def peak_normalize(self) -> Waveform:
        """按峰值缩放到 ``[-1, 1]``。

        非有限值先置 0。若剩余峰值大于 1，整段除以该峰值后再钳位。
        峰值不超过 1 时波形不变。

        Returns:
            不修改原对象的新 Waveform。

        Examples:
            >>> import numpy as np
            >>> from asr_data import Waveform
            >>> Waveform(np.array([0.0, 2.0, -2.0], dtype=np.float32), 16000).peak_normalize().samples.tolist()
            [0.0, 1.0, -1.0]
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
    def metadata(self) -> dict[str, Any]: ...
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

class StreamingResampler:
    """跨块复用的有状态 PCM 重采样器。

    Args:
        from_hz: 输入采样率。
        to_hz: 输出采样率。
        channels: 交错 PCM 的声道数，默认为 1。

    Raises:
        AsrDataError: 无法按给定参数创建重采样器。

    Examples:
        >>> from asr_data import StreamingResampler
        >>> resampler = StreamingResampler(8000, 16000)
        >>> len(resampler.process([0.0] * 8000, is_final=True))
        16000
    """
    def __init__(self, from_hz: int, to_hz: int, channels: int = 1) -> None: ...
    def process(self, samples: npt.ArrayLike, *, is_final: bool = False) -> list[float]:
        """处理一块交错 PCM，返回本次可以交出的输出样本。

        Args:
            samples: 一维 float32 兼容数组；多声道按帧交错。
            is_final: 最后一块时为真，冲刷滤波器尾巴。

        Returns:
            本次输出的 float32 样本；可能为空（还在凑输入窗）。

        Raises:
            ValueError: 样本不是一维 C 连续 float32，或不能整除声道数。
            AsrDataError: 单次重采样失败。

        Examples:
            >>> from asr_data import StreamingResampler
            >>> resampler = StreamingResampler(8000, 16000)
            >>> first = resampler.process([0.0] * 4000, is_final=False)
            >>> rest = resampler.process([0.0] * 4000, is_final=True)
            >>> len(first) + len(rest)
            16000
        """

class AudioStream:
    """与 Audio 平级、随 chunk 迭代持续增长的流式音频文档。"""
    @staticmethod
    def from_path(
        path: str | PathLike[str], chunk_size_ms: int = 100, *, id: str | None = None
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
    def from_modelscope(
        repo_id: str,
        file_path: str,
        chunk_size_ms: int = 100,
        *,
        revision: str | None = None,
        id: str | None = None,
    ) -> AudioStream:
        """下载 ModelScope 数据集中的单个音频文件并创建流。

        Args:
            repo_id: ModelScope 数据集仓库 ID。
            file_path: 仓库内相对路径。
            chunk_size_ms: chunk 目标时长。
            revision: 可选仓库 revision，默认 master。
            id: 可选的文档 ID。

        Returns:
            新的 AudioStream。

        Raises:
            ValueError: repo_id、file_path、revision 为空，或 chunk_size_ms 为零。
            AsrDataError: 文件无法下载或探测。

        Examples:
            >>> stream = AudioStream.from_modelscope("org/name", "wav/a.wav")
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

class Sentence:
    """一句转写。时间是时间轴上的绝对毫秒。

    Args:
        text: 整句文本。可以为空，表示这段假设被整段删除。
        start_ms: 起始时间，单位为毫秒。
        end_ms: 结束时间，单位为毫秒。
        tokens: 可选 Token 列表。

    Raises:
        ValueError: 结束时间不晚于起始时间。

    Examples:
        >>> from asr_data.annotation import Sentence
        >>> Sentence("你好", 0, 300).text
        '你好'
    """
    def __init__(
        self,
        text: str,
        start_ms: int,
        end_ms: int,
        *,
        tokens: list[Token] | None = None,
    ) -> None: ...
    @property
    def text(self) -> str: ...
    @property
    def start_ms(self) -> int: ...
    @property
    def end_ms(self) -> int: ...
    @property
    def tokens(self) -> list[Token]: ...

class Transcription:
    """一段语音里的转写。至少要有全文，句子和 token 都可以空着。

    Args:
        text: 整段转写文本。
        sentences: 可选句子列表。省略或空列表都合法。
        confidence: 可选整段置信度。

    Examples:
        >>> from asr_data.annotation import Sentence, Transcription
        >>> Transcription("你好").text
        '你好'
        >>> Transcription("你好", sentences=[Sentence("你好", 0, 300)]).sentences[0].tokens
        []
    """
    def __init__(
        self,
        text: str,
        *,
        sentences: list[Sentence] | None = None,
        confidence: float | None = None,
    ) -> None: ...
    @property
    def text(self) -> str: ...
    @property
    def sentences(self) -> list[Sentence]: ...
    @property
    def confidence(self) -> float | None: ...

class Speaker:
    """一段语音上的说话人。一段语音最多一个说话人。

    Args:
        name: 说话人名称或业务标识。
        gender: 可选性别，取 ``male``、``female`` 或 ``unknown``。省略表示还没标。

    Raises:
        ValueError: 名字是空白，或性别不是这三个值。

    Examples:
        >>> from asr_data.annotation import Speaker
        >>> Speaker("agent", gender="female").gender
        'female'
    """
    def __init__(self, name: str, *, gender: str | None = None) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def gender(self) -> str | None: ...

class AudioEvent:
    """音频事件基类。音乐、噪声、静音以及其他非语音事件都用它。

    ``Speech`` 从本类派生。直接构造时事件名不能是 ``speech``。

    Args:
        name: 事件名，例如 ``music``、``noise``、``silence``。
        confidence: 可选检测置信度。

    Raises:
        ValueError: 事件名为空白，或者是 ``speech``。

    Examples:
        >>> from asr_data.annotation import AudioEvent
        >>> AudioEvent("music").name
        'music'
    """
    def __init__(self, name: str, *, confidence: float | None = None) -> None: ...
    @property
    def id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def start_ms(self) -> int: ...
    @property
    def end_ms(self) -> int: ...
    @property
    def source(self) -> str | None: ...
    @property
    def confidence(self) -> float | None: ...

class Speech(AudioEvent):
    """从 AudioEvent 派生的语音事件。语种、说话人和转写都可以缺。

    Args:
        language: 可选 BCP-47 语种标签。
        speaker: 可选说话人。一段语音只有一个。
        transcription: 可选转写。
        confidence: 可选活动检测置信度。

    Examples:
        >>> from asr_data.annotation import Speech
        >>> isinstance(Speech(), AudioEvent)
        True
    """
    def __init__(
        self,
        *,
        language: str | None = None,
        speaker: Speaker | None = None,
        transcription: Transcription | None = None,
        confidence: float | None = None,
    ) -> None: ...
    @property
    def language(self) -> str | None: ...
    @property
    def speaker(self) -> Speaker | None: ...
    @property
    def transcription(self) -> Transcription | None: ...

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
    def reference(self) -> list[AudioEvent]:
        """参考事件。语音是 Speech，其余是 AudioEvent。"""
    @property
    def prediction(self) -> list[AudioEvent]:
        """预测事件。每条都带非空 source。"""
    @property
    def reference_transcript(self) -> Transcript:
        """从参考语音事件拼出的转写。"""
    def annotate(
        self,
        start_ms: int,
        end_ms: int,
        event: AudioEvent,
        *,
        is_reference: bool = True,
        source: str | None = None,
    ) -> AudioEvent:
        """写入一条 AudioEvent 或派生的 Speech。

        Args:
            start_ms: 起始时间，单位为毫秒。
            end_ms: 结束时间，单位为毫秒。
            event: 基类事件或 Speech。
            is_reference: ``True`` 表示参考，``False`` 表示预测。默认为 ``True``。
            source: 预测来源；reference 必须省略。

        Returns:
            写入后的事件。语音返回 Speech，其他事件返回 AudioEvent。

        Raises:
            ValueError: 时间范围无效、reference 携带 source，或 prediction 缺少 source。
            AsrDataError: 事件与已有内容冲突。

        Examples:
            >>> from asr_data import AudioSource
            >>> from asr_data.annotation import Speech
            >>> timeline = AudioSource.from_pcm(b"\0\0" * 10, 1000).load().timeline("mono")
            >>> timeline.annotate(0, timeline.duration_ms, Speech()).name
            'speech'
        """
    def remove(self, event_id: str) -> bool:
        """按 ID 删除一条事件。

        Args:
            event_id: 要删除的事件 ID。

        Returns:
            找到并删除时为 ``True``。

        Examples:
            >>> from asr_data import AudioSource
            >>> from asr_data.annotation import Speech
            >>> timeline = AudioSource.from_pcm(b"\0\0" * 10, 1000).load().timeline("mono")
            >>> event = timeline.annotate(0, timeline.duration_ms, Speech())
            >>> timeline.remove(event.id)
            True
        """
    def prediction_transcript(self, source: str) -> Transcript:
        """某个预测来源的转写。

        Args:
            source: 预测来源。

        Returns:
            按句子时间拼出的 Transcript。没有文本时文本为空。

        Examples:
            >>> from asr_data import AudioSource
            >>> timeline = AudioSource.from_pcm(b"\0\0" * 10, 1000).load().timeline("mono")
            >>> timeline.prediction_transcript("asr").text
            ''
        """
    def remove_predictions(self, source: str) -> int:
        """删除某个预测来源的全部事件。

        Args:
            source: 预测来源。

        Returns:
            删除的事件条数。

        Examples:
            >>> from asr_data import AudioSource
            >>> timeline = AudioSource.from_pcm(b"\0\0" * 10, 1000).load().timeline("mono")
            >>> timeline.remove_predictions("asr")
            0
        """
    def relabel_prediction_source(self, from_source: str, to_source: str) -> int:
        """把预测来源从 ``from_source`` 改成 ``to_source``。

        Args:
            from_source: 原来源。
            to_source: 新来源。

        Returns:
            修改的事件条数。

        Raises:
            ValueError: 来源为空。
            AsrDataError: 改名后同一来源里的事件会重叠。

        Examples:
            >>> from asr_data import AudioSource
            >>> timeline = AudioSource.from_pcm(b"\0\0" * 10, 1000).load().timeline("mono")
            >>> timeline.relabel_prediction_source("asr", "asr-v2")
            0
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
        traditional_to_simple: bool = True,
        full_to_half: bool = True,
        remove_erhua: bool = True,
        remove_interjections: bool = True,
        remove_puncts: bool = True,
    ) -> TimelineEvaluation:
        """评测一个或多个 prediction source。

        Args:
            transcription: 转写来源或来源名称列表。
            activity: Activity 来源或来源名称列表。
            normalize: 是否在计算 CER 前执行中文文本标准化。
            traditional_to_simple: 是否将繁体中文转换为简体中文。
            full_to_half: 是否将全角字符转换为半角字符。
            remove_erhua: 是否去除儿化音“儿”。
            remove_interjections: 是否去除“嗯”“啊”“呃”等语气词。
            remove_puncts: 是否去除标点符号。

        Returns:
            按任务和 source 分组的 TimelineEvaluation。

        Raises:
            AsrDataError: reference 缺失、source 不存在或没有可评测内容。
            TypeError: source 参数不是字符串或字符串序列。
            ValueError: source 是空字符串。

        Examples:
            >>> from asr_data import Audio, AudioSource
            >>> from asr_data.annotation import Speech, Transcription
            >>> timeline = Audio(
            ...     AudioSource.from_pcm(b"\0\0" * 10, 16000)
            ... ).timeline("mono")
            >>> end = timeline.duration_ms
            >>> speech = Speech(transcription=Transcription("你好"))
            >>> _ = timeline.annotate(0, end, speech)
            >>> _ = timeline.annotate(0, end, speech, is_reference=False, source="qwen-asr")
            >>> result = timeline.eval(transcription="qwen-asr")
            >>> result.transcription["qwen-asr"].cer
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

class DatasetSpeakerEvaluation:
    """单个 source 的标签无关说话人分离聚合结果。"""
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
    def reference_speaker_ms(self) -> int: ...
    @property
    def predicted_speaker_ms(self) -> int: ...
    @property
    def correct_speaker_ms(self) -> int: ...
    @property
    def missed_speaker_ms(self) -> int: ...
    @property
    def false_alarm_ms(self) -> int: ...
    @property
    def speaker_confusion_ms(self) -> int: ...
    @property
    def der(self) -> float: ...
    @property
    def coverage(self) -> float: ...

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
    traditional_to_simple: bool = True,
    full_to_half: bool = True,
    remove_erhua: bool = True,
    remove_interjections: bool = True,
    remove_puncts: bool = True,
) -> DatasetEvaluation:
    """聚合内存中多个 Audio 的评测统计量。

    Args:
        docs: 要评测的 Audio 列表。
        transcription: 转写来源或来源列表；省略时自动发现。
        activity: Activity 来源或来源列表；省略时自动发现。
        normalize: 是否在计算 CER 前执行中文文本标准化。
        traditional_to_simple: 是否将繁体中文转换为简体中文。
        full_to_half: 是否将全角字符转换为半角字符。
        remove_erhua: 是否去除儿化音“儿”。
        remove_interjections: 是否去除“嗯”“啊”“呃”等语气词。
        remove_puncts: 是否去除标点符号。

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
        >>> from asr_data.annotation import Speech, Transcription
        >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
        >>> timeline = doc.timeline("mono")
        >>> end = timeline.duration_ms
        >>> speech = Speech(transcription=Transcription("你好"))
        >>> _ = timeline.annotate(0, end, speech)
        >>> _ = timeline.annotate(0, end, speech, is_reference=False, source="asr")
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
    def from_path(path: str | PathLike[str], *, id: str | None = None) -> Audio:
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
    def from_modelscope(
        repo_id: str,
        file_path: str,
        *,
        revision: str | None = None,
        id: str | None = None,
    ) -> Audio:
        """下载 ModelScope 数据集中的单个音频文件并完整加载。

        Args:
            repo_id: ModelScope 数据集仓库 ID。
            file_path: 仓库内相对路径。
            revision: 可选仓库 revision，默认 master。
            id: 可选的文档 ID。

        Returns:
            完整 Audio。

        Raises:
            ValueError: repo_id、file_path 或 revision 为空。
            AsrDataError: 下载失败或音频无法解码。

        Examples:
            >>> audio = Audio.from_modelscope("org/name", "wav/a.wav", id="sample")
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
    def timeline(self, channel: str | int = "mono") -> Timeline | None:
        """查询指定声道的 timeline。

        Args:
            channel: 声道名称或索引，默认为 ``"mono"``。

        Returns:
            对应 Timeline；不存在时为 None。

        Raises:
            ValueError: 声道名称或索引无效。

        Examples:
            >>> from asr_data import Audio, AudioSource
            >>> doc = Audio(AudioSource.from_pcm(b"\0\0" * 10, 16000))
            >>> doc.timeline().duration_ms
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

class AudioDataset:
    """具名、带版本，并由可选 train、val、test AudioDB 支撑的数据集。"""
    def __init__(
        self,
        train: AudioDB | None = None,
        val: AudioDB | None = None,
        test: AudioDB | None = None,
    ) -> None: ...
    @staticmethod
    def from_modelscope(
        repo_id: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> AudioDataset:
        """通过 modelhub 下载完整 ModelScope 数据集仓库。

        name 使用 repo_id，version 使用实际 revision，license 从同一
        revision 的 README.md front matter 读取。仓库中不存在的切分数据库
        返回 None，存在的数据库只读打开。

        Args:
            repo_id: ModelScope 数据集仓库 ID。
            revision: 可选仓库 revision，默认 master。
            cache_dir: 可选 ModelScope 缓存根目录。

        Returns:
            train、val、test 为只读 AudioDB 或 None 的 AudioDataset。

        Raises:
            ValueError: repo_id、revision 或 README.md license 无效。
            AsrDataError: 整仓下载失败，或已有切分数据库不是受支持的 AudioDB。

        Examples:
            >>> from asr_data import AudioDataset
            >>> dataset = AudioDataset.from_modelscope("di-osc/aishell-1")
        """
    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def license(self) -> str: ...
    @property
    def train(self) -> AudioDB | None: ...
    @property
    def val(self) -> AudioDB | None: ...
    @property
    def test(self) -> AudioDB | None: ...

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
    @staticmethod
    def from_modelscope(
        repo_id: str,
        file_path: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> AudioDB:
        """下载 ModelScope 数据集中的一个数据库文件，并以只读方式打开。

        Args:
            repo_id: ModelScope 数据集仓库 ID。
            file_path: 仓库内数据库路径，例如 ``train.db``。
            revision: 可选仓库 revision，默认 master。
            cache_dir: 可选 modelhub 缓存根目录。

        Returns:
            只读打开的 AudioDB。``path`` 是下载后的本地文件。

        Raises:
            ValueError: repo_id、file_path 或 revision 为空。
            AsrDataError: 下载失败，或文件不是受支持的 AudioDB。

        Examples:
            >>> from asr_data import AudioDB
            >>> db = AudioDB.from_modelscope("di-osc/aishell-1", "train.db")
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
    def eval_transcription(
        self,
        source: str | list[str] | None = None,
        *,
        normalize: bool = True,
        traditional_to_simple: bool = True,
        full_to_half: bool = True,
        remove_erhua: bool = True,
        remove_interjections: bool = True,
        remove_puncts: bool = True,
        batch_size: int = 100,
        after: str | None = None,
        min_duration_ms: int | None = None,
        max_duration_ms: int | None = None,
        created_from: datetime | None = None,
        created_until: datetime | None = None,
        updated_from: datetime | None = None,
        updated_until: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, DatasetTranscriptionEvaluation]:
        """评测转写，并按 prediction source 返回结果。

        ``normalize=False`` 时忽略其他文本标准化选项。

        Args:
            source: 转写 prediction 来源或来源列表；省略时自动发现。
            normalize: 是否执行中文 TN 和 CER 清洗。
            traditional_to_simple: 是否将繁体中文转换为简体中文。
            full_to_half: 是否将全角字符转换为半角字符。
            remove_erhua: 是否去除儿化音“儿”。
            remove_interjections: 是否去除“嗯”“啊”“呃”等语气词。
            remove_puncts: 是否去除标点符号。

        Returns:
            按 prediction source 分组的转写评测结果。

        Examples:
            在已填充数据库上调用 ``db.eval_transcription("qwen-asr")``。
        """
    def eval_activity(
        self,
        source: str | list[str] | None = None,
        *,
        batch_size: int = 100,
        after: str | None = None,
        min_duration_ms: int | None = None,
        max_duration_ms: int | None = None,
        created_from: datetime | None = None,
        created_until: datetime | None = None,
        updated_from: datetime | None = None,
        updated_until: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, DatasetActivityEvaluation]:
        """评测语音和其他音频事件，并按 prediction source 返回结果。

        Args:
            source: Activity prediction 来源或来源列表；省略时自动发现。

        Returns:
            按 prediction source 分组的 Activity 评测结果。

        Examples:
            在已填充数据库上调用 ``db.eval_activity("silero-vad")``。
        """
    def eval_speaker(
        self,
        source: str | list[str] | None = None,
        *,
        batch_size: int = 100,
        after: str | None = None,
        min_duration_ms: int | None = None,
        max_duration_ms: int | None = None,
        created_from: datetime | None = None,
        created_until: datetime | None = None,
        updated_from: datetime | None = None,
        updated_until: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, DatasetSpeakerEvaluation]:
        """使用标签无关的最佳映射评测 Speaker DER。

        Args:
            source: Speaker prediction 来源或来源列表；省略时自动发现。

        Returns:
            按 prediction source 分组的说话人评测结果。

        Examples:
            在已填充数据库上调用 ``db.eval_speaker("diarizer")``。
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
