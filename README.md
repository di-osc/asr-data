<p align="center">
  <img src="assets/logo.png" alt="asr-data logo" width="160" />
</p>

<h1 align="center">asr-data</h1>

`asr-data` 是一个面向 ASR（Automatic Speech Recognition，自动语音识别）
数据管理的 Rust / Python 库。它把音频、转写、说话人、语言、模型预测等信息组织成统一的数据模型，并使用 SQLite `.sqlite` 文件持久化，方便构建语音数据集、标注流水线和模型评测工具。

## 特点

- **统一的数据模型**：用 `AudioDoc` 表示一条音频记录，用 `Timeline` 管理语音段、转写、说话人、语言、热词、诊断信息等时间轴标注；音频来源由 `AudioPath`、`AudioUrl`、`AudioBytes`、`AudioBase64`、`AudioPcm` 等类型描述。
- **SQLite 本地存储**：`AudioDB` 将数据保存为 `.sqlite` 文件，支持按 ID 查询、分页遍历、元数据过滤和时长过滤。
- **音频加载与处理**：支持从文件、URL、字节流和 PCM 构造音频；可解码为 `Waveform`，并进行声道拆分、转单声道、重采样、归一化等操作。
- **Rust 核心，Python 易用**：核心能力由 Rust 实现，同时提供 PyO3 Python 绑定，适合在脚本、Notebook 和数据处理流水线中使用。
- **面向 ASR 流程**：可区分人工标注、模型输出、系统阶段产物等来源，便于保存参考文本、模型预测和后处理结果。

## Breaking change / Migration

旧版 `Audio` 已重命名为 `AudioDoc`，音频来源改为显式类型；加载入口也拆分到 source 或 `Waveform` 工厂方法。

| 旧 API | 新 API |
| --- | --- |
| `Audio` | `AudioDoc` |
| `Audio.from_file(p)` | `AudioDoc(AudioPath(p))` |
| `Audio.from_url(...)` | `AudioDoc(AudioUrl(...))` |
| `Audio.from_bytes(...)` | `AudioDoc(AudioBytes(...))` |
| `Audio.from_pcm(...)` | `AudioDoc(AudioPcm(...))` |
| `audio.load()` | `doc.source.load()` 或 `Waveform.from_*` |
| `await audio.aload()` | `await doc.source.aload()` |

## 安装

### Python

```bash
pip install asr-data
```

### Rust

作为依赖添加到项目：

```bash
cargo add asr-data
```

## Python 使用示例

### 创建音频记录

```python
from asr_data import AudioDoc, AudioPath, AudioPcm, Waveform

# 从本地文件创建
doc = AudioDoc(AudioPath("audio.wav"), id="call-001")

# 添加元数据
doc.metadata["split"] = "train"
doc.metadata["speaker"] = "alice"
```

也可以从 URL、原始字节或 PCM 数据创建：

```python
from asr_data import AudioBytes, AudioDoc, AudioPcm, AudioUrl

doc_from_url = AudioDoc(AudioUrl("https://example.com/audio.wav"), id="remote-001")
doc_from_bytes = AudioDoc(AudioBytes(open("audio.wav", "rb").read()), id="bytes-001")
doc_from_pcm = AudioDoc(AudioPcm(b"\0\0" * 16000, sample_rate=16000), id="pcm-001")

# 音频 ID 和时长属于 AudioDoc，并同步到所有声道 Timeline。
doc_from_pcm.duration_ms = 1_000
```

### 添加时间轴标注

```python
# 语音段
doc.timeline("mono").add_speech(0, 1200, confidence=0.98)

# 转写文本
doc.timeline("mono").add_transcription(
    0,
    1200,
    "hello world",
    source="whisper-large",
    source_kind="model",
    language="en",
    confidence=0.91,
)

# 说话人
doc.timeline("mono").add_speaker(0, 1200, "speaker-1")

print(doc.timeline("mono").transcript_by_source("whisper-large").text)
```

`doc.timeline("mono")` 表示 mono 时间轴。双声道通话可以把左右声道的标注分别保存在同一条 `AudioDoc` 记录中：

```python
waveform = doc.source.load()
caller_waveform = waveform.channel(0)
agent_waveform = waveform.channel(1)

# 识别器只处理提取后的 mono waveform；调用方把结果写回对应声道。
doc.ensure_timeline("left").add_transcription(0, 1200, "caller text")
doc.ensure_timeline("right").add_transcription(0, 1200, "agent text")
```

`timeline()` 只查询，不会修改 AudioDoc；声道不存在时返回 `None`。需要创建时显式使用 `ensure_timeline()`：

```python
right = doc.timeline("right")
if right is None:
    right = doc.ensure_timeline("right")

doc.remove_timeline("right")
```

需要混音时仍显式调用 `waveform.to_mono()`，不会自动合并左右声道的 Timeline。

### 加载和处理波形

```python
from asr_data import Waveform

# 从文件直接加载
wf = Waveform.from_path("audio.wav")

# 或通过 source 对象加载
wf = doc.source.load()
# 或 AudioPath("audio.wav").load()

wf = wf.resample(16000).normalize()

print(wf.sample_rate)
print(wf.channels)

mono = wf.to_mono()
left = wf.channel(0)
```

异步加载：

```python
waveform = await doc.source.aload()
```

### 使用 AudioDB 持久化

```python
from asr_data import AudioDB, AudioDoc, AudioPath

doc = AudioDoc(AudioPath("audio.wav"), id="call-001")
db = AudioDB("dataset.sqlite")

# 写入
db.insert(doc)

# 按 ID 读取
loaded = db["call-001"]

# 更新
loaded.metadata["checked"] = True
db.update(loaded)

# 批量更新使用单个 SQLite 事务
db.update_many([loaded])

# 数据库级元数据适合保存模型运行信息
db.set_metadata("annotation_runs", {
    "whisper-large": {"language": "en"},
})

# 删除
db.delete("call-001")
```

### 查询和分页

`query` 按 `audio_id` 排序。分页时，把上一页最后一条 ID 作为下一页的 `after`：

```python
first_page = db.query(limit=100, metadata={"split": "train"})

if first_page:
    second_page = db.query(
        limit=100,
        after=first_page[-1].id,
        metadata={"split": "train"},
    )
```

也可以按时长过滤：

```python
items = db.query(
    limit=50,
    min_duration_ms=1_000,
    max_duration_ms=30_000,
)
```

直接迭代数据库会按内部游标懒加载：

```python
for doc in db:
    print(doc.id)
```

## 数据格式

`asr-data` 使用标准 SQLite 数据库文件保存数据，推荐使用 `.sqlite` 后缀。库会写入 SQLite application ID，用于识别和校验数据库格式。

## 许可证

本项目采用 [MIT License](LICENSE) 开源。
