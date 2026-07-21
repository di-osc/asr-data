# AudioSource 加载与流式选项设计

## 目标

保留独立的同步与异步加载、流式接口，为四个接口增加一致的可选采样率和单声道转换，并保证所有解码输出都是有限的 `float32`，范围为 `[-1.0, 1.0]`。项目不再提供峰值归一化状态或方法。

## Python API

`AudioPath`、`AudioUrl`、`AudioBytes`、`AudioBase64` 和 `AudioPcm` 使用相同签名：

```python
source.load(
    *,
    sample_rate: int | None = None,
    mono: bool | None = None,
) -> Audio

source.stream(
    chunk_size_ms: int = 100,
    *,
    sample_rate: int | None = None,
    mono: bool | None = None,
) -> Iterator[AudioChunk]

await source.aload(
    *,
    sample_rate: int | None = None,
    mono: bool | None = None,
) -> Audio

source.astream(
    chunk_size_ms: int = 100,
    *,
    sample_rate: int | None = None,
    mono: bool | None = None,
) -> AsyncIterator[AudioChunk]
```

- `sample_rate=None` 保留源采样率；正整数表示目标采样率；零值报错。
- `mono=True` 将多声道平均为单声道；`mono=None` 或 `False` 保留源声道。
- `chunk_size_ms` 表示输出 chunk 的目标时长，默认 100ms；零值报错；最后一块允许更短且不补零。
- `load()` 返回完整 `Audio`，`stream()` 返回惰性 `AudioChunk` 迭代器。
- `aload()` 是 `load()` 的非阻塞异步版本，参数和结果语义相同。
- `astream()` 返回真正的异步迭代器，通过 `async for` 消费；网络读取和解码不会阻塞 Python 事件循环。

## Rust 与内部结构

- 解码层负责把所有受支持格式转换成 `f32`，并统一把 `NaN`、正负无穷替换为 `0.0`，其余值截断到 `[-1.0, 1.0]`。
- 完整加载先解码，再按需转单声道、重采样，最后再次执行范围清理，以处理重采样可能产生的轻微 overshoot。
- 流式加载使用有状态的转换迭代器，保持重采样器跨解码块连续，避免逐块重置滤波器造成边界伪影。转换后的样本按目标采样率重新组成固定时长 chunk。
- `astream()` 在后台异步任务中读取和解码，通过有界通道逐块交付给 Python 异步迭代器；消费者取消或释放迭代器后，生产任务停止。
- `AudioLoadOptions` 改为可选 `sample_rate` 和 `mono`，移除 `normalize` 配置；默认值表示保留原采样率和声道。
- 从 Rust 与 Python API、序列化结构和类型声明中删除 `Audio.is_normalized`、`Audio.normalize()`、`AudioChunk.is_normalized` 与 `AudioChunk.normalize()`。
- 读取旧版 map 形式的 MessagePack/SQLite 记录时忽略遗留的 `is_normalized` 字段，确保已有数据库仍可打开；新写入的数据不再包含该字段。

## 数据与错误行为

处理顺序固定为：

```text
格式解码 -> 可选转单声道 -> 可选重采样 -> 有限值与 [-1, 1] 范围清理
```

文件、URL、编码字节、Base64 与 PCM 路径必须得到相同语义。源格式信息保留在 `source_format`；输出 `sample_rate` 和 `channels` 反映转换后的数据，offset 根据输出帧位置计算。

## 兼容性

- 现有无参数 `load()` 调用继续有效。
- 现有显式 `stream(chunk_size_ms)` 调用继续有效。
- `stream()` 新增 100ms 默认值。
- `aload()` 的现有无参数调用继续有效，并新增与 `load()` 相同的选项。
- 新增真正的异步迭代接口 `astream()`。
- 峰值归一化相关方法和状态属于明确的破坏性删除。

## 测试

- 先新增 Python API 测试，覆盖所有 source 类型的四个接口、默认行为、转单声道、重采样和组合选项。
- 验证 `load()` 与拼接后的 `stream()` 在采样率、声道数、帧序列和范围上保持一致。
- 验证 `aload()` 与 `load()` 结果一致，`astream()` 可用 `async for` 消费、不会阻塞事件循环，并与同步流式结果一致。
- 使用包含 `NaN`、无穷和越界浮点值的内部单元测试验证样本清理规则。
- 验证默认 100ms chunk、最后一块不补零、offset 单调且最终块标记正确。
- 验证所有归一化字段、方法和类型声明均已删除，并验证带遗留 `is_normalized` 字段的旧记录仍能读取。
- 运行 Rust 测试、Python bindings 测试、格式检查和类型声明检查。
