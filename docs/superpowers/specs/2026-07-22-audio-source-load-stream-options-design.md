# AudioSource 加载与流式选项设计

## 目标

保留独立的 `load()` 与 `stream()` 接口，为二者增加一致的可选采样率和单声道转换，并保证所有解码输出都是有限的 `float32`，范围为 `[-1.0, 1.0]`。加载接口不提供自动峰值归一化参数；需要峰值归一化时仍由调用方显式调用 `Audio.normalize()` 或 `AudioChunk.normalize()`。

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
```

- `sample_rate=None` 保留源采样率；正整数表示目标采样率；零值报错。
- `mono=True` 将多声道平均为单声道；`mono=None` 或 `False` 保留源声道。
- `chunk_size_ms` 表示输出 chunk 的目标时长，默认 100ms；零值报错；最后一块允许更短且不补零。
- `load()` 返回完整 `Audio`，`stream()` 返回惰性 `AudioChunk` 迭代器。
- `aload()` 本次不增加新参数，但同样继承统一的解码样本范围保证。

## Rust 与内部结构

- 解码层负责把所有受支持格式转换成 `f32`，并统一把 `NaN`、正负无穷替换为 `0.0`，其余值截断到 `[-1.0, 1.0]`。
- 完整加载先解码，再按需转单声道、重采样，最后再次执行范围清理，以处理重采样可能产生的轻微 overshoot。
- 流式加载使用有状态的转换迭代器，保持重采样器跨解码块连续，避免逐块重置滤波器造成边界伪影。转换后的样本按目标采样率重新组成固定时长 chunk。
- `AudioLoadOptions` 改为可选 `sample_rate` 和 `mono`，移除自动 `normalize` 配置；默认值表示保留原采样率和声道。
- `is_normalized` 继续只表示是否显式执行过峰值归一化；仅做浮点范围清理不会把它设为 `true`。

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
- 不删除 `Audio.normalize()`、`AudioChunk.normalize()` 或 `is_normalized`，只是不在 source 加载方法中自动调用它们。

## 测试

- 先新增 Python API 测试，覆盖所有 source 类型的签名、默认行为、转单声道、重采样和组合选项。
- 验证 `load()` 与拼接后的 `stream()` 在采样率、声道数、帧序列和范围上保持一致。
- 使用包含 `NaN`、无穷和越界浮点值的内部单元测试验证样本清理规则。
- 验证默认 100ms chunk、最后一块不补零、offset 单调且最终块标记正确。
- 运行 Rust 测试、Python bindings 测试、格式检查和类型声明检查。
