<p align="center">
  <img src="assets/logo.png" alt="asr-data logo" width="160" />
</p>

<h1 align="center">asr-data</h1>

<p align="center">
  <a href="https://crates.io/crates/asr-data"><img src="https://img.shields.io/crates/v/asr-data?label=crates.io" alt="crates.io" /></a>
  <a href="https://pypi.org/project/asr-data/"><img src="https://img.shields.io/pypi/v/asr-data?label=PyPI" alt="PyPI" /></a>
  <a href="https://di-osc.github.io/asr-data/"><img src="https://img.shields.io/badge/docs-latest-blue" alt="documentation" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license" /></a>
</p>

`asr-data` 是一个面向 ASR（Automatic Speech Recognition，自动语音识别）工作流的 Rust / Python 库，提供统一的音频、时间轴、标注和评测数据模型，并支持使用 SQLite 持久化语音数据。

## 特性

- Rust 核心实现，提供 Python 绑定和类型支持
- 支持本地文件、URL、编码字节、Base64 和 PCM 音频
- 支持音频解码、波形处理和流式读取
- 支持转写、Token、活动事件和说话人标注
- 支持 SQLite 数据库、数据集管理和 ModelScope 数据集加载
- 支持转写 CER、活动检测和说话人分离等评测

## 安装

### Python

```bash
pip install asr-data
```

### Rust

```bash
cargo add asr-data
```

默认开启 SQLite、评测指标和 ModelScope 数据集。只要波形和时间轴时：

```toml
asr-data = { version = "0.1.1", default-features = false }
```

## 文档

完整的使用指南、API 参考、数据模型、评测说明和示例请查看在线文档：

**[https://di-osc.github.io/asr-data/](https://di-osc.github.io/asr-data/)**

其他资源：

- [crates.io](https://crates.io/crates/asr-data)
- [PyPI](https://pypi.org/project/asr-data/)
- [GitHub Repository](https://github.com/di-osc/asr-data)

## 许可证

本项目采用 [MIT License](LICENSE) 开源。
