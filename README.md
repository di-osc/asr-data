<p align="center">
  <img src="assets/logo.png" alt="asr-data logo" width="160" />
</p>

<h1 align="center">asr-data</h1>

<p align="center">
  <a href="https://crates.io/crates/asr-data"><img src="https://img.shields.io/crates/v/asr-data?label=crates.io" alt="crates.io" /></a>
  <a href="https://pypi.org/project/asr-data/"><img src="https://img.shields.io/pypi/v/asr-data?label=PyPI" alt="PyPI" /></a>
  <a href="https://libraries-793f13szd-di-osc1.vercel.app/asr-data"><img src="https://img.shields.io/badge/docs-latest-blue" alt="docs" /></a>
</p>

`asr-data` 是一个面向 ASR（Automatic Speech Recognition，自动语音识别）数据管理的 Rust / Python 库。它提供统一的音频与标注数据模型，并使用 SQLite 持久化，适合构建语音数据集、标注流水线和模型评测工具。

## 特点

- **统一数据模型**：集中管理音频、转写、说话人、语言和模型预测等信息。
- **SQLite 本地存储**：数据保存为易于管理和迁移的 `.sqlite` 文件。
- **音频处理**：支持从文件、URL、字节流和 PCM 加载音频，并提供声道与重采样等操作。
- **Rust 核心，Python 易用**：兼顾 Rust 性能与 Python 脚本、Notebook 的使用体验。
- **面向 ASR 工作流**：适用于数据集构建、人工标注、模型输出和评测结果管理。

## 安装

### Python

```bash
pip install asr-data
```

### Rust

```bash
cargo add asr-data
```

## 快速开始

```python
from asr_data import AudioDB, AudioDoc, AudioPath

doc = AudioDoc(AudioPath("audio.wav"), id="call-001")
timeline = doc.ensure_timeline("mono", duration_ms=1_200)
timeline.add_transcription(0, 1_200, "hello world", language="en")

db = AudioDB("dataset.sqlite")
db.insert(doc)
```

## 文档

完整的 API 与使用说明请查看[在线文档](https://libraries-793f13szd-di-osc1.vercel.app/asr-data)。

## 许可证

本项目采用 [MIT License](LICENSE) 开源。
