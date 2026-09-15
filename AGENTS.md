# AGENTS.md

## 项目概览

`asr-data` 是一个 Rust 核心、Python 绑定的 ASR 音频数据管理库，使用 SQLite 持久化音频、时间轴和标注数据。

## 开发约定

- 优先保持 Rust 核心 API、Python 绑定和类型存根的一致性。
- 修改公共 API 时，同时检查相关测试和文档。
- 编写代码的时候要遵循 Rust 的代码风格和最佳实践，确保代码的正确性和可读性，每个函数与方法都要添加详细的文档说明，关键代码要有详细的注释。

## 网页文档

文档地址：[https://di-osc.github.io/asr-data/](https://di-osc.github.io/asr-data/)

编写规范：
- 主页面编写章节的主要内容，右侧补充代码示例。
- 示例代码需要有rust和python两种语言的示例。
