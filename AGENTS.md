# AGENTS.md

## 项目概览

`asr-data` 是一个 Rust 核心、Python 绑定的 ASR 音频数据管理库，使用 SQLite 持久化音频、时间轴和标注数据。

## 开发约定

- 优先保持 Rust 核心 API、Python 绑定和类型存根的一致性。
- 修改公共 API 时，同时检查相关测试和文档。
- 不要提交构建产物、缓存文件或本地环境文件。
- 提交前运行与修改内容相关的测试和检查。

## 网页文档

文档地址：[https://di-osc.github.io/asr-data/](https://di-osc.github.io/asr-data/)

编写规范：
- 主页面编写章节的主要内容，右侧补充代码示例。
- 示例代码需要有rust和python两种。
