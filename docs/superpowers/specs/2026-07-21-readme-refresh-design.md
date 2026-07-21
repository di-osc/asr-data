# README 精简设计

## 目标

将 README 从详细 API 指南精简为项目首页，让访问者快速了解 `asr-data`、完成安装并运行一个最小示例；完整 API 说明统一引导至在线文档。

## 页面结构

1. Logo、项目名，以及 `crates.io`、`PyPI`、`docs` 三个可点击 badge。
2. 一段简短的项目定位说明。
3. 四至五条核心特点，覆盖统一数据模型、SQLite 存储、音频处理、Rust/Python 双语言支持和 ASR 工作流。
4. Python 与 Rust 安装命令。
5. 一个最小 Python 示例，展示创建 `AudioDoc`、添加转写并写入 `AudioDB`。
6. 指向完整在线文档的明确链接。
7. MIT 许可证说明。

## 链接

- crates.io：`https://crates.io/crates/asr-data`
- PyPI：`https://pypi.org/project/asr-data/`
- docs：`https://libraries-793f13szd-di-osc1.vercel.app/asr-data`

Badge 图片使用 shields.io，并让每个 badge 链接到对应页面。docs badge 使用静态 `latest` 标签，避免依赖文档站提供专用 badge 接口。

## 内容边界

README 不再保留迁移说明、完整音频加载与处理方法、分页查询等 API 细节。示例仅使用当前公开 API，不引入新功能或修改代码。

## 验证

- 检查所有 badge 图片和目标 URL。
- 检查 README 中的安装命令与包元数据一致。
- 检查最小示例中的类和方法仍存在于当前源码。
- 检查 Markdown 结构及仓库状态，确保不改动用户已有的无关文件。
