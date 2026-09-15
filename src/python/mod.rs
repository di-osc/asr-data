//! Python 绑定：把 Rust 核心类型导出为 `_native` 扩展模块。

mod annotation;
mod audio;
mod common;
mod db;
mod doc;
mod evaluation;
mod metrics;
mod timeline;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(_native, AsrDataError, PyException);

/// Python 扩展模块入口：注册类型、函数和 `AsrDataError`。
#[pymodule]
fn _native(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = audio::async_runtime();
    module.add("AsrDataError", py.get_type::<AsrDataError>())?;
    annotation::register(module)?;
    audio::register(module)?;
    metrics::register(module)?;
    timeline::register(module)?;
    doc::register(module)?;
    evaluation::register(module)?;
    db::register(module)?;
    Ok(())
}
