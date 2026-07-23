use pyo3::prelude::*;

use crate::normalize_zh as rust_normalize_zh;

use super::common::py_error;

/// 使用内嵌中文 TN 资源标准化文本。
///
/// Args:
///     text: 要标准化的原始文本。
///
/// Returns:
///     转换为口语形式的文本。
///
/// Raises:
///     AsrDataError: 内嵌 FST 无法执行。
///
/// Examples:
///     >>> from asr_data import normalize_zh
///     >>> normalize_zh("2024年")
///     '二零二四年'
#[pyfunction]
fn normalize_zh(text: &str) -> PyResult<String> {
    rust_normalize_zh(text).map_err(py_error)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(normalize_zh, module)?)?;
    Ok(())
}
