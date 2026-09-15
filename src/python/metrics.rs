use pyo3::prelude::*;

use crate::{ChineseTextNormalizationOptions, normalize_zh_with_options};

use super::common::py_error;

/// 使用内嵌中文 TN 资源标准化文本。
///
/// Args:
///     text: 要标准化的原始文本。
///     traditional_to_simple: 是否将繁体中文转换为简体中文。
///     full_to_half: 是否将全角字符转换为半角字符。
///     remove_erhua: 是否去除儿化音“儿”。
///     remove_interjections: 是否去除“嗯”“啊”“呃”等语气词。
///     remove_puncts: 是否去除标点符号。
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
#[pyo3(signature = (
    text,
    *,
    traditional_to_simple=true,
    full_to_half=true,
    remove_erhua=true,
    remove_interjections=true,
    remove_puncts=true
))]
fn normalize_zh(
    text: &str,
    traditional_to_simple: bool,
    full_to_half: bool,
    remove_erhua: bool,
    remove_interjections: bool,
    remove_puncts: bool,
) -> PyResult<String> {
    normalize_zh_with_options(
        text,
        ChineseTextNormalizationOptions {
            traditional_to_simple,
            full_to_half,
            remove_erhua,
            remove_interjections,
            remove_puncts,
        },
    )
    .map_err(py_error)
}

/// 把本模块的 Python 类型和函数注册进 `_native`。
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(normalize_zh, module)?)?;
    Ok(())
}
