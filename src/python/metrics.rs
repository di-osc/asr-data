use pyo3::prelude::*;

use crate::normalize_zh as rust_normalize_zh;

use super::common::py_error;

#[pyfunction]
fn normalize_zh(text: &str) -> PyResult<String> {
    rust_normalize_zh(text).map_err(py_error)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(normalize_zh, module)?)?;
    Ok(())
}
