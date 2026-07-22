mod annotation;
mod audio;
mod common;
mod db;
mod doc;
mod metrics;
mod timeline;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(_native, AsrDataError, PyException);

#[pymodule]
fn _native(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = audio::async_runtime();
    module.add("AsrDataError", py.get_type::<AsrDataError>())?;
    annotation::register(module)?;
    audio::register(module)?;
    metrics::register(module)?;
    timeline::register(module)?;
    doc::register(module)?;
    db::register(module)?;
    Ok(())
}
