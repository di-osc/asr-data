mod cer;
mod normalization;
pub use cer::{CerStats, compute_cer, normalize_for_cer};
pub(crate) use normalization::normalize_zh_without_tn;
pub use normalization::{
    ChineseTextNormalizationOptions, TextNormalizationError, normalize_zh,
    normalize_zh_with_options,
};
