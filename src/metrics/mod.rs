mod cer;
mod normalization;
pub use cer::{CerStats, compute_cer, normalize_for_cer};
pub use normalization::{TextNormalizationError, normalize_zh};
