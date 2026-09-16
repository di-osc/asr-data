//! 时间区间。

use serde::{Deserialize, Serialize};

/// 半开时间区间 `[start_ms, end_ms)`，单位毫秒。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeRange {
    /// 区间起点（含），单位毫秒。
    #[serde(alias = "start")]
    pub start_ms: usize,
    /// 区间终点（不含），单位毫秒。
    #[serde(alias = "end")]
    pub end_ms: usize,
}

impl TimeRange {
    /// 构造时间区间，不检查 `start_ms <= end_ms`。
    pub fn new(start_ms: usize, end_ms: usize) -> Self {
        Self { start_ms, end_ms }
    }

    /// 区间长度；若 `end_ms < start_ms` 则饱和为 0。
    pub fn duration(self) -> usize {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// 两个半开区间是否相交。
    pub fn overlaps(&self, other: &TimeRange) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }

    /// 时间点是否落在 `[start_ms, end_ms)` 内。
    pub fn contains(&self, point_ms: usize) -> bool {
        self.start_ms <= point_ms && point_ms < self.end_ms
    }
}
