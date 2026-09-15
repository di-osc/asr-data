//! 时间与样本位置的轻量新类型。

use serde::{Deserialize, Serialize};

/// 以毫秒表示的时长或时间戳。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DurationMs(pub u64);

impl DurationMs {
    /// 转换成秒（浮点）。
    pub fn seconds(self) -> f64 {
        self.0 as f64 / 1000.0
    }
}

/// 波形中的采样帧下标。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SampleIndex(pub u64);

/// 半开时间区间 `[start, end)`，单位毫秒。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeRange {
    /// 区间起点（含）。
    pub start: DurationMs,
    /// 区间终点（不含）。
    pub end: DurationMs,
}

impl TimeRange {
    /// 构造时间区间，不检查 `start <= end`。
    pub fn new(start: DurationMs, end: DurationMs) -> Self {
        Self { start, end }
    }

    /// 区间长度；若 `end < start` 则饱和为 0。
    pub fn duration(self) -> DurationMs {
        DurationMs(self.end.0.saturating_sub(self.start.0))
    }

    /// 两个半开区间是否相交。
    pub fn overlaps(&self, other: &TimeRange) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// 时间点是否落在 `[start, end)` 内。
    pub fn contains(&self, point: DurationMs) -> bool {
        self.start <= point && point < self.end
    }
}
