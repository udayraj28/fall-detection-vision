"""Rolling, thread-safe person bounding-box size statistics.

Used to normalize a track's current bbox size against what "normal" person
size looks like in this camera's view, without knowing physical camera
height. Only reliable (non-truncated, confident, standing-like) detections
are used, so a person mid-fall never contaminates the "standing" baseline.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from src.config import CalibrationConfig, RollingSizeConfig


@dataclass
class SizeStatsSnapshot:
    """Point-in-time readout of the rolling statistics."""

    sample_count: int
    median_height: float | None
    p25_height: float | None
    p75_height: float | None
    height_std: float | None
    median_width: float | None
    p25_width: float | None
    p75_width: float | None
    calibration_confidence: float  # 0.0 (no data) to 1.0 (fully calibrated)

    @property
    def is_calibrated(self) -> bool:
        return self.median_height is not None


def is_reliable_standing_sample(
    bbox: tuple[float, float, float, float],
    mean_keypoint_conf: float,
    num_critical_keypoints_visible: int,
    frame_width: int,
    frame_height: int,
    cal: CalibrationConfig,
) -> bool:
    """Decide whether a detection is trustworthy enough to feed calibration.

    Rejects: low-confidence detections, boxes touching the frame edge
    (likely truncated), and boxes whose aspect ratio doesn't look like an
    upright/standing person (so a real fall doesn't skew the "normal size"
    baseline).
    """
    if mean_keypoint_conf < cal.min_keypoint_confidence:
        return False
    if num_critical_keypoints_visible < cal.min_critical_keypoints:
        return False

    x1, y1, x2, y2 = bbox
    margin = cal.frame_edge_margin_px
    if x1 <= margin or y1 <= margin or x2 >= frame_width - margin or y2 >= frame_height - margin:
        return False

    width, height = x2 - x1, y2 - y1
    if height <= 0:
        return False
    aspect = width / height
    if not (cal.min_aspect_ratio_for_standing <= aspect <= cal.max_aspect_ratio_for_standing):
        return False

    return True


class RollingPersonSizeStats:
    """Adaptive rolling window of reliable person bbox sizes for one camera.

    Fills up to `window_size` samples one at a time, then switches to
    stride-based batch updates: buffers `update_stride` new reliable
    samples, then evicts the oldest `update_stride` and admits the newest
    `update_stride` in one step. Thread-safe; all locking is a short
    critical section around plain list/array operations, so it never
    blocks the main inference loop for long.
    """

    def __init__(self, config: RollingSizeConfig | None = None):
        self.config = config or RollingSizeConfig()
        self._heights: list[float] = []
        self._widths: list[float] = []
        self._pending_heights: list[float] = []
        self._pending_widths: list[float] = []
        self._total_samples_seen = 0
        self._lock = threading.Lock()

    def add_sample(self, width: float, height: float) -> None:
        """Add one reliable bbox size sample. Caller must have already
        filtered for reliability via `is_reliable_standing_sample`."""
        with self._lock:
            self._total_samples_seen += 1

            if len(self._heights) < self.config.window_size:
                self._heights.append(height)
                self._widths.append(width)
                return

            self._pending_heights.append(height)
            self._pending_widths.append(width)
            if len(self._pending_heights) >= self.config.update_stride:
                n = self.config.update_stride
                del self._heights[:n]
                del self._widths[:n]
                self._heights.extend(self._pending_heights[:n])
                self._widths.extend(self._pending_widths[:n])
                del self._pending_heights[:n]
                del self._pending_widths[:n]

    def snapshot(self, cal: CalibrationConfig | None = None) -> SizeStatsSnapshot:
        """Compute robust statistics over the current window."""
        cal = cal or CalibrationConfig()
        with self._lock:
            heights = np.array(self._heights, dtype=np.float64)
            widths = np.array(self._widths, dtype=np.float64)
            n = len(heights)

        if n < cal.min_samples_for_confidence:
            return SizeStatsSnapshot(
                sample_count=n,
                median_height=None, p25_height=None, p75_height=None, height_std=None,
                median_width=None, p25_width=None, p75_width=None,
                calibration_confidence=0.0,
            )

        heights = _reject_outliers(heights, self.config.outlier_reject_iqr_multiplier)
        widths = _reject_outliers(widths, self.config.outlier_reject_iqr_multiplier)

        confidence = min(1.0, n / max(self.config.full_confidence_samples, 1))

        return SizeStatsSnapshot(
            sample_count=n,
            median_height=float(np.median(heights)),
            p25_height=float(np.percentile(heights, 25)),
            p75_height=float(np.percentile(heights, 75)),
            height_std=float(np.std(heights)),
            median_width=float(np.median(widths)),
            p25_width=float(np.percentile(widths, 25)),
            p75_width=float(np.percentile(widths, 75)),
            calibration_confidence=confidence,
        )

    @property
    def total_samples_seen(self) -> int:
        with self._lock:
            return self._total_samples_seen

    def reset(self) -> None:
        with self._lock:
            self._heights.clear()
            self._widths.clear()
            self._pending_heights.clear()
            self._pending_widths.clear()
            self._total_samples_seen = 0


def _reject_outliers(values: np.ndarray, iqr_multiplier: float) -> np.ndarray:
    if len(values) < 4:
        return values
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return values
    lo, hi = q25 - iqr_multiplier * iqr, q75 + iqr_multiplier * iqr
    filtered = values[(values >= lo) & (values <= hi)]
    return filtered if len(filtered) > 0 else values
