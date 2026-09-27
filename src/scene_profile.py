"""Camera-independent scene normalization.

Does NOT attempt to measure physical camera height. Instead builds a
per-camera profile from runtime observations: how expected person bbox
size varies with full 2D image position (row AND column -- a
ground-plane-referenced perspective-scale regression, not a row-only
one), so the same real-world posture produces comparable normalized
features regardless of camera tilt/height/zoom/column.

If too few reliable samples have been observed, parameters are marked
unavailable/low-confidence rather than invented — callers should fall
back to plain frame-relative normalization in that case.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import CalibrationConfig, FloorRegionConfig
from src.person_size_stats import RollingPersonSizeStats, SizeStatsSnapshot


@dataclass
class FloorRegion:
    """Estimated image-row band where this camera's tracked people actually
    touch the ground, from observed ankle y-positions — not assumed to be
    the bottom of the frame. Not a 3D reconstruction: purely an empirical
    band in image space, with an explicit confidence."""

    band_low: float = 0.0
    band_high: float = 0.0
    confidence: float = 0.0
    num_samples: int = 0

    @property
    def is_valid(self) -> bool:
        return self.confidence > 0.0 and self.band_high > self.band_low

    def contact_likelihood(self, y: float) -> float:
        """1.0 when y falls inside the floor band, decaying smoothly outside
        it (normalized by the band's own half-width). 0.0 (no evidence)
        when the band isn't calibrated — never a fabricated value."""
        if not self.is_valid:
            return 0.0
        half_width = max((self.band_high - self.band_low) / 2.0, 1e-3)
        center = (self.band_high + self.band_low) / 2.0
        distance = max(0.0, abs(y - center) - half_width)
        return float(np.exp(-distance / half_width)) * self.confidence


@dataclass
class ScenePerspectiveFit:
    """Ground-plane-referenced fit of expected bbox height/width as a
    function of full 2D image position (row AND column), not row alone.

    height_at(x, y) ~= coef_h_x*x + coef_h_y*y + intercept_h, fit over
    (ankle_x, bbox_bottom_y, bbox_height) triples from reliable
    standing-like detections. A camera looking down at an angle (any real
    mounted CCTV camera) has perspective scale that varies across BOTH
    axes, not just vertically -- a person near a side edge of the frame at
    the same row as someone near the center is not the same real distance
    from the camera in general (this is most visible on a wide/fisheye-ish
    lens, where the earlier row-only fit's "expected size" was only
    correct along one column and increasingly wrong toward the frame
    edges). Still an empirical, auto-calibrating regression from
    accumulated observations -- not a true metric ground-plane homography
    (that needs known real-world reference distances this system never
    has), but the 2D-position dependence is the specific gap a 1D,
    row-only fit had."""

    coef_h_x: float = 0.0
    coef_h_y: float = 0.0
    intercept_h: float = 0.0
    coef_w_x: float = 0.0
    coef_w_y: float = 0.0
    intercept_w: float = 0.0
    r_squared: float = 0.0
    num_samples: int = 0

    def height_at(self, x: float, y: float) -> float:
        return self.coef_h_x * x + self.coef_h_y * y + self.intercept_h

    @property
    def is_valid(self) -> bool:
        return self.num_samples >= 15 and self.r_squared > 0.1


@dataclass
class SceneProfile:
    """Per-camera scene/normalization profile, built and refined at runtime."""

    camera_id: str
    frame_width: int
    frame_height: int
    calibration_confidence: float = 0.0
    perspective: ScenePerspectiveFit = field(default_factory=ScenePerspectiveFit)
    size_stats: SizeStatsSnapshot | None = None
    # "flat" | "tilted_up_left" | "tilted_up_right" | "unknown"
    dominant_floor_direction: str = "unknown"
    floor_region: FloorRegion = field(default_factory=FloorRegion)

    def normalize_height(
        self, bbox_bottom_y: float, observed_height: float, bbox_center_x: float
    ) -> float:
        """Return observed_height / expected_standing_height, i.e. a
        camera-position-independent ratio (~1.0 = normal standing size,
        <1.0 = shorter than expected — e.g. lying reduces vertical extent).

        Prefers the position-aware perspective regression (accounts for
        near/far perspective scaling across BOTH image axes, not just
        row); falls back to the position-agnostic rolling median when the
        regression isn't calibrated yet; falls back to a neutral 1.0 (no
        signal) when nothing is calibrated at all — never a fabricated
        value. `bbox_center_x` is required (not optional/defaulted) since
        the fit now genuinely depends on it — a caller that doesn't track
        horizontal position has no correct value to substitute.
        """
        if self.perspective.is_valid:
            expected = self.perspective.height_at(bbox_center_x, bbox_bottom_y)
            if expected > 1e-3:
                return observed_height / expected

        if self.size_stats is not None and self.size_stats.is_calibrated:
            expected = self.size_stats.median_height
            if expected and expected > 1e-3:
                return observed_height / expected

        return 1.0

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "calibration_confidence": round(self.calibration_confidence, 3),
            "dominant_floor_direction": self.dominant_floor_direction,
            "perspective": {
                "coef_h_x": self.perspective.coef_h_x,
                "coef_h_y": self.perspective.coef_h_y,
                "intercept_h": self.perspective.intercept_h,
                "coef_w_x": self.perspective.coef_w_x,
                "coef_w_y": self.perspective.coef_w_y,
                "intercept_w": self.perspective.intercept_w,
                "r_squared": self.perspective.r_squared,
                "num_samples": self.perspective.num_samples,
                "is_valid": self.perspective.is_valid,
            },
            "floor_region": {
                "band_low": self.floor_region.band_low,
                "band_high": self.floor_region.band_high,
                "confidence": round(self.floor_region.confidence, 3),
                "num_samples": self.floor_region.num_samples,
                "is_valid": self.floor_region.is_valid,
            },
            "size_stats": (
                {
                    "sample_count": self.size_stats.sample_count,
                    "median_height": self.size_stats.median_height,
                    "p25_height": self.size_stats.p25_height,
                    "p75_height": self.size_stats.p75_height,
                    "height_std": self.size_stats.height_std,
                    "median_width": self.size_stats.median_width,
                    "p25_width": self.size_stats.p25_width,
                    "p75_width": self.size_stats.p75_width,
                    "calibration_confidence": round(self.size_stats.calibration_confidence, 3),
                    "is_calibrated": self.size_stats.is_calibrated,
                }
                if self.size_stats is not None
                else None
            ),
        }


class SceneProfileBuilder:
    """Accumulates (bbox_bottom_y, height, width, ankle_x) samples for one
    camera and periodically refits the perspective regression. Thread-safe
    and cheap enough to update every frame; the expensive refit only runs
    every `refit_stride` new samples, not on every observation."""

    def __init__(
        self,
        camera_id: str,
        frame_width: int,
        frame_height: int,
        cal: CalibrationConfig | None = None,
        floor_region_config: FloorRegionConfig | None = None,
        refit_stride: int = 10,
        max_samples: int = 500,
        sample_log_dir: str | Path | None = None,
        sample_log_max_rows: int = 50,
    ):
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cal = cal or CalibrationConfig()
        self.floor_region_config = floor_region_config or FloorRegionConfig()
        self.refit_stride = refit_stride
        self.max_samples = max_samples

        self._ys: list[float] = []
        self._heights: list[float] = []
        self._widths: list[float] = []
        self._ankle_xs: list[float] = []
        self._ankle_ys: list[float] = []
        self._new_since_fit = 0
        self._lock = threading.Lock()

        self._fit = ScenePerspectiveFit()
        self._floor_region = FloorRegion()
        self.size_stats = RollingPersonSizeStats()

        # Real-time, per-reliable-sample audit log (one JSON line per
        # sample) — the raw per-person record a live/RTSP deployment needs
        # for manual inspection, since the stream may run for hours and
        # could be killed without ever hitting the "stream ended" save path.
        # Distinct from `save()`'s aggregate snapshot, and from the 30-sample
        # window RollingPersonSizeStats actually uses to make its decision
        # (see module docstring reference in add_sample below) — this file
        # is bounded to the most recent `sample_log_max_rows` rows so a
        # 24/7 camera doesn't grow it unbounded; older rows are dropped
        # automatically. Seeded from the existing file (if any) so a
        # restart doesn't blank the recent history to zero.
        self._sample_log_path = None
        self._sample_log_rows: deque[str] | None = None
        if sample_log_dir is not None:
            sample_log_dir = Path(sample_log_dir)
            sample_log_dir.mkdir(parents=True, exist_ok=True)
            self._sample_log_path = sample_log_dir / f"{camera_id}_samples.jsonl"
            self._sample_log_rows = deque(maxlen=sample_log_max_rows)
            if self._sample_log_path.exists():
                with open(self._sample_log_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._sample_log_rows.append(line)

    def add_sample(
        self,
        bbox: tuple[float, float, float, float],
        ankle_x: float | None,
        ankle_y: float | None = None,
        track_id: int | None = None,
        frame_num: int | None = None,
    ) -> None:
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        self.size_stats.add_sample(width, height)

        if self._sample_log_rows is not None:
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "frame_num": frame_num,
                "track_id": track_id,
                "bbox_height": round(height, 2),
                "bbox_width": round(width, 2),
            }
            # deque(maxlen=...) auto-evicts the oldest row once at capacity;
            # rewriting the whole (bounded, small) file keeps disk state in
            # sync with that eviction -- an append-only write can't drop
            # old rows on its own.
            self._sample_log_rows.append(json.dumps(record))
            with open(self._sample_log_path, "w") as f:
                f.write("\n".join(self._sample_log_rows) + "\n")

        with self._lock:
            self._ys.append(y2)
            self._heights.append(height)
            self._widths.append(width)
            self._ankle_xs.append(ankle_x if ankle_x is not None else (x1 + x2) / 2)
            self._ankle_ys.append(ankle_y if ankle_y is not None else y2)

            if len(self._ys) > self.max_samples:
                overflow = len(self._ys) - self.max_samples
                del self._ys[:overflow]
                del self._heights[:overflow]
                del self._widths[:overflow]
                del self._ankle_xs[:overflow]
                del self._ankle_ys[:overflow]

            self._new_since_fit += 1
            if self._new_since_fit >= self.refit_stride:
                self._refit_locked()
                self._new_since_fit = 0

    def _refit_locked(self) -> None:
        """Refit the perspective regression. Caller must hold self._lock."""
        n = len(self._ys)
        if n < self.cal.min_samples_for_confidence:
            return

        ys = np.array(self._ys, dtype=np.float64)
        heights = np.array(self._heights, dtype=np.float64)
        widths = np.array(self._widths, dtype=np.float64)
        ankle_xs = np.array(self._ankle_xs, dtype=np.float64)

        coef_h_x, coef_h_y, intercept_h, r2 = _fit_plane_with_r2(ankle_xs, ys, heights)
        coef_w_x, coef_w_y, intercept_w, _ = _fit_plane_with_r2(ankle_xs, ys, widths)

        direction = "flat"
        if abs(coef_h_y) > 1e-4:
            x_corr = np.corrcoef(ankle_xs, ys)[0, 1] if n > 1 and np.std(ankle_xs) > 1e-6 else 0.0
            if not np.isfinite(x_corr):
                x_corr = 0.0
            if x_corr > 0.2:
                direction = "tilted_up_right"
            elif x_corr < -0.2:
                direction = "tilted_up_left"
            else:
                direction = "flat"

        self._fit = ScenePerspectiveFit(
            coef_h_x=coef_h_x, coef_h_y=coef_h_y, intercept_h=intercept_h,
            coef_w_x=coef_w_x, coef_w_y=coef_w_y, intercept_w=intercept_w,
            r_squared=r2, num_samples=n,
        )
        self._last_direction = direction
        self._floor_region = self._compute_floor_region(np.array(self._ankle_ys, dtype=np.float64))

    def _compute_floor_region(self, ankle_ys: np.ndarray) -> FloorRegion:
        n = len(ankle_ys)
        frc = self.floor_region_config
        if n < frc.min_samples_for_confidence:
            return FloorRegion(num_samples=n)

        q25, q75 = np.percentile(ankle_ys, [25, 75])
        iqr = max(float(q75 - q25), 1e-3)
        median = float(np.median(ankle_ys))
        half_width = frc.band_half_width_ratio * iqr
        confidence = min(1.0, n / max(frc.full_confidence_samples, 1))
        return FloorRegion(
            band_low=float(median - half_width), band_high=float(median + half_width),
            confidence=confidence, num_samples=n,
        )

    def build_profile(self) -> SceneProfile:
        with self._lock:
            fit = self._fit
            floor_region = self._floor_region
            n = len(self._ys)
            direction = getattr(self, "_last_direction", "unknown")

        if n >= self.cal.min_samples_for_confidence:
            confidence = min(1.0, n / max(self.cal.full_confidence_samples, 1))
        else:
            confidence = 0.0
        stats_snapshot = self.size_stats.snapshot(self.cal)

        return SceneProfile(
            camera_id=self.camera_id,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            calibration_confidence=confidence,
            perspective=fit,
            size_stats=stats_snapshot,
            dominant_floor_direction=direction if fit.is_valid else "unknown",
            floor_region=floor_region,
        )

    def close(self) -> None:
        """No-op: the per-sample audit log (if any) writes+closes its file
        on every sample rather than holding an open handle, so there's
        nothing left open to release. Kept for API stability."""

    def save(self, directory: str | Path = "configs/scene_profiles") -> Path:
        """Persist the current profile so future runs on this camera don't
        need to recalibrate from scratch."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.camera_id}.json"
        with open(path, "w") as f:
            json.dump(self.build_profile().to_dict(), f, indent=2)
        return path

    @staticmethod
    def load_seed(camera_id: str, directory: str | Path = "configs/scene_profiles") -> dict | None:
        """Load a previously saved profile dict, if any, to seed logging/
        display only (the builder still recalibrates from live samples;
        this just avoids reporting "unavailable" for a camera already seen)."""
        path = Path(directory) / f"{camera_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)


def _fit_plane_with_r2(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float, float, float]:
    """Least-squares plane fit z = coef_x*x + coef_y*y + intercept, plus R^2
    — the 2D generalization of the old single-variable line fit, since
    expected bbox size genuinely depends on both image column and row for
    a camera looking down at an angle, not row alone. Degrades gracefully
    (falls back toward a row-only or constant fit) when there isn't enough
    variation in x yet to constrain that term, rather than producing a
    numerically unstable fit from an ill-conditioned system."""
    n = len(x)
    if n < 3:
        return 0.0, 0.0, float(np.mean(z)) if n else 0.0, 0.0

    design = np.column_stack([x, y, np.ones(n)])
    if np.std(x) < 1e-6:
        # No horizontal variation observed yet -- solving for coef_x would
        # be numerically degenerate (a column of near-zero variance), so
        # drop it and fit row-only instead of letting lstsq return a
        # meaningless/unstable value for it.
        design = design[:, 1:]
        (coef_y, intercept), *_ = np.linalg.lstsq(design, z, rcond=None)
        coef_x = 0.0
    else:
        (coef_x, coef_y, intercept), *_ = np.linalg.lstsq(design, z, rcond=None)

    pred = coef_x * x + coef_y * y + intercept
    ss_res = np.sum((z - pred) ** 2)
    ss_tot = np.sum((z - np.mean(z)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    return float(coef_x), float(coef_y), float(intercept), float(max(0.0, r2))
