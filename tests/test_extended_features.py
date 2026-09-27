"""Tests for the extended feature pool used by the feature-based Mandown logic."""

import math

import numpy as np
import pytest

from src.feature_extraction import ExtendedFeatureExtractor


def _make_standing() -> np.ndarray:
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[:] = [320, 240, 0.9]
    kps[0] = (320, 50, 0.95)
    kps[5] = (290, 130, 0.9)
    kps[6] = (350, 130, 0.9)
    kps[11] = (295, 280, 0.9)
    kps[12] = (345, 280, 0.9)
    kps[13] = (295, 380, 0.9)
    kps[14] = (345, 380, 0.9)
    kps[15] = (295, 450, 0.9)
    kps[16] = (345, 450, 0.9)
    return kps


def _make_lying() -> np.ndarray:
    """Lying flat: head, shoulders, hips, knees, ankles all near the same row."""
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[:] = [320, 400, 0.9]
    kps[0] = (100, 400, 0.95)
    kps[5] = (180, 400, 0.9)
    kps[6] = (180, 420, 0.9)
    kps[11] = (350, 400, 0.9)
    kps[12] = (350, 420, 0.9)
    kps[13] = (450, 400, 0.9)
    kps[14] = (450, 420, 0.9)
    kps[15] = (550, 400, 0.9)
    kps[16] = (550, 420, 0.9)
    return kps


class TestExtendedFeatureExtractor:
    def setup_method(self):
        self.extractor = ExtendedFeatureExtractor()

    def test_standing_small_floor_proximity_cue(self):
        kps = _make_standing()
        ext = self.extractor.extract(kps, (270, 40, 360, 460), 640, 480)
        # Standing: head and knees should be far apart vertically.
        assert abs(ext.head_height - ext.knee_height) > 0.3
        assert ext.torso_angle < 0.3

    def test_lying_collapses_head_knee_span(self):
        kps = _make_lying()
        ext = self.extractor.extract(kps, (80, 380, 580, 440), 640, 480)
        # Lying: head and knees at similar row.
        assert abs(ext.head_height - ext.knee_height) < 0.2
        assert ext.torso_angle > 0.5

    def test_no_nan_or_inf(self):
        for kps, bbox in [
            (_make_standing(), (270, 40, 360, 460)),
            (_make_lying(), (80, 380, 580, 440)),
            (np.zeros((17, 3), dtype=np.float32), (0, 0, 0, 0)),
        ]:
            ext = self.extractor.extract(kps, bbox, 640, 480)
            for field in (
                ext.torso_angle, ext.hip_angle, ext.shoulder_angle, ext.head_height,
                ext.knee_height, ext.horizontal_velocity, ext.acceleration,
                ext.angle_change, ext.body_height_norm, ext.shoulder_hip_distance,
                ext.pose_quality,
            ):
                assert math.isfinite(field)

    def test_first_frame_zero_deltas(self):
        kps = _make_standing()
        ext = self.extractor.extract(kps, (270, 40, 360, 460), 640, 480, prev=None)
        assert ext.angle_change == 0.0
        assert ext.acceleration == 0.0
        assert ext.horizontal_velocity == 0.0

    def test_acceleration_from_velocity_delta(self):
        kps = _make_standing()
        prev_ext = ExtendedFeatureExtractor().extract(kps, (270, 40, 360, 460), 640, 480)
        ext = self.extractor.extract(
            kps, (270, 40, 360, 460), 640, 480,
            prev=prev_ext, vertical_velocity_now=0.4, prev_vertical_velocity=0.0,
        )
        assert ext.acceleration == pytest.approx(0.4, abs=1e-5)

    def test_body_height_norm_neutral_without_scene_profile(self):
        kps = _make_standing()
        ext = self.extractor.extract(kps, (270, 40, 360, 460), 640, 480, expected_height_fn=None)
        assert ext.body_height_norm == pytest.approx(1.0, abs=1e-5)

    def test_body_height_norm_uses_expected_height_fn(self):
        kps = _make_standing()
        ext = self.extractor.extract(
            kps, (270, 40, 360, 460), 640, 480,
            expected_height_fn=lambda y, h, x: h / 200.0,
        )
        assert ext.body_height_norm == pytest.approx(420.0 / 200.0, abs=1e-3)

    def test_critical_keypoints_count(self):
        kps = _make_standing()
        ext = self.extractor.extract(kps, (270, 40, 360, 460), 640, 480)
        assert ext.num_critical_keypoints_visible == 6

        kps[11, 2] = 0.0  # hide left hip
        ext2 = self.extractor.extract(kps, (270, 40, 360, 460), 640, 480)
        assert ext2.num_critical_keypoints_visible == 5

    def test_cog_x_helper(self):
        kps = _make_standing()
        cx = self.extractor.cog_x(kps, (270, 40, 360, 460))
        assert 290 <= cx <= 350
