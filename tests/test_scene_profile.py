"""Tests for camera-independent scene profile normalization."""

import json

from src.config import CalibrationConfig
from src.scene_profile import SceneProfileBuilder


class TestSceneProfileBuilder:
    def test_uncalibrated_marks_unavailable(self):
        builder = SceneProfileBuilder("cam1", 640, 480, refit_stride=1)
        for _ in range(5):
            builder.add_sample((100, 50, 160, 300), ankle_x=130)
        profile = builder.build_profile()
        assert profile.calibration_confidence == 0.0
        assert not profile.perspective.is_valid
        assert profile.dominant_floor_direction == "unknown"

    def test_calibrates_with_enough_samples(self):
        builder = SceneProfileBuilder("cam1", 640, 480, refit_stride=1)
        # Simulate a flat-floor camera: person height roughly constant
        # regardless of row (weak/no real perspective slope, but enough
        # samples to reach calibration confidence).
        for i in range(30):
            y2 = 300 + (i % 5)
            builder.add_sample((100, y2 - 250, 160, y2), ankle_x=130)
        profile = builder.build_profile()
        assert profile.calibration_confidence > 0.0
        assert profile.size_stats.is_calibrated

    def test_normalize_height_neutral_when_uncalibrated(self):
        builder = SceneProfileBuilder("cam1", 640, 480, refit_stride=1)
        profile = builder.build_profile()
        # No samples at all -> must not invent a value, return neutral 1.0.
        assert profile.normalize_height(
            bbox_bottom_y=300, observed_height=250, bbox_center_x=130
        ) == 1.0

    def test_normalize_height_uses_perspective_when_valid(self):
        builder = SceneProfileBuilder(
            "cam1", 640, 480, refit_stride=1,
            cal=CalibrationConfig(min_samples_for_confidence=15),
        )
        # Perspective: height grows linearly with row (near bottom = closer = taller).
        # ankle_x held constant (no horizontal variance in these samples) --
        # the fit should gracefully degrade to a row-only model rather than
        # producing an unstable coef_h_x from an ill-conditioned column.
        for y2 in range(200, 460, 5):
            height = 100 + (y2 - 200) * 0.5
            builder.add_sample((100, y2 - height, 100 + height * 0.3, y2), ankle_x=130)
        profile = builder.build_profile()
        assert profile.perspective.is_valid
        assert profile.perspective.coef_h_x == 0.0
        # A person at y2=400 with the expected height for that row -> ratio ~1.0
        expected_height_at_400 = 100 + (400 - 200) * 0.5
        ratio = profile.normalize_height(
            bbox_bottom_y=400, observed_height=expected_height_at_400, bbox_center_x=130
        )
        assert 0.8 < ratio < 1.2
        # A much shorter box at the same row (e.g. lying down) -> ratio well below 1.0
        ratio_lying = profile.normalize_height(
            bbox_bottom_y=400, observed_height=expected_height_at_400 * 0.3, bbox_center_x=130
        )
        assert ratio_lying < 0.5

    def test_normalize_height_learns_real_horizontal_perspective(self):
        """The actual point of the 2D upgrade: a camera where expected
        height genuinely varies across columns too (e.g. a wide/fisheye-ish
        lens, or a camera angled so one side of frame is nearer than the
        other) -- a row-only fit would apply the SAME expected height to a
        near-column and far-column person at the same row, silently wrong
        for one of them. The fit must learn a real, non-zero coef_h_x and
        predict correctly at two different columns, same row."""
        builder = SceneProfileBuilder(
            "cam1", 640, 480, refit_stride=1,
            cal=CalibrationConfig(min_samples_for_confidence=15),
        )
        # height depends on BOTH row and column: taller near the bottom
        # AND taller toward the right column (simulating a camera where
        # the right side of frame is genuinely closer to the camera).
        for i in range(60):
            x = 50 + (i % 6) * 100
            y2 = 200 + (i % 8) * 30
            height = 80 + (y2 - 200) * 0.4 + (x - 50) * 0.3
            builder.add_sample((x - 30, y2 - height, x + 30, y2), ankle_x=x)
        profile = builder.build_profile()
        assert profile.perspective.is_valid
        assert profile.perspective.coef_h_x > 0.1, "should learn a real positive x-dependence"

        expected_left = 80 + (350 - 200) * 0.4 + (50 - 50) * 0.3
        expected_right = 80 + (350 - 200) * 0.4 + (550 - 50) * 0.3
        assert expected_right > expected_left  # sanity on the synthetic setup itself

        ratio_left = profile.normalize_height(
            bbox_bottom_y=350, observed_height=expected_left, bbox_center_x=50
        )
        ratio_right = profile.normalize_height(
            bbox_bottom_y=350, observed_height=expected_right, bbox_center_x=550
        )
        # Same real posture (each box matches ITS OWN column's expectation)
        # -> both should read as ~normal standing, despite one being a much
        # larger box in raw pixels than the other.
        assert 0.8 < ratio_left < 1.2
        assert 0.8 < ratio_right < 1.2

    def test_save_and_load_seed(self, tmp_path):
        builder = SceneProfileBuilder("cam_test", 640, 480, refit_stride=1)
        for i in range(20):
            builder.add_sample((100, 50, 160, 300), ankle_x=130)
        path = builder.save(directory=tmp_path)
        assert path.exists()

        seed = SceneProfileBuilder.load_seed("cam_test", directory=tmp_path)
        assert seed is not None
        assert seed["camera_id"] == "cam_test"

        missing = SceneProfileBuilder.load_seed("nonexistent_cam", directory=tmp_path)
        assert missing is None

    def test_floor_region_uncalibrated_below_min_samples(self):
        builder = SceneProfileBuilder("cam1", 640, 480, refit_stride=1)
        for _ in range(5):
            builder.add_sample((100, 50, 160, 300), ankle_x=130, ankle_y=300)
        profile = builder.build_profile()
        assert not profile.floor_region.is_valid
        assert profile.floor_region.contact_likelihood(300) == 0.0

    def test_floor_region_calibrates_with_varied_ankle_y(self):
        """Real-variance ankle y-positions (not all identical) — regression
        test for a numpy-bool-vs-Python-bool bug that only surfaced with
        actual variance and broke JSON serialization of the profile."""
        builder = SceneProfileBuilder("cam1", 640, 480, refit_stride=1)
        for i in range(30):
            y = 400 + (i % 7)  # small real variance around y=400
            builder.add_sample((100, y - 250, 160, y), ankle_x=130, ankle_y=y)
        profile = builder.build_profile()
        assert profile.floor_region.is_valid
        assert profile.floor_region.contact_likelihood(403) > 0.5  # center of the observed band
        assert profile.floor_region.contact_likelihood(50) < 0.2   # far from it
        # Must be plain JSON-serializable types, not numpy scalars.
        json.dumps(profile.to_dict())

    def test_floor_region_contact_likelihood_neutral_when_uncalibrated(self):
        builder = SceneProfileBuilder("cam1", 640, 480, refit_stride=1)
        profile = builder.build_profile()
        assert profile.floor_region.contact_likelihood(100) == 0.0
