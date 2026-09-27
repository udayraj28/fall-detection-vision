"""Tests for rolling person-size statistics."""

from src.config import CalibrationConfig, RollingSizeConfig
from src.person_size_stats import RollingPersonSizeStats, is_reliable_standing_sample


class TestIsReliableStandingSample:
    def setup_method(self):
        self.cal = CalibrationConfig()

    def test_accepts_good_standing_box(self):
        assert is_reliable_standing_sample(
            bbox=(100, 50, 160, 300), mean_keypoint_conf=0.8,
            num_critical_keypoints_visible=6, frame_width=640, frame_height=480, cal=self.cal,
        )

    def test_rejects_low_confidence(self):
        assert not is_reliable_standing_sample(
            bbox=(100, 50, 160, 300), mean_keypoint_conf=0.2,
            num_critical_keypoints_visible=6, frame_width=640, frame_height=480, cal=self.cal,
        )

    def test_rejects_truncated_box(self):
        assert not is_reliable_standing_sample(
            bbox=(0, 50, 160, 300), mean_keypoint_conf=0.8,
            num_critical_keypoints_visible=6, frame_width=640, frame_height=480, cal=self.cal,
        )

    def test_rejects_wide_short_box(self):
        """A fallen-looking (wide/short) box should never seed the standing baseline."""
        assert not is_reliable_standing_sample(
            bbox=(100, 200, 400, 260), mean_keypoint_conf=0.8,
            num_critical_keypoints_visible=6, frame_width=640, frame_height=480, cal=self.cal,
        )

    def test_rejects_too_few_critical_keypoints(self):
        assert not is_reliable_standing_sample(
            bbox=(100, 50, 160, 300), mean_keypoint_conf=0.8,
            num_critical_keypoints_visible=1, frame_width=640, frame_height=480, cal=self.cal,
        )


class TestRollingPersonSizeStats:
    def test_uncalibrated_below_min_samples(self):
        stats = RollingPersonSizeStats(RollingSizeConfig())
        for _ in range(5):
            stats.add_sample(width=60.0, height=180.0)
        snap = stats.snapshot(CalibrationConfig(min_samples_for_confidence=15))
        assert not snap.is_calibrated
        assert snap.calibration_confidence == 0.0

    def test_calibrates_after_enough_samples(self):
        stats = RollingPersonSizeStats(RollingSizeConfig())
        for _ in range(20):
            stats.add_sample(width=60.0, height=180.0)
        cal = CalibrationConfig(min_samples_for_confidence=15, full_confidence_samples=50)
        snap = stats.snapshot(cal)
        assert snap.is_calibrated
        assert snap.median_height == 180.0
        assert 0.0 < snap.calibration_confidence < 1.0

    def test_full_confidence_at_window_size(self):
        cal = CalibrationConfig(min_samples_for_confidence=15, full_confidence_samples=50)
        stats = RollingPersonSizeStats(RollingSizeConfig(window_size=50))
        for _ in range(50):
            stats.add_sample(width=60.0, height=180.0)
        snap = stats.snapshot(cal)
        assert snap.calibration_confidence == 1.0

    def test_rolling_stride_update_shifts_window(self):
        """After the window fills, values should update via stride-10 batches,
        not grow unboundedly."""
        cfg = RollingSizeConfig(window_size=50, update_stride=10)
        stats = RollingPersonSizeStats(cfg)
        for _ in range(50):
            stats.add_sample(width=60.0, height=180.0)
        # Push 10 new, very different samples — should fully replace the oldest 10.
        for _ in range(10):
            stats.add_sample(width=60.0, height=90.0)
        cal = CalibrationConfig(min_samples_for_confidence=15, full_confidence_samples=50)
        snap = stats.snapshot(cal)
        assert snap.sample_count == 50
        # Median should now reflect a mix, still bounded to the 50-sample window.
        assert snap.median_height <= 180.0

    def test_outlier_rejection(self):
        stats = RollingPersonSizeStats(RollingSizeConfig())
        for _ in range(19):
            stats.add_sample(width=60.0, height=180.0)
        stats.add_sample(width=60.0, height=5000.0)  # extreme outlier
        cal = CalibrationConfig(min_samples_for_confidence=15)
        snap = stats.snapshot(cal)
        assert snap.median_height == 180.0
        assert snap.p75_height < 1000.0

    def test_thread_safety_smoke(self):
        """Concurrent add_sample calls should not corrupt internal state."""
        import threading

        stats = RollingPersonSizeStats(RollingSizeConfig(window_size=50, update_stride=10))

        def worker():
            for _ in range(100):
                stats.add_sample(width=60.0, height=180.0)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert stats.total_samples_seen == 400
        snap = stats.snapshot(CalibrationConfig(min_samples_for_confidence=15))
        assert snap.sample_count == 50
