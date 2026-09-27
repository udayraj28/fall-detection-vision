"""Tests for monocular 2D->3D pose lifting (src/pose_3d.py)."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from src.pose_3d import Pose3DEstimator


@dataclass
class _FakeLandmark:
    x: float
    y: float
    z: float
    visibility: float


def _make_estimator_with_fake_result(pose_world_landmarks, asymmetry_threshold=0.3):
    """A Pose3DEstimator whose `_landmarker.detect()` is monkeypatched to
    return a controllable fake result, so the per-side/monotonic-max
    logic in `estimate()` can be tested precisely without a real image or
    the real MediaPipe model."""
    estimator = Pose3DEstimator(
        model_path="/does/not/exist.task", asymmetry_threshold=asymmetry_threshold
    )
    landmarks = [_FakeLandmark(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    for idx, lm in pose_world_landmarks.items():
        landmarks[idx] = lm
    fake_result = SimpleNamespace(pose_world_landmarks=[landmarks])
    estimator._landmarker = SimpleNamespace(detect=lambda image: fake_result)
    return estimator


class TestPose3DPerSideMonotonicMax:
    """V6 Fix Set A (§2.2): when one side's shoulder+hip landmarks are
    confidently more visible than the other's, that side's own angle may
    only RAISE torso_angle_3d relative to the bilateral (midpoint)
    reading -- never lower it. This is what makes the fix provably
    monotonic w.r.t. every existing confirmed case."""

    def _estimate(self, l_sh, r_sh, l_hip, r_hip, asymmetry_threshold=0.3):
        estimator = _make_estimator_with_fake_result(
            {11: l_sh, 12: r_sh, 23: l_hip, 24: r_hip}, asymmetry_threshold=asymmetry_threshold
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return estimator.estimate(frame, (100.0, 100.0, 300.0, 460.0))

    def test_symmetric_visibility_uses_bilateral_only_unchanged(self):
        """Both sides equally (confidently) visible and posed
        symmetrically — behavior must be byte-for-byte the pre-V6
        bilateral-only calculation (no regression risk for the common
        case)."""
        # A body lying flat toward the camera: torso vector mostly along
        # Z (depth), small Y — real large angle from vertical.
        l_sh = _FakeLandmark(x=-0.1, y=0.0, z=0.0, visibility=0.9)
        r_sh = _FakeLandmark(x=0.1, y=0.0, z=0.0, visibility=0.9)
        l_hip = _FakeLandmark(x=-0.1, y=0.05, z=0.5, visibility=0.9)
        r_hip = _FakeLandmark(x=0.1, y=0.05, z=0.5, visibility=0.9)
        result = self._estimate(l_sh, r_sh, l_hip, r_hip)
        assert result is not None
        # Bilateral-only expectation: torso_vec = hip_mid - sh_mid = (0, 0.05, 0.5)
        expected_angle = float(
            np.clip(np.arctan2(np.hypot(0.0, 0.5), abs(0.05)) / (np.pi / 2), 0.0, 1.0)
        )
        assert abs(result.torso_angle_3d - expected_angle) < 1e-9

    def test_asymmetric_visibility_can_only_raise_the_angle(self):
        """Real §7.1 shape: a raised knee shifts the RIGHT hip toward the
        torso in the bilateral average, artificially deflating the
        bilateral angle below what the confidently-visible LEFT side
        alone (genuinely lying) shows. The per-side reading must win via
        max(), raising torso_angle_3d above the corrupted bilateral value."""
        l_sh = _FakeLandmark(x=-0.1, y=0.0, z=0.0, visibility=0.95)
        r_sh = _FakeLandmark(x=0.1, y=0.0, z=0.0, visibility=0.55)
        # Left hip: genuinely far out (flat lying leg). Right hip: pulled
        # up/close to the shoulder by a raised knee -- corrupts the
        # bilateral midpoint toward "upright".
        l_hip = _FakeLandmark(x=-0.1, y=0.05, z=0.6, visibility=0.95)
        r_hip = _FakeLandmark(x=0.1, y=0.02, z=0.05, visibility=0.55)
        result = self._estimate(l_sh, r_sh, l_hip, r_hip)
        assert result is not None

        bilateral_sh = np.array([0.0, 0.0, 0.0])
        bilateral_hip = np.array([0.0, 0.035, 0.325])
        bilateral_vec = bilateral_hip - bilateral_sh
        bilateral_angle = float(
            np.clip(
                np.arctan2(np.hypot(bilateral_vec[0], bilateral_vec[2]), abs(bilateral_vec[1]))
                / (np.pi / 2), 0.0, 1.0,
            )
        )
        left_vec = np.array([l_hip.x - l_sh.x, l_hip.y - l_sh.y, l_hip.z - l_sh.z])
        left_angle = float(
            np.clip(
                np.arctan2(np.hypot(left_vec[0], left_vec[2]), abs(left_vec[1])) / (np.pi / 2),
                0.0, 1.0,
            )
        )
        assert left_angle > bilateral_angle, "test fixture must actually be corrupted"
        assert abs(result.torso_angle_3d - left_angle) < 1e-9
        assert result.torso_angle_3d >= bilateral_angle

    def test_asymmetric_but_below_threshold_stays_bilateral(self):
        """A visibility gap that doesn't clear `asymmetry_threshold` must
        not switch to the single-side reading — avoids over-triggering
        the correction on ordinary, mild visibility noise."""
        l_sh = _FakeLandmark(x=-0.1, y=0.0, z=0.0, visibility=0.85)
        r_sh = _FakeLandmark(x=0.1, y=0.0, z=0.0, visibility=0.75)
        l_hip = _FakeLandmark(x=-0.1, y=0.05, z=0.5, visibility=0.85)
        r_hip = _FakeLandmark(x=0.1, y=0.05, z=0.5, visibility=0.75)
        result = self._estimate(l_sh, r_sh, l_hip, r_hip, asymmetry_threshold=0.3)
        expected_angle = float(
            np.clip(np.arctan2(np.hypot(0.0, 0.5), abs(0.05)) / (np.pi / 2), 0.0, 1.0)
        )
        assert abs(result.torso_angle_3d - expected_angle) < 1e-9

    def test_asymmetric_standing_person_stays_low(self):
        """A genuinely upright, standing person with an incidentally
        asymmetric visibility reading (e.g. one arm partially self-
        occluding a hip) must NOT be pushed toward "lying" by this fix —
        both the bilateral AND single-side readings are low for a real
        standing posture, so max() correctly stays low too."""
        l_sh = _FakeLandmark(x=-0.1, y=-0.5, z=0.0, visibility=0.95)
        r_sh = _FakeLandmark(x=0.1, y=-0.5, z=0.0, visibility=0.5)
        l_hip = _FakeLandmark(x=-0.1, y=0.0, z=0.02, visibility=0.95)
        r_hip = _FakeLandmark(x=0.1, y=0.0, z=0.02, visibility=0.5)
        result = self._estimate(l_sh, r_sh, l_hip, r_hip)
        assert result is not None
        assert result.torso_angle_3d < 0.2


class TestPose3DEstimatorGracefulDegradation:
    """Never a hard dependency — missing model file (or, by the same
    code path, a missing mediapipe install) must disable lifting cleanly,
    not raise, so the rest of the pipeline runs exactly as it did before
    this feature existed."""

    def test_missing_model_file_disables_cleanly(self, tmp_path):
        estimator = Pose3DEstimator(model_path=tmp_path / "does_not_exist.task")
        assert not estimator.is_available

    def test_estimate_returns_none_when_unavailable(self, tmp_path):
        estimator = Pose3DEstimator(model_path=tmp_path / "does_not_exist.task")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert estimator.estimate(frame, (100.0, 100.0, 200.0, 300.0)) is None

    def test_degenerate_bbox_returns_none_without_crashing(self, tmp_path):
        """A zero-area or out-of-frame bbox (e.g. a stale/corrupted
        detection) must not raise — cheap to check before ever touching
        the (possibly unavailable) landmarker."""
        estimator = Pose3DEstimator(model_path=tmp_path / "does_not_exist.task")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert estimator.estimate(frame, (100.0, 100.0, 100.0, 100.0)) is None
        assert estimator.estimate(frame, (-50.0, -50.0, -10.0, -10.0)) is None


class TestPose3DEstimatorRealModel:
    """Exercises the actual downloaded model + mediapipe install this
    project's own dev environment has. Skips itself (rather than failing)
    on a machine where the model asset hasn't been fetched — this file's
    presence is an environment/deployment concern, not a code-correctness
    one this suite should block on."""

    def setup_method(self):
        self.estimator = Pose3DEstimator()
        if not self.estimator.is_available:
            import pytest
            pytest.skip("pose_landmarker model asset not present in this environment")

    def test_blank_frame_returns_none(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert self.estimator.estimate(frame, (100.0, 100.0, 300.0, 460.0)) is None

    def test_real_standing_crop_reads_low_angle(self):
        """A real, clearly-standing person (from this repo's own
        validation footage) must read a LOW torso_angle_3d — the same
        sanity check that caught the crop-padding requirement during
        development."""
        import cv2

        cap = cv2.VideoCapture("input/mandown_2.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame = cap.read()
        cap.release()
        assert ret, "mandown_2.mp4 must be present for this repo's real-video validation"

        from src.detector import PersonDetector

        det = PersonDetector(confidence_threshold=0.3, imgsz=960)
        result = det.detect_frame(frame, frame_num=30)
        standing = max(result.persons, key=lambda p: p.bbox[3] - p.bbox[1])

        pose_3d = self.estimator.estimate(frame, standing.bbox)
        assert pose_3d is not None
        assert pose_3d.torso_angle_3d < 0.35


class TestBodyAxisAngle3D:
    """V6.1 (Objective 1): `body_axis_angle_3d` -- a long-baseline
    shoulder->ankle angle on ONE side, anchored on whichever leg the
    CALLER has already confirmed via 2D keypoint confidence is the
    flat/extended one, so it never asks the hip landmark anything (the
    landmark V6's per-side torso_angle_3d correction found MediaPipe
    biases for an unusual-but-valid raised-knee configuration, without
    that bias ever showing up as a detectable visibility asymmetry).

    Mandatory negative controls (per the V6.1 spec) BEFORE this could be
    wired into fusion at all: re-run the real V4.7/V4.8.1 crouch/deep-bend
    regression geometry through this new feature and confirm it stays
    safely low -- these fixtures mirror the exact real postures at the
    center of those two reverted/hard-won fixes.
    """

    def _estimate_with_ankle(
        self, l_sh, r_sh, l_hip, r_hip, l_ankle, r_ankle, dominant_leg_side="L"
    ):
        estimator = _make_estimator_with_fake_result(
            {11: l_sh, 12: r_sh, 23: l_hip, 24: r_hip, 27: l_ankle, 28: r_ankle}
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return estimator.estimate(frame, (100.0, 100.0, 300.0, 460.0), dominant_leg_side=dominant_leg_side)

    def test_no_dominant_side_gives_no_reading(self):
        """When the caller has no confident 2D-only dominant leg (e.g.
        neither leg's keypoints cleared confidence), no body_axis_angle_3d
        reading is fabricated."""
        l_sh = _FakeLandmark(-0.1, -0.5, 0.0, 0.9)
        r_sh = _FakeLandmark(0.1, -0.5, 0.0, 0.9)
        l_hip = _FakeLandmark(-0.1, 0.0, 0.0, 0.9)
        r_hip = _FakeLandmark(0.1, 0.0, 0.0, 0.9)
        l_ankle = _FakeLandmark(-0.1, 0.5, 0.0, 0.9)
        r_ankle = _FakeLandmark(0.1, 0.5, 0.0, 0.9)
        result = self._estimate_with_ankle(
            l_sh, r_sh, l_hip, r_hip, l_ankle, r_ankle, dominant_leg_side=None
        )
        assert result is not None
        assert result.body_axis_angle_3d == 0.0
        assert result.body_axis_angle_3d_evidence_quality == 0.0

    def test_real_v47_crouch_geometry_stays_low(self):
        """Mirrors `crouching_keypoints()`'s real 2D geometry
        (tests/test_already_lying_integration.py): hip/knee/ankle nearly
        collinear and directly below the shoulder (a compact, vertical
        crouch, not a forward pitch) -- the long shoulder->ankle baseline
        must stay well below the lying floor, the same way the 2D fixture
        reads torso_angle~0."""
        l_sh = _FakeLandmark(-0.1, -0.55, 0.0, 0.95)
        r_sh = _FakeLandmark(0.1, -0.55, 0.0, 0.95)
        l_hip = _FakeLandmark(-0.1, -0.15, 0.02, 0.9)
        r_hip = _FakeLandmark(0.1, -0.15, 0.02, 0.9)
        l_ankle = _FakeLandmark(-0.1, 0.35, 0.03, 0.85)
        r_ankle = _FakeLandmark(0.1, 0.35, 0.03, 0.85)
        result = self._estimate_with_ankle(
            l_sh, r_sh, l_hip, r_hip, l_ankle, r_ankle, dominant_leg_side="L"
        )
        assert result is not None
        assert result.body_axis_angle_3d < 0.3

    def test_real_v481_deep_bend_geometry_stays_low(self):
        """Mirrors the REAL V4.8.1 deep-bend regression (straight,
        planted legs; only the torso pitches forward) -- torso_angle_3d
        reads high here (the real regression this exact case caused), but
        body_axis_angle_3d must stay well below the lying floor: the
        ankle stays anchored straight down from the hip (legs never bend
        or lift in a hip-hinge bend), so the long shoulder-to-ankle
        baseline is far less affected by the torso's forward pitch than
        the short torso vector is."""
        l_sh = _FakeLandmark(-0.1, -0.30, 0.45, 0.95)
        r_sh = _FakeLandmark(0.1, -0.30, 0.45, 0.95)
        l_hip = _FakeLandmark(-0.1, 0.0, 0.0, 0.9)
        r_hip = _FakeLandmark(0.1, 0.0, 0.0, 0.9)
        l_ankle = _FakeLandmark(-0.1, 0.85, 0.0, 0.85)
        r_ankle = _FakeLandmark(0.1, 0.85, 0.0, 0.85)
        result = self._estimate_with_ankle(
            l_sh, r_sh, l_hip, r_hip, l_ankle, r_ankle, dominant_leg_side="R"
        )
        assert result is not None
        # Sanity: torso_angle_3d itself DOES read high for this fixture
        # (confirming this is a faithful reproduction of the real
        # regression geometry) -- the new feature's safety must not rest
        # on an unrealistic fixture that doesn't even trigger the danger
        # this negative control exists to check.
        assert result.torso_angle_3d >= 0.5
        assert result.body_axis_angle_3d < 0.3

    def test_low_visibility_ankle_gives_no_reading(self):
        """The chosen side's ankle visibility below min_visibility must
        not be trusted -- an unreliable 3D reading is worse than none."""
        l_sh = _FakeLandmark(-0.1, -0.5, 0.0, 0.9)
        r_sh = _FakeLandmark(0.1, -0.5, 0.0, 0.9)
        l_hip = _FakeLandmark(-0.1, 0.0, 0.0, 0.9)
        r_hip = _FakeLandmark(0.1, 0.0, 0.0, 0.9)
        l_ankle = _FakeLandmark(-0.1, 0.02, 0.5, 0.2)  # low visibility
        r_ankle = _FakeLandmark(0.1, 0.5, 0.0, 0.9)
        result = self._estimate_with_ankle(
            l_sh, r_sh, l_hip, r_hip, l_ankle, r_ankle, dominant_leg_side="L"
        )
        assert result is not None
        assert result.body_axis_angle_3d == 0.0
        assert result.body_axis_angle_3d_evidence_quality == 0.0

    def test_genuine_foreshortened_lying_on_dominant_side_reads_high(self):
        """Positive case: a genuinely foreshortened-lying body where the
        dominant (flat/extended) leg's ankle sits far in Z (depth) from
        the shoulder, with little Y drop -- the long baseline correctly
        reads as lying."""
        l_sh = _FakeLandmark(-0.1, -0.05, 0.0, 0.95)
        r_sh = _FakeLandmark(0.1, -0.05, 0.0, 0.95)
        l_hip = _FakeLandmark(-0.1, 0.0, 0.3, 0.9)
        r_hip = _FakeLandmark(0.1, 0.02, 0.05, 0.55)  # the "raised knee" (corrupted) side
        l_ankle = _FakeLandmark(-0.1, 0.03, 0.75, 0.9)
        r_ankle = _FakeLandmark(0.1, 0.3, 0.2, 0.5)
        result = self._estimate_with_ankle(
            l_sh, r_sh, l_hip, r_hip, l_ankle, r_ankle, dominant_leg_side="L"
        )
        assert result is not None
        assert result.body_axis_angle_3d >= 0.55
