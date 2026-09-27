"""Tests for the direct feature-based Mandown scoring logic."""

from collections import deque

import numpy as np

from src.config import FeatureMandownConfig, KneelingConfig
from src.feature_extraction import ExtendedFeatureExtractor, ExtendedFeatureVector, FeatureVector
from src.feature_mandown import FeatureMandownScorer


def _base(
    body_angle=0.05, cog_height=0.5, vertical_velocity=0.0,
    bbox_aspect_ratio=0.35, keypoint_confidence=0.9,
):
    return FeatureVector(
        body_angle, cog_height, vertical_velocity, bbox_aspect_ratio, keypoint_confidence
    )


def _ext(torso_angle=0.05, hip_angle=0.05, shoulder_angle=0.05, head_height=0.1, knee_height=0.8,
         horizontal_velocity=0.0, acceleration=0.0, angle_change=0.0, body_height_norm=1.0,
         shoulder_hip_distance=0.4, leg_extension_ratio=1.0, leg_evidence_quality=1.0,
         num_critical=6, pose_quality=0.9, body_axis_angle=0.0, body_axis_evidence_quality=0.0,
         torso_angle_evidence_quality=1.0, hip_angle_evidence_quality=1.0,
         torso_angle_3d=0.0, torso_angle_3d_evidence_quality=0.0):
    return ExtendedFeatureVector(
        torso_angle=torso_angle, hip_angle=hip_angle, shoulder_angle=shoulder_angle,
        head_height=head_height, knee_height=knee_height, horizontal_velocity=horizontal_velocity,
        acceleration=acceleration, angle_change=angle_change, body_height_norm=body_height_norm,
        shoulder_hip_distance=shoulder_hip_distance, leg_extension_ratio=leg_extension_ratio,
        leg_evidence_quality=leg_evidence_quality,
        num_critical_keypoints_visible=num_critical, pose_quality=pose_quality,
        body_axis_angle=body_axis_angle, body_axis_evidence_quality=body_axis_evidence_quality,
        torso_angle_evidence_quality=torso_angle_evidence_quality,
        hip_angle_evidence_quality=hip_angle_evidence_quality,
        torso_angle_3d=torso_angle_3d, torso_angle_3d_evidence_quality=torso_angle_3d_evidence_quality,
    )


class TestFeatureMandownScorer:
    def setup_method(self):
        self.scorer = FeatureMandownScorer(FeatureMandownConfig())

    def test_standing_scores_low(self):
        base = _base()
        ext = _ext()
        base_hist = deque([base] * 15, maxlen=30)
        ext_hist = deque([ext] * 15, maxlen=30)
        scores = self.scorer.score(base, ext, base_hist, ext_hist)
        assert scores.mandown_confidence < 0.4
        assert scores.down_score < 0.4
        assert scores.floor_proximity_score < 0.4

    def test_lying_scores_high_even_without_motion_history(self):
        """Case B: already-down person, no fall transition in history at all."""
        base = _base(body_angle=0.9, cog_height=0.85, vertical_velocity=0.0, bbox_aspect_ratio=2.5)
        ext = _ext(
            torso_angle=0.9, hip_angle=0.85, head_height=0.75,
            knee_height=0.78, body_height_norm=0.35,
        )
        base_hist = deque([base] * 30, maxlen=30)  # static — no motion at all
        ext_hist = deque([ext] * 30, maxlen=30)
        scores = self.scorer.score(base, ext, base_hist, ext_hist)
        assert scores.down_score > 0.6
        assert scores.floor_proximity_score > 0.6
        assert scores.persistence_score > 0.8
        assert scores.mandown_confidence > 0.5

    def test_active_fall_transition_scores_high_motion(self):
        """Case A: mid-fall, still upright-ish posture but a sharp recent
        velocity/angle spike should drive fall_motion_score up."""
        base = _base()
        ext = _ext()
        base_hist = deque([_base()] * 10, maxlen=30)
        base_hist.append(_base(vertical_velocity=0.5))
        ext_hist = deque([_ext()] * 10, maxlen=30)
        ext_hist.append(_ext(angle_change=0.4, acceleration=0.3))
        scores = self.scorer.score(base, ext, base_hist, ext_hist)
        assert scores.fall_motion_score > 0.5

    def test_low_pose_quality_dampens_confidence(self):
        base = _base(bbox_aspect_ratio=2.5, body_angle=0.9, cog_height=0.85)
        ext_good = _ext(torso_angle=0.9, hip_angle=0.85, head_height=0.75, knee_height=0.78,
                         body_height_norm=0.35, pose_quality=0.95, num_critical=6)
        ext_bad = _ext(torso_angle=0.9, hip_angle=0.85, head_height=0.75, knee_height=0.78,
                        body_height_norm=0.35, pose_quality=0.1, num_critical=1)
        base_hist = deque([base] * 30, maxlen=30)
        good = self.scorer.score(base, ext_good, base_hist, deque([ext_good] * 30, maxlen=30))
        bad = self.scorer.score(base, ext_bad, base_hist, deque([ext_bad] * 30, maxlen=30))
        assert bad.mandown_confidence < good.mandown_confidence
        # Dampened, not zeroed — never fabricate certainty either way from bad pose.
        assert bad.mandown_confidence > 0.0

    def test_empty_history_does_not_crash(self):
        base = _base()
        ext = _ext()
        scores = self.scorer.score(base, ext, deque(), deque())
        assert 0.0 <= scores.mandown_confidence <= 1.0

    def test_no_single_feature_alone_dominates(self):
        """A large body_angle alone (e.g. camera angle quirk) with otherwise
        normal geometry/motion should not alone push confidence high."""
        base = _base(body_angle=0.9, bbox_aspect_ratio=0.35, cog_height=0.5)
        ext = _ext(
            torso_angle=0.9, hip_angle=0.1, head_height=0.1,
            knee_height=0.8, body_height_norm=1.0,
        )
        base_hist = deque([base] * 30, maxlen=30)
        ext_hist = deque([ext] * 30, maxlen=30)
        scores = self.scorer.score(base, ext, base_hist, ext_hist)
        assert scores.mandown_confidence < 0.6


class TestKneelingDiscrimination:
    """Regression coverage for the officers-kneeling-beside-a-downed-person
    false positive found in real-footage testing (M_D_2.mp4)."""

    def setup_method(self):
        self.scorer = FeatureMandownScorer(FeatureMandownConfig(), KneelingConfig())

    def test_kneeling_upright_suppresses_down_score(self):
        """Bent legs + upright torso (classic kneeling) should score far
        lower than the same posture with extended legs."""
        base = _base(bbox_aspect_ratio=1.2, body_angle=0.5, cog_height=0.7)
        kneeling = _ext(
            torso_angle=0.3, hip_angle=0.3, head_height=0.4, knee_height=0.6,
            body_height_norm=0.6, leg_extension_ratio=0.5,
        )
        standing_equivalent = _ext(
            torso_angle=0.3, hip_angle=0.3, head_height=0.4, knee_height=0.6,
            body_height_norm=0.6, leg_extension_ratio=1.0,
        )
        hist = deque([base] * 30, maxlen=30)
        kneeling_scores = self.scorer.score(base, kneeling, hist, deque([kneeling] * 30, maxlen=30))
        extended_scores = self.scorer.score(
            base, standing_equivalent, hist, deque([standing_equivalent] * 30, maxlen=30)
        )
        assert kneeling_scores.kneeling_penalty > 0.5
        assert kneeling_scores.down_score < extended_scores.down_score

    def test_kneeling_leaning_forward_still_partially_suppressed(self):
        """Regression: a kneeling person leaning sharply forward reads a
        high torso_angle too, not just genuine lying. The gate floor
        (`min_gate`) must keep some suppression even then — this is exactly
        the case that slipped through before `min_gate` was added."""
        base = _base(bbox_aspect_ratio=1.5, body_angle=0.8, cog_height=0.75)
        bent = _ext(
            torso_angle=0.8, hip_angle=0.6, head_height=0.55, knee_height=0.65,
            body_height_norm=0.4, leg_extension_ratio=0.5,
        )
        extended = _ext(
            torso_angle=0.8, hip_angle=0.6, head_height=0.55, knee_height=0.65,
            body_height_norm=0.4, leg_extension_ratio=1.0,
        )
        hist = deque([base] * 30, maxlen=30)
        bent_scores = self.scorer.score(base, bent, hist, deque([bent] * 30, maxlen=30))
        extended_scores = self.scorer.score(base, extended, hist, deque([extended] * 30, maxlen=30))
        assert bent_scores.kneeling_penalty > 0.0
        assert bent_scores.down_score < extended_scores.down_score

    def test_lying_with_bent_knee_not_fully_suppressed(self):
        """A genuinely lying person (torso flat) with one bent knee (e.g.
        recovery position) should still score meaningfully — the gate
        should be near its weakest here, not near its strongest."""
        base = _base(bbox_aspect_ratio=2.5, body_angle=0.95, cog_height=0.85)
        ext = _ext(
            torso_angle=0.95, hip_angle=0.9, head_height=0.75, knee_height=0.78,
            body_height_norm=0.35, leg_extension_ratio=0.5,
        )
        hist = deque([base] * 30, maxlen=30)
        scores = self.scorer.score(base, ext, hist, deque([ext] * 30, maxlen=30))
        assert scores.mandown_confidence > 0.4

    def test_side_lying_bent_knees_strongly_down_and_not_kneeling(self):
        """V4: a flat torso with a moderately bent knee (natural side-lying,
        e.g. hip_angle reduced by a folded thigh) must read as strongly
        "down" via the torso-only orientation cue AND stay correctly
        unsuppressed by the kneeling gate — the two-part proof that
        `_orientation_cue` fixes the side-lying-with-bent-knees weakness
        without reopening the kneeling/sitting/crouching false-positive
        class `_kneeling_penalty` already protects against."""
        base = _base(bbox_aspect_ratio=2.2, body_angle=0.9, cog_height=0.8)
        ext = _ext(
            torso_angle=0.95, hip_angle=0.55, head_height=0.7, knee_height=0.75,
            body_height_norm=0.4, leg_extension_ratio=0.6, leg_evidence_quality=1.0,
        )
        hist = deque([base] * 30, maxlen=30)
        scores = self.scorer.score(base, ext, hist, deque([ext] * 30, maxlen=30))
        assert scores.down_score > 0.6
        assert scores.kneeling_penalty < 0.2

    def test_foreshortened_fall_with_unevidenced_legs_relents_kneeling_penalty(self):
        """V4.6 regression test for a real RTSP miss: a person fallen toward
        the camera, legs unevidenced (ankle confidence too low to trust —
        the real miss measured ~0.05-0.09), was never flagged. torso_angle
        reads ~0.02 (foreshortened, no lateral spread to measure) — the
        SAME reading a genuinely upright/kneeling person facing the camera
        produces — so `upright_gate` applied the bent-leg-fallback penalty
        at near-full strength (measured on the real miss: 0.28, cutting
        down_score from ~0.70 to 0.50) purely from that ambiguous angle.
        Real per-camera-calibrated body_height_norm (0.34, i.e. this
        person's bbox is a third of this camera's expected standing height
        at that image row) leaves no doubt this is a genuine collapse, not
        kneeling — the penalty must relent given that corroboration."""
        base = _base(bbox_aspect_ratio=1.0, cog_height=0.99)
        ext = _ext(
            torso_angle=0.024, hip_angle=0.0, head_height=0.816, knee_height=0.8,
            body_height_norm=0.34, leg_evidence_quality=0.0, leg_extension_ratio=1.0,
        )
        hist = deque([base] * 30, maxlen=30)
        scores = self.scorer.score(base, ext, hist, deque([ext] * 30, maxlen=30))
        assert scores.kneeling_penalty < 0.15
        assert scores.down_score > 0.55
        assert scores.floor_proximity_score > 0.65

    def test_upright_facing_camera_with_moderate_collapse_not_relented(self):
        """Adversarial case the floor above must still exclude: torso_angle
        near-zero (facing the camera, same ambiguous reading as the real
        miss above) with a large head-to-hip gap (torso clearly extends a
        normal vertical distance — the fallback's own signal that this is
        NOT a collapsed torso) and only moderately reduced body_height_norm
        (0.65, below the relent floor — ordinary standing-close-to-camera
        or mild crouching variance, nowhere near a genuine fall's degree).
        Must stay suppressed, not relented."""
        base = _base(bbox_aspect_ratio=1.0, cog_height=0.6)
        ext = _ext(
            torso_angle=0.02, hip_angle=0.0, head_height=0.15, knee_height=0.8,
            body_height_norm=0.65, leg_evidence_quality=0.0, leg_extension_ratio=1.0,
        )
        hist = deque([base] * 30, maxlen=30)
        scores = self.scorer.score(base, ext, hist, deque([ext] * 30, maxlen=30))
        assert scores.kneeling_penalty > 0.5

    def test_pose3d_confident_lying_relents_kneeling_penalty_alone(self):
        """V4.8: monocular 3D lifting can relent this penalty independent
        of the V4.6 2D-angle+height_collapse combination — torso_angle_3d
        =0.857 is the REAL value measured (via a real MediaPipe pose-
        lifting run on real footage) for a clean, flat, real fall in
        mandown_2.mp4. V4.8.1: still requires height_collapse_cue to
        confirm a genuine collapse (torso_angle_3d alone doesn't
        distinguish a deep bend from lying — see fusion.py's V4.8.1) —
        body_height_norm=0.3 here is a realistic real-lying-person
        reading (a real floor-level fall measured ~0.34 this session),
        not the deliberately-uncorroborated 1.0 an earlier version of
        this test used."""
        base = _base(bbox_aspect_ratio=1.0, cog_height=0.6)
        ext = _ext(
            torso_angle=0.02, hip_angle=0.0, head_height=0.15, knee_height=0.8,
            body_height_norm=0.3, leg_evidence_quality=0.0, leg_extension_ratio=1.0,
            torso_angle_3d=0.857, torso_angle_3d_evidence_quality=0.95,
        )
        hist = deque([base] * 30, maxlen=30)
        scores = self.scorer.score(base, ext, hist, deque([ext] * 30, maxlen=30))
        # Same fixture minus torso_angle_3d (see
        # test_upright_facing_camera_with_moderate_collapse_not_relented)
        # reads kneeling_penalty ~0.92 (unrelented, raw*upright_gate~1.0).
        # Relented here to raw*min_gate (~0.917*0.35~0.32) -- a clear,
        # large drop, not a tight arbitrary number.
        assert scores.kneeling_penalty < 0.35

    def test_pose3d_bend_with_bent_knees_not_relented_below_height_gate(self):
        """V4.8.1 regression test: a moderate-to-deep forward bend WITH
        bent knees (e.g. crouching to reach something) — bent knees
        already drive the raw bend signal high (as intended, real
        kneeling/crouching evidence), and BOTH torso_angle (0.5) and
        torso_angle_3d (0.6) read as real, substantial tilt (a real bend
        genuinely tilts the torso in true 3D, not just 2D projection —
        the live regression this session found measured torso_angle_3d
        =0.777 for a bend, almost identical to its own 2D torso_angle
        =0.778). Without the V4.8.1 height_collapse_cue requirement, the
        3D relent would fire here too (torso_angle_3d clears its 0.55
        floor) and wrongly pull the gate down to min_gate even though
        this posture never brought the hip toward the floor
        (body_height_norm=1.0, height_collapse_cue~0.15, well below the
        0.7 floor that requirement checks) — exactly the same bend-vs-
        lying ambiguity torso_angle_3d cannot resolve alone."""
        base = _base(bbox_aspect_ratio=1.0, cog_height=0.6)
        ext = _ext(
            torso_angle=0.5, hip_angle=0.0, head_height=0.3, knee_height=0.6,
            body_height_norm=1.0, leg_evidence_quality=1.0, leg_extension_ratio=0.5,
            torso_angle_3d=0.6, torso_angle_3d_evidence_quality=0.95,
        )
        hist = deque([base] * 30, maxlen=30)
        scores = self.scorer.score(base, ext, hist, deque([ext] * 30, maxlen=30))
        # Unrelented (fixed behavior): raw(~0.985) * upright_gate(~0.5) ~
        # 0.49. Wrongly relented (pre-fix bug): raw * min_gate(0.35) ~
        # 0.34. The gap is decisive enough to tell the two apart.
        assert scores.kneeling_penalty > 0.4

    def test_pose3d_low_angle_does_not_relent_crouching(self):
        """The exact case an earlier, reverted 2D-only attempt (V4.7)
        couldn't safely exclude: crouching, where the torso really IS
        vertical in 3D even though bent knees shorten the observed bbox.
        A low torso_angle_3d (confidently NOT lying) must not relent the
        penalty, even with a confident evidence_quality."""
        base = _base(bbox_aspect_ratio=1.0, cog_height=0.6)
        ext = _ext(
            torso_angle=0.02, hip_angle=0.0, head_height=0.15, knee_height=0.8,
            body_height_norm=0.65, leg_evidence_quality=0.0, leg_extension_ratio=1.0,
            torso_angle_3d=0.1, torso_angle_3d_evidence_quality=0.95,
        )
        hist = deque([base] * 30, maxlen=30)
        scores = self.scorer.score(base, ext, hist, deque([ext] * 30, maxlen=30))
        assert scores.kneeling_penalty > 0.5


class TestOrientationCue:
    """V4: orientation-invariant lying geometry — the torso-only,
    boundary-sigmoid `_orientation_cue` and its `body_axis_angle`
    corroboration."""

    def setup_method(self):
        self.scorer = FeatureMandownScorer(FeatureMandownConfig(), KneelingConfig())

    def test_down_score_rises_with_torso_angle_past_upright(self):
        """down_score should climb sharply once torso_angle is clearly past
        upright (30/45/60/90 deg). hip_angle tracks torso_angle closely
        here (a real diagonally-lying body tilts as one consistent unit)
        rather than a low, fixed value, simply to keep this fixture
        representative of that case -- note hip_angle isn't actually
        consulted by down_score/_orientation_cue at all (an attempt to add
        a torso-hip-mismatch bend signal, V4.9, was reverted after it also
        suppressed a real M_D_2.mp4 true positive; see config.py)."""
        base = _base(bbox_aspect_ratio=1.0, body_angle=0.5, cog_height=0.6)
        prev_down = None
        for torso_angle in (0.33, 0.5, 0.67, 1.0):  # ~30/45/60/90 deg
            ext = _ext(
                torso_angle=torso_angle, hip_angle=torso_angle * 0.85, head_height=0.3,
                knee_height=0.7, body_height_norm=0.6,
            )
            hist = deque([base] * 30, maxlen=30)
            scores = self.scorer.score(base, ext, hist, deque([ext] * 30, maxlen=30))
            if torso_angle >= 0.5:
                assert scores.down_score > 0.55, f"torso_angle={torso_angle}: {scores.down_score}"
            if prev_down is not None:
                assert scores.down_score >= prev_down - 1e-6
            prev_down = scores.down_score

    def test_floor_proximity_corroborated_by_body_axis_angle(self):
        """A diagonally-lying body's image-Y-projection head/knee span stays
        large (the old, sole signal), but body_axis_angle corroboration
        should still raise floor_proximity_score once the torso itself
        shows non-trivial lean."""
        ext_no_axis_evidence = _ext(
            torso_angle=0.7, head_height=0.3, knee_height=0.75,
            body_axis_angle=0.0, body_axis_evidence_quality=0.0,
        )
        ext_with_axis_evidence = _ext(
            torso_angle=0.7, head_height=0.3, knee_height=0.75,
            body_axis_angle=0.8, body_axis_evidence_quality=1.0,
        )
        cue_no_axis = self.scorer._floor_proximity_score(ext_no_axis_evidence)
        cue_with_axis = self.scorer._floor_proximity_score(ext_with_axis_evidence)
        assert cue_with_axis >= cue_no_axis

    def test_noisy_axis_alone_cannot_override_upright_torso(self):
        """The corroboration gate must never let a single noisy/occlusion-
        affected body_axis_angle push a clearly-upright torso toward "down"
        — standing, sitting, kneeling, crouching, and a moderate bend must
        all stay suppressed regardless of how high body_axis_angle reads.
        Swept strictly below upright_boundary (0.4), where the gate must
        stay fully closed — at/past the boundary itself, torso_cue's own
        sigmoid legitimately approaches 0.5+ regardless of any axis
        corroboration, which is by design, not a leak."""
        for torso_angle in (0.0, 0.1, 0.2, 0.3, 0.35):
            no_axis = _ext(
                torso_angle=torso_angle, head_height=0.15, knee_height=0.7,
                body_axis_angle=0.0, body_axis_evidence_quality=0.0,
            )
            noisy_axis = _ext(
                torso_angle=torso_angle, head_height=0.15, knee_height=0.7,
                body_axis_angle=1.0, body_axis_evidence_quality=1.0,  # maximally noisy/high
            )
            cue_no_axis = self.scorer._orientation_cue(no_axis)
            cue_noisy_axis = self.scorer._orientation_cue(noisy_axis)
            assert cue_noisy_axis == cue_no_axis, (
                f"torso_angle={torso_angle}: noisy axis changed the cue "
                f"({cue_no_axis} -> {cue_noisy_axis}) despite the gate being closed"
            )
            assert cue_noisy_axis < 0.45, f"torso_angle={torso_angle}: {cue_noisy_axis}"

    def test_body_axis_angle_generalizes_to_rotation(self):
        """`body_axis_angle` (head/shoulder-mid -> ankle-mid) should read
        proportionally to the actual rotation angle at 0/30/45/60/90 deg,
        the same way torso_angle already does via
        `_segment_angle_from_vertical` — proportional, not just monotonic."""
        import math

        import numpy as np

        from src.feature_extraction import ExtendedFeatureExtractor

        extractor = ExtendedFeatureExtractor()
        top = np.array([320.0, 200.0])
        length = 200.0
        for deg in (0, 30, 45, 60, 90):
            rad = math.radians(deg)
            bottom = top + length * np.array([math.sin(rad), math.cos(rad)])
            kps = np.zeros((17, 3), dtype=np.float32)
            kps[:] = [320, 240, 0.9]
            kps[0] = [top[0], top[1], 0.9]  # nose
            kps[15] = [bottom[0], bottom[1], 0.9]  # left ankle
            kps[16] = [bottom[0], bottom[1], 0.9]  # right ankle
            angle, evidence_quality = extractor._body_axis_angle(kps)
            assert evidence_quality == 1.0
            assert abs(angle - deg / 90.0) < 0.02, f"deg={deg}: got {angle}"


class TestOcclusionAwareKneelingPenalty:
    """V3 Phase 6: when legs are occluded, leg_extension_ratio's neutral
    default (1.0, 'no bend evidence') must not be trusted as if it were
    confirmed 'extended' — that was the root cause of a real, unresolved
    V2 false positive (an occluded kneeling officer). A leg-independent
    torso-upright signal must fill the gap instead."""

    def setup_method(self):
        self.scorer = FeatureMandownScorer(FeatureMandownConfig(), KneelingConfig())

    def test_occluded_legs_upright_torso_still_suppressed(self):
        """Legs occluded (leg_evidence_quality=0.0) but torso clearly
        upright (large head-hip gap) — must still be suppressed, unlike
        V2's behavior where this fell through as an unsuppressed false
        positive."""
        base = _base(bbox_aspect_ratio=1.3, body_angle=0.4, cog_height=0.7)
        occluded_kneeling = _ext(
            torso_angle=0.4, head_height=0.35, knee_height=0.6, body_height_norm=0.6,
            leg_extension_ratio=1.0, leg_evidence_quality=0.0,  # occluded -> neutral default
        )
        hist = deque([base] * 30, maxlen=30)
        ext_hist = deque([occluded_kneeling] * 30, maxlen=30)
        scores = self.scorer.score(base, occluded_kneeling, hist, ext_hist)
        assert scores.kneeling_penalty > 0.3

    def test_occluded_legs_flat_torso_not_over_suppressed(self):
        """Legs occluded AND torso itself also compact/flat (small head-hip
        gap, consistent with genuine lying) — must NOT be penalized just
        because the legs happen to be occluded. The torso itself is the
        deciding evidence here, not the missing legs."""
        base = _base(bbox_aspect_ratio=2.5, body_angle=0.9, cog_height=0.85)
        occluded_lying = _ext(
            torso_angle=0.9, head_height=0.82, knee_height=0.8, body_height_norm=0.35,
            leg_extension_ratio=1.0, leg_evidence_quality=0.0,
        )
        hist = deque([base] * 30, maxlen=30)
        ext_hist = deque([occluded_lying] * 30, maxlen=30)
        scores = self.scorer.score(base, occluded_lying, hist, ext_hist)
        assert scores.mandown_confidence > 0.4

    def test_visible_legs_use_leg_geometry_not_torso_fallback(self):
        """When legs ARE confidently visible, the normal leg-based penalty
        applies even if the torso-fallback signal would have said
        something different — visible ground-truth beats a fallback.
        hip_angle set close to torso_angle simply to keep this fixture
        representative of a consistent posture (not actually consulted by
        this code path)."""
        base = _base(bbox_aspect_ratio=1.2, body_angle=0.5, cog_height=0.7)
        visible_extended = _ext(
            torso_angle=0.5, hip_angle=0.45, head_height=0.4, knee_height=0.6, body_height_norm=0.6,
            leg_extension_ratio=1.0, leg_evidence_quality=1.0,  # confidently extended
        )
        hist = deque([base] * 30, maxlen=30)
        ext_hist = deque([visible_extended] * 30, maxlen=30)
        scores = self.scorer.score(base, visible_extended, hist, ext_hist)
        assert scores.kneeling_penalty < 0.3

    def test_leg_evidence_quality_reported_in_diagnostics(self):
        base = _base()
        ext = _ext(leg_evidence_quality=0.5)
        hist = deque([base] * 5, maxlen=30)
        scores = self.scorer.score(base, ext, hist, deque([ext] * 5, maxlen=30))
        assert scores.leg_evidence_quality == 0.5
        assert "leg_evidence_quality" in scores.to_dict()


def _full_kps(overrides: dict) -> np.ndarray:
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[:] = [320, 240, 0.0]  # unlisted keypoints default to fully occluded
    for idx, val in overrides.items():
        kps[idx] = val
    return kps


class TestDifficultOcclusionCases:
    """V4 Step 11 (spec cases A-F): keypoint-level occlusion coverage using
    the real ExtendedFeatureExtractor -> FeatureMandownScorer chain (not
    just hand-picked ExtendedFeatureVector values), so the full missing-
    keypoint-handling path is actually exercised end to end.

    Case F (person partially hidden / brief detection gap) is covered
    separately by `TrackStateMachine.on_gap` — see
    tests/test_state_machine.py::test_on_gap_never_resets_confirmed_state
    and tests/test_mandown_track_context.py — not duplicated here."""

    def setup_method(self):
        self.extractor = ExtendedFeatureExtractor()
        self.scorer = FeatureMandownScorer(FeatureMandownConfig(), KneelingConfig())

    def _score(self, kps, bbox=(280.0, 320.0, 560.0, 400.0)):
        ext = self.extractor.extract(kps, bbox, 640, 480)
        base_hist = deque(maxlen=30)
        ext_hist = deque([ext] * 10, maxlen=30)
        # Minimal FeatureVector standin — down_score/floor_proximity/
        # persistence don't depend on it; only bbox_aspect_ratio (shape_cue)
        # and fall_motion (empty history -> 0) do.
        from src.feature_extraction import FeatureVector as _FV
        base = _FV(0.0, 0.5, 0.0, (bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1e-6), 0.9)
        return ext, self.scorer.score(base, ext, base_hist, ext_hist)

    def test_case_a_upper_visible_legs_occluded_lying_still_confirms(self):
        """Case A: head/shoulders/hips confidently visible and clearly
        lying flat; both legs (knees+ankles) fully occluded."""
        kps = _full_kps({
            0: (150, 300, 0.9),
            5: (230, 300, 0.9), 6: (230, 320, 0.9),
            11: (400, 300, 0.9), 12: (400, 320, 0.9),
        })
        ext, scores = self._score(kps)
        assert ext.leg_evidence_quality == 0.0
        # down_score/floor_proximity carry the geometric evidence directly;
        # mandown_confidence itself is additionally, structurally capped by
        # zero fall_motion/floor_contact in this bare, uncalibrated,
        # zero-history test harness (see the V4 report's diagonal-lying
        # discussion) — not what this occlusion-specific test is about.
        assert scores.down_score > 0.6
        assert scores.floor_proximity_score > 0.6

    def test_case_b_lower_visible_upper_occluded_lying_still_confirms(self):
        """Case B: hips/knees/ankles confidently visible and clearly lying
        flat; head and both shoulders fully occluded. Regression coverage
        for a real gap found this session: torso_angle silently defaults to
        0.0 ("upright") when shoulders are missing, which without the
        hip_angle fallback in `_effective_torso_angle` would misread a
        clearly-down lower body as standing."""
        kps = _full_kps({
            11: (380, 340, 0.9), 12: (370, 360, 0.9),
            13: (450, 350, 0.9), 14: (440, 365, 0.9),
            15: (520, 355, 0.9), 16: (510, 370, 0.9),
        })
        ext, scores = self._score(kps, bbox=(370.0, 330.0, 520.0, 380.0))
        assert ext.torso_angle_evidence_quality == 0.0
        assert ext.hip_angle_evidence_quality == 1.0
        assert ext.body_axis_evidence_quality == 0.0, (
            "body_axis_angle must not report confident evidence when its "
            "own top endpoint (nose/shoulders) is entirely missing"
        )
        assert scores.down_score > 0.5
        assert scores.floor_proximity_score > 0.5

    def test_case_c_one_leg_visible_one_missing(self):
        """Case C: a fully lying posture with only one leg's keypoints
        visible (the other leg entirely occluded, e.g. behind the body or
        out of frame) — leg_evidence_quality must read partial (0.5), and
        the posture must still be recognized as lying via the visible
        side."""
        kps = _full_kps({
            0: (150, 300, 0.9),
            5: (230, 300, 0.9), 6: (230, 320, 0.9),
            11: (400, 300, 0.9), 12: (400, 320, 0.9),
            13: (480, 300, 0.9),  # only left knee
            15: (560, 300, 0.9),  # only left ankle
        })
        ext, scores = self._score(kps)
        assert ext.leg_evidence_quality == 0.5
        assert scores.mandown_confidence > 0.4

    def test_case_d_both_legs_missing_kneeling_penalty_uses_torso_fallback(self):
        """Case D: both legs entirely missing on an otherwise-standing
        (upright torso) person — must NOT be read as extended/lying just
        because there's no leg evidence to contradict it; the torso-gap
        fallback in `_kneeling_penalty` must still suppress this."""
        kps = _full_kps({
            0: (320, 100, 0.9),
            5: (300, 160, 0.9), 6: (340, 160, 0.9),
            11: (305, 300, 0.9), 12: (335, 300, 0.9),
        })
        ext, scores = self._score(kps, bbox=(280.0, 80.0, 360.0, 320.0))
        assert ext.leg_evidence_quality == 0.0
        assert scores.mandown_confidence < 0.4

    def test_case_e_torso_partially_occluded_one_shoulder_missing(self):
        """Case E: one shoulder occluded but the other shoulder + both hips
        visible, on a clearly lying posture — `_midpoint`'s single-point
        fallback should still let torso_angle read correctly rather than
        losing evidence entirely."""
        kps = _full_kps({
            0: (150, 300, 0.9),
            6: (230, 320, 0.9),  # only right shoulder
            11: (400, 300, 0.9), 12: (400, 320, 0.9),
            13: (480, 300, 0.9), 14: (480, 320, 0.9),
            15: (560, 300, 0.9), 16: (560, 320, 0.9),
        })
        ext, scores = self._score(kps)
        assert ext.torso_angle_evidence_quality == 1.0
        assert scores.mandown_confidence > 0.4


class TestFloorContactScore:
    def setup_method(self):
        self.scorer = FeatureMandownScorer(FeatureMandownConfig())

    def test_higher_floor_contact_score_raises_confidence(self):
        base = _base()
        ext = _ext()
        hist = deque([base] * 15, maxlen=30)
        ext_hist = deque([ext] * 15, maxlen=30)
        low = self.scorer.score(base, ext, hist, ext_hist, floor_contact_score=0.0)
        high = self.scorer.score(base, ext, hist, ext_hist, floor_contact_score=1.0)
        assert high.mandown_confidence > low.mandown_confidence
        assert high.floor_contact_score == 1.0
