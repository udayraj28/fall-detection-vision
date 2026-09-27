"""Tests for the evidence fusion layer."""

from src.config import FusionConfig
from src.fusion import EvidenceFusion


class TestEvidenceFusion:
    def setup_method(self):
        self.fusion = EvidenceFusion(FusionConfig())

    def test_both_weak_is_normal(self):
        result = self.fusion.combine(
            lstm_mandown_prob=0.1, feature_confidence=0.1, persistence_score=0.1,
            pose_quality=0.9, down_score=0.1,
        )
        assert result.suggested_tier == "NORMAL"

    def test_lstm_strong_feature_moderate_is_at_least_possible(self):
        """down_score=0.7 (not the old 0.5) — V4.1 found that a MODERATE
        down_score (~0.5, consistent with a sustained standing bend, not
        genuine lying) combined with a strong LSTM is exactly the real
        false-positive pattern being fixed (see
        test_moderate_down_score_with_wrong_lstm_does_not_confirm below);
        0.7 keeps this test's original intent (basic "strong signal" wiring)
        without exercising that danger zone."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.9, feature_confidence=0.5, persistence_score=0.3,
            pose_quality=0.9, down_score=0.7,
        )
        assert result.suggested_tier in ("POSSIBLE_MANDOWN", "MANDOWN")

    def test_single_strong_lstm_signal_with_some_grounding_not_gated_by_and(self):
        """A single very strong signal is still sufficient on its own — no
        strict AND requirement between LSTM and feature evidence — as long
        as there is *real* geometric grounding (down_score at a genuinely
        lying-consistent level, not merely "some"). down_score=0.3 (this
        test's original value) turned out to be exactly the V4.1 danger
        zone — a moderate-but-not-lying reading a wrong LSTM can exploit —
        so it no longer belongs in a "should NOT be NORMAL" test; 0.7 tests
        the same no-AND-gate wiring on genuinely-grounded evidence."""
        result_lstm_only = self.fusion.combine(
            lstm_mandown_prob=0.95, feature_confidence=0.0, persistence_score=0.0,
            pose_quality=0.9, down_score=0.7,
        )
        result_feature_only = self.fusion.combine(
            lstm_mandown_prob=0.0, feature_confidence=0.95, persistence_score=0.0,
            pose_quality=0.9, down_score=0.9,
        )
        assert result_lstm_only.suggested_tier != "NORMAL"
        assert result_feature_only.suggested_tier != "NORMAL"

    def test_moderate_down_score_with_wrong_lstm_does_not_confirm(self):
        """V4.1 regression test for a real RTSP false positive: a person
        sustained bending forward (straight legs — kneeling_penalty can't
        help) reads down_score up to ~0.55 at the most extreme synthetic
        angle tested (see tests/test_already_lying_integration.py
        deep_bend_keypoints/very-deep variants) — never genuinely "lying"
        (real confirmed events in this repo's 4 validation videos all have
        down_score in [0.704, 0.933]). A confidently-wrong LSTM must not be
        able to push evidence like this to MANDOWN, whether via a single
        frame or (critically) sustained over many frames — a per-frame
        fusion_score capped only into POSSIBLE_MANDOWN would still let
        TrackStateMachine's streak eventually confirm it, since that state
        machine treats POSSIBLE_MANDOWN and MANDOWN identically for "up"
        streak purposes. Must land at NORMAL, not merely below MANDOWN."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.95, feature_confidence=0.5, persistence_score=1.0,
            pose_quality=0.9, down_score=0.55,
        )
        assert result.suggested_tier == "NORMAL"

    def test_small_lowquality_sitting_with_wrong_lstm_does_not_confirm(self):
        """V4.1 regression test for a real RTSP false positive: a small,
        distant, low-confidence sitting detection. down_score/persistence
        stay low even under a confidently-wrong LSTM and low pose_quality —
        must not confirm."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.95, feature_confidence=0.1, persistence_score=0.0,
            pose_quality=0.35, down_score=0.05,
        )
        assert result.suggested_tier == "NORMAL"

    def test_lstm_only_with_zero_geometric_grounding_is_normal(self):
        """V4 regression test for a real false positive (M_D_2.mp4 track
        694): a confident LSTM alone, with down_score and persistence_score
        both at/near zero (the person's own geometry never once read as
        down-like), must not escalate past NORMAL — otherwise the state
        machine's sustained-streak pathway can confirm a full MANDOWN purely
        from a wrong/spurious LSTM read with zero physical corroboration.
        Values below mirror the real track's observed range exactly
        (down_score max 0.204, persistence_score constant 0.0 across the
        whole ~60-frame event, lstm_mandown_prob ~0.99)."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.99, feature_confidence=0.245, persistence_score=0.0,
            pose_quality=0.9, down_score=0.20,
        )
        assert result.suggested_tier == "NORMAL"

    def test_sustained_moderate_evidence_with_real_grounding_still_confirms(self):
        """Regression guard for a real, legitimate confirmed event
        (man_down_c08.mp4's second track fragment of the same physically-down
        person). Values are the exact measured diagnostics for that track
        (dedup disabled, so its own independent evidence is visible rather
        than merged into track 1's): down_score=0.709, feature_confidence=
        0.62, persistence=1.0, LSTM cold (its own 30-frame buffer hadn't
        filled). This already independently qualifies via
        geometric_confirm_threshold/geometric_confirm_persistence_threshold
        (0.62>=0.60, 1.0>=0.65) and clears the V4.1 down_score floor
        (0.709>=0.65) — both by real margin, not by coincidence of the old
        approximated values (0.6/0.56) this test previously used, which
        understated the real evidence and no longer reflects genuine
        production data now that it's been re-measured. Confirms in a
        SINGLE call now (via the geometric-alone-confirm override directly)
        rather than needing the sustained-streak pathway the old,
        understated values relied on — the real evidence was stronger than
        previously assumed, not weaker."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.0, feature_confidence=0.62, persistence_score=1.0,
            pose_quality=0.9, down_score=0.709,
        )
        assert result.suggested_tier == "MANDOWN"

    def test_both_strong_confirms_mandown(self):
        result = self.fusion.combine(
            lstm_mandown_prob=0.95, feature_confidence=0.9, persistence_score=0.9,
            pose_quality=0.9, down_score=0.9,
        )
        assert result.suggested_tier == "MANDOWN"

    def test_low_pose_quality_discounts_not_zeroes(self):
        strong = self.fusion.combine(
            lstm_mandown_prob=0.9, feature_confidence=0.9, persistence_score=0.9,
            pose_quality=0.9, down_score=0.9,
        )
        weak_pose = self.fusion.combine(
            lstm_mandown_prob=0.9, feature_confidence=0.9, persistence_score=0.9,
            pose_quality=0.05, down_score=0.9,
        )
        assert weak_pose.fusion_score < strong.fusion_score
        assert weak_pose.fusion_score > 0.0

    def test_output_bounded(self):
        for lstm in (0.0, 0.5, 1.0):
            for feat in (0.0, 0.5, 1.0):
                result = self.fusion.combine(lstm, feat, 0.5, 0.8, down_score=feat)
                assert 0.0 <= result.fusion_score <= 1.0

    def test_strong_persistent_geometric_alone_confirms_mandown(self):
        """V3: an already-lying person must reach full MANDOWN on geometric
        evidence alone — a cold/wrong LSTM (0.0) must not cap confirmation
        at POSSIBLE_MANDOWN when the geometry is strong and sustained."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.0, feature_confidence=0.7, persistence_score=0.9,
            pose_quality=0.9, down_score=0.7,
        )
        assert result.suggested_tier == "MANDOWN"

    def test_moderate_geometric_alone_does_not_confirm(self):
        """Below the geometric-confirm bar, a cold LSTM should NOT be
        bypassed — avoids over-triggering on moderate, ambiguous evidence."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.0, feature_confidence=0.5, persistence_score=0.9,
            pose_quality=0.9, down_score=0.5,
        )
        assert result.suggested_tier != "MANDOWN"

    def test_strong_geometric_without_persistence_does_not_confirm_alone(self):
        """A single strong-but-unsustained geometric reading (persistence
        low) should not instantly confirm — still needs to prove itself
        over time, avoiding a one-frame geometric spike causing MANDOWN."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.0, feature_confidence=0.9, persistence_score=0.1,
            pose_quality=0.9, down_score=0.9,
        )
        assert result.suggested_tier != "MANDOWN"

    def test_floor_proximity_path_lets_vertical_fall_confirm_with_moderate_lstm(self):
        """V4.2 regression test for a real RTSP report: a person fallen
        toward/away from the camera reads down_score well below
        confirmed_mandown_down_floor even after height-collapse
        corroboration (shape_cue/angle_cue both stay weak — the fall has no
        lateral image-plane spread to measure), but floor_proximity_score,
        height_collapse_cue, and pose_quality all clear their own bars —
        this is the alternate grounding path that lets a realistic (not
        maximal) LSTM contribution complete confirmation. torso_angle~0.0
        with evidence available reflects the real fixture (a foreshortened
        fall has no lateral spread to read an angle from at all) — see
        test_floor_proximity_path_requires_torso_angle_near_zero below for
        the V4.5 gate this must still pass through."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.75, feature_confidence=0.45, persistence_score=1.0,
            pose_quality=0.5, down_score=0.55, floor_proximity_score=0.75,
            height_collapse_cue=0.9, torso_angle=0.0, torso_angle_evidence_quality=1.0,
        )
        assert result.suggested_tier == "MANDOWN"

    def test_floor_proximity_path_requires_pose_quality_above_its_floor(self):
        """The exact adversarial case this path must not reopen: a
        partially-occluded seated/standing person (V4.1's real sitting
        false positive) can also show an elevated floor_proximity_score AND
        height_collapse_cue from its naturally-short visible bbox, but with
        far fewer visible keypoints (low pose_quality) than a genuine fall.
        Requiring pose_quality above its own established floor blocks this
        even under a confidently-wrong LSTM. torso_angle passed as if it
        were a genuine near-zero foreshortened reading, so this test
        isolates the pose_quality gate specifically, not the V4.5
        torso-angle gate below."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.95, feature_confidence=0.45, persistence_score=1.0,
            pose_quality=0.15, down_score=0.55, floor_proximity_score=0.75,
            height_collapse_cue=0.9, torso_angle=0.0, torso_angle_evidence_quality=1.0,
        )
        assert result.suggested_tier != "MANDOWN"

    def test_floor_proximity_path_requires_floor_proximity_above_its_floor(self):
        """A moderate (not strong) floor_proximity_score must not qualify —
        avoids the alternate grounding path becoming a second, looser
        version of the same LSTM-confirms-weak-geometry bug it exists to
        close. torso_angle passed as a genuine near-zero foreshortened
        reading, isolating the floor_proximity gate specifically."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.95, feature_confidence=0.45, persistence_score=1.0,
            pose_quality=0.5, down_score=0.55, floor_proximity_score=0.5,
            height_collapse_cue=0.9, torso_angle=0.0, torso_angle_evidence_quality=1.0,
        )
        assert result.suggested_tier != "MANDOWN"

    def test_floor_proximity_path_requires_height_collapse_cue_above_its_floor(self):
        """V4.2 regression test: the real bug found via a real screenshot
        re-check. A deep forward bend near a table (V4.1's real bending
        false positive) drives floor_proximity_score to ~0.89 via
        orientation_cue (torso angle) — *higher* than a genuine
        foreshortened fall's ~0.74 — so floor_proximity_score and
        pose_quality alone do NOT exclude it (a clearly-visible bend has
        high pose_quality too). height_collapse_cue is driven only by
        body_height_norm, never by torso angle, so it correctly stays low
        (~0.15, uncalibrated-equivalent) for a bend even though
        floor_proximity_score and pose_quality both read as if it were a
        genuine fall. torso_angle passed as a genuine near-zero
        foreshortened reading, isolating the height_collapse_cue gate
        specifically — see test_floor_proximity_path_requires_torso_angle_
        near_zero below for the case where height_collapse_cue does NOT
        stay this low under real per-camera calibration."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.95, feature_confidence=0.5, persistence_score=1.0,
            pose_quality=0.9, down_score=0.48, floor_proximity_score=0.89,
            height_collapse_cue=0.15, torso_angle=0.0, torso_angle_evidence_quality=1.0,
        )
        assert result.suggested_tier != "MANDOWN"

    def test_floor_proximity_path_requires_torso_angle_near_zero(self):
        """V4.5 regression test for a real RTSP false positive: a person
        bending forward on their feet to reach the ground was confirmed
        MANDOWN through this exact alternate path. Under real per-camera
        height calibration, a deep bend's height_collapse_cue does NOT
        reliably stay as low as the ~0.15 "uncalibrated-equivalent" figure
        used in the test above — tests/test_already_lying_integration.py's
        test_standing_then_deep_bend_never_confirms measured a calibrated
        deep bend at ~0.499, a hair below the confirm floor, and a real
        camera then crossed it. This path's actual justification is that
        torso_angle is *blind* (~0.0) for a foreshortened fall; a deep
        bend's torso_angle (~0.64, per deep_bend_keypoints()) is not blind
        at all and must not get a pass just because the other two cues
        happened to read high too."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.95, feature_confidence=0.5, persistence_score=1.0,
            pose_quality=0.9, down_score=0.48, floor_proximity_score=0.89,
            height_collapse_cue=0.9, torso_angle=0.64, torso_angle_evidence_quality=1.0,
        )
        assert result.suggested_tier != "MANDOWN"

    def test_floor_proximity_path_requires_torso_angle_evidence_present(self):
        """No torso evidence at all (both shoulders and hips missing) must
        not get the benefit of the doubt either — this path's blind-torso
        justification can only be trusted when actually confirmed, not
        merely unknown."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.95, feature_confidence=0.5, persistence_score=1.0,
            pose_quality=0.9, down_score=0.48, floor_proximity_score=0.89,
            height_collapse_cue=0.9, torso_angle=0.0, torso_angle_evidence_quality=0.0,
        )
        assert result.suggested_tier != "MANDOWN"


class TestPose3DGrounding:
    """V4.8: monocular 2D->3D pose lifting (src/pose_3d.py) as an
    independent grounding path — a direct, camera-calibration-independent
    torso-orientation measurement. torso_angle_3d values below are REAL
    measurements (via an actual MediaPipe pose-lifting run on real
    footage): 0.857/0.952 from clean, flat, real falls in mandown_2.mp4
    and man_down_c08.mp4 respectively.

    V4.8.1: this path is NOT independent of height_collapse_cue after
    all — torso_angle_3d turned out to also read high (not blind, unlike
    a camera-facing fall) for a deep-enough forward bend (real regression:
    torso_angle_3d=0.777, almost identical to that bend's own 2D
    torso_angle=0.778), so height_collapse_cue (which a bend can never
    satisfy — see test_deep_bend_high_3d_angle_still_excluded below) is
    now required here too, same floor the 2D alternate-grounding path
    above already uses."""

    def setup_method(self):
        self.fusion = EvidenceFusion(FusionConfig())

    def test_confident_3d_lying_confirms_with_minimal_geometric_grounding(self):
        result = self.fusion.combine(
            lstm_mandown_prob=0.75, feature_confidence=0.5, persistence_score=1.0,
            pose_quality=0.9, down_score=0.3, floor_proximity_score=0.2,
            height_collapse_cue=0.6, torso_angle=0.3, torso_angle_evidence_quality=1.0,
            torso_angle_3d=0.857, torso_angle_3d_evidence_quality=0.95,
        )
        assert result.suggested_tier == "MANDOWN"

    def test_deep_bend_high_3d_angle_still_excluded(self):
        """V4.8.1 regression test for a real RTSP false positive: a deep
        forward bend (straight legs, reaching down to a low table)
        measured torso_angle_3d=0.777 live — a bend genuinely tilts the
        torso in real 3D space, not just in 2D projection, so a high
        reading alone doesn't distinguish it from genuine lying. What a
        bend can never do is bring the hip down toward the floor —
        height_collapse_cue stayed low (0.148, measured on the same real
        bend, uncalibrated) precisely because of that, and correctly
        excludes it here even with a confident LSTM and a down_score
        already fairly substantial on its own (0.632, also measured
        live)."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.95, feature_confidence=0.5, persistence_score=1.0,
            pose_quality=0.9, down_score=0.632, floor_proximity_score=0.938,
            height_collapse_cue=0.148, torso_angle=0.778, torso_angle_evidence_quality=1.0,
            torso_angle_3d=0.777, torso_angle_3d_evidence_quality=0.998,
        )
        assert result.suggested_tier != "MANDOWN"

    # NOTE (V4.9, attempted and reverted): the same real bend under this
    # camera's real per-camera calibration also clears the V4.8.1 height_
    # collapse_cue floor (body_height_norm=0.626 -> height_collapse_cue=
    # 0.53), so that fix alone does not exclude it once calibration is
    # real rather than the neutral uncalibrated default. A torso-hip-
    # angle-mismatch exclusion was tried and reverted — see config.py's
    # FusionConfig/KneelingConfig V4.9 notes for why (it also suppressed a
    # real M_D_2.mp4 true positive). This specific real gap (a straight-
    # legged deep bend under real calibration) remains open, documented
    # here rather than forced closed with an unsafe fix.

    def test_no_3d_evidence_does_not_use_this_path(self):
        """Same otherwise-favorable numbers, but no 3D reading at all
        (evidence_quality=0.0) — must not confirm through this path."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.75, feature_confidence=0.4, persistence_score=1.0,
            pose_quality=0.9, down_score=0.3, floor_proximity_score=0.2,
            height_collapse_cue=0.1, torso_angle=0.3, torso_angle_evidence_quality=1.0,
            torso_angle_3d=0.0, torso_angle_3d_evidence_quality=0.0,
        )
        assert result.suggested_tier != "MANDOWN"

    def test_low_3d_angle_does_not_confirm_even_if_confident(self):
        """A confident 3D reading that says "not lying" (e.g. a genuine
        crouch, torso vertical in 3D) must not confirm — confidence alone
        isn't the gate, the angle itself must show real lying."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.95, feature_confidence=0.4, persistence_score=1.0,
            pose_quality=0.9, down_score=0.3, floor_proximity_score=0.2,
            height_collapse_cue=0.1, torso_angle=0.0, torso_angle_evidence_quality=1.0,
            torso_angle_3d=0.15, torso_angle_3d_evidence_quality=0.95,
        )
        assert result.suggested_tier != "MANDOWN"

    def test_confident_3d_lying_still_requires_minimal_geometric_grounding(self):
        """A lone 3D reading with essentially ZERO geometric backing
        (down_score far below even the minimal floor) must not confirm —
        this path corroborates real geometric evidence, it doesn't
        replace the need for any. height_collapse_cue kept high (0.6,
        clearing its own floor) so this test isolates the down_score
        floor specifically, not a compound failure."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.95, feature_confidence=0.3, persistence_score=1.0,
            pose_quality=0.9, down_score=0.05, floor_proximity_score=0.1,
            height_collapse_cue=0.6, torso_angle=0.3, torso_angle_evidence_quality=1.0,
            torso_angle_3d=0.952, torso_angle_3d_evidence_quality=0.9,
        )
        assert result.suggested_tier != "MANDOWN"


class TestBodyAxisAngle3DGrounding:
    """V6.1 (Objective 1): `body_axis_angle_3d` is an INDEPENDENT
    corroborator alongside torso_angle_3d, not a replacement -- either
    one confirming (under the same gating) is sufficient."""

    def setup_method(self):
        self.fusion = EvidenceFusion(FusionConfig())

    def test_axis_confirms_when_torso_disagrees(self):
        """The exact disagreement this feature exists for: torso_angle_3d
        reads too low (below its own floor -- the real raised-knee case's
        hip-landmark bias), but body_axis_angle_3d independently confirms.
        Same recipe as the existing
        test_confident_3d_lying_confirms_with_minimal_geometric_grounding
        (lstm+feature+persistence alone already reach the weighted-average
        confirm threshold; what's under test is that `confident_3d_lying`
        -- via the axis path, since the torso path's own floor isn't
        cleared -- is what stops the "no strong-enough grounding" safeguard
        from capping it back down to POSSIBLE_MANDOWN)."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.75, feature_confidence=0.5, persistence_score=1.0,
            pose_quality=0.9, down_score=0.3, floor_proximity_score=0.2,
            height_collapse_cue=0.6, torso_angle=0.3, torso_angle_evidence_quality=1.0,
            torso_angle_3d=0.32, torso_angle_3d_evidence_quality=0.95,
            body_axis_angle_3d=0.62, body_axis_angle_3d_evidence_quality=0.9,
        )
        assert result.suggested_tier == "MANDOWN"

    def test_axis_alone_still_requires_height_collapse(self):
        """Same as the existing torso_angle_3d path's own requirement --
        body_axis_angle_3d confirming alone, without a real height
        collapse, must not bypass that floor."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.0, feature_confidence=0.3, persistence_score=0.2,
            pose_quality=0.9, down_score=0.3, floor_proximity_score=0.3,
            height_collapse_cue=0.1, torso_angle=0.1, torso_angle_evidence_quality=1.0,
            body_axis_angle_3d=0.9, body_axis_angle_3d_evidence_quality=0.9,
        )
        assert result.suggested_tier != "MANDOWN"

    def test_axis_low_angle_does_not_confirm(self):
        """A confident axis reading that says "not lying" (mirrors the
        crouch/deep-bend negative controls in test_pose_3d.py) must not
        confirm."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.0, feature_confidence=0.3, persistence_score=0.2,
            pose_quality=0.9, down_score=0.3, floor_proximity_score=0.3,
            height_collapse_cue=0.6, torso_angle=0.1, torso_angle_evidence_quality=1.0,
            body_axis_angle_3d=0.2, body_axis_angle_3d_evidence_quality=0.9,
        )
        assert result.suggested_tier != "MANDOWN"

    def test_axis_excluded_by_standing_bend_signature(self):
        """The same hip/torso-angle-mismatch bend-exclusion gate that
        protects the torso_angle_3d path must also protect this one --
        it's the same `standing_bend_signature` check, shared by both."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.0, feature_confidence=0.3, persistence_score=0.2,
            pose_quality=0.9, down_score=0.3, floor_proximity_score=0.3,
            height_collapse_cue=0.6, torso_angle=0.8, torso_angle_evidence_quality=1.0,
            hip_angle=0.2, hip_angle_evidence_quality=1.0,
            body_axis_angle_3d=0.7, body_axis_angle_3d_evidence_quality=0.9,
        )
        assert result.suggested_tier != "MANDOWN"


class TestCollapseThenStillnessFusionPath:
    """V6 Fix Set B, rule 8: `collapse_then_stillness` is computed
    upstream (MandownTrackContext) as a hard 0/1 gate already backed by
    real geometric evidence (a genuine height-collapse spike, sustained
    stillness, and a real calibrated collapse right now) -- fusion.py's
    only job is to let that True flow through to a MANDOWN confirmation,
    the same way V4.10's direct down_score path does, and to still
    require a pose-quality floor so an unreliable skeleton can't ride
    this path to confirmation."""

    def setup_method(self):
        self.fusion = EvidenceFusion(FusionConfig())

    def test_collapse_then_stillness_true_confirms_even_with_cold_lstm(self):
        result = self.fusion.combine(
            lstm_mandown_prob=0.0, feature_confidence=0.3, persistence_score=0.2,
            pose_quality=0.6, down_score=0.3, floor_proximity_score=0.3,
            height_collapse_cue=0.6, torso_angle=0.1, torso_angle_evidence_quality=1.0,
            collapse_then_stillness=True,
        )
        assert result.suggested_tier == "MANDOWN"
        assert result.fusion_score >= 0.65

    def test_collapse_then_stillness_false_does_not_confirm_on_its_own(self):
        """Same otherwise-weak evidence, but the caller never asserted the
        gate — must not confirm just because the OTHER fields happen to
        look plausible."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.0, feature_confidence=0.3, persistence_score=0.2,
            pose_quality=0.6, down_score=0.3, floor_proximity_score=0.3,
            height_collapse_cue=0.6, torso_angle=0.1, torso_angle_evidence_quality=1.0,
            collapse_then_stillness=False,
        )
        assert result.suggested_tier != "MANDOWN"

    def test_collapse_then_stillness_gated_by_pose_quality_floor(self):
        """An unreliable skeleton (pose_quality below the established
        floor) must not be able to confirm through this path either —
        matches the same discipline as every other confirmation path in
        this module."""
        result = self.fusion.combine(
            lstm_mandown_prob=0.0, feature_confidence=0.2, persistence_score=0.1,
            pose_quality=0.1, down_score=0.2, floor_proximity_score=0.2,
            height_collapse_cue=0.6, torso_angle=0.1, torso_angle_evidence_quality=1.0,
            collapse_then_stillness=True,
        )
        assert result.suggested_tier != "MANDOWN"
