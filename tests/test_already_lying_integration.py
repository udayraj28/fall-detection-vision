"""V3 Phase 8: already-lying-at-start integration tests.

Wires the REAL component chain (base+extended feature extraction -> feature
scorer -> evidence fusion -> temporal state machine) exactly as
`pipeline.py::_process_frame` does per person, but with a static "already
lying" posture fed from frame 0 of a brand-new track — no prior standing
history, no fall velocity/acceleration, and critically, the LSTM is
simulated cold (`lstm_mandown_prob=0.0` throughout, as if its 30-frame
buffer never fills or it's simply wrong) to prove the geometric/floor
evidence alone is sufficient, per the V3 requirement.

This is UNIT/INTEGRATION TESTED with synthetic keypoints — not real-video
validated. Real footage validated only the "back-ish" lying orientation
that happened to appear in the available test videos; side/prone/diagonal
orientations have no real footage available this session. See
MANDOWN_V3_IMPLEMENTATION_REPORT.md for the explicit UNIT TESTED vs REAL
VIDEO VALIDATED distinction.
"""

import math
from collections import deque

import numpy as np
import pytest

from src.config import FeatureMandownConfig, FusionConfig, HipDropConfig, KneelingConfig, StateMachineConfig
from src.feature_extraction import ExtendedFeatureExtractor, FeatureExtractor
from src.feature_mandown import FeatureMandownScorer
from src.fusion import EvidenceFusion
from src.mandown_track_context import MandownTrackContext
from src.state_machine import MandownState, TrackStateMachine

FRAME_W, FRAME_H = 640, 480


def _kps(overrides: dict) -> np.ndarray:
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[:] = [FRAME_W / 2, FRAME_H / 2, 0.9]
    for idx, val in overrides.items():
        kps[idx] = val
    return kps


def standing_keypoints() -> np.ndarray:
    return _kps({
        0: (320, 60, 0.95),
        5: (295, 140, 0.9), 6: (345, 140, 0.9),
        11: (300, 280, 0.9), 12: (340, 280, 0.9),
        13: (300, 370, 0.9), 14: (340, 370, 0.9),
        15: (300, 450, 0.9), 16: (340, 450, 0.9),
    })


def supine_lying_keypoints() -> np.ndarray:
    """Lying on back, roughly horizontal across the frame, legs extended."""
    return _kps({
        0: (120, 300, 0.9),
        5: (200, 300, 0.9), 6: (200, 320, 0.9),
        11: (380, 300, 0.9), 12: (380, 320, 0.9),
        13: (480, 300, 0.9), 14: (480, 320, 0.9),
        15: (560, 300, 0.9), 16: (560, 320, 0.9),
    })


def prone_lying_keypoints() -> np.ndarray:
    """Face-down lying — geometrically similar 2D layout to supine (COCO
    2D keypoints don't distinguish face-up/face-down orientation on their
    own); included for explicit test coverage per the spec's request."""
    return _kps({
        0: (130, 310, 0.85),
        5: (210, 305, 0.9), 6: (210, 325, 0.9),
        11: (390, 305, 0.9), 12: (390, 325, 0.9),
        13: (490, 305, 0.9), 14: (490, 325, 0.9),
        15: (570, 305, 0.9), 16: (570, 325, 0.9),
    })


def side_lying_keypoints() -> np.ndarray:
    """Side lying with knees drawn up (natural side-lying posture) — torso
    flat/horizontal but legs bent, exercising the kneeling-penalty gate's
    "genuinely lying with a bent knee" exception."""
    return _kps({
        0: (150, 300, 0.85),
        5: (230, 295, 0.9), 6: (230, 315, 0.85),
        11: (360, 300, 0.9), 12: (360, 320, 0.85),
        13: (420, 340, 0.85), 14: (420, 355, 0.8),
        15: (400, 310, 0.7), 16: (400, 325, 0.65),
    })


def diagonal_lying_keypoints() -> np.ndarray:
    """Lying at a diagonal across the frame (not purely horizontal in
    image-space) — stresses whether the geometry still reads as "down"
    when the body axis isn't image-horizontal."""
    return _kps({
        0: (150, 200, 0.9),
        5: (220, 240, 0.9), 6: (210, 260, 0.9),
        11: (380, 340, 0.9), 12: (370, 360, 0.9),
        13: (450, 390, 0.9), 14: (440, 405, 0.9),
        15: (520, 440, 0.9), 16: (510, 455, 0.9),
    })


class _StaticTrackSimulator:
    """Feeds identical keypoints frame after frame through the real
    base+extended feature -> scorer -> fusion -> state-machine chain,
    simulating a track that starts already in this posture with the LSTM
    cold (0.0) the entire time."""

    def __init__(self, expected_height_fn=None, floor_contact_score=0.0, hip_drop_config=None):
        self.base_extractor = FeatureExtractor()
        self.ext_extractor = ExtendedFeatureExtractor()
        self.scorer = FeatureMandownScorer(FeatureMandownConfig(), KneelingConfig())
        self.fusion = EvidenceFusion(FusionConfig())
        self.state_machine = TrackStateMachine(StateMachineConfig())
        self.expected_height_fn = expected_height_fn
        # In real deployment, floor-region calibration and standing-height
        # calibration come from the same scene-profiling pipeline (both need
        # enough reliable samples) -- a test wiring one on should usually
        # wire the other on too, rather than leaving this at its default-0.0
        # "uncalibrated" reading while claiming height calibration is active.
        self.floor_contact_score = floor_contact_score
        # V4.4: a real MandownTrackContext, not a re-implementation -- this
        # is exactly what carries last_standing_cog_height/hip_drop_evidence
        # in the real pipeline, so using it here keeps this simulator
        # genuinely representative rather than a parallel approximation
        # that could silently drift from the real logic.
        self.hip_drop_config = hip_drop_config or HipDropConfig()
        self.ctx = MandownTrackContext(history_maxlen=30)

        self.base_history = deque(maxlen=30)
        self.ext_history = deque(maxlen=30)
        self.prev_ext = None
        self.prev_cog_x = None
        self.prev_vertical_velocity = None

    def step(self, keypoints: np.ndarray, bbox: tuple, lstm_mandown_prob: float = 0.0) -> MandownState:
        prev_cog = self.base_history[-1].cog_height if self.base_history else None
        base = self.base_extractor.extract(keypoints, bbox, FRAME_H, prev_cog)
        self.base_history.append(base)

        ext = self.ext_extractor.extract(
            keypoints, bbox, FRAME_W, FRAME_H,
            prev=self.prev_ext, prev_cog_x=self.prev_cog_x,
            vertical_velocity_now=base.vertical_velocity,
            prev_vertical_velocity=self.prev_vertical_velocity,
            expected_height_fn=self.expected_height_fn,
        )
        self.ext_history.append(ext)
        self.prev_ext = ext
        self.prev_cog_x = self.ext_extractor.cog_x(keypoints, bbox)
        self.prev_vertical_velocity = base.vertical_velocity

        hip_drop_evidence = self.ctx.hip_drop_evidence(base.cog_height, self.hip_drop_config)
        # torso_angle alone isn't enough -- see pipeline.py's identical
        # check for why body_height_norm must also look standing-height.
        is_confidently_upright = (
            ext.torso_angle_evidence_quality > 0.0
            and ext.torso_angle < self.hip_drop_config.upright_torso_angle_threshold
            and ext.body_height_norm >= self.hip_drop_config.upright_body_height_norm_threshold
        )
        self.ctx.update_standing_baseline(base.cog_height, is_confidently_upright)

        scores = self.scorer.score(
            base, ext, self.base_history, self.ext_history,
            floor_contact_score=self.floor_contact_score,
            hip_drop_evidence=hip_drop_evidence,
        )
        fusion_result = self.fusion.combine(
            # cold (0.0) LSTM by default — the whole point of most tests in
            # this file; some V4.2 tests pass a non-zero value explicitly.
            lstm_mandown_prob=lstm_mandown_prob,
            feature_confidence=scores.mandown_confidence,
            persistence_score=scores.persistence_score,
            pose_quality=scores.pose_quality_score,
            down_score=scores.down_score,
            floor_proximity_score=scores.floor_proximity_score,
            height_collapse_cue=scores.height_collapse_cue,
            torso_angle=ext.torso_angle,
            torso_angle_evidence_quality=ext.torso_angle_evidence_quality,
        )
        step_result = self.state_machine.step(fusion_result.suggested_tier)
        return step_result.state


def _bbox_for(keypoints: np.ndarray, pad=20) -> tuple:
    valid = keypoints[keypoints[:, 2] > 0.3]
    x1, y1 = valid[:, 0].min() - pad, valid[:, 1].min() - pad
    x2, y2 = valid[:, 0].max() + pad, valid[:, 1].max() + pad
    return (float(x1), float(y1), float(x2), float(y2))


def _run_static_posture(keypoints_fn, max_frames=40) -> tuple[MandownState, int]:
    sim = _StaticTrackSimulator()
    kps = keypoints_fn()
    bbox = _bbox_for(kps)
    for frame_num in range(max_frames):
        state = sim.step(kps, bbox)
        if state == MandownState.MANDOWN:
            return state, frame_num
    return state, max_frames


class TestAlreadyLyingNoHistory:
    """No fall history, no velocity, no acceleration, no prior standing
    frames — LSTM cold throughout. Each case starts a BRAND NEW track
    directly in the given posture."""

    def test_supine_lying_reaches_mandown(self):
        state, frame = _run_static_posture(supine_lying_keypoints)
        assert state == MandownState.MANDOWN, f"never confirmed (last state {state})"
        assert frame < 30, f"took {frame} frames — slower than expected"

    def test_prone_lying_reaches_mandown(self):
        state, frame = _run_static_posture(prone_lying_keypoints)
        assert state == MandownState.MANDOWN, f"never confirmed (last state {state})"
        assert frame < 30

    def test_side_lying_with_bent_knees_not_suppressed_by_kneeling_penalty(self):
        """Legs bent (natural side-lying) must not be suppressed by the
        kneeling penalty the way a true kneeling posture is — the torso
        itself is flat, which the upright_gate must recognize."""
        sim = _StaticTrackSimulator()
        kps = side_lying_keypoints()
        bbox = _bbox_for(kps)
        sim.step(kps, bbox)
        scores = sim.scorer.score(
            sim.base_history[-1], sim.ext_history[-1], sim.base_history, sim.ext_history
        )
        assert scores.kneeling_penalty < 0.2, "torso-flat side-lying wrongly read as kneeling"

    def test_side_lying_with_bent_knees_now_confirms(self):
        """Formerly a documented, honest limitation (see git history): a
        side-lying posture with significantly bent knees read as
        meaningfully elevated (mandown_confidence ~0.54, torso_angle
        correctly ~0.98, near-maximally "down") but didn't reach full
        MANDOWN — mandown_confidence blends down_score with
        fall_motion_score and floor_contact_score, both structurally 0 in
        this zero-history, never-calibrated synthetic scenario, diluting
        the composite below geometric_confirm_threshold regardless of how
        strong the orientation evidence itself was.

        V4.10 fixes this at the actual root: down_score itself (not the
        diluted blend) is the most direct signal of "is this person down",
        and was already independently strong enough to justify bypassing
        the LSTM-cap entirely -- it now also gets a direct path to force
        the confirmed-tier floor (paired with the same persistence
        requirement already used for the analogous feature_confidence
        path), instead of only being usable indirectly through a blend
        that dilutes it. Real per-camera-calibrated falls (see
        TestVerticalFallDetection below) benefit from the exact same fix."""
        sim = _StaticTrackSimulator()
        kps = side_lying_keypoints()
        bbox = _bbox_for(kps)
        state = None
        for _ in range(40):
            state = sim.step(kps, bbox)
        scores = sim.scorer.score(
            sim.base_history[-1], sim.ext_history[-1], sim.base_history, sim.ext_history
        )
        assert scores.mandown_confidence > 0.5
        assert scores.down_score >= 0.65
        assert state == MandownState.MANDOWN

    def test_standing_baseline_never_reaches_mandown(self):
        """Negative control — must NOT confirm for a static standing posture."""
        sim = _StaticTrackSimulator()
        kps = standing_keypoints()
        bbox = _bbox_for(kps)
        for _ in range(40):
            state = sim.step(kps, bbox)
        assert state == MandownState.NORMAL

    def test_diagonal_lying_reaches_mandown(self):
        """Formerly xfail (see git history): V4's torso-only boundary-
        sigmoid orientation cue substantially narrowed this gap (fusion
        score ~0.35 -> ~0.44), but fall_motion_score and floor_contact_score
        are both structurally 0 in this zero-history, never-calibrated
        synthetic scenario, diluting the composite mandown_confidence below
        geometric_confirm_threshold independent of how strong the
        orientation evidence (down_score) itself was.

        V4.10 closes the remainder: down_score now has a direct path to
        the confirmed-tier floor (paired with the same persistence
        requirement already used for the analogous feature_confidence
        path), instead of only being usable indirectly through a blend
        that dilutes it — the same fix that also closed the side-lying-
        bent-knee and calibrated-vertical-fall-cold-lstm gaps."""
        state, frame = _run_static_posture(diagonal_lying_keypoints)
        assert state == MandownState.MANDOWN, f"never confirmed (last state {state})"
        assert frame < 30


def deep_bend_keypoints() -> np.ndarray:
    """Standing, bent forward ~60 deg at the hip, legs straight/vertical —
    the highest-risk case for the V4 orientation fix: torso_angle reads
    almost as high as the diagonal-lying fixture (~0.64), purely from a
    forward bend, with kneeling_penalty unable to help (legs stay
    extended). Held statically for the full test (a real bend is
    transient); this is the worst-case sustained stress test."""
    return _kps({
        0: (500, 390, 0.9),
        5: (430, 350, 0.9), 6: (430, 370, 0.9),
        11: (320, 280, 0.9), 12: (320, 300, 0.9),
        13: (320, 380, 0.9), 14: (320, 400, 0.9),
        15: (320, 460, 0.9), 16: (320, 480, 0.9),
    })


def lunge_keypoints() -> np.ndarray:
    """Forward lunge: one leg bent deeply forward, the other extended back,
    torso still upright-ish."""
    return _kps({
        0: (330, 90, 0.9),
        5: (305, 160, 0.9), 6: (355, 160, 0.9),
        11: (310, 290, 0.9), 12: (350, 290, 0.9),
        13: (260, 360, 0.85), 14: (400, 350, 0.85),
        15: (240, 440, 0.8), 16: (460, 445, 0.8),
    })


def sitting_keypoints() -> np.ndarray:
    """Sitting in a chair: torso upright, hips/knees bent ~90 deg, lower
    legs roughly vertical below the knees."""
    return _kps({
        0: (320, 150, 0.9),
        5: (295, 210, 0.9), 6: (345, 210, 0.9),
        11: (300, 330, 0.9), 12: (340, 330, 0.9),
        13: (300, 400, 0.85), 14: (340, 400, 0.85),
        15: (300, 480, 0.8), 16: (340, 480, 0.8),
    })


def crouching_keypoints() -> np.ndarray:
    """Full crouch/squat: hips low and close to knees/ankles, torso
    upright-ish, both knees deeply bent."""
    return _kps({
        0: (320, 220, 0.9),
        5: (300, 270, 0.9), 6: (340, 270, 0.9),
        11: (300, 360, 0.9), 12: (340, 360, 0.9),
        13: (295, 400, 0.85), 14: (345, 400, 0.85),
        15: (295, 440, 0.8), 16: (345, 440, 0.8),
    })


def kneeling_keypoints() -> np.ndarray:
    """Classic kneeling: torso upright, both knees on the ground, lower
    legs folded flat/back under the body."""
    return _kps({
        0: (320, 150, 0.9),
        5: (300, 200, 0.9), 6: (340, 200, 0.9),
        11: (305, 320, 0.9), 12: (335, 320, 0.9),
        13: (300, 400, 0.9), 14: (340, 400, 0.9),
        15: (290, 405, 0.85), 16: (350, 405, 0.85),
    })


def semi_kneeling_keypoints() -> np.ndarray:
    """Semi-kneeling / genuflect: one knee down and folded under, the other
    leg planted forward bent ~90 deg — torso still upright."""
    return _kps({
        0: (330, 140, 0.9),
        5: (305, 190, 0.9), 6: (345, 190, 0.9),
        11: (310, 310, 0.9), 12: (335, 310, 0.9),
        13: (300, 400, 0.85), 14: (375, 350, 0.85),
        15: (290, 405, 0.8), 16: (385, 425, 0.8),
    })


def mild_camera_roll_standing_keypoints() -> np.ndarray:
    """A genuinely standing person, photographed by a camera with a ~18 deg
    Dutch-angle roll — the whole skeleton is rotated in image-space, but
    the person is physically upright. Must NOT be read as "down": catches
    `upright_boundary`/`boundary_steepness` being tuned too aggressively
    (this is the case most likely to expose an over-correction)."""
    base = standing_keypoints()
    angle = math.radians(18)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    cx, cy = FRAME_W / 2, FRAME_H / 2
    rotated = base.copy()
    for i in range(base.shape[0]):
        x, y, conf = base[i]
        dx, dy = x - cx, y - cy
        rotated[i] = [
            cx + dx * cos_a - dy * sin_a,
            cy + dx * sin_a + dy * cos_a,
            conf,
        ]
    return rotated


class TestOrientationNegativeControls:
    """V4: the new torso-only, boundary-sigmoid orientation cue and its
    body_axis_angle corroboration must not create a new false-positive
    class for postures that are emphatically not "down" — including the
    hardest cases (a deep sustained bend, a mild camera roll on a standing
    person) that most resemble the diagonal/side-lying cases being fixed."""

    def test_deep_bend_never_reaches_up_tier(self):
        sim = _StaticTrackSimulator()
        kps = deep_bend_keypoints()
        bbox = _bbox_for(kps)
        state = None
        for _ in range(40):
            state = sim.step(kps, bbox)
        assert state == MandownState.NORMAL

    def test_deep_bend_never_confirms_even_under_a_confidently_wrong_lstm(self):
        """V4.2 regression test for a real production bug: this exact real
        posture (bending near a table, screenshotted live) was re-confirmed
        as MANDOWN after the V4.2 height-collapse fix landed, via a route
        the cold-LSTM test above can't exercise — floor_proximity_score for
        a deep bend (~0.89, driven by orientation_cue/torso angle) is
        already higher than a genuine foreshortened fall's (~0.74), so once
        the LSTM contributes even moderately, the naive floor_proximity-
        based alternate grounding path let it through. Sweeps a range of
        LSTM confidences, cold through maximal, all of which must stay
        NORMAL — not just fail to reach MANDOWN on the final frame, but
        never transiently enter it either."""
        kps = deep_bend_keypoints()
        bbox = _bbox_for(kps)
        for lstm_prob in (0.0, 0.5, 0.75, 0.9, 0.95, 1.0):
            sim = _StaticTrackSimulator()
            for _ in range(60):
                state = sim.step(kps, bbox, lstm_mandown_prob=lstm_prob)
                assert state != MandownState.MANDOWN, f"confirmed at lstm={lstm_prob}"

    def test_standing_then_deep_bend_never_confirms(self):
        """V4.4 regression test for a second real production bug: with
        scene calibration active (this session's RTSP camera measured
        calibration_confidence=1.0), a deep-but-standing bend near a table
        shrinks body_height_norm enough that height_collapse_cue read
        ~0.499 -- a hair's breadth below the 0.5 gate, and a slightly
        deeper real bend (or a different camera's exact calibration
        numbers) would easily cross it, reaching mandown_confidence~0.565
        with the LSTM cold, dangerously close to the PRE-EXISTING V3
        geometric-alone-confirm threshold (0.60) with ZERO LSTM needed at
        all.

        This is the realistic shape of the real bug: a real person is seen
        STANDING before they bend down (they don't teleport into a bent
        posture), so this test captures that transition -- standing first
        (long enough to record a hip baseline), then bending -- rather than
        starting already bent. A V4.3 attempt gated this via a scene-wide
        FloorRegion evaluated at the hip; V4.4 replaced it with this
        per-track, self-relative "has the hip actually moved down" signal,
        which holds with real margin here (hip barely moves during a bend)
        rather than by luck of one fixture's exact numbers matching one
        camera's exact floor calibration."""
        standing_kps = standing_keypoints()
        standing_bbox = _bbox_for(standing_kps)
        bend_kps = deep_bend_keypoints()
        bend_bbox = _bbox_for(bend_kps)
        for lstm_prob in (0.0, 0.5, 0.75, 0.9, 0.95, 1.0):
            sim = _StaticTrackSimulator(
                expected_height_fn=_calibrated_expected_height_fn, floor_contact_score=0.9,
            )
            for _ in range(20):  # establish the standing hip baseline
                sim.step(standing_kps, standing_bbox, lstm_mandown_prob=lstm_prob)
            for _ in range(80):  # then bend, and stay bent
                state = sim.step(bend_kps, bend_bbox, lstm_mandown_prob=lstm_prob)
                assert state != MandownState.MANDOWN, f"confirmed at lstm={lstm_prob}"

    def test_deep_bend_with_no_prior_standing_history_stays_below_confirm(self):
        """Honest limitation, not hidden: a track that appears ALREADY bent
        from its very first frame (no captured standing-to-bending
        transition — e.g. the tracker only picks the person up mid-bend)
        has no hip baseline to compare against, so hip_drop_evidence is
        None and height_collapse_cue is NOT gated for it. This case is
        protected only by the pre-existing, narrower margin (down_score's
        own corroboration tops out ~0.60, and mandown_confidence ~0.565 —
        both below their respective 0.65/0.60 thresholds) established
        before this session's hip-drop work, not by the new signal. LSTM
        held cold, since a confidently-wrong LSTM combined with zero hip
        protection is exactly the scenario with the least margin."""
        kps = deep_bend_keypoints()
        bbox = _bbox_for(kps)
        sim = _StaticTrackSimulator(
            expected_height_fn=_calibrated_expected_height_fn, floor_contact_score=0.9,
        )
        state = None
        for _ in range(80):
            state = sim.step(kps, bbox, lstm_mandown_prob=0.0)
        assert state != MandownState.MANDOWN

    def test_lunge_never_reaches_up_tier(self):
        state, _ = _run_static_posture(lunge_keypoints)
        assert state == MandownState.NORMAL

    def test_sitting_never_reaches_up_tier(self):
        state, _ = _run_static_posture(sitting_keypoints)
        assert state == MandownState.NORMAL

    def test_crouching_never_reaches_up_tier(self):
        state, _ = _run_static_posture(crouching_keypoints)
        assert state == MandownState.NORMAL

    def test_mild_camera_roll_standing_never_reaches_up_tier(self):
        state, _ = _run_static_posture(mild_camera_roll_standing_keypoints)
        assert state == MandownState.NORMAL

    def test_kneeling_never_reaches_up_tier(self):
        state, _ = _run_static_posture(kneeling_keypoints)
        assert state == MandownState.NORMAL

    def test_semi_kneeling_never_reaches_up_tier(self):
        state, _ = _run_static_posture(semi_kneeling_keypoints)
        assert state == MandownState.NORMAL


# `_kps()` defaults every unset keypoint to a CONFIDENT (0.9) point at the
# frame center — harmless for angle-based fixtures (angles are computed
# from specific keypoint pairs, never from the bbox), but it silently
# pollutes `_bbox_for()` for a small, off-center cluster like the two
# fixtures below, since body_height_norm is directly bbox-height-sensitive.
# Explicitly zeroing the unused eyes/ears/elbows/wrists keeps the bbox
# honest for these two fixtures specifically.
_UNUSED_KEYPOINTS_ZEROED = {i: (0.0, 0.0, 0.0) for i in (1, 2, 3, 4, 7, 8, 9, 10)}


def vertical_fall_keypoints() -> np.ndarray:
    """Fallen toward/away from the camera ("vertically aligned" in a real
    RTSP report): all keypoints stay in a narrow x-band with only ~38px of
    raw vertical spread (vs. a calibrated expected standing height of
    ~430px in this file's frame scale, matching standing_keypoints()'s own
    bbox height). torso_angle/hip_angle read ~0.0 (genuinely "vertical" in
    2D) since there's no lateral spread at all -- this is the case
    angle-based cues are structurally blind to; only a calibrated
    body_height_norm shortfall reveals it."""
    return _kps({
        **_UNUSED_KEYPOINTS_ZEROED,
        0: (400, 440, 0.9),
        5: (398, 448, 0.9), 6: (402, 448, 0.9),
        11: (398, 458, 0.9), 12: (402, 458, 0.9),
        13: (398, 468, 0.9), 14: (402, 468, 0.9),
        15: (398, 478, 0.9), 16: (402, 478, 0.9),
    })


def occluded_upper_body_keypoints() -> np.ndarray:
    """Adversarial case: only the upper body is visible (legs hidden behind
    a desk/furniture) -- the real shape of the V4.1 sitting false positive.
    The visible bbox is naturally short, which could ALSO trip a
    body_height_norm shortfall; must stay suppressed via pose_quality
    (far fewer keypoints visible) even under a confidently-wrong LSTM.
    Knees/ankles explicitly zeroed too (not just the usual unused
    keypoints) -- they must read as genuinely occluded, not confidently
    placed at the frame center by `_kps()`'s default."""
    return _kps({
        **_UNUSED_KEYPOINTS_ZEROED,
        13: (0.0, 0.0, 0.0), 14: (0.0, 0.0, 0.0), 15: (0.0, 0.0, 0.0), 16: (0.0, 0.0, 0.0),
        0: (400, 300, 0.9),
        5: (392, 330, 0.9), 6: (408, 330, 0.9),
        11: (394, 380, 0.85), 12: (406, 380, 0.85),
    })


# Matches this file's own standing_keypoints() bbox height (nose y=60 to
# ankle y=450, default _bbox_for pad=20 -> ~430px) so a genuinely standing
# person in these fixtures' frame scale correctly reads body_height_norm~1.0
# under this stand-in calibration.
_CALIBRATED_STANDING_HEIGHT = 430.0


def _calibrated_expected_height_fn(
    bbox_bottom_y: float, observed_height: float, bbox_center_x: float = 0.0
) -> float:
    """Stand-in for a confident SceneProfile.normalize_height on a real,
    well-calibrated camera (this session's RTSP camera measured
    calibration_confidence=1.0, perspective r-squared=0.96). Flat in x —
    these fixtures test posture discrimination, not the 2D perspective
    fit's own horizontal-position term (see test_scene_profile.py for
    that)."""
    return observed_height / _CALIBRATED_STANDING_HEIGHT


class TestVerticalFallDetection:
    """V4.2: a person fallen toward/away from the camera reads as "standing"
    to every angle-based cue (torso_angle/hip_angle/body_axis_angle all stay
    near 0 -- there's no lateral spread to measure), which is exactly the
    real-world gap reported against this session's RTSP camera. Only a
    calibrated body_height_norm shortfall (the bbox is much shorter than
    this camera's known standing height at this image row) reveals it, so
    these tests specifically wire a calibration function through the real
    chain -- unlike every other test in this file, which deliberately tests
    the calibration-unavailable case."""

    def test_uncalibrated_vertical_fall_does_not_confirm(self):
        """Baseline: without scene calibration, body_height_norm has no
        real signal (neutral default) and the fix must not fabricate one —
        matches the pre-fix behavior exactly."""
        sim = _StaticTrackSimulator(expected_height_fn=None)
        kps = vertical_fall_keypoints()
        bbox = _bbox_for(kps, pad=10)
        state = None
        for _ in range(40):
            state = sim.step(kps, bbox)
        assert state == MandownState.NORMAL

    def test_calibrated_vertical_fall_with_realistic_lstm_reaches_mandown(self):
        """The real fix: with scene calibration active (as this session's
        RTSP camera has) and a realistic — not maximal — LSTM contribution
        (0.75, not the 0.95+ used elsewhere for "single strong signal"
        tests), the height-collapse-corroborated geometry plus persistence
        clears the confirmed_mandown_floor_proximity_floor alternate
        grounding path. Cold LSTM alone does not confirm this (see
        test_calibrated_vertical_fall_cold_lstm_stays_below_mandown) — real
        deployment needs the LSTM to contribute something, same as it does
        for every other posture in this codebase; this test proves the
        geometry no longer actively blocks it. floor_contact_score=0.9
        simulates this camera's floor_region also being calibrated (the
        same real camera this fix targets has floor_region.is_valid=true
        alongside its height calibration — both come from the same
        scene-profiling pipeline, so a bare 0.0 here would understate a
        realistic deployment). This track has no prior standing history
        (an "already lying at start" case, per this class's docstring), so
        hip_drop_evidence is None and height_collapse_cue is NOT gated —
        exactly the intended behavior: the V4.4 hip-drop signal only ever
        applies extra scrutiny to a track that WAS seen standing (see
        TestOrientationNegativeControls.test_standing_then_deep_bend_never_confirms),
        never to a genuinely-already-down person with nothing to compare
        against."""
        sim = _StaticTrackSimulator(
            expected_height_fn=_calibrated_expected_height_fn, floor_contact_score=0.9,
        )
        kps = vertical_fall_keypoints()
        bbox = _bbox_for(kps, pad=10)
        state = None
        for _ in range(40):
            state = sim.step(kps, bbox, lstm_mandown_prob=0.75)
        assert state == MandownState.MANDOWN

    def test_calibrated_vertical_fall_confirms_even_with_cold_lstm(self):
        """Formerly an honest limitation (see git history): geometry alone
        (cold LSTM) substantially improved down_score for this orientation
        via height-collapse corroboration, but mandown_confidence (a blend
        of down_score with several other components) stayed below
        geometric_confirm_threshold, so the old fusion logic couldn't
        confirm without LSTM help despite down_score itself already
        clearing confirmed_mandown_down_floor (0.713, measured here).

        V4.10 fixes this: down_score now has a direct path to the
        confirmed-tier floor (paired with the same persistence requirement
        already used for the analogous feature_confidence path), instead
        of only being usable indirectly through a blend that dilutes it.
        This is the real V3 promise realized more fully -- "strong,
        persistent geometric evidence alone" should not need the LSTM's
        help, and now genuinely doesn't for this orientation either."""
        sim = _StaticTrackSimulator(expected_height_fn=_calibrated_expected_height_fn)
        kps = vertical_fall_keypoints()
        bbox = _bbox_for(kps, pad=10)
        state = None
        for _ in range(40):
            state = sim.step(kps, bbox, lstm_mandown_prob=0.0)
        scores = sim.scorer.score(
            sim.base_history[-1], sim.ext_history[-1], sim.base_history, sim.ext_history
        )
        assert scores.down_score >= 0.65
        assert state == MandownState.MANDOWN

    def test_occluded_upper_body_not_falsely_confirmed_even_with_strong_wrong_lstm(self):
        """The adversarial case this fix must not reopen: a partially-
        occluded seated/standing person (V4.1's real sitting false
        positive) also shows a body_height_norm shortfall from its
        naturally-short visible bbox, but with far lower pose_quality
        (fewer visible keypoints) and a lower floor_proximity_score than a
        genuine fall — both fail confirmed_mandown_floor_proximity_floor's
        AND condition. Tested against a MORE confident wrong LSTM (0.95)
        than the genuine-fall test above (0.75) precisely because this is
        the case that must hold up hardest. No prior standing history
        here either (see TestVerticalFallDetection's equivalent note), so
        hip_drop_evidence is None -- this case is protected by
        pose_quality and floor_proximity_score alone, which is sufficient
        on its own."""
        sim = _StaticTrackSimulator(expected_height_fn=_calibrated_expected_height_fn)
        kps = occluded_upper_body_keypoints()
        bbox = _bbox_for(kps, pad=10)
        state = None
        for _ in range(40):
            state = sim.step(kps, bbox, lstm_mandown_prob=0.95)
        assert state == MandownState.NORMAL

    def test_calibrated_standing_not_falsely_confirmed(self):
        """Sanity: a genuinely standing person, even with scene calibration
        active and a moderately-confident wrong LSTM, must stay NORMAL —
        body_height_norm reads ~1.0 (matches expectation), so height_cue
        stays near its own uncalibrated-equivalent low reading."""
        sim = _StaticTrackSimulator(expected_height_fn=_calibrated_expected_height_fn)
        kps = standing_keypoints()
        bbox = _bbox_for(kps)
        state = None
        for _ in range(40):
            state = sim.step(kps, bbox, lstm_mandown_prob=0.75)
        assert state != MandownState.MANDOWN
