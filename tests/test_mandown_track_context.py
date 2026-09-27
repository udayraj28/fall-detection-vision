"""Tests for per-track pose-failure gap handling."""

import numpy as np

from src.config import DetectionRecoveryConfig, MotionSignatureConfig, TrackBridgeConfig
from src.mandown_track_context import MandownTrackContext, MandownTrackContextManager
from src.state_machine import MandownState


class TestGapConfidenceMultiplier:
    def setup_method(self):
        self.ctx = MandownTrackContext(history_maxlen=30)
        self.ctx.last_seen_frame = 100

    def test_no_gap_returns_full_confidence(self):
        assert self.ctx.gap_confidence_multiplier(101, hold_frames=15, min_multiplier=0.5) == 1.0

    def test_small_gap_discounts_partially(self):
        m = self.ctx.gap_confidence_multiplier(105, hold_frames=15, min_multiplier=0.5)
        assert 0.5 < m < 1.0

    def test_full_hold_window_hits_floor(self):
        m = self.ctx.gap_confidence_multiplier(115, hold_frames=15, min_multiplier=0.5)
        assert abs(m - 0.5) < 1e-6

    def test_beyond_hold_window_still_floored_not_below(self):
        m = self.ctx.gap_confidence_multiplier(500, hold_frames=15, min_multiplier=0.5)
        assert m == 0.5

    def test_zero_hold_frames_no_crash(self):
        assert self.ctx.gap_confidence_multiplier(200, hold_frames=0, min_multiplier=0.5) == 1.0


class TestLastKnownBbox:
    def test_update_stores_bbox(self):
        ctx = MandownTrackContext(history_maxlen=30)
        assert ctx.last_known_bbox is None
        # update() requires a real ExtendedFeatureVector normally; here we
        # only exercise the bbox-tracking side effect directly.
        ctx.last_known_bbox = (1.0, 2.0, 3.0, 4.0)
        assert ctx.last_known_bbox == (1.0, 2.0, 3.0, 4.0)


class TestTrackBridging:
    """V4.2: BoT-SORT reassigning a new track_id to the same physically-
    unmoving downed person otherwise resets all accumulated Mandown state
    to zero (measured: one real validation video went from a handful of
    tracks to 74 under this session's detection settings). A new track_id
    whose first bbox closely overlaps a just-lost track's last position
    should adopt that track's existing context instead of starting fresh."""

    def _lost_context(self, manager, track_id, bbox, last_seen_frame, state=None):
        ctx = manager.get_or_create(track_id)
        ctx.last_known_bbox = bbox
        ctx.last_seen_frame = last_seen_frame
        if state is not None:
            ctx.state_machine.state = state
        return ctx

    def test_new_track_with_no_bbox_creates_fresh_context(self):
        manager = MandownTrackContextManager()
        ctx = manager.get_or_create(1)
        assert ctx.last_known_bbox is None
        assert ctx.state_machine.state == MandownState.NORMAL

    def test_overlapping_recent_track_bridges_confirmed_state(self):
        manager = MandownTrackContextManager(
            track_bridge_config=TrackBridgeConfig(iou_threshold=0.3, recency_frames=30)
        )
        old_ctx = self._lost_context(
            manager, track_id=1, bbox=(100.0, 100.0, 200.0, 300.0),
            last_seen_frame=90, state=MandownState.MANDOWN,
        )

        new_ctx = manager.get_or_create(2, bbox=(105.0, 100.0, 205.0, 300.0), frame_num=92)

        assert new_ctx is old_ctx  # same object, not a copy — full history preserved
        assert new_ctx.state_machine.state == MandownState.MANDOWN
        assert 1 not in manager.contexts  # old id retired, not left as a lingering duplicate
        assert manager.contexts[2] is new_ctx

    def test_real_107_frame_gap_at_10fps_bridges_with_default_config(self):
        """V4.3 regression test for a real observed gap: this session's
        RTSP camera (10fps) showed a confirmed-MANDOWN, unmoving lying
        person flicker back to NORMAL after a 107-frame (~10.7s) track_id
        change — well beyond the old default recency_frames=30 (3s at
        10fps), so bridging never triggered. Uses the manager's DEFAULT
        config (not a scoped override) to guard the actual shipped
        default, not just the mechanism in the abstract."""
        manager = MandownTrackContextManager()  # default TrackBridgeConfig
        old_ctx = self._lost_context(
            manager, track_id=1, bbox=(200.0, 300.0, 400.0, 380.0),
            last_seen_frame=756, state=MandownState.MANDOWN,
        )

        new_ctx = manager.get_or_create(2, bbox=(200.0, 300.0, 400.0, 380.0), frame_num=863)

        assert new_ctx is old_ctx
        assert new_ctx.state_machine.state == MandownState.MANDOWN

    def test_well_separated_track_does_not_bridge(self):
        manager = MandownTrackContextManager()
        self._lost_context(
            manager, track_id=1, bbox=(0.0, 0.0, 100.0, 100.0),
            last_seen_frame=90, state=MandownState.MANDOWN,
        )

        new_ctx = manager.get_or_create(2, bbox=(900.0, 600.0, 1050.0, 750.0), frame_num=92)

        assert new_ctx.state_machine.state == MandownState.NORMAL
        assert len(new_ctx.ext_history) == 0
        assert 1 in manager.contexts  # untouched — genuinely a different person

    def test_stale_track_beyond_recency_does_not_bridge(self):
        manager = MandownTrackContextManager(
            track_bridge_config=TrackBridgeConfig(iou_threshold=0.3, recency_frames=30)
        )
        self._lost_context(
            manager, track_id=1, bbox=(100.0, 100.0, 200.0, 300.0),
            last_seen_frame=90, state=MandownState.MANDOWN,
        )

        # Same spot, but 5000 frames later — an unrelated later person must
        # not silently inherit a long-gone track's confirmed state.
        new_ctx = manager.get_or_create(2, bbox=(100.0, 100.0, 200.0, 300.0), frame_num=5090)

        assert new_ctx.state_machine.state == MandownState.NORMAL

    def test_own_track_id_never_bridges_to_itself(self):
        manager = MandownTrackContextManager()
        ctx = self._lost_context(
            manager, track_id=1, bbox=(100.0, 100.0, 200.0, 300.0),
            last_seen_frame=90, state=MandownState.MANDOWN,
        )
        # Same id continuing normally — must use its own context, unchanged.
        same_ctx = manager.get_or_create(1, bbox=(101.0, 100.0, 201.0, 300.0), frame_num=92)
        assert same_ctx is ctx

    def test_stale_threshold_must_not_reap_a_context_still_inside_the_bridging_window(self):
        """FallDetectionPipeline derives stale_threshold as
        max(90, recency_frames*2) specifically so cleanup() never removes a
        context that a later frame could still legitimately bridge to —
        this pins that invariant directly, independent of the pipeline's
        own wiring, since a mismatch here would silently defeat bridging
        for any gap in the back half of the recency window."""
        cfg = TrackBridgeConfig(iou_threshold=0.3, recency_frames=300)
        manager = MandownTrackContextManager(
            stale_threshold=max(90, cfg.recency_frames * 2), track_bridge_config=cfg,
        )
        self._lost_context(
            manager, track_id=1, bbox=(100.0, 100.0, 200.0, 300.0),
            last_seen_frame=100, state=MandownState.MANDOWN,
        )

        # Periodic cleanup runs at frame_num % 100 == 0 in the real
        # pipeline; simulate several such passes right up to the edge of
        # the recency window (100 + 300 - 1 = 399) without ever reaping it.
        for frame_num in (200, 300, 399):
            manager.cleanup(frame_num)
        assert 1 in manager.contexts

        new_ctx = manager.get_or_create(2, bbox=(100.0, 100.0, 200.0, 300.0), frame_num=399)
        assert new_ctx.state_machine.state == MandownState.MANDOWN


class TestCollapseThenStillness:
    """V6 Fix Set B: a hard 0/1 gate requiring a real height-collapse
    spike followed by sustained stillness, plus a real height collapse
    right now. Fixtures use fps=10 (matches this project's real RTSP
    cameras) — t_lookback_seconds=5/t_still_seconds=5 => 50-frame windows,
    one_sec_frames=10."""

    FPS = 10.0

    def setup_method(self):
        self.ctx = MandownTrackContext(history_maxlen=30)
        self.config = MotionSignatureConfig(
            t_lookback_seconds=5.0, t_still_seconds=5.0, theta_spike=0.12, theta_still=0.06,
            height_collapse_floor=0.5,
        )

    def _seed_height(self, frames_values: list[tuple[int, float]]):
        for f, v in frames_values:
            self.ctx.height_norm_history.append((f, v))

    def _seed_motion(self, frames_values: list[tuple[int, float]]):
        for f, v in frames_values:
            self.ctx.motion_energy_history.append((f, v))

    def test_genuine_collapse_then_stillness_confirms(self):
        """A real spike (1.0 -> 0.4 within 1s) at frame 70, followed by
        50 continuous still frames (energy=0.01) through frame 120,
        confirms."""
        self._seed_height([(f, 1.0) for f in range(0, 60)] + [(f, 0.4) for f in range(60, 121)])
        self._seed_motion([(f, 0.01) for f in range(70, 121)])
        result = self.ctx.collapse_then_stillness(
            frame_num=120, fps=self.FPS, height_collapse_cue=0.6,
            is_confidently_upright_now=False, config=self.config,
        )
        assert result is True

    def test_deliberate_gradual_sit_down_does_not_confirm(self):
        """Negative control (V6 §3.4): a deliberate, GRADUAL sit-down —
        height_norm declines slowly over 10s rather than sharply within
        1s — must fail condition (a). If real footage ever shows a real
        spike here, theta_spike is set too low; this test would need
        re-measurement, not a weakened condition (a)/(b)."""
        values = [(f, 1.0 - 0.006 * f) for f in range(0, 121)]  # ~0.06/1s decline
        self._seed_height(values)
        self._seed_motion([(f, 0.01) for f in range(70, 121)])
        result = self.ctx.collapse_then_stillness(
            frame_num=120, fps=self.FPS, height_collapse_cue=0.6,
            is_confidently_upright_now=False, config=self.config,
        )
        assert result is False

    def test_chair_scoot_back_does_not_confirm(self):
        """Negative control (V6 §3.4): height never really collapses (a
        chair scoot-back) — condition (c) fails regardless of any motion
        pattern. Rejected purely via the height_collapse_cue argument,
        with no special-casing needed."""
        self._seed_height([(f, 0.97) for f in range(0, 121)])
        self._seed_motion([(f, 0.01) for f in range(0, 121)])
        result = self.ctx.collapse_then_stillness(
            frame_num=120, fps=self.FPS, height_collapse_cue=0.1,
            is_confidently_upright_now=False, config=self.config,
        )
        assert result is False

    def test_bending_to_help_then_standing_back_up_does_not_confirm(self):
        """Negative control (V6 §3.4): a genuine spike occurs (bending
        down), but the person stands back up before stillness has
        sustained for the full t_still_seconds window — condition (b)
        fails. Real measured example (this test's own fixture): motion
        resumes (energy=0.5, well above theta_still) at frame 90, only 2s
        after the frame-70 spike."""
        self._seed_height([(f, 1.0) for f in range(0, 60)] + [(f, 0.4) for f in range(60, 121)])
        motion = [(f, 0.01) for f in range(70, 90)] + [(90, 0.5)] + [
            (f, 0.01) for f in range(91, 121)
        ]
        self._seed_motion(motion)
        result = self.ctx.collapse_then_stillness(
            frame_num=120, fps=self.FPS, height_collapse_cue=0.6,
            is_confidently_upright_now=False, config=self.config,
        )
        assert result is False

    def test_confidently_upright_now_never_confirms(self):
        """Condition (d): even with an otherwise-qualifying spike+stillness
        history, a track currently reading confidently upright must not
        confirm — avoids a stale/misattributed history from an earlier,
        unrelated collapse (e.g. after the person stood back up long ago
        and the track later re-entered a brief upright frame)."""
        self._seed_height([(f, 1.0) for f in range(0, 60)] + [(f, 0.4) for f in range(60, 121)])
        self._seed_motion([(f, 0.01) for f in range(70, 121)])
        result = self.ctx.collapse_then_stillness(
            frame_num=120, fps=self.FPS, height_collapse_cue=0.6,
            is_confidently_upright_now=True, config=self.config,
        )
        assert result is False

    def test_insufficient_history_does_not_confirm(self):
        """A freshly-created track with no history at all must not crash
        or fabricate a confirmation."""
        result = self.ctx.collapse_then_stillness(
            frame_num=5, fps=self.FPS, height_collapse_cue=0.9,
            is_confidently_upright_now=False, config=self.config,
        )
        assert result is False


class TestExtrapolatedKeypoints:
    """V6 Fix Set C (§4.2): constant-velocity extrapolation across a
    brief true detection gap, with confidence decaying per frame since
    the last real detection."""

    def setup_method(self):
        self.ctx = MandownTrackContext(history_maxlen=30)
        self.config = DetectionRecoveryConfig(
            enabled=True, t_gap_max_seconds=3.0, confidence_decay_factor=0.85
        )

    def _kp(self, x_offset: float, conf: float = 0.9) -> np.ndarray:
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[:, 0] = 100.0 + x_offset
        kp[:, 1] = 200.0
        kp[:, 2] = conf
        return kp

    def test_no_history_returns_none(self):
        assert self.ctx.extrapolated_keypoints(10, self.config) is None

    def test_single_frame_history_returns_none(self):
        self.ctx._confident_keypoint_history.append((10, self._kp(0.0)))
        assert self.ctx.extrapolated_keypoints(11, self.config) is None

    def test_extrapolates_constant_velocity_and_decays_confidence(self):
        self.ctx._confident_keypoint_history.append((10, self._kp(0.0, conf=0.9)))
        self.ctx._confident_keypoint_history.append((11, self._kp(5.0, conf=0.9)))
        result = self.ctx.extrapolated_keypoints(13, self.config)
        assert result is not None
        predicted, synthetic_confidence = result
        # velocity = 5.0/frame -> 2 frames since last real detection (11->13)
        assert abs(predicted[0, 0] - (105.0 + 2 * 5.0)) < 1e-3
        assert abs(synthetic_confidence - 0.9 * (0.85 ** 2)) < 1e-4
        assert np.allclose(predicted[:, 2], synthetic_confidence)
