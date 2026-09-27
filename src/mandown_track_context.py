"""Per-track state for the feature-based Mandown / fusion / state-machine
layer. Kept separate from `TrackManager` (which only manages the LSTM's
30-frame sliding window) so that class's tested behavior is untouched.
"""

from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from src.config import (
    DetectionRecoveryConfig,
    HipDropConfig,
    MotionSignatureConfig,
    StateMachineConfig,
    TrackBridgeConfig,
)
from src.feature_extraction import ExtendedFeatureVector
from src.geometry import Box, iou
from src.state_machine import TrackStateMachine

logger = logging.getLogger(__name__)

# Generous enough to cover MotionSignatureConfig's default 5s+5s=10s
# window at up to ~40fps, with margin -- a fixed cap (not derived from a
# runtime fps) since MandownTrackContext is constructed before any
# per-camera fps is necessarily known.
_MOTION_HISTORY_MAXLEN = 450


def _sigmoid(x: float, steepness: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-steepness * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _bbox_diagonal(bbox: Box) -> float:
    x1, y1, x2, y2 = bbox
    return max(math.hypot(x2 - x1, y2 - y1), 1e-6)


def _motion_energy(
    keypoints: np.ndarray, prev_keypoints: np.ndarray, bbox: Box, conf_threshold: float = 0.3,
) -> float:
    """Mean per-keypoint displacement (for keypoints confidently visible
    in BOTH frames), normalized by this frame's bbox diagonal so it's
    comparable across distances/resolutions. 0.0 when no keypoint is
    confidently visible in both frames (never fabricates motion from
    missing data)."""
    diag = _bbox_diagonal(bbox)
    conf_now = keypoints[:, 2]
    conf_prev = prev_keypoints[:, 2]
    valid = (conf_now >= conf_threshold) & (conf_prev >= conf_threshold)
    if not np.any(valid):
        return 0.0
    dists = np.hypot(
        keypoints[valid, 0] - prev_keypoints[valid, 0],
        keypoints[valid, 1] - prev_keypoints[valid, 1],
    )
    return float(np.mean(dists) / diag)


class MandownTrackContext:
    """Extended-feature history and state machine for one tracked person."""

    def __init__(self, history_maxlen: int, state_machine_config: StateMachineConfig | None = None):
        self.ext_history: deque[ExtendedFeatureVector] = deque(maxlen=history_maxlen)
        self.state_machine = TrackStateMachine(state_machine_config)
        self.prev_cog_x: float | None = None
        self.prev_vertical_velocity: float | None = None
        self.last_seen_frame: int = 0
        self.last_known_bbox: tuple[float, float, float, float] | None = None
        # V4.4: this track's own most recent hip height (cog_height) while
        # confidently upright -- a self-relative baseline for "has the hip
        # since moved down toward the floor", which needs no scene/floor
        # calibration at all (unlike the floor_region-based hip check it
        # replaces, which proved too imprecise for real camera perspective
        # and blocked genuine falls on top of the bends it was meant to
        # exclude). None until this track is ever observed confidently
        # upright -- deliberately never fabricated for a track that
        # appeared already lying down, so the V3 already-lying-at-start
        # guarantee (confirm from geometry alone, no prior standing
        # history required) is completely unaffected by this signal's
        # absence.
        self.last_standing_cog_height: float | None = None
        # V6 Fix Set B: (frame_num, body_height_norm) / (frame_num,
        # motion_energy) history for `collapse_then_stillness` -- see
        # MotionSignatureConfig. motion_energy needs a PREVIOUS frame's
        # raw keypoints to compute a displacement, so the previous raw
        # array is kept here (not in ExtendedFeatureVector, which never
        # carries raw keypoints by design).
        self.height_norm_history: deque[tuple[int, float]] = deque(maxlen=_MOTION_HISTORY_MAXLEN)
        self.motion_energy_history: deque[tuple[int, float]] = deque(maxlen=_MOTION_HISTORY_MAXLEN)
        self._prev_keypoints: np.ndarray | None = None
        # V6 Fix Set C (§4.1): last angle a rotation-search re-acquisition
        # won at for this track, decayed back toward 0 when not
        # re-triggered -- see RotationSearchConfig.
        self.last_rotation_search_angle: float = 0.0
        # V6 Fix Set C (§4.2): last 2 CONFIDENT raw keypoint arrays +
        # frame numbers, for constant-velocity extrapolation across a
        # brief true detection gap -- see DetectionRecoveryConfig.
        self._confident_keypoint_history: deque[tuple[int, np.ndarray]] = deque(maxlen=2)

    def update(
        self, ext: ExtendedFeatureVector, cog_x: float, vertical_velocity: float, frame_num: int,
        bbox: tuple[float, float, float, float] | None = None,
        keypoints: "np.ndarray | None" = None,
    ) -> None:
        self.ext_history.append(ext)
        self.prev_cog_x = cog_x
        self.prev_vertical_velocity = vertical_velocity
        self.last_seen_frame = frame_num
        if bbox is not None:
            self.last_known_bbox = bbox
        self.height_norm_history.append((frame_num, ext.body_height_norm))
        if keypoints is not None:
            if self._prev_keypoints is not None and bbox is not None:
                motion = _motion_energy(keypoints, self._prev_keypoints, bbox)
                self.motion_energy_history.append((frame_num, motion))
            self._prev_keypoints = keypoints.copy()
            self._confident_keypoint_history.append((frame_num, keypoints.copy()))

    def update_standing_baseline(self, cog_height: float, is_confidently_upright: bool) -> None:
        """Call once per frame with a check independent of this frame's own
        Mandown classification (e.g. torso_angle clearly upright) -- avoids
        the circular risk of using the fusion/state-machine's own output to
        decide whether to trust this frame as a "standing" baseline."""
        if is_confidently_upright:
            self.last_standing_cog_height = cog_height

    def hip_drop_evidence(self, current_cog_height: float, config: HipDropConfig) -> float | None:
        """How confident is a genuine hip collapse, based purely on this
        track's own history — None when this track has never been observed
        confidently upright (nothing to compare against; never fabricated).
        cog_height is 0=top/1=bottom of frame, so a positive
        current-minus-baseline delta means the hip has moved DOWN."""
        if self.last_standing_cog_height is None:
            return None
        drop = current_cog_height - self.last_standing_cog_height
        return _sigmoid(drop - config.drop_threshold, config.steepness)

    def gap_confidence_multiplier(
        self, frame_num: int, hold_frames: int, min_multiplier: float
    ) -> float:
        """Linear discount for evidence coming from a track that just went
        through a detection gap — reflects genuine uncertainty about what
        happened during the gap, without fabricating or fully discarding it."""
        gap = frame_num - self.last_seen_frame
        if gap <= 1 or hold_frames <= 0:
            return 1.0
        frac = min(1.0, gap / hold_frames)
        return 1.0 - frac * (1.0 - min_multiplier)

    def collapse_then_stillness(
        self,
        frame_num: int,
        fps: float,
        height_collapse_cue: float,
        is_confidently_upright_now: bool,
        config: MotionSignatureConfig,
    ) -> bool:
        """V6 Fix Set B: hard 0/1 gate, True only when ALL of (a)-(d) in
        `MotionSignatureConfig`'s docstring hold. Call with THIS frame's
        `ext`/keypoints already recorded via `update()` -- the window
        below includes the current frame.

        Explicit negative controls this is designed to reject (see
        MANDOWN_IMPLEMENTATION_REPORT_V6 §3.4 for the reasoning and any
        real numbers logged against them):
          - a deliberate, gradual sit-down (no genuine 1s spike -- fails (a))
          - a chair scoot-back (no real height collapse -- fails (c))
          - bending to help someone then standing back up (stillness never
            sustains for `t_still_seconds` -- fails (b))
        """
        if fps <= 0:
            return False
        if is_confidently_upright_now:
            return False
        if height_collapse_cue < config.height_collapse_floor:
            return False

        lookback_frames = max(int(config.t_lookback_seconds * fps), 1)
        still_frames_needed = max(int(config.t_still_seconds * fps), 1)
        one_sec_frames = max(int(round(fps)), 1)

        hn_by_frame = dict(self.height_norm_history)
        if len(hn_by_frame) < 2:
            return False
        available_frames = sorted(hn_by_frame.keys())

        # (a) a genuine 1s collapse spike at some tau_0 -- searched over
        # [frame_num - T_lookback - T_still, frame_num] (per the V6 spec
        # §3.2: tau_0 must be recent enough that the required STILLNESS
        # duration since it could still fit before frame_num, not just
        # within the lookback window alone).
        search_start = frame_num - lookback_frames - still_frames_needed
        spike_frame: int | None = None
        # Earliest qualifying tau_0 first: maximizes the elapsed-since-
        # spike duration available to satisfy (b) below, matching "at
        # SOME tau_0" in the spec -- a later, still-qualifying point
        # within the same transition window (a step change keeps
        # re-triggering the 1s-window comparison for several frames after
        # it happens) would only make (b)'s duration requirement harder
        # to satisfy for no real reason.
        for t in available_frames:
            if t < search_start or t > frame_num:
                continue
            target_prev = t - one_sec_frames
            candidates = [f for f in available_frames if f <= target_prev]
            if not candidates:
                continue
            t_prev = max(candidates)
            collapse_velocity = hn_by_frame[t_prev] - hn_by_frame[t]
            if collapse_velocity >= config.theta_spike:
                spike_frame = t
                break
        if spike_frame is None:
            return False

        # (b) stillness has held continuously since that spike, for at
        # least t_still_seconds.
        if frame_num - spike_frame < still_frames_needed:
            return False
        me_by_frame = dict(self.motion_energy_history)
        for f in range(spike_frame, frame_num + 1):
            energy = me_by_frame.get(f)
            if energy is not None and energy > config.theta_still:
                return False

        return True

    def extrapolated_keypoints(
        self, frame_num: int, config: DetectionRecoveryConfig
    ) -> tuple[np.ndarray, float] | None:
        """V6 Fix Set C (§4.2): constant-velocity extrapolation of this
        track's last 2 confident raw keypoint arrays across a brief TRUE
        detection gap (straight-crop AND rotation-search re-acquisition
        both already failed this frame -- see pipeline.py). Returns
        (predicted_keypoints, synthetic_confidence), or None when there's
        no 2-frame history to extrapolate from, or the gap has exceeded
        `t_gap_max_seconds` worth of frames (using the last known
        inter-frame spacing as the timestep -- this class has no direct
        fps, so the caller is expected to have already checked the
        wall-clock gap via `last_seen_frame`/fps before calling this;
        this method only refuses an obviously-stale 2-frame history).

        Confidence at every keypoint is set uniformly to
        `synthetic_confidence`, decayed by `confidence_decay_factor` per
        frame since the last real detection -- this feeds the EXACT same
        downstream feature/fusion pipeline a real detection would (pose_
        quality_score's existing confidence discount already handles
        decaying trust), composing with existing logic instead of adding
        a new one."""
        if len(self._confident_keypoint_history) < 2:
            return None
        (f0, kp0), (f1, kp1) = self._confident_keypoint_history
        dt = f1 - f0
        if dt <= 0:
            return None
        frames_since = frame_num - f1
        if frames_since <= 0:
            return None

        velocity = (kp1[:, :2] - kp0[:, :2]) / dt
        predicted_xy = kp1[:, :2] + velocity * frames_since
        base_confidence = float(np.clip(np.mean(kp1[:, 2]), 0.0, 1.0))
        synthetic_confidence = base_confidence * (config.confidence_decay_factor ** frames_since)
        predicted = np.column_stack([predicted_xy, np.full(17, synthetic_confidence, dtype=np.float32)])
        return predicted.astype(np.float32), synthetic_confidence


class MandownTrackContextManager:
    """Per-track_id registry of `MandownTrackContext`, mirroring
    `TrackManager`'s create-on-demand / stale-cleanup pattern."""

    def __init__(self, history_maxlen: int = 30, stale_threshold: int = 90,
                 state_machine_config: StateMachineConfig | None = None,
                 track_bridge_config: TrackBridgeConfig | None = None):
        self.history_maxlen = history_maxlen
        self.stale_threshold = stale_threshold
        self.state_machine_config = state_machine_config
        self.track_bridge_config = track_bridge_config or TrackBridgeConfig()
        self.contexts: dict[int, MandownTrackContext] = {}

    def get_or_create(
        self, track_id: int, bbox: Box | None = None, frame_num: int | None = None,
    ) -> MandownTrackContext:
        """Get this track's context, creating one on first sight — unless a
        brand-new track_id's first bbox closely overlaps a just-lost
        track's last known position, in which case that track's EXISTING
        context (history + state machine) is adopted instead of starting
        fresh. `bbox`/`frame_num` are optional so existing callers that
        don't need bridging are unaffected.

        The caller's own gap handling (checking `last_seen_frame` and
        calling `TrackStateMachine.on_gap()`) runs unchanged immediately
        after this returns — a bridged context's `last_seen_frame` still
        reflects the OLD track_id's last update, so the real elapsed gap is
        computed and decayed exactly as it already is for a same-ID
        detection gap. This is not a new mechanism, just a new way to reach
        the existing one."""
        if track_id not in self.contexts:
            bridged_from = None
            if bbox is not None and frame_num is not None:
                bridged_from = self._find_bridge_source(bbox, frame_num)
            if bridged_from is not None:
                self.contexts[track_id] = self.contexts.pop(bridged_from)
                logger.info(
                    f"Bridged mandown context: track {bridged_from} -> {track_id} "
                    f"(same physical person, fragmented tracking)"
                )
            else:
                self.contexts[track_id] = MandownTrackContext(
                    self.history_maxlen, self.state_machine_config
                )
                logger.debug(f"New mandown track context created: {track_id}")
        return self.contexts[track_id]

    def _find_bridge_source(self, bbox: Box, frame_num: int) -> int | None:
        cfg = self.track_bridge_config
        best_id, best_iou = None, 0.0
        for tid, ctx in self.contexts.items():
            if ctx.last_known_bbox is None:
                continue
            gap = frame_num - ctx.last_seen_frame
            if gap <= 0 or gap > cfg.recency_frames:
                continue
            score = iou(bbox, ctx.last_known_bbox)
            if score >= cfg.iou_threshold and score > best_iou:
                best_id, best_iou = tid, score
        return best_id

    def cleanup(self, current_frame: int) -> list[int]:
        stale_ids = [
            tid for tid, ctx in self.contexts.items()
            if current_frame - ctx.last_seen_frame > self.stale_threshold
        ]
        for tid in stale_ids:
            del self.contexts[tid]
        return stale_ids

    def reset(self) -> None:
        self.contexts.clear()
