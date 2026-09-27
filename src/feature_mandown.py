"""Deterministic, feature-based Mandown scoring — independent of the LSTM.

Detects strong evidence a person is down using explicit, continuous
(non hard-thresholded) combinations of posture, motion, and geometry cues,
so it can catch Mandown events the LSTM never saw in training. Distinguishes
two cases, both of which should score high:

  A. Falling/down transition — caught by `fall_motion_score` (a recent spike
     in vertical velocity / angle-change / acceleration).
  B. Already-down/lying person — caught by `down_score` (posture/shape) and
     `floor_proximity_score` (head-to-knee vertical span collapsing), which
     stay high even with no motion history at all.

No single feature (e.g. body angle alone) can drive the score alone; each
sub-score blends multiple cues through smooth sigmoids rather than a single
hard-coded threshold.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.config import FeatureMandownConfig, KneelingConfig
from src.feature_extraction import ExtendedFeatureVector, FeatureVector


@dataclass
class MandownFeatureScores:
    """Diagnostic sub-scores plus the final combined feature-based confidence."""

    down_score: float
    fall_motion_score: float
    floor_proximity_score: float
    floor_contact_score: float
    persistence_score: float
    pose_quality_score: float
    kneeling_penalty: float
    leg_evidence_quality: float  # 0.0 (occluded) / 0.5 / 1.0 — transparency for §6-style audits
    # V4.2: exposed separately from floor_proximity_score, which also
    # absorbs orientation_cue via max() -- a forward torso pitch (deep
    # bending) drives floor_proximity_score just as high as a genuine
    # height collapse does (measured: bend=0.893 > a real foreshortened
    # fall's 0.742), so floor_proximity_score alone cannot gate the
    # LSTM-assisted confirmation path in fusion.py without reopening the
    # V4.1 bending false positive through this new path. height_collapse_cue
    # is driven ONLY by body_height_norm, so it stays low (~0.15) for a
    # bend (uncalibrated angle-driven cue, not a real collapse) while
    # staying high for a genuine fallen-toward-camera person.
    height_collapse_cue: float
    mandown_confidence: float

    def to_dict(self) -> dict:
        return {
            "down_score": round(self.down_score, 3),
            "fall_motion_score": round(self.fall_motion_score, 3),
            "floor_proximity_score": round(self.floor_proximity_score, 3),
            "floor_contact_score": round(self.floor_contact_score, 3),
            "persistence_score": round(self.persistence_score, 3),
            "pose_quality_score": round(self.pose_quality_score, 3),
            "kneeling_penalty": round(self.kneeling_penalty, 3),
            "leg_evidence_quality": round(self.leg_evidence_quality, 3),
            "height_collapse_cue": round(self.height_collapse_cue, 3),
            "mandown_confidence": round(self.mandown_confidence, 3),
        }


def _sigmoid(x: float, steepness: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-steepness * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class FeatureMandownScorer:
    """Stateless scorer: operates on a track's recent feature history, which
    the caller (TrackManager) is responsible for maintaining per track_id."""

    def __init__(
        self,
        config: FeatureMandownConfig | None = None,
        kneeling_config: KneelingConfig | None = None,
    ):
        self.config = config or FeatureMandownConfig()
        self.kneeling_config = kneeling_config or KneelingConfig()

    def score(
        self,
        base: FeatureVector,
        ext: ExtendedFeatureVector,
        base_history: deque[FeatureVector],
        ext_history: deque[ExtendedFeatureVector],
        floor_contact_score: float = 0.0,
        hip_drop_evidence: float | None = None,
    ) -> MandownFeatureScores:
        kneeling_penalty = self._kneeling_penalty(base, ext)
        gate = 1.0 - kneeling_penalty

        # Kneeling/sitting (bent legs) can otherwise mimic lying: a forward
        # pitch collapses the head/knee span and reads as a lying-like
        # posture. Gate both cues by leg-bend evidence so bent legs can't
        # drive a Mandown score even if the torso momentarily pitches over.
        #
        # hip_drop_evidence is intentionally NOT threaded into down_score/
        # floor_proximity_score/persistence_score here — a real M_D_2.mp4
        # true positive's down_score (0.707, already comfortably above
        # confirmed_mandown_down_floor on its own) dropped enough to stop
        # confirming when an earlier version of this fix (using a scene-
        # wide, floor_region-based hip check instead of this per-track,
        # self-relative one) gated height_collapse_cue everywhere it's
        # used. Only the exposed `height_collapse_cue` field below (used
        # exclusively by fusion's alternate grounding path, which only
        # matters when down_score ALONE isn't already sufficient) is
        # hip-gated -- genuine falls that already clear
        # confirmed_mandown_down_floor via down_score's own (ungated)
        # corroboration are completely unaffected by this gate, since that
        # "OR" condition short-circuits before the alternate path is ever
        # consulted.
        down_score = self._down_score(base, ext) * gate
        floor_proximity_score = self._floor_proximity_score(ext) * gate
        fall_motion_score = self._fall_motion_score(base_history, ext_history)
        persistence_score = self._persistence_score(ext_history)
        pose_quality_score = self._pose_quality_score(ext)
        floor_contact_score = float(np.clip(floor_contact_score, 0.0, 1.0))

        w = self.config.weights
        weight_sum = (
            w.down_score + w.fall_motion_score + w.floor_proximity_score
            + w.persistence_score + w.floor_contact_score
        )
        raw = (
            w.down_score * down_score
            + w.fall_motion_score * fall_motion_score
            + w.floor_proximity_score * floor_proximity_score
            + w.persistence_score * persistence_score
            + w.floor_contact_score * floor_contact_score
        ) / max(weight_sum, 1e-6)

        # Dampen (never zero out) confidence when pose is unreliable, instead
        # of fabricating certainty from a barely-visible skeleton.
        pose_gate = 0.5 + 0.5 * pose_quality_score
        mandown_confidence = float(np.clip(raw * pose_gate, 0.0, 1.0))

        return MandownFeatureScores(
            down_score=down_score,
            fall_motion_score=fall_motion_score,
            floor_proximity_score=floor_proximity_score,
            floor_contact_score=floor_contact_score,
            persistence_score=persistence_score,
            pose_quality_score=pose_quality_score,
            kneeling_penalty=kneeling_penalty,
            leg_evidence_quality=ext.leg_evidence_quality,
            height_collapse_cue=self._height_collapse_cue(ext, hip_drop_evidence),
            mandown_confidence=mandown_confidence,
        )

    def _kneeling_penalty(self, base: FeatureVector, ext: ExtendedFeatureVector) -> float:
        """Bent-leg evidence, gated by torso uprightness. A bent knee — the
        geometric signature of kneeling/sitting/crouching — drives this up
        regardless of camera angle; it weakens as the torso itself approaches
        horizontal, since a lying person's legs are extended but their torso
        is also flat (so `down_score`/`floor_proximity_score` aren't fully
        gated away for a genuinely lying person with a bent knee, e.g. a
        recovery position). The gate has a floor (`min_gate`) rather than
        vanishing entirely — a kneeling person leaning sharply forward also
        reads a high torso_angle, not just genuine lying.

        When legs aren't confidently visible (`leg_evidence_quality` low),
        `leg_extension_ratio`'s neutral "no bend evidence" default must NOT
        be read as "confirmed extended" — that would let an occluded
        kneeling person through unsuppressed. Falls back to a leg-independent
        signal instead: a large head-to-hip vertical gap means the torso
        itself is still upright regardless of what the unseen legs are
        doing — itself evidence against lying.

        V4.6: `upright_gate` below reads torso_angle~0 as "confidently
        upright" — but that's the same reading a person fallen toward/away
        from the camera produces (foreshortened, no lateral spread to
        measure), so this penalty was applying at near-full strength for a
        real such miss (legs unevidenced, torso_angle~0.02) instead of
        relenting. Corroborated below with height_collapse_cue, the same
        way fusion.py's alternate grounding path resolves the identical
        ambiguity."""
        cfg = self.kneeling_config
        if ext.leg_evidence_quality >= cfg.leg_evidence_quality_threshold:
            bend = cfg.extension_ratio_threshold - ext.leg_extension_ratio
            raw = _sigmoid(bend, cfg.sigmoid_steepness)
        else:
            torso_gap = abs(ext.head_height - base.cog_height)
            raw = _sigmoid(torso_gap - cfg.torso_upright_gap_threshold, cfg.sigmoid_steepness)

        # V4.9: real false positive the bent-leg check above structurally
        # cannot catch -- hinging forward at the hip with straight legs
        # (bend correctly reads "not bent" here, so raw stays low) lets
        # down_score's torso-only angle cue climb unopposed. Legs staying
        # planted/vertical while the torso tilts far over (hip_angle much
        # lower than torso_angle) is itself real bend-not-lying evidence,
        # independent of leg straightness -- a genuinely diagonally-lying
        # body tilts as one unit, so torso_angle and hip_angle read close
        # together instead. Combined via max(), not replacing the bent-leg
        # signal, and still gated by the same upright_gate below.
        #
        # A first attempt at this was reverted after an UNCALIBRATED
        # per-frame check of M_D_2.mp4's protected true positive appeared
        # to suppress its entire confirmed event. Re-verified with that
        # camera's REAL saved calibration: at every frame where this track
        # actually reaches confirming down_score levels, the real
        # torso-hip mismatch is small (0.01-0.23, well under the
        # threshold below) -- only an already-below-threshold, non-
        # confirming frame showed a large mismatch. The uncalibrated test
        # was not representative; this reflects the real, calibrated
        # picture.
        if ext.torso_angle_evidence_quality > 0.0 and ext.hip_angle_evidence_quality > 0.0:
            mismatch = ext.torso_angle - ext.hip_angle
            bend_at_hip_raw = _sigmoid(
                mismatch - cfg.hip_torso_mismatch_threshold, cfg.sigmoid_steepness
            )
            raw = max(raw, bend_at_hip_raw)

        # The min_gate floor itself fades out above extreme_lying_torso_angle
        # — it exists to stop a forward-leaning kneeling posture (moderately
        # high torso_angle) from fully cancelling the penalty, not to floor
        # a genuinely flat lying posture (very high torso_angle) too.
        base_gate = (1.0 - ext.torso_angle) ** cfg.upright_gate_power
        floor_bump = cfg.min_gate * _sigmoid(
            cfg.extreme_lying_torso_angle - ext.torso_angle, cfg.sigmoid_steepness
        )
        upright_gate = max(base_gate, floor_bump)

        torso_confirmed_near_zero = (
            ext.torso_angle_evidence_quality > 0.0
            and ext.torso_angle <= cfg.foreshortened_relent_torso_angle_ceiling
        )
        if torso_confirmed_near_zero:
            height_collapse = self._height_collapse_cue(ext)
            if height_collapse >= cfg.foreshortened_relent_height_collapse_floor:
                upright_gate = min(upright_gate, cfg.min_gate)

        # V4.8: monocular 2D->3D pose lifting gives a DIRECT torso-
        # orientation reading, immune to the 2D foreshortening blind spot
        # above -- when available and confident, it can relent this
        # penalty independent of the 2D torso_angle reading, since it
        # isn't sourced from the same blind signal. Crouching reads LOW
        # here (torso genuinely upright in 3D despite bent knees
        # shortening the bbox), so this doesn't reopen the V4.7 crouching
        # false positive the way a 2D-derived proxy did.
        #
        # V4.8.1: torso_angle_3d is NOT blind to a deep forward bend the
        # way it is to a camera-facing fall -- a real bend genuinely tilts
        # the torso in 3D too (measured live: torso_angle_3d=0.777 for a
        # real bend, almost identical to its own 2D torso_angle=0.778), so
        # a high reading alone doesn't distinguish "deep bend, legs
        # straight, hip still at standing height" from genuine lying.
        # Still requires height_collapse_cue to confirm the hip has
        # actually come down -- a bend can never do that (see
        # fusion.py's V4.8.1 for the full rationale; same discriminator).
        if (
            ext.torso_angle_3d_evidence_quality > 0.0
            and ext.torso_angle_3d >= cfg.pose3d_lying_relent_floor
            and self._height_collapse_cue(ext) >= cfg.foreshortened_relent_height_collapse_floor
        ):
            upright_gate = min(upright_gate, cfg.min_gate)

        return float(np.clip(raw * upright_gate, 0.0, 1.0))

    def _effective_torso_angle(self, ext: ExtendedFeatureVector) -> tuple[float, bool]:
        """torso_angle when it has real evidence (shoulders AND hips
        visible); otherwise falls back to hip_angle when THAT has evidence
        (hips AND knees visible, e.g. shoulders/head occluded but the lower
        body is visible) — never trusts torso_angle's silent 0.0
        ("vertical/upright") default, which would otherwise misrepresent a
        fully-occluded upper body as confirmed-not-lying evidence. Returns
        (angle, has_evidence); has_evidence=False only when neither segment
        has any visible endpoint at all."""
        if ext.torso_angle_evidence_quality > 0.0:
            return ext.torso_angle, True
        if ext.hip_angle_evidence_quality > 0.0:
            return ext.hip_angle, True
        return 0.0, False

    def _orientation_cue(self, ext: ExtendedFeatureVector) -> float:
        """Torso-only (falling back to hip-only under upper-body occlusion,
        see `_effective_torso_angle`) down/lying cue via a boundary sigmoid
        (not a linear 0-90 deg ramp): a standing body's torso axis reliably
        stays near image-vertical for a non-rolled camera, while a lying
        body's axis legitimately spans a wide angular range
        (yaw/perspective), so "clearly past upright" is treated as strongly
        down rather than only saturating at fully horizontal. This directly
        fixes diagonal lying (image-projected angle well short of 90 deg)
        without needing any camera-rotation assumption — see
        `_segment_angle_from_vertical`, which already generalizes correctly
        to any rotation direction.

        Deliberately torso-first rather than averaging in hip_angle (as the
        old `(torso_angle+hip_angle)/2` did): a bent knee/hip corrupts
        hip_angle even when the torso itself is unambiguously flat, which
        was the direct cause of the side-lying-with-bent-knees false
        negative — hip_angle is used only as a fallback, when torso_angle
        itself has no evidence at all. `body_axis_angle` (long-baseline
        head->ankle) may corroborate — but only when the primary cue
        already shows non-trivial lean (agreement-gated), and even then
        only partially (bounded assist) — a single noisy/occlusion-affected
        long-baseline reading must never be able to unilaterally push a
        clearly-upright standing/sitting/kneeling/crouching/bending posture
        toward "down"."""
        cfg = self.config.orientation
        angle, has_evidence = self._effective_torso_angle(ext)
        if not has_evidence:
            return 0.0
        primary_cue = _sigmoid(angle - cfg.upright_boundary, cfg.boundary_steepness)
        if ext.body_axis_evidence_quality > 0.0 and primary_cue > cfg.corroboration_floor:
            axis_cue = _sigmoid(
                ext.body_axis_angle - cfg.upright_boundary, cfg.boundary_steepness
            )
            primary_cue = primary_cue + cfg.corroboration_weight * max(
                0.0, axis_cue - primary_cue
            )
        return float(np.clip(primary_cue, 0.0, 1.0))

    def _height_collapse_cue(
        self, ext: ExtendedFeatureVector, hip_drop_evidence: float | None = None
    ) -> float:
        """body_height_norm vs. this camera's calibrated expected standing
        height at this image row -- orientation-independent (a body
        collapsed toward/away from the camera loses most of its vertical
        image extent same as one that falls sideways). Defaults to a
        neutral ~0.15 reading whenever scene calibration is unavailable
        (body_height_norm's own neutral default is exactly 1.0 in that
        case), so this never fabricates a collapse signal from an
        uncalibrated camera.

        V4.3/V4.4: a sufficiently deep bend (still standing, e.g. reaching
        for something on a low table) also shrinks the overall bbox enough
        to read as a partial height collapse — measured on a real
        production false positive, a deep-but-standing bend reached
        body_height_norm readings uncomfortably close to a genuine fall's.
        What a bend can NEVER do is bring the HIP down toward the floor —
        bending pivots the torso around a hip that stays at normal standing
        height, whereas any genuine fall (regardless of orientation) brings
        the hip down close to floor level along with everything else.

        `hip_drop_evidence` gates this cue at its source (only the exposed
        `MandownFeatureScores.height_collapse_cue` field, used exclusively
        by fusion's alternate grounding path — see `score()`'s docstring
        comment for why down_score/floor_proximity_score/persistence_score
        are deliberately NOT gated by it). V4.3 first tried this using
        FloorRegion.contact_likelihood evaluated at the hip -- a SCENE-WIDE
        calibration built from many different people's ankle positions
        across the room. That proved too imprecise for real camera
        perspective: a genuine fall's hip doesn't reliably land inside an
        aggregate floor band that was never designed to be perspective-
        correct for every image column, so it blocked real falls on top of
        the bends it was meant to exclude (two separate real regressions
        found via real-video validation in one session). V4.4 replaced it
        with a PER-TRACK, self-relative signal instead — has THIS
        specific track's hip moved down substantially from where it was
        the last time it was confidently upright (see
        MandownTrackContext.last_standing_cog_height) — which needs no
        scene calibration at all and isn't subject to aggregate-band
        imprecision. None whenever no prior standing observation exists for
        this track (including every "already lying at start" case, by
        design — never fabricates a gate from nothing)."""
        steep = self.config.sigmoid_steepness
        raw = _sigmoid(0.65 - ext.body_height_norm, steep * 0.5)
        if hip_drop_evidence is not None:
            raw = raw * hip_drop_evidence
        return raw

    def _down_score(self, base: FeatureVector, ext: ExtendedFeatureVector) -> float:
        """Instantaneous posture/shape cue — catches an already-down person
        with no motion history at all (case B).

        Weighted (not averaged equally): shape_cue/angle_cue need no scene
        calibration and are reliable from frame one; height_cue needs the
        scene profile to have calibrated a standing-height baseline first
        (V1/V2 SceneProfile), which never happens for a camera whose only
        subject enters already lying down — down_score must not be
        structurally capped by a signal that may never become available.

        V4.2: a person who falls toward/away from the camera keeps
        shape_cue/angle_cue both weak (foreshortened body still looks
        roughly narrow/upright in 2D) even though they're genuinely down.
        height_cue doesn't depend on fall direction, so once it's clearly
        past its own midpoint (a real, calibrated shortfall, not just
        "slightly short"), it may corroborate the composite the same way
        body_axis_angle corroborates angle_cue — bounded, agreement-gated,
        never a full override."""
        steep = self.config.sigmoid_steepness
        shape_cue = _sigmoid(base.bbox_aspect_ratio - 1.0, steep * 0.5)
        angle_cue = self._orientation_cue(ext)
        height_cue = self._height_collapse_cue(ext)
        weighted = 0.4 * shape_cue + 0.4 * angle_cue + 0.2 * height_cue

        hc = self.config.height_collapse
        if height_cue > hc.corroboration_floor:
            weighted = weighted + hc.corroboration_weight * max(0.0, height_cue - weighted)
        return float(np.clip(weighted, 0.0, 1.0))

    def _fall_motion_score(
        self, base_history: deque[FeatureVector], ext_history: deque[ExtendedFeatureVector]
    ) -> float:
        """Recent spike in downward velocity / angle-change / acceleration —
        catches an active falling transition (case A). Supporting evidence
        only — down/floor state (§_down_score, §_floor_proximity_score) is
        the primary Mandown criterion and must reach confirmation without
        this ever firing."""
        window = self.config.motion_window_frames
        if not base_history or not ext_history:
            return 0.0

        recent_base = list(base_history)[-window:]
        recent_ext = list(ext_history)[-window:]

        max_velocity = max((f.vertical_velocity for f in recent_base), default=0.0)
        max_angle_change = max((f.angle_change for f in recent_ext), default=0.0)
        max_accel = max((abs(f.acceleration) for f in recent_ext), default=0.0)

        # Steepness*offset chosen so exactly-zero motion reads near 0, not a
        # ~0.3 floor (a real bug: the original offset/steepness combination
        # left a substantial non-zero score even for a perfectly static
        # track, inflating "motion" evidence that should be absent).
        steep = self.config.sigmoid_steepness * 3.0
        velocity_cue = _sigmoid(max_velocity - 0.1, steep)
        angle_cue = _sigmoid(max_angle_change - 0.15, steep)
        accel_cue = _sigmoid(max_accel - 0.1, steep)

        return float(np.clip((velocity_cue + angle_cue + accel_cue) / 3.0, 0.0, 1.0))

    def _floor_proximity_score(self, ext: ExtendedFeatureVector) -> float:
        """Head-to-knee vertical span collapsing is a camera-position-independent
        lying cue: standing has a large head/knee gap anywhere in frame, lying
        collapses that gap regardless of where in the image it happens.

        This image-Y-projection span underestimates a diagonally-lying body
        (the projected span stays large even though the body is collapsed
        along its own axis), so it's corroborated with the same torso-based
        `_orientation_cue` used by `_down_score` — taking the max can only
        improve on the existing span cue, never regress it, since
        `_orientation_cue` is itself gated to near-zero for any upright
        posture (standing/sitting/kneeling/crouching/bending).

        V4.2: also corroborated with `_height_collapse_cue` for the same
        reason as `_down_score` — a person fallen toward/away from the
        camera can keep a large head/knee image-Y span (foreshortening) even
        though `body_height_norm` correctly shows their overall vertical
        extent has collapsed. Plain max(), not gated: height_cue is already
        near-zero (~0.15) whenever scene calibration is unavailable, so it
        can only ever help here, never introduce a false positive on an
        uncalibrated camera."""
        span = abs(ext.head_height - ext.knee_height)
        span_cue = _sigmoid(0.15 - span, self.config.sigmoid_steepness)
        return float(max(span_cue, self._orientation_cue(ext), self._height_collapse_cue(ext)))

    def _persistence_score(self, ext_history: deque[ExtendedFeatureVector]) -> float:
        """Fraction of the recent window with consistently down-like posture,
        computed independently of `down_score` to avoid circularity.

        Torso-first (via `_effective_torso_angle`, not
        `(torso_angle+hip_angle)/2`) for the same reason as
        `_orientation_cue`: a bent knee/hip must not dilute a torso that is
        itself already reading flat, and a fully-occluded upper body must
        not silently read as "upright" when the visible lower body says
        otherwise.

        V4.2: a frame also counts as down-like when `_height_collapse_cue`
        clears its own midpoint, even if the angle reads "upright" — a
        person fallen toward/away from the camera has a torso_angle stuck
        near 0 (foreshortened, no lateral spread) for the entire event, so
        without this, persistence_score could never accumulate for that
        orientation at all, permanently blocking both its 10% weight in the
        composite and eligibility for the geometric-alone-confirm override
        (which needs persistence_score, not just feature_confidence)."""
        window = self.config.persistence_window_frames
        if not ext_history:
            return 0.0
        recent = list(ext_history)[-window:]
        hc_floor = self.config.height_collapse.corroboration_floor
        down_like = 0
        for f in recent:
            angle, has_evidence = self._effective_torso_angle(f)
            if (has_evidence and angle > 0.45) or self._height_collapse_cue(f) > hc_floor:
                down_like += 1
        return float(down_like / len(recent))

    def _pose_quality_score(self, ext: ExtendedFeatureVector) -> float:
        raw = ext.pose_quality * (ext.num_critical_keypoints_visible / 6.0)
        return float(np.clip(raw, 0.0, 1.0))
