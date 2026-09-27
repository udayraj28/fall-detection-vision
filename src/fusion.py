"""Evidence fusion: combines LSTM confidence with independent feature-based
confidence into a single fusion score. Transparent weighted rule-based
fusion (not another learned model), so behavior stays debuggable.

Does not require strict AND between LSTM and feature evidence — a single
strong signal can push toward Mandown on its own, matching the fusion
examples in the target spec:
    LSTM strong + feature moderate               -> possible
    LSTM moderate + feature strong + persistence  -> possible/confirmed
    LSTM weak + feature weak                      -> normal

V3: current down/floor-state evidence is the *primary* Mandown criterion,
not falling motion or LSTM corroboration — an already-lying person (LSTM
buffer not yet full, or LSTM simply wrong) must be able to reach full
MANDOWN on strong, persistent geometric evidence alone. See
`geometric_confirm_threshold` below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import FusionConfig


@dataclass
class FusionResult:
    fusion_score: float
    lstm_mandown_prob: float
    feature_confidence: float
    persistence_score: float
    pose_quality: float
    suggested_tier: str  # "NORMAL" | "POSSIBLE_MANDOWN" | "MANDOWN" — diagnostic only;
                          # the state machine applies temporal hysteresis on top of this.

    def to_dict(self) -> dict:
        return {
            "fusion_score": round(self.fusion_score, 3),
            "lstm_mandown_prob": round(self.lstm_mandown_prob, 3),
            "feature_confidence": round(self.feature_confidence, 3),
            "persistence_score": round(self.persistence_score, 3),
            "pose_quality": round(self.pose_quality, 3),
            "suggested_tier": self.suggested_tier,
        }


class EvidenceFusion:
    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()

    def combine(
        self,
        lstm_mandown_prob: float,
        feature_confidence: float,
        persistence_score: float,
        pose_quality: float,
        down_score: float,
        floor_proximity_score: float = 0.0,
        height_collapse_cue: float = 0.0,
        torso_angle: float = 0.0,
        torso_angle_evidence_quality: float = 0.0,
        torso_angle_3d: float = 0.0,
        torso_angle_3d_evidence_quality: float = 0.0,
        hip_angle: float = 0.0,
        hip_angle_evidence_quality: float = 0.0,
        collapse_then_stillness: bool = False,
        body_axis_angle_3d: float = 0.0,
        body_axis_angle_3d_evidence_quality: float = 0.0,
    ) -> FusionResult:
        c = self.config
        weight_sum = c.weight_lstm + c.weight_feature + c.weight_persistence
        weighted = (
            c.weight_lstm * lstm_mandown_prob
            + c.weight_feature * feature_confidence
            + c.weight_persistence * persistence_score
        ) / max(weight_sum, 1e-6)

        # A single very strong signal is sufficient evidence on its own —
        # avoid requiring both sources to agree (keeps recall high).
        strongest = max(lstm_mandown_prob, feature_confidence)
        if strongest >= c.strong_signal_threshold:
            weighted = max(weighted, c.possible_mandown_threshold)

        # Strong + persistent geometric (down/floor-state) evidence alone can
        # fully confirm MANDOWN — the LSTM is supporting evidence, not a
        # gate. Without this, an already-lying person with a cold/wrong LSTM
        # could never clear confirmed_mandown_threshold no matter how
        # confident and sustained the geometry was.
        #
        # V4.10: real RTSP miss -- a genuine, sustained vertical/
        # foreshortened fall measured down_score=0.925 (near-maximal) and
        # persistence=1.0 for 42 real seconds, yet never confirmed. Root
        # cause: feature_confidence (mandown_confidence) is down_score
        # blended with several OTHER weighted components (fall_motion_score
        # correctly reads ~0 for a static lying person, diluting the
        # average) and then discounted by pose_quality (0.53 here, from
        # partial keypoint visibility) -- landing at 0.602, a hair above
        # geometric_confirm_threshold, not comfortably above it. Any small
        # frame-to-frame dip (pose_quality or fall_motion varying slightly
        # over many real seconds) drops it below 0.60, breaking the
        # override and the state machine's need for consecutive confirming
        # frames -- while down_score itself, the single most direct signal
        # of "is this person down", stayed far above its own floor the
        # whole time. Giving down_score alone the SAME direct path (paired
        # with the same persistence requirement) removes that fragility
        # without loosening anything: confirmed_mandown_down_floor is the
        # exact bar already required elsewhere to justify bypassing the cap
        # entirely, so this doesn't admit any case that wasn't already
        # being let through some other way.
        if persistence_score >= c.geometric_confirm_persistence_threshold and (
            feature_confidence >= c.geometric_confirm_threshold
            or down_score >= c.confirmed_mandown_down_floor
        ):
            weighted = max(weighted, c.confirmed_mandown_threshold)

        # V6 Fix Set B: `collapse_then_stillness` is a purely TEMPORAL
        # confirmation path (see MandownTrackContext.collapse_then_stillness
        # and MotionSignatureConfig) for a foreshortened/asymmetric lying
        # posture where every single-frame geometric signal lands
        # moderately elevated but none individually clears its own
        # threshold (the open §7.1 case from the V5 report). The caller
        # only ever passes True when a genuine height-collapse spike was
        # already followed by several real seconds of continuous
        # stillness AND a real calibrated height collapse holds right
        # now -- i.e. real geometric evidence already backs this, the
        # same style as the V4.10 direct down_score path above. A hard
        # 0/1 gate, not blended, per explicit design intent to prioritize
        # zero false positives for this path over sensitivity.
        if collapse_then_stillness and pose_quality >= c.pose_quality_floor:
            weighted = max(weighted, c.confirmed_mandown_threshold)

        # Discount (don't zero) when pose is unreliable — reflects genuine
        # uncertainty rather than fabricating confidence from bad keypoints.
        if pose_quality < c.pose_quality_floor:
            discount = 0.5 + 0.5 * (pose_quality / max(c.pose_quality_floor, 1e-6))
            weighted *= discount

        fusion_score = float(np.clip(weighted, 0.0, 1.0))

        # Mirror-image safeguard for the geometric-alone-confirms override
        # above: a confident LSTM, sustained over many frames via the state
        # machine's streak (which does not distinguish *why* a tier reads
        # "up"), must not be able to reach full MANDOWN with essentially no
        # geometric grounding at any point. This is a necessary-not-
        # sufficient floor, not a stricter global threshold — a track with
        # real down/floor-state evidence (even weak/still-developing, as at
        # the onset of every genuine fall in this repo's real-video
        # validation set) is completely unaffected; it only forces NORMAL
        # when there is literally no geometric grounding at all.
        has_geometric_grounding = (
            down_score >= c.lstm_only_geometric_grounding_down_floor
            or persistence_score >= c.lstm_only_geometric_grounding_persistence_floor
        )
        if not has_geometric_grounding:
            fusion_score = min(fusion_score, c.possible_mandown_threshold - 1e-6)

        # V4.1: a higher-bar version of the floor above — a confident-but-
        # wrong LSTM (or a moderately-elevated feature_confidence) must not
        # be able to reach MANDOWN off the back of a MODERATE (bend/sit-
        # level, not lying-level) down_score. Real case: a sustained deep
        # bend or a small/low-quality sitting detection, both with
        # down_score well below genuine lying but well above the 0.22 floor
        # above (a sustained bend trivially satisfies that looser floor via
        # persistence_score alone, since persistence saturates to 1.0 for
        # any posture held steady long enough). Geometry-alone confirmation
        # (the override earlier in this method) is unaffected — every real
        # confirmed MANDOWN in this repo's validation set already qualifies
        # through that path independent of down_score's value here.
        #
        # Applied unconditionally (not gated on the current frame's fusion
        # score already being >=0.65): TrackStateMachine's "up" streak
        # treats POSSIBLE_MANDOWN and MANDOWN identically by design — a
        # sustained possible-tier signal is *meant* to eventually escalate
        # to confirmed — so a per-frame cap that only demotes MANDOWN to
        # POSSIBLE_MANDOWN does not actually prevent eventual confirmation
        # once sustained long enough (verified: it still reached MANDOWN
        # after 10 more frames with only that weaker cap in place). Must
        # cap all the way to NORMAL, mirroring the floor above.
        # V4.2: alternate path for a person fallen toward/away from the
        # camera — angle-based cues stay near "upright" (foreshortened), so
        # down_score is structurally capped below the floor above even with
        # height-collapse corroboration.
        #
        # IMPORTANT: floor_proximity_score alone is NOT a safe gate here —
        # it also absorbs orientation_cue via max() in feature_mandown.py,
        # so a deep forward torso pitch (the V4.1 bending false positive)
        # drives it just as high as a genuine collapse does (measured:
        # bend=0.893, actually *higher* than a real foreshortened fall's
        # 0.742) — an earlier version of this gate used floor_proximity_
        # score alone and reopened that exact bending false positive
        # through this new path once the LSTM was even moderately
        # confident. height_collapse_cue is driven ONLY by body_height_norm
        # (never by torso angle), so it correctly stays low (~0.15,
        # uncalibrated-equivalent) for a bend while staying high (~0.9) for
        # a genuine fall — this is the condition that must exclude bending.
        # pose_quality above its own established floor additionally
        # excludes the occluded-seated-person case (both height_collapse_
        # cue and floor_proximity_score read high there too — only far
        # fewer visible keypoints distinguish it from a genuine fall).
        # V4.5: this alternate path's whole premise is that torso_angle is
        # structurally blind (~0.0) for a foreshortened fall -- so it must
        # only fire when that premise actually holds. A deep bend's
        # torso_angle (~0.64) is NOT blind and already carries real "not
        # down" evidence that down_score/kneeling_penalty apply -- without
        # this check, this path bypassed that evidence entirely and
        # confirmed a real bending false positive. No torso evidence at all
        # (quality<=0) does not get the benefit of the doubt either: the
        # blind-torso justification can't be confirmed, so this path
        # requires it be checked and low, not merely unknown.
        torso_confirmed_near_zero = (
            torso_angle_evidence_quality > 0.0
            and torso_angle <= c.alternate_grounding_torso_angle_ceiling
        )
        # V4.8: monocular 2D->3D pose lifting (src/pose_3d.py) gives a
        # direct, camera-calibration-independent measurement of real torso
        # orientation -- unlike the 2D-derived paths above, it isn't
        # blind to a camera-facing fall. A genuine crouch reads LOW on
        # torso_angle_3d (the torso really is vertical in 3D even though
        # bent knees shorten the observed bbox), which is exactly the
        # discriminator an earlier, reverted attempt at a similar
        # combined-evidence path (V4.7, 2D-only) was missing.
        #
        # V4.8.1 real-camera regression, found via live testing: a deep
        # forward bend (straight legs, e.g. reaching down to a low table)
        # measured torso_angle_3d=0.777 -- almost identical to its own 2D
        # torso_angle=0.778 -- because a sufficiently deep bend genuinely
        # tilts the torso in real 3D space, not just in 2D projection.
        # torso_angle_3d is NOT blind to this the way it is to a camera-
        # facing fall -- it correctly reports a real, large angle for
        # BOTH a deep bend and genuine lying, so angle alone (2D or 3D)
        # cannot tell them apart here. What a bend can never do is bring
        # the hip down toward the floor (legs stay extended, hip stays at
        # standing height) -- height_collapse_cue stayed low (0.148) for
        # this real bend precisely because of that, so requiring it here
        # (the same floor the 2D alternate-grounding path above already
        # uses) is the correct, already-established discriminator this
        # path was missing. Real falls this path targets clear this floor
        # easily (a person actually lying down has genuinely collapsed).
        #
        # V4.9: that height_collapse_cue requirement alone still isn't
        # enough -- a real bend measured live also partially collapses it
        # (body_height_norm=0.626, height_collapse_cue=0.53, clearing the
        # floor above). The reliable discriminator is hip_angle: a body
        # hinging at the hip with planted, straight legs keeps hip_angle
        # much lower than torso_angle, whereas genuine diagonal lying
        # tilts as one unit (both read close together) -- same signal
        # added to kneeling_penalty's V4.9 fix, needed here too since this
        # path bypasses kneeling_penalty entirely. (An initial check of
        # this against M_D_2.mp4's protected true positive without real
        # calibration wired in appeared to suppress it; re-verified with
        # that camera's real saved calibration, the mismatch at its
        # actual confirming frames is small -- see feature_mandown.py's
        # V4.9 note for the full account.)
        standing_bend_signature = (
            torso_angle_evidence_quality > 0.0
            and hip_angle_evidence_quality > 0.0
            and (torso_angle - hip_angle) >= c.hip_torso_mismatch_threshold
        )
        # V6.1 (Objective 1): body_axis_angle_3d is an INDEPENDENT
        # corroborator, not a replacement -- it's anchored on a different
        # landmark pair (shoulder+ankle on one confirmed-extended side)
        # than torso_angle_3d's (shoulder-center+hip-center), so the two
        # can and do disagree on the exact real frame this was written
        # for (the hip-center-based torso_angle_3d reads too low there,
        # since MediaPipe's kinematic prior biases the hip landmark for
        # an unusual-but-valid raised-knee configuration; body_axis_angle_3d
        # never asks that landmark anything and should read correctly).
        # Either one confirming independently is sufficient -- same
        # gating (height_collapse_cue + the minimal down_score floor +
        # not a standing-bend signature) as the existing torso_angle_3d
        # path, since a long shoulder-to-ankle baseline is subject to the
        # exact same "deep bend also has a large real 3D angle" confound
        # (see the V4.8.1 deep-bend regression this same gating already
        # protects against) -- verified this doesn't reopen the V4.7/V4.8.1
        # crouching/bending false positives (see feature_mandown.py's
        # V6.1 note and tests/test_pose_3d.py's negative-control tests).
        confident_3d_lying_torso = (
            torso_angle_3d_evidence_quality > 0.0
            and torso_angle_3d >= c.pose3d_lying_angle_floor
            and down_score >= c.lstm_only_geometric_grounding_down_floor
            and height_collapse_cue >= c.confirmed_mandown_height_collapse_floor
            and not standing_bend_signature
        )
        confident_3d_lying_axis = (
            body_axis_angle_3d_evidence_quality > 0.0
            and body_axis_angle_3d >= c.pose3d_axis_lying_angle_floor
            and down_score >= c.lstm_only_geometric_grounding_down_floor
            and height_collapse_cue >= c.confirmed_mandown_height_collapse_floor
            and not standing_bend_signature
        )
        confident_3d_lying = confident_3d_lying_torso or confident_3d_lying_axis
        has_strong_enough_grounding_for_up_tier = (
            down_score >= c.confirmed_mandown_down_floor
            or (
                feature_confidence >= c.geometric_confirm_threshold
                and persistence_score >= c.geometric_confirm_persistence_threshold
            )
            or (
                floor_proximity_score >= c.confirmed_mandown_floor_proximity_floor
                and height_collapse_cue >= c.confirmed_mandown_height_collapse_floor
                and pose_quality >= c.pose_quality_floor
                and torso_confirmed_near_zero
            )
            or confident_3d_lying
            or collapse_then_stillness
        )
        if not has_strong_enough_grounding_for_up_tier:
            fusion_score = min(fusion_score, c.possible_mandown_threshold - 1e-6)

        if fusion_score >= c.confirmed_mandown_threshold:
            tier = "MANDOWN"
        elif fusion_score >= c.possible_mandown_threshold:
            tier = "POSSIBLE_MANDOWN"
        else:
            tier = "NORMAL"

        return FusionResult(
            fusion_score=fusion_score,
            lstm_mandown_prob=lstm_mandown_prob,
            feature_confidence=feature_confidence,
            persistence_score=persistence_score,
            pose_quality=pose_quality,
            suggested_tier=tier,
        )
