"""Central configuration for the Mandown algorithmic core (scene normalization,
rolling person-size stats, feature-based scoring, fusion, state machine).

Loaded once from configs/mandown_config.yaml. All fields have defaults matching
that file, so callers that don't pass a config still get sane behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "mandown_config.yaml"


@dataclass
class CalibrationConfig:
    min_keypoint_confidence: float = 0.5
    min_critical_keypoints: int = 4
    frame_edge_margin_px: float = 4.0
    min_aspect_ratio_for_standing: float = 0.15
    max_aspect_ratio_for_standing: float = 0.75
    min_samples_for_confidence: int = 15
    full_confidence_samples: int = 50


@dataclass
class RollingSizeConfig:
    window_size: int = 30
    update_stride: int = 10
    outlier_reject_iqr_multiplier: float = 2.5
    # Own confidence-saturation point, decoupled from CalibrationConfig's
    # (which also governs the unrelated, much-larger perspective/floor-region
    # sample buffers) -- keeps calibration_confidence able to reach 1.0 once
    # this window is actually full, whatever window_size is set to.
    full_confidence_samples: int = 30


@dataclass
class FeatureMandownWeights:
    down_score: float = 0.40
    fall_motion_score: float = 0.20
    floor_proximity_score: float = 0.20
    persistence_score: float = 0.10
    floor_contact_score: float = 0.10  # absolute floor-band evidence; 0 until calibrated


@dataclass
class OrientationConfig:
    # V4: torso_angle's boundary past which "clearly not upright" is read as
    # strongly down rather than climbing linearly only to saturate at fully
    # horizontal (90 deg) -- a lying body's image-plane angle legitimately
    # varies with yaw/perspective, so it shouldn't need to reach 90 deg to
    # register as strongly down. ~0.40 = ~36 deg from vertical.
    upright_boundary: float = 0.40
    boundary_steepness: float = 10.0  # matches this module's existing sigmoid_steepness default
    # Agreement gate for body_axis_angle corroboration: the long-baseline
    # axis (ankle-sensitive, just as leg-bend-sensitive as hip_angle) may
    # only ADD to a cue when the torso ALREADY reads unambiguously past
    # upright (torso_cue > 0.5, i.e. torso_angle itself already clears
    # upright_boundary) -- never let a single noisy/occlusion-affected axis
    # reading unilaterally push a torso that's merely close to (but still on
    # the upright side of) the boundary toward "down". A lower floor here
    # was found, via a dedicated adversarial test, to let a maximally noisy
    # axis reading (1.0) push torso_angle=0.3 -- clearly still upright --
    # up to a 0.63 cue on its own; 0.5 closes that gap.
    corroboration_floor: float = 0.5
    # Even when gated open, the axis may only contribute this fraction of
    # its gap over the torso cue -- bounded assist, never a full override.
    corroboration_weight: float = 0.5


@dataclass
class HeightCollapseConfig:
    # V4.2: a person who falls toward/away from the camera (foreshortened --
    # torso/hip/body-axis angles all still read close to vertical in 2D,
    # since the fall direction is along the depth axis, not across the image
    # plane) is invisible to angle-based cues. body_height_norm (bbox height
    # vs. this camera's calibrated expected standing height at that image
    # row) doesn't depend on fall direction at all -- a collapsed body loses
    # most of its vertical image extent regardless of which way it fell. Lets
    # a strong, calibrated height-collapse reading corroborate down_score/
    # floor_proximity_score directly, rather than being diluted to a fixed
    # 20% weight alongside two cues that are specifically unreliable for
    # this orientation. Only ever activates on a REAL calibrated shortfall --
    # body_height_norm defaults to a neutral 1.0 (no signal) whenever scene
    # calibration is unavailable, which reads as height_cue~0.15, well below
    # this floor.
    corroboration_floor: float = 0.5
    corroboration_weight: float = 0.75


@dataclass
class HipDropConfig:
    # V4.4: replaces a V4.3 attempt that gated height_collapse_cue using
    # FloorRegion.contact_likelihood (a SCENE-WIDE floor band built from
    # many people's ankle positions across the room) evaluated at the hip.
    # That proved too imprecise for real camera perspective -- a genuine
    # fall's hip doesn't reliably land inside an aggregate band that was
    # never designed to be perspective-correct for every image column, and
    # it blocked real falls on top of the bends it was meant to exclude
    # (two separate real-video regressions found in one session). This is
    # a PER-TRACK, self-relative signal instead: has THIS track's hip
    # (cog_height, 0=top/1=bottom of frame) moved down by at least this
    # much, normalized to frame height, since the last time it was
    # confidently upright (MandownTrackContext.last_standing_cog_height)?
    # Needs no scene calibration at all, and is never computed (stays None)
    # for a track with no prior standing observation -- including every
    # "already lying at start" case, by design.
    drop_threshold: float = 0.08
    steepness: float = 15.0
    # A track counts as "confidently upright" (safe to record as the new
    # standing baseline) only when torso_angle is unambiguous evidence of
    # near-vertical -- well below the general upright_boundary used
    # elsewhere, since a wrongly-recorded baseline (e.g. from a transient
    # partial bend) would corrupt every future hip-drop reading for this
    # track until it happens to stand fully upright again.
    upright_torso_angle_threshold: float = 0.15
    # CRITICAL second condition, found via real-video regression testing:
    # a person fallen toward/away from the camera ALSO reads torso_angle
    # near 0 -- that foreshortening is exactly why angle-based cues are
    # blind to that orientation in the first place. Without also requiring
    # body_height_norm to look genuinely standing-height, a foreshortened
    # fall's own posture would get wrongly recorded as its "standing"
    # baseline (comparing the fall against itself gives ~zero drop),
    # completely defeating hip_drop_evidence for the exact orientation it
    # exists to help with. Neutral-default body_height_norm (1.0, whenever
    # scene calibration is unavailable) still passes this -- without
    # calibration there's no reliable way to detect a height collapse at
    # all here either, matching the rest of this codebase's stance.
    upright_body_height_norm_threshold: float = 0.85


@dataclass
class Pose3DConfig:
    """V4.8: monocular 2D->3D pose lifting (src/pose_3d.py), the
    root-cause fix for the foreshortening blind spot every earlier fix
    this session worked around indirectly. `enabled=False` (or MediaPipe/
    the model file simply not being available) makes every 3D-derived
    feature read as "no evidence" (evidence_quality=0.0) everywhere it's
    consumed -- the pipeline behaves exactly as it did before this
    feature existed, never a hard dependency."""

    enabled: bool = True
    model_path: str = "models/mediapipe/pose_landmarker_lite.task"
    min_pose_detection_confidence: float = 0.3
    # Below this mean landmark visibility, treat as no reading at all
    # rather than trust an uncertain 3D lift -- an unreliable 3D reading
    # is worse than none, since callers treat this as strong, direct
    # evidence rather than one signal among several.
    min_visibility: float = 0.5
    # BlazePose expects to see a full body with some margin, not a tight
    # box -- padding the YOLO bbox by this fraction (of its own width/
    # height) before cropping avoids clipping hands/feet right at the
    # crop edge, which real testing showed causes spurious detection
    # failures on an otherwise-clean pose.
    crop_pad_ratio: float = 0.25
    # Real-time cost gate: 3D lifting is meaningfully more expensive per
    # person per frame than every other feature here (a second model
    # inference), so it must not run unconditionally on every normal
    # standing frame in a live deployment. It only runs when the cheap 2D
    # height signal ALREADY looks suspicious (body_height_norm below this
    # -- crouching/bending/sitting/lying, or simply "shorter than expected
    # standing"), or when that signal isn't calibrated yet at all (can't
    # use a height gate to skip when there's no calibrated height to
    # check against -- exactly the situation for a freshly-started
    # camera, which must not silently disable the one signal that could
    # catch a fall before any calibration exists). A generously high
    # threshold errs toward running 3D when uncertain, not toward
    # skipping it.
    height_gate_threshold: float = 0.85
    # V6 Fix Set A (§2.2): per-side (asymmetric) torso-orientation
    # correction. A raised knee / reaching arm / arm tucked under the
    # torso corrupts MediaPipe's BILATERAL (midpoint-of-both-sides)
    # shoulder/hip center, artificially deflating torso_angle_3d for a
    # genuinely foreshortened-lying body -- real measured case (2026-09-14
    # 18:19:32 screenshot, prone person with one knee bent): the
    # bilateral reading landed at 0.319, well below the 0.55 "confident
    # lying" floor. When one side's shoulder+hip landmarks are
    # confidently more visible than the other's by more than this margin,
    # that side's OWN shoulder->hip vector is computed as a second
    # candidate reading, and the LARGER of the two (bilateral vs.
    # single-side) is kept -- never the single-side reading alone. This
    # makes the fix provably monotonic: it can only ever RAISE
    # torso_angle_3d relative to the pre-V6 bilateral-only behavior, never
    # lower it, so it cannot regress any existing confirmed case
    # (including the protected M_D_2.mp4 track) even before per-camera
    # re-verification. DESIGN MARGIN: the V6 spec's own suggested
    # starting point ("tune against real MediaPipe outputs") -- not yet
    # validated against a large real visibility-asymmetry dataset.
    torso3d_asymmetry_threshold: float = 0.3


@dataclass
class MotionSignatureConfig:
    """V6 Fix Set B: `collapse_then_stillness` -- a purely TEMPORAL
    confirmation path for a foreshortened/asymmetric lying posture that
    Fix Set A's per-frame geometric corrections don't fully resolve (see
    the open §7.1 case in the V5 report: every single-frame signal landed
    moderately elevated but none individually cleared its own threshold).
    A hard 0/1 gate (never a blended/continuous score, per explicit user
    instruction prioritizing zero false positives for this path) requiring
    ALL of: a genuine height-collapse SPIKE within the last
    `t_lookback_seconds`, continuous STILLNESS since that spike for
    `t_still_seconds`, a real calibrated height collapse right now, and
    the track NOT currently inside its own confidently-standing baseline.
    This explicitly rejects a deliberate slow sit-down (no spike, since
    the collapse is gradual rather than a genuine 1-second drop), a chair
    scoot-back (no real height collapse), and bending-to-help-then-
    standing-back-up (stillness never sustains for the full window). See
    `MandownTrackContext.collapse_then_stillness` and `EvidenceFusion`'s
    rule 8.

    theta_spike/theta_still MEASURED from this repo's 2 real single-fall
    videos (mandown_2.mp4 track 1, man_down_c08.mp4 track 1) via a
    dedicated diagnostic pass -- not guessed. Deliberately conservative
    given the small (n=2) real sample: set with margin toward NOT falsely
    triggering rather than toward maximum sensitivity, since this is a
    SUPPLEMENTARY confirmation path -- every real confirmed case in this
    repo's validation set already confirms through an existing path
    (V4.10 direct down_score, geometric-alone confirm, or the 3D
    grounding path) independent of whether this rule ever fires.
    """

    t_lookback_seconds: float = 5.0
    t_still_seconds: float = 5.0
    # MEASURED: the weaker of the 2 real fall onsets' 1-second
    # body_height_norm collapse -- man_down_c08.mp4 track 1's own onset
    # measured only 0.137 (a more gradual descent than mandown_2.mp4
    # track 1's much sharper 0.560) -- sits just above this floor, so
    # both real events would clear it.
    theta_spike: float = 0.12
    # MEASURED: comfortably above both real events' own post-onset
    # "settled/lying" per-frame motion_energy 95th percentile
    # (mandown_2.mp4: 0.012, man_down_c08.mp4: 0.053 -- real footage is
    # never perfectly motionless; ongoing minor jitter/breathing/pose-
    # estimation noise is normal) -- a genuinely still lying person
    # should read below this on nearly every frame of the required
    # window.
    theta_still: float = 0.06
    # Reuses the SAME calibrated height-collapse floor already
    # established and validated across V4.2-V4.9
    # (HeightCollapseConfig.corroboration_floor /
    # FusionConfig.confirmed_mandown_height_collapse_floor) rather than a
    # third, independently-drifting copy of the same real-measured number.
    height_collapse_floor: float = 0.5


@dataclass
class RotationSearchConfig:
    """V6 Fix Set C (§4.1): narrow-range rotation search, run ONLY as a
    last-resort fallback when the normal straight-crop re-acquisition
    (`PersonDetector.redetect_in_roi`, unchanged) fails to find a
    previously-tracked person at all. Runs the SAME frozen YOLO-pose
    model (no retraining) on a small number of rotated copies of the
    padded ROI and keeps whichever gives the highest mean keypoint
    confidence -- a body oriented steeply off-vertical is read far more
    reliably once rotated closer to the model's canonical
    upright-training-distribution orientation. Narrowing the search to a
    window around the last successful angle (rather than a full 0-360 deg
    sweep every frame) is what keeps this affordable; running it only on
    confirmed re-acquisition failure (not on every frame, nor even on
    every re-acquisition attempt) keeps it rarer still than the V6 spec's
    own proposal.
    """

    enabled: bool = True
    angle_step_degrees: float = 15.0
    # Decays theta_prev_best back toward 0 by this factor per frame this
    # path ISN'T triggered, so a track that recovers normally (no longer
    # needs rotation) doesn't keep biasing future searches toward a stale
    # angle from an old, resolved occlusion.
    angle_decay_factor: float = 0.9


@dataclass
class DetectionRecoveryConfig:
    """V6 Fix Set C (§4.2): confidence-decayed synthetic keypoint
    carry-forward for a brief TRUE detection gap (the pose model produced
    literally nothing for this track this frame, even after the straight-
    crop and rotation-search re-acquisition attempts both failed). Not a
    new model and not a new fusion rule -- the synthetic keypoints/
    confidence are fed into the exact same feature/fusion pipeline a real
    detection would use (`pose_quality_score`'s existing confidence
    discount already handles decaying trust in a noisy skeleton), so this
    composes with existing logic instead of requiring new thresholds
    downstream. Hard-capped at `t_gap_max_seconds` of real, continuous
    non-detection -- beyond that, the track is dropped for real rather
    than extrapolated indefinitely, to avoid a ghost track.
    """

    enabled: bool = True
    t_gap_max_seconds: float = 3.0
    confidence_decay_factor: float = 0.85


@dataclass
class FeatureMandownConfig:
    motion_window_frames: int = 15
    persistence_window_frames: int = 45
    weights: FeatureMandownWeights = field(default_factory=FeatureMandownWeights)
    sigmoid_steepness: float = 10.0
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    height_collapse: HeightCollapseConfig = field(default_factory=HeightCollapseConfig)


@dataclass
class FusionConfig:
    # V3: current down/floor-state evidence is the primary signal, LSTM is
    # supporting — rebalanced from V2's 0.5/0.35/0.15 (LSTM-majority).
    weight_lstm: float = 0.35
    weight_feature: float = 0.45
    weight_persistence: float = 0.20
    strong_signal_threshold: float = 0.85
    possible_mandown_threshold: float = 0.45
    confirmed_mandown_threshold: float = 0.65
    pose_quality_floor: float = 0.3
    # V3: strong + persistent geometric evidence alone can fully confirm
    # MANDOWN without LSTM corroboration — closes a gap where an already-lying
    # person (LSTM buffer not yet full, or LSTM simply wrong) capped out at
    # POSSIBLE_MANDOWN regardless of how strong/persistent the geometry was.
    geometric_confirm_threshold: float = 0.60
    geometric_confirm_persistence_threshold: float = 0.65
    # V4: mirror-image safeguard for the override above — a confident LSTM
    # alone, sustained via the state machine's streak, must not be able to
    # confirm MANDOWN with literally no geometric grounding. Real case:
    # M_D_2.mp4 track 694 — persistence_score=0.0 and down_score<=0.20 for
    # the entire ~60-frame event, yet lstm_mandown_prob~0.99 alone plateaued
    # fusion_score at ~0.46 long enough to trigger a full MANDOWN confirm.
    # Calibrated against 4 real confirmed true positives (down_score>=0.27,
    # persistence>0 at their earliest up-tier frame) with margin to spare.
    lstm_only_geometric_grounding_down_floor: float = 0.22
    lstm_only_geometric_grounding_persistence_floor: float = 0.05
    # V4.1: mirror-image safeguard, one tier up — the floor above stops an
    # LSTM confirming with *zero* geometric grounding (down_score<=0.20),
    # but a sustained deep bend (straight legs, torso pitched forward) or a
    # small/low-quality detection can land down_score in a MODERATE
    # 0.2-0.6ish range that clears that floor easily while still not being
    # genuinely "lying". Real case: an RTSP false positive where a person
    # bending down for several seconds (and, separately, a small/distant
    # sitting person) reached confirmed MANDOWN. Calibrated against this
    # repo's 4 real validation videos: every genuine confirmed MANDOWN has
    # down_score in [0.704, 0.933] at its peak-confidence frame, and every
    # one already independently qualifies via geometric_confirm_threshold/
    # geometric_confirm_persistence_threshold above (unaffected by this).
    # The worst-case synthetic standing bend (~85 deg, straight legs, held
    # statically) tops out at down_score~0.55. This floor sits with margin
    # on both sides and only caps the LSTM-assisted weighted/strong-signal
    # paths -- genuine geometric-alone confirmation is untouched.
    confirmed_mandown_down_floor: float = 0.65
    # V4.2: alternate path to the floor above, for a person fallen toward/
    # away from the camera (foreshortened -- angle-based cues stay near
    # "upright", capping down_score structurally below the floor above even
    # after height-collapse corroboration).
    #
    # Three conditions, each excluding a DIFFERENT adversarial case --
    # floor_proximity_score and height_collapse_cue are NOT interchangeable,
    # despite both being "how close to the floor" signals:
    #   - floor_proximity_score also absorbs orientation_cue (torso angle)
    #     via max() in feature_mandown.py, so it alone does NOT exclude a
    #     deep forward bend -- measured: bend=0.893, actually higher than a
    #     genuine foreshortened fall's 0.742. An earlier version of this
    #     gate used floor_proximity_score alone and reopened the V4.1
    #     bending false positive through this new path.
    #   - height_collapse_cue is driven only by body_height_norm (never by
    #     torso angle), so it correctly excludes bending (~0.15,
    #     uncalibrated-equivalent) while staying high (~0.9) for a genuine
    #     fall -- but it does NOT exclude an occluded-seated-person case on
    #     its own (both read ~0.89-0.93).
    #   - pose_quality excludes the occluded-seated-person case (~0.17 vs a
    #     genuine fall's ~0.48 -- far fewer visible keypoints), but not
    #     bending (a clearly-visible posture, pose_quality~0.9).
    # All three together, measured on synthetic fixtures:
    #   genuine foreshortened fall: floor_prox=0.74, height_collapse=0.93, pose_q=0.48 (passes all)
    #   deep bend (V4.1 case):      floor_prox=0.89, height_collapse=0.15, pose_q=0.90 (fails height_collapse)
    #   occluded seated person:     floor_prox=0.59, height_collapse=0.89, pose_q=0.17 (fails floor_prox AND pose_q)
    confirmed_mandown_floor_proximity_floor: float = 0.70
    confirmed_mandown_height_collapse_floor: float = 0.5
    # V4.5: real RTSP false positive -- a person bending forward on their
    # feet to reach the ground got confirmed MANDOWN through this exact
    # alternate path. Root cause: this path's entire justification is that
    # torso_angle is *structurally blind* for a foreshortened fall (reads
    # ~0.0 same as standing), so floor_proximity/height_collapse have to
    # carry the evidence instead -- but the path never actually checked
    # that torso_angle WAS near-zero. It fired just the same for a deep
    # bend, where torso_angle is emphatically NOT blind (~0.64, per
    # deep_bend_keypoints()) and already carries clear "not down" evidence
    # that down_score/kneeling_penalty are supposed to apply -- this path
    # bypasses both entirely. Made worse under real per-camera height
    # calibration (vs. the simpler uncalibrated fixture the ~0.15 figure
    # above came from): tests/test_already_lying_integration.py's
    # test_standing_then_deep_bend_never_confirms measured a *calibrated*
    # deep bend's height_collapse_cue at ~0.499, a hair below the floor
    # above -- exactly the margin that a real camera then crossed.
    # torso_angle=0.0 for a genuine foreshortened fall vs. ~0.64 for a deep
    # bend is a wide, camera-calibration-independent gap (unlike
    # body_height_norm, which depends on per-camera height calibration
    # accuracy) -- restrict this path to when torso evidence is actually
    # available AND genuinely near-zero, matching the path's own stated
    # justification instead of ignoring torso entirely.
    alternate_grounding_torso_angle_ceiling: float = 0.2
    # V4.7 (attempted, REVERTED): tried a fourth path combining moderate-
    # but-simultaneously-elevated down_score/floor_proximity_score/
    # height_collapse_cue readings (targeting a real RTSP miss: a person
    # lying flat, legs extended, torso_angle=0.22 -- too shallow for the
    # alternate_grounding path above but too foreshortened for down_score's
    # own angle cue). Reverted before shipping: a synthetic crouching
    # fixture (torso_angle=0.0, kneeling_penalty only ~0.145 since its 2D
    # leg geometry doesn't read as bent) measured down_score=0.385,
    # floor_proximity=0.476, height_collapse_cue=0.556 -- all HIGHER than
    # or comparable to the real miss's own numbers on two of three axes,
    # so no threshold on these three signals alone could admit the real
    # miss without also reopening this crouching false positive under a
    # confident LSTM (verified: reached fusion_score=0.683, MANDOWN).
    # Needs a genuinely new discriminating signal (not just a looser
    # threshold on existing ones) before revisiting -- see the V4.7
    # discussion in session notes for the researched alternatives (ground-
    # plane homography calibration, monocular 3D pose lifting).
    #
    # V4.8: that genuinely new discriminator, via monocular 2D->3D pose
    # lifting (src/pose_3d.py). torso_angle_3d is a direct measurement of
    # real torso orientation, not a 2D-derived proxy -- crouching (torso
    # vertical in 3D despite a shortened observed bbox) reads LOW here,
    # unlike height_collapse_cue which can't tell it apart from genuine
    # lying. When 3D lifting confidently confirms lying AND there's at
    # least the same minimal geometric grounding the LSTM-alone safeguard
    # above already requires (down_score>=lstm_only_geometric_grounding_
    # down_floor), that's sufficient corroboration on its own -- this is
    # NOT gated behind the alternate path's floor_proximity/height_collapse
    # combination, since 3D evidence is a different, independent kind of
    # signal from those 2D-derived ones. Reuses
    # lstm_only_geometric_grounding_down_floor directly (not a separate
    # duplicated value) so the two floors can't silently drift apart.
    pose3d_lying_angle_floor: float = 0.55
    # V4.9: a deep forward bend at the hip also partially collapses
    # height_collapse_cue under real per-camera calibration (measured
    # live: body_height_norm=0.626, height_collapse_cue=0.53, clearing the
    # V4.8.1 floor above), so that requirement alone isn't sufficient --
    # hip_angle much lower than torso_angle (legs planted/vertical while
    # only the torso hinges over) is the reliable discriminator, same
    # signal added to KneelingConfig/_kneeling_penalty (see its V4.9 note
    # for the full rationale, including the uncalibrated-test false alarm
    # this was re-verified against with M_D_2.mp4's real calibration).
    hip_torso_mismatch_threshold: float = 0.35
    # V6.1 (Objective 1): floor for `body_axis_angle_3d` (the long-baseline
    # shoulder->ankle 3D angle, anchored on the leg already confirmed via
    # 2D keypoint confidence to be flat/extended -- see
    # src/pose_3d.py::Pose3DResult.body_axis_angle_3d) to count as
    # confirming lying, in EvidenceFusion's `confident_3d_lying` check.
    # DESIGN MARGIN: defaults to the SAME value as pose3d_lying_angle_floor
    # (0.55) since both are the same [0,1] angle-from-vertical convention
    # and no live footage was available this round to independently
    # re-tune it -- kept as its own field (not a shared reference) so the
    # two can be independently recalibrated later without a code change,
    # once real data distinguishes them.
    pose3d_axis_lying_angle_floor: float = 0.55


@dataclass
class StateMachineConfig:
    frames_to_confirm_mandown: int = 8
    frames_to_enter_possible: int = 3
    frames_to_recover_normal: int = 45
    cooldown_frames: int = 90


@dataclass
class PoseFailureConfig:
    hold_frames: int = 15               # how long a lost track is eligible for ROI re-acquisition
    gap_tolerance_frames: int = 2        # gaps at/below this don't decay hysteresis progress
    roi_reacquisition_padding_ratio: float = 0.6
    max_reacquisitions_per_frame: int = 5
    min_confidence_multiplier: float = 0.5  # floor for the gap-based confidence discount
    # V4.4: how long (wall-clock seconds) a confirmed-fallen track's last
    # known box keeps drawing after it drops out of detection entirely —
    # was a hardcoded 10s (fps*10 frames) tied to nothing, and real
    # feedback found it created a visibly "stuck" red box + MANDOWN banner
    # that persisted long after the person may have already gotten up and
    # left the scene, reading as a false alert with nobody there. 3s (2x
    # hold_frames' own re-acquisition window at a typical 10fps camera)
    # keeps a brief occlusion from causing a jarring flicker, without
    # leaving a stale box on screen for many seconds after real detection
    # is gone. Purely a rendering choice -- the underlying MANDOWN state
    # itself (sticky until sustained normal evidence, per TrackStateMachine)
    # is completely unaffected by how long this marker is drawn.
    occluded_marker_seconds: float = 3.0


@dataclass
class EventDedupConfig:
    # V4: same physical person, fragmented across two simultaneously-active
    # BoT-SORT track IDs (real case: man_down_c08.mp4, tracks 1 and 2 --
    # near-identical head/shoulder keypoints, diverging leg pose estimates,
    # ~0.28-0.40 bbox IoU sustained for 80+ concurrent frames), must not
    # produce a second independent alert. Calibrated with margin below the
    # observed real range; two genuinely different people rarely sustain
    # this much bbox overlap for a new event's very first frame.
    iou_threshold: float = 0.25
    # A candidate "same event" match must have been updated this recently —
    # without this, an event from early in a long video would stay eligible
    # for deduplication against an unrelated later event that happens to
    # start in a similar image region (events are never individually
    # cleared, only on a full pipeline reset).
    recency_frames: int = 30


@dataclass
class TrackBridgeConfig:
    # V4.2: BoT-SORT reassigning a new track_id to the same physically-
    # unmoving downed person (measured this session: one real validation
    # video went from a handful of tracks to 74 under current settings)
    # otherwise discards all accumulated Mandown state -- a brand new
    # MandownTrackContext starts at NORMAL with zero history, needing
    # frames_to_enter_possible+frames_to_confirm_mandown fresh "up" frames
    # to reconfirm, which reads as a flicker between MANDOWN and NORMAL for
    # someone who never got up. When a brand-new track_id's first bbox
    # closely overlaps a just-lost track's last known position, the SAME
    # context object (history + state machine) is adopted for the new ID
    # instead of creating a fresh one, then TrackStateMachine.on_gap() is
    # called on it exactly as it already is for a same-ID detection gap --
    # this is not a new risk category, it's the same already-accepted
    # "MANDOWN persists through a gap, but decays over sustained genuinely-
    # normal evidence afterward" behavior, just triggered by an ID change
    # instead of a same-ID gap. If the new ID actually belongs to a
    # different, healthy person, their own next-frame geometry reads
    # NORMAL regardless of the bridged state, and the existing 45-frame
    # recover-to-NORMAL countdown applies exactly as it already would.
    iou_threshold: float = 0.3
    # V4.3: raised from 30 -- calibrated frame counts elsewhere in this repo
    # (e.g. EventDedupConfig's 30) assume a 25-30fps recorded video, where
    # 30 frames is ~1-1.2s. This session's real RTSP camera runs at 10fps,
    # and a real observed same-physical-person track gap (confirmed MANDOWN
    # at one frame, reappearing at a nearby frame count later) measured
    # 107 frames (~10.7s) -- well beyond the old 30-frame/3s window, so
    # bridging never triggered and the person flickered back to NORMAL.
    # 300 frames (30s at 10fps, ~10-12s at 25-30fps) gives comfortable
    # margin above that observed real gap. The primary safety constraint
    # remains iou_threshold, not this window -- an unrelated person would
    # need to occupy nearly the exact same on-screen bbox, which a merely
    # nearby passerby won't, regardless of how long the window is.
    recency_frames: int = 300


@dataclass
class KneelingConfig:
    extension_ratio_threshold: float = 0.85  # below this, legs read as bent (kneeling/sitting)
    sigmoid_steepness: float = 12.0
    upright_gate_power: float = 1.0          # penalty scales with (1 - torso_angle) ** this
    min_gate: float = 0.35                   # floor so a sharp forward lean doesn't fully cancel
                                              # the bent-leg penalty (kneeling leaning far forward
                                              # reads high torso_angle too, not just genuine lying)
                                              # -- but the floor fades above
                                              # extreme_lying_torso_angle, so genuinely flat lying
                                              # with a bent knee isn't wrongly suppressed either
    extreme_lying_torso_angle: float = 0.9   # above this torso_angle, min_gate itself fades to ~0
    # V3: occlusion handling (Phase 6) — when legs aren't confidently visible,
    # leg_extension_ratio's neutral default must NOT be trusted as "confirmed
    # extended". Fall back to a leg-independent signal instead.
    leg_evidence_quality_threshold: float = 0.5  # below this, legs are "occluded"
    torso_upright_gap_threshold: float = 0.25    # head-to-hip normalized gap above which
                                                  # the torso alone reads as upright
    # V4.6: real RTSP miss -- a person fallen toward the camera (legs
    # unevidenced -- ankle confidence ~0.05, well below leg_evidence_
    # quality_threshold) was never flagged: kneeling_penalty applied at
    # near-full strength (upright_gate~0.98) purely because torso_angle
    # read ~0.02. Root cause: `upright_gate` above is the SAME torso-angle
    # blind spot fixed in fusion.py's alternate grounding path -- torso_
    # angle reads near-zero for BOTH a genuinely upright/kneeling person
    # facing the camera AND a person fallen toward/away from it, so this
    # gate can't tell them apart and defaults to "fully upright" (full
    # suppression) for both. Measured on the real miss: this gate alone
    # was cutting down_score from ~0.70 (would have confirmed) to 0.50,
    # and floor_proximity_score from ~0.82 to 0.59. Corroborate with
    # height_collapse_cue (driven by calibrated body_height_norm, never by
    # torso angle) the same way fusion.py's alternate path does: only
    # relent when torso is confirmed near-zero AND independent height
    # evidence strongly confirms genuine collapse -- kneeling/sitting also
    # shorten the observed bbox somewhat, but nowhere near a genuine
    # fall's degree, so the floor below sits with margin above ordinary
    # kneeling/sitting shortening.
    foreshortened_relent_torso_angle_ceiling: float = 0.2
    foreshortened_relent_height_collapse_floor: float = 0.7
    # V4.8: monocular 2D->3D pose lifting (src/pose_3d.py) gives a DIRECT
    # measurement of real torso orientation, immune to the foreshortening
    # blind spot the 2D-only gate above works around indirectly (via
    # height_collapse_cue, which needs per-camera height calibration and
    # only imperfectly separates "genuinely collapsed" from "moderately
    # shortened"). When 3D lifting confidently reads a body as lying
    # (torso_angle_3d past this floor), that's real, camera-calibration-
    # independent evidence the torso itself is not upright -- crouching
    # (bent knees, torso still vertical in 3D even though it lowers the
    # observed bbox height) should read LOW here, unlike the 2D height-
    # collapse proxy which can't tell the two apart (this was the exact
    # reason an earlier attempt at a similar "combined moderate evidence"
    # fusion path was reverted -- see fusion.py's V4.7 history). 0.55 is
    # a deliberately conservative floor (~50 deg from vertical) with
    # margin above genuine kneeling/crouching/bending, not tuned to the
    # bare minimum that admits any one real case.
    pose3d_lying_relent_floor: float = 0.55
    # V4.9: real RTSP false positive -- a deep forward bend at the hip
    # with straight legs (real measured example: torso_angle=0.82,
    # hip_angle=0.27) reached MANDOWN because kneeling_penalty's bent-leg
    # check correctly reads "not bent" (legs genuinely ARE straight here),
    # so down_score's torso-angle-driven cue climbed unopposed,
    # independent of the LSTM, 3D lifting, or height_collapse_cue. This is
    # a genuinely different signature than bent-knee kneeling OR real
    # diagonal/side lying: hinging at the hip while legs stay planted
    # keeps hip_angle near vertical even as torso_angle tilts far over --
    # whereas a real diagonally-lying body tilts as one consistent unit,
    # so torso_angle and hip_angle read close together (verified against
    # diagonal_lying_keypoints(): hip_angle~0.62 vs torso_angle~0.64, a
    # gap of ~0.02, nothing like this bend's ~0.55). A large gap between
    # the two, with both confidently evidenced, is itself independent
    # evidence of "bending at the hip", added as an alternative raw bend
    # signal alongside (not replacing) the existing bent-leg one -- still
    # gated by the same upright_gate as always, so genuinely-flat lying
    # with a folded thigh keeps its existing protection.
    #
    # An initial attempt at this value was checked against M_D_2.mp4's
    # protected true positive using an UNCALIBRATED per-frame test and
    # appeared to suppress its entire confirmed event -- reverted, then
    # re-verified with that camera's REAL saved height calibration wired
    # in: at every frame where that track actually reaches confirming
    # down_score levels, the real torso-hip mismatch is small (0.01-0.23,
    # well under the threshold below); only an already-below-threshold,
    # non-confirming frame showed a large mismatch. The uncalibrated test
    # was not representative of real deployment and gave a false alarm.
    hip_torso_mismatch_threshold: float = 0.35


@dataclass
class FloorRegionConfig:
    min_samples_for_confidence: int = 15
    full_confidence_samples: int = 50
    band_half_width_ratio: float = 0.5  # half-width of the floor band, as a multiple of ankle-y IQR


@dataclass
class MandownConfig:
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    rolling_size: RollingSizeConfig = field(default_factory=RollingSizeConfig)
    feature_mandown: FeatureMandownConfig = field(default_factory=FeatureMandownConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    pose_failure: PoseFailureConfig = field(default_factory=PoseFailureConfig)
    kneeling: KneelingConfig = field(default_factory=KneelingConfig)
    floor_region: FloorRegionConfig = field(default_factory=FloorRegionConfig)
    event_dedup: EventDedupConfig = field(default_factory=EventDedupConfig)
    track_bridge: TrackBridgeConfig = field(default_factory=TrackBridgeConfig)
    hip_drop: HipDropConfig = field(default_factory=HipDropConfig)
    pose_3d: Pose3DConfig = field(default_factory=Pose3DConfig)
    motion_signature: MotionSignatureConfig = field(default_factory=MotionSignatureConfig)
    rotation_search: RotationSearchConfig = field(default_factory=RotationSearchConfig)
    detection_recovery: DetectionRecoveryConfig = field(default_factory=DetectionRecoveryConfig)
    logging_level: str = "INFO"

    @staticmethod
    def from_yaml(path: str | Path = DEFAULT_CONFIG_PATH) -> "MandownConfig":
        """Load config from YAML, falling back to defaults for missing keys/file."""
        path = Path(path)
        if not path.exists():
            return MandownConfig()

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        cal = raw.get("calibration", {})
        rs = raw.get("rolling_size", {})
        fm = raw.get("feature_mandown", {})
        fm_weights = fm.get("weights", {})
        fm_orientation = fm.get("orientation", {})
        fu = raw.get("fusion", {})
        sm = raw.get("state_machine", {})
        pf = raw.get("pose_failure", {})
        kn = raw.get("kneeling", {})
        fr = raw.get("floor_region", {})
        ed = raw.get("event_dedup", {})
        tb = raw.get("track_bridge", {})
        hd = raw.get("hip_drop", {})
        p3 = raw.get("pose_3d", {})
        ms = raw.get("motion_signature", {})
        rs_search = raw.get("rotation_search", {})
        dr = raw.get("detection_recovery", {})
        fm_height_collapse = fm.get("height_collapse", {})

        return MandownConfig(
            calibration=CalibrationConfig(**cal),
            rolling_size=RollingSizeConfig(**rs),
            feature_mandown=FeatureMandownConfig(
                motion_window_frames=fm.get("motion_window_frames", 15),
                persistence_window_frames=fm.get("persistence_window_frames", 45),
                weights=FeatureMandownWeights(**fm_weights),
                sigmoid_steepness=fm.get("sigmoid_steepness", 10.0),
                orientation=OrientationConfig(**fm_orientation),
                height_collapse=HeightCollapseConfig(**fm_height_collapse),
            ),
            fusion=FusionConfig(**fu),
            state_machine=StateMachineConfig(**sm),
            pose_failure=PoseFailureConfig(**pf),
            kneeling=KneelingConfig(**kn),
            floor_region=FloorRegionConfig(**fr),
            event_dedup=EventDedupConfig(**ed),
            track_bridge=TrackBridgeConfig(**tb),
            hip_drop=HipDropConfig(**hd),
            pose_3d=Pose3DConfig(**p3),
            motion_signature=MotionSignatureConfig(**ms),
            rotation_search=RotationSearchConfig(**rs_search),
            detection_recovery=DetectionRecoveryConfig(**dr),
            logging_level=raw.get("logging_level", "INFO"),
        )
