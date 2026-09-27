"""Monocular 2D->3D pose lifting via MediaPipe Pose Landmarker.

Root-cause fix for the failure mode every earlier fix this session worked
around instead of solving: a person's torso pointed toward/away from the
camera has almost no lateral spread in the 2D image (shoulders and hips
project to nearly the same x,y), so EVERY 2D-only angle feature
(`torso_angle`, `body_axis_angle`) reads "near vertical" whether the
person is standing or lying flat along the camera's own viewing axis.
That's a real information loss, not a bug in how those features are
computed -- a single 2D projection genuinely cannot separate "upright"
from "foreshortened-lying" along the camera's depth axis.

MediaPipe's Pose Landmarker (BlazePose) recovers an approximate metric 3D
skeleton from the same single image, via a learned prior over plausible
human body configurations (trained on motion-capture datasets), not a
real depth measurement -- it's an inference, not a measurement, and its
accuracy on an unusual camera angle (elevated, oblique, wide/fisheye-ish
CCTV, vs. the mostly eye-level datasets these models are trained on) is
not guaranteed without real validation on this project's own cameras.

World-landmark convention (verified empirically against this project's
own real footage, not just assumed from docs): `pose_world_landmarks` are
metric, hip-centered, with Y increasing DOWNWARD for a person in a normal
(not upside-down/heavily rolled) camera view -- confirmed on a real
standing person (torso_vec dominated by +Y, ~0.47 vs ~0.01 lateral) and a
real clearly-lying-flat person (angle from the Y axis ~77-86 deg, vs. a
2D-only torso_angle that would read near-zero for the same person if they
were foreshortened instead of lying broadside-on). This is a documented,
stable BlazePose convention (not a per-camera-varying quantity the way
2D pixel calibration is), so no per-camera calibration step is needed for
this axis the way `SceneProfile`'s height calibration needs one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

# MediaPipe BlazePose landmark indices -- a different topology/ordering
# than the COCO keypoints YOLO produces elsewhere in this codebase; do not
# mix the two index sets.
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_HIP, _R_HIP = 23, 24
_L_ANKLE, _R_ANKLE = 27, 28

DEFAULT_MODEL_PATH = "models/mediapipe/pose_landmarker_lite.task"


@dataclass
class Pose3DResult:
    # Angle between the 3D shoulder->hip vector and the vertical (Y) axis,
    # normalized to [0, 1] the same way the existing 2D torso_angle is
    # (0 = vertical/upright, 1 = fully horizontal/lying) -- so it can be
    # compared against the same thresholds without a unit conversion at
    # every call site.
    torso_angle_3d: float
    # Mean landmark visibility across the four joints used, in [0, 1].
    # 0.0 is never returned by `estimate()` (that returns None instead,
    # see `min_visibility`) -- this is the confidence a caller should
    # weigh the reading by, not a presence/absence flag.
    evidence_quality: float
    # V6.1 (Objective 1): angle between the 3D shoulder->ankle vector on
    # ONE side -- the side the caller has already confirmed (via 2D
    # keypoint confidence alone, see feature_extraction.dominant_leg_side)
    # is the flat/extended leg -- and vertical. Same [0,1] convention as
    # torso_angle_3d.
    #
    # Root cause this targets: V6's per-side torso_angle_3d correction
    # (gated on a VISIBILITY asymmetry between the two hips) was found,
    # via real re-measurement, to not engage on the real raised-knee case
    # at all -- MediaPipe reports ~equally high visibility (~0.997) on
    # BOTH hips even though its inferred hip position is wrong for the
    # raised-knee side. The bias is in MediaPipe's kinematic prior
    # reacting to an unusual-but-valid limb configuration -- a
    # confidently WRONG estimate, not an uncertain one -- so no
    # confidence-based gate on the hip landmark itself can ever detect
    # it. This feature sidesteps the problem instead of trying to detect
    # it: it never asks the biased hip landmark anything at all. A long
    # shoulder-to-ankle baseline on the leg already independently
    # confirmed (via ordinary 2D keypoint confidence, not a MediaPipe-
    # derived value) to be flat/extended is far less sensitive to a
    # single joint's kinematic-prior bias than the short torso vector.
    #
    # 0.0 (default) means no reading was available for this frame -- see
    # `body_axis_angle_3d_evidence_quality`.
    body_axis_angle_3d: float = 0.0
    # Mean visibility of the shoulder+ankle pair actually used, in [0,1].
    # 0.0 means "no side was confidently identified in 2D, or the chosen
    # side's own 3D landmarks read below min_visibility" -- never a
    # fabricated reading.
    body_axis_angle_3d_evidence_quality: float = 0.0


class Pose3DEstimator:
    """Per-person, per-frame 2D->3D lifting. Crops the given bbox (with
    padding, since BlazePose expects to see a full body with some margin,
    not a tight box) and runs MediaPipe's pose landmarker on just that
    crop -- YOLO+BoT-SORT already handles multi-person detection/tracking
    upstream; this only needs to lift ONE already-localized person at a
    time, run per detection rather than once per frame."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        min_pose_detection_confidence: float = 0.3,
        min_visibility: float = 0.5,
        crop_pad_ratio: float = 0.25,
        asymmetry_threshold: float = 0.3,
    ):
        self.min_visibility = min_visibility
        self.crop_pad_ratio = crop_pad_ratio
        # V6 Fix Set A (§2.2): see `estimate()` -- gates the per-side
        # (asymmetric) angle correction.
        self.asymmetry_threshold = asymmetry_threshold
        self._landmarker = None

        if not _MEDIAPIPE_AVAILABLE:
            logger.warning(
                "mediapipe is not installed -- 3D pose lifting disabled, "
                "falls back to 2D-only orientation features everywhere."
            )
            return

        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning(
                f"Pose3DEstimator model not found at {model_path} -- 3D pose "
                "lifting disabled, falls back to 2D-only orientation features."
            )
            return

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    @property
    def is_available(self) -> bool:
        return self._landmarker is not None

    def estimate(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        dominant_leg_side: str | None = None,
    ) -> Pose3DResult | None:
        """Returns None (never a fabricated reading) when: lifting is
        unavailable, the crop is degenerate, MediaPipe finds no pose in
        it, or the found pose's landmark visibility is below
        `min_visibility` -- an unreliable 3D reading is worse than none,
        since callers treat this as strong, direct evidence.

        `dominant_leg_side` ("L" or "R", or None): V6.1 Objective 1 --
        the leg the CALLER has already identified, from ordinary 2D
        keypoint confidence alone (see
        `feature_extraction.dominant_leg_side`), as the flat/extended one.
        When given, also computes `body_axis_angle_3d` from that side's
        OWN shoulder->ankle vector -- deliberately never touching the hip
        landmark, which is the one MediaPipe's kinematic prior was found
        to bias for an unusual-but-valid limb configuration (see
        `Pose3DResult.body_axis_angle_3d`'s docstring)."""
        if self._landmarker is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        bbox_w, bbox_h = x2 - x1, y2 - y1
        pad_x, pad_y = bbox_w * self.crop_pad_ratio, bbox_h * self.crop_pad_ratio
        cx1, cy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        cx2, cy2 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            return None

        crop = frame[cy1:cy2, cx1:cx2]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        )
        try:
            result = self._landmarker.detect(mp_image)
        except Exception:
            logger.debug("MediaPipe pose detection raised", exc_info=True)
            return None

        if not result.pose_world_landmarks:
            return None

        wl = result.pose_world_landmarks[0]
        l_sh, r_sh, l_hip, r_hip = wl[_L_SHOULDER], wl[_R_SHOULDER], wl[_L_HIP], wl[_R_HIP]
        visibilities = [l_sh.visibility, r_sh.visibility, l_hip.visibility, r_hip.visibility]
        evidence_quality = float(np.clip(np.mean(visibilities), 0.0, 1.0))
        if evidence_quality < self.min_visibility:
            return None

        bilateral_angle = self._torso_angle(l_sh, r_sh, l_hip, r_hip)
        if bilateral_angle is None:
            return None
        torso_angle_3d = bilateral_angle

        # V6 Fix Set A (§2.2): a raised knee / reaching arm / arm tucked
        # under the torso can shift MediaPipe's inferred hip or shoulder
        # position asymmetrically between the two sides -- the BILATERAL
        # midpoint used above then averages a corrupted side into a clean
        # one, artificially deflating the resulting angle for a genuinely
        # foreshortened-lying body (real measured case: 2026-09-14
        # 18:19:32 screenshot, prone person with one knee bent --
        # bilateral torso_angle_3d landed at 0.319, below the 0.55
        # "confident lying" floor). When one side's own two landmarks are
        # confidently more visible than the other's, that side's own
        # shoulder->hip vector is a second, independent candidate
        # reading -- take the LARGER of the two (never the single-side
        # reading alone), so this can only ever RAISE torso_angle_3d
        # relative to the pre-V6 bilateral-only behavior, never lower it.
        # This makes the fix provably monotonic w.r.t. every existing
        # confirmed case (a body that reads unambiguously lying on both
        # sides is completely unaffected -- max() just keeps the
        # bilateral value), without needing a live-camera re-verification
        # pass to rule out a same-session regression.
        w_l = l_sh.visibility * l_hip.visibility
        w_r = r_sh.visibility * r_hip.visibility
        if abs(w_l - w_r) > self.asymmetry_threshold:
            side_sh, side_hip = (l_sh, l_hip) if w_l > w_r else (r_sh, r_hip)
            side_angle = self._torso_angle(side_sh, side_sh, side_hip, side_hip)
            if side_angle is not None:
                torso_angle_3d = max(torso_angle_3d, side_angle)

        body_axis_angle_3d = 0.0
        body_axis_angle_3d_evidence_quality = 0.0
        if dominant_leg_side in ("L", "R"):
            ankle = wl[_L_ANKLE] if dominant_leg_side == "L" else wl[_R_ANKLE]
            shoulder = l_sh if dominant_leg_side == "L" else r_sh
            # min(), not mean(): a single unreliable landmark in the pair
            # makes the WHOLE vector unreliable -- an artificially high
            # partner visibility must not paper over it.
            axis_evidence = float(np.clip(min(shoulder.visibility, ankle.visibility), 0.0, 1.0))
            if axis_evidence >= self.min_visibility:
                axis_angle = self._angle_from_vertical(
                    np.array([shoulder.x, shoulder.y, shoulder.z]),
                    np.array([ankle.x, ankle.y, ankle.z]),
                )
                if axis_angle is not None:
                    body_axis_angle_3d = axis_angle
                    body_axis_angle_3d_evidence_quality = axis_evidence

        return Pose3DResult(
            torso_angle_3d=torso_angle_3d,
            evidence_quality=evidence_quality,
            body_axis_angle_3d=body_axis_angle_3d,
            body_axis_angle_3d_evidence_quality=body_axis_angle_3d_evidence_quality,
        )

    @staticmethod
    def _angle_from_vertical(top: np.ndarray, bottom: np.ndarray) -> float | None:
        """Angle between the `top`->`bottom` 3D vector and vertical (Y),
        normalized to [0, 1] (0=vertical/upright, 1=horizontal/lying)."""
        vec = bottom - top
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None
        angle_rad = float(np.arctan2(np.hypot(vec[0], vec[2]), abs(vec[1])))
        return float(np.clip(angle_rad / (np.pi / 2), 0.0, 1.0))

    @classmethod
    def _torso_angle(cls, l_sh, r_sh, l_hip, r_hip) -> float | None:
        """Angle between the 3D shoulder->hip vector and vertical (Y),
        normalized to [0, 1] (0=vertical/upright, 1=horizontal/lying).
        Pass the SAME landmark twice for both `l_*`/`r_*` args to get a
        single-side (not bilateral-midpoint) reading."""
        sh_mid = np.array([(l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2, (l_sh.z + r_sh.z) / 2])
        hip_mid = np.array([(l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2, (l_hip.z + r_hip.z) / 2])
        return cls._angle_from_vertical(sh_mid, hip_mid)
