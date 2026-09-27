"""Feature extraction from COCO skeleton keypoints for fall detection."""

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.pose_3d import Pose3DResult

logger = logging.getLogger(__name__)

# COCO keypoint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Number of features extracted per frame per person
NUM_FEATURES = 5


@dataclass
class FeatureVector:
    """Features extracted from a single person in a single frame.

    Attributes:
        body_angle: Angle between hip-midpoint→nose line and vertical (degrees, 0-180).
        cog_height: Normalized center of gravity height (0.0=top, 1.0=bottom).
        vertical_velocity: Change in CoG height from previous frame (-1.0 to 1.0).
        bbox_aspect_ratio: Bounding box width / height (clamped to 0-5).
        keypoint_confidence: Mean confidence of all 17 keypoints (0.0-1.0).
    """

    body_angle: float
    cog_height: float
    vertical_velocity: float
    bbox_aspect_ratio: float
    keypoint_confidence: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array of shape (5,)."""
        return np.array(
            [
                self.body_angle,
                self.cog_height,
                self.vertical_velocity,
                self.bbox_aspect_ratio,
                self.keypoint_confidence,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def from_array(arr: np.ndarray) -> "FeatureVector":
        """Create FeatureVector from numpy array of shape (5,)."""
        return FeatureVector(
            body_angle=float(arr[0]),
            cog_height=float(arr[1]),
            vertical_velocity=float(arr[2]),
            bbox_aspect_ratio=float(arr[3]),
            keypoint_confidence=float(arr[4]),
        )


class FeatureExtractor:
    """Extracts fall-relevant features from skeleton keypoints.

    Features:
        1. Body angle (0-180 degrees, normalized to 0-1)
        2. Center of gravity height (normalized by frame height)
        3. Vertical velocity (delta CoG between frames)
        4. Bounding box aspect ratio (width/height)
        5. Mean keypoint confidence
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        normalize_angle: bool = True,
    ):
        """Initialize the feature extractor.

        Args:
            confidence_threshold: Minimum keypoint confidence to consider valid.
            normalize_angle: If True, normalize body angle to 0-1 range.
        """
        self.confidence_threshold = confidence_threshold
        self.normalize_angle = normalize_angle

    def extract(
        self,
        keypoints: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_height: int,
        prev_cog_height: float | None = None,
    ) -> FeatureVector:
        """Extract features from a single person's keypoints and bbox.

        Args:
            keypoints: Array of shape (17, 3) with [x, y, confidence].
            bbox: Bounding box as (x1, y1, x2, y2).
            frame_height: Height of the video frame in pixels.
            prev_cog_height: CoG height from previous frame for velocity computation.

        Returns:
            FeatureVector with all 5 features.
        """
        body_angle = self._compute_body_angle(keypoints)
        cog_height = self._compute_cog_height(keypoints, frame_height)
        vertical_velocity = self._compute_velocity(cog_height, prev_cog_height)
        bbox_aspect_ratio = self._compute_bbox_aspect_ratio(bbox)
        keypoint_confidence = self._compute_mean_confidence(keypoints)

        return FeatureVector(
            body_angle=body_angle,
            cog_height=cog_height,
            vertical_velocity=vertical_velocity,
            bbox_aspect_ratio=bbox_aspect_ratio,
            keypoint_confidence=keypoint_confidence,
        )

    def _compute_body_angle(self, keypoints: np.ndarray) -> float:
        """Compute angle between hip-midpoint→nose line and vertical axis.

        Returns:
            Angle in degrees (0-180), or normalized to 0-1 if normalize_angle is True.
            Standing ≈ 0, lying ≈ 90 (or 0.5 normalized).
            Returns 0.0 if required keypoints are not visible.
        """
        nose = keypoints[NOSE]
        left_hip = keypoints[LEFT_HIP]
        right_hip = keypoints[RIGHT_HIP]

        # Check if we have the needed keypoints with sufficient confidence
        has_nose = nose[2] >= self.confidence_threshold
        has_left_hip = left_hip[2] >= self.confidence_threshold
        has_right_hip = right_hip[2] >= self.confidence_threshold

        if not has_nose or not (has_left_hip or has_right_hip):
            # Try shoulder midpoint as fallback for torso angle
            left_shoulder = keypoints[LEFT_SHOULDER]
            right_shoulder = keypoints[RIGHT_SHOULDER]
            has_ls = left_shoulder[2] >= self.confidence_threshold
            has_rs = right_shoulder[2] >= self.confidence_threshold

            if has_nose and (has_ls or has_rs):
                if has_ls and has_rs:
                    mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
                    mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
                elif has_ls:
                    mid_x, mid_y = left_shoulder[0], left_shoulder[1]
                else:
                    mid_x, mid_y = right_shoulder[0], right_shoulder[1]
            else:
                return 0.0
        else:
            # Compute hip midpoint
            if has_left_hip and has_right_hip:
                mid_x = (left_hip[0] + right_hip[0]) / 2
                mid_y = (left_hip[1] + right_hip[1]) / 2
            elif has_left_hip:
                mid_x, mid_y = left_hip[0], left_hip[1]
            else:
                mid_x, mid_y = right_hip[0], right_hip[1]

        # Vector from midpoint to nose
        dx = float(nose[0] - mid_x)
        dy = float(mid_y - nose[1])  # Invert Y since image Y goes downward

        # Guard against NaN/Inf from corrupted keypoint coordinates
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return 0.0

        # Angle with vertical (upward direction)
        # abs(dx) so left/right lean gives same angle. Range: 0 (standing) to 180 (inverted)
        angle_rad = math.atan2(abs(dx), dy) if (dx != 0 or dy != 0) else 0.0
        angle_deg = math.degrees(angle_rad)

        # Clamp to 0-180
        angle_deg = max(0.0, min(180.0, angle_deg))

        if self.normalize_angle:
            return angle_deg / 180.0

        return angle_deg

    def _compute_cog_height(self, keypoints: np.ndarray, frame_height: int) -> float:
        """Compute normalized center of gravity height.

        Uses hip keypoints as CoG proxy, falls back to all visible keypoints.

        Returns:
            Normalized height (0.0=top, 1.0=bottom of frame).
            Returns 0.5 if no valid keypoints.
        """
        if frame_height <= 0:
            return 0.5

        left_hip = keypoints[LEFT_HIP]
        right_hip = keypoints[RIGHT_HIP]
        has_left_hip = left_hip[2] >= self.confidence_threshold
        has_right_hip = right_hip[2] >= self.confidence_threshold

        if has_left_hip and has_right_hip:
            cog_y = (left_hip[1] + right_hip[1]) / 2
        elif has_left_hip:
            cog_y = left_hip[1]
        elif has_right_hip:
            cog_y = right_hip[1]
        else:
            # Fallback: use mean Y of all visible keypoints
            valid_mask = keypoints[:, 2] >= self.confidence_threshold
            if not np.any(valid_mask):
                return 0.5
            cog_y = np.mean(keypoints[valid_mask, 1])

        result = float(cog_y / frame_height)
        if not math.isfinite(result):
            return 0.5
        return float(np.clip(result, 0.0, 1.0))

    def _compute_velocity(
        self, cog_height: float, prev_cog_height: float | None
    ) -> float:
        """Compute vertical velocity as delta CoG height.

        Returns:
            Velocity clamped to [-1.0, 1.0]. Positive = moving down.
            Returns 0.0 for the first frame of a track.
        """
        if prev_cog_height is None:
            return 0.0
        velocity = cog_height - prev_cog_height
        return float(np.clip(velocity, -1.0, 1.0))

    def _compute_bbox_aspect_ratio(
        self, bbox: tuple[float, float, float, float]
    ) -> float:
        """Compute bounding box width/height ratio.

        Returns:
            Aspect ratio clamped to [0.0, 5.0].
            Standing person ≈ 0.3-0.5 (tall/narrow).
            Fallen person ≈ 1.5-3.0 (wide/short).
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if height <= 0:
            return 0.0

        ratio = width / height
        return float(np.clip(ratio, 0.0, 5.0))

    def _compute_mean_confidence(self, keypoints: np.ndarray) -> float:
        """Compute mean confidence of all 17 keypoints.

        Returns:
            Mean confidence in [0.0, 1.0].
        """
        if keypoints is None or len(keypoints) == 0:
            return 0.0
        return float(np.clip(np.mean(keypoints[:, 2]), 0.0, 1.0))


# Keypoints considered "critical" for reliable posture reasoning.
_CRITICAL_KEYPOINTS = [
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE,
]


@dataclass
class ExtendedFeatureVector:
    """Richer per-frame features for the independent feature-based Mandown
    logic and evidence fusion. Kept separate from `FeatureVector` so the
    already-trained LSTM's 5-feature input contract is never touched.

    All angle features are normalized to 0-1 (0=vertical/upright, 1=horizontal).
    All height/position features are normalized 0-1 by frame height.
    Velocity/acceleration are per-frame deltas, positive = downward/rightward.
    """

    torso_angle: float
    hip_angle: float
    shoulder_angle: float
    head_height: float
    knee_height: float
    horizontal_velocity: float
    acceleration: float
    angle_change: float
    body_height_norm: float
    shoulder_hip_distance: float
    leg_extension_ratio: float
    leg_evidence_quality: float  # 0.0 (no leg visible) / 0.5 (one leg) / 1.0 (both legs)
    num_critical_keypoints_visible: int
    pose_quality: float
    body_axis_angle: float  # long-baseline head->ankle-mid axis, angle-from-vertical (0-1)
    body_axis_evidence_quality: float  # 0.0 (top or bottom endpoint entirely missing) / 1.0 (both present)
    torso_angle_evidence_quality: float  # 0.0 (shoulders or hips entirely missing) / 1.0 (both present)
    hip_angle_evidence_quality: float  # 0.0 (hips or knees entirely missing) / 1.0 (both present)
    # Monocular 2D->3D lifted torso angle (see src/pose_3d.py) -- same 0-1
    # convention (0=vertical, 1=horizontal) as torso_angle, but computed
    # from an inferred 3D skeleton so it isn't blind to a body oriented
    # toward/away from the camera the way the pure-2D angle features are.
    # evidence_quality 0.0 (default) means no 3D reading was available for
    # this frame (lifting disabled, unavailable, or below its own
    # confidence floor) -- never a fabricated angle.
    torso_angle_3d: float = 0.0
    torso_angle_3d_evidence_quality: float = 0.0
    # V6.1 (Objective 1): long-baseline 3D shoulder->ankle angle on the
    # single leg already confirmed, via 2D keypoint confidence alone (see
    # `dominant_leg_side` below), to be the flat/extended one -- never
    # touches the hip landmark MediaPipe's kinematic prior was found to
    # bias for an unusual-but-valid raised-knee configuration. Same [0,1]
    # convention as torso_angle_3d. 0.0/evidence_quality=0.0 (default)
    # means no reading was available (no confident dominant leg in 2D, no
    # 3D pose lifted this frame, or the chosen side's own landmarks read
    # below the pose_3d min_visibility floor) -- never fabricated.
    body_axis_angle_3d: float = 0.0
    body_axis_angle_3d_evidence_quality: float = 0.0


def dominant_leg_side(keypoints: np.ndarray, confidence_threshold: float = 0.3) -> str | None:
    """V6.1 (Objective 1): which leg ("L"/"R"), by ORDINARY 2D keypoint
    confidence alone (never a MediaPipe-derived value -- that's the whole
    point, see `Pose3DResult.body_axis_angle_3d`'s docstring), is the
    flat/extended one -- i.e. the straighter of the two legs among those
    with all 3 keypoints (hip, knee, ankle) confidently visible. Returns
    None when neither leg has all 3 keypoints confidently visible (never
    fabricates a side to anchor a 3D reading on).

    Deliberately independent of (and does not modify)
    `ExtendedFeatureExtractor._leg_extension_ratio`, which stays exactly
    as it was in V6 -- this is a new, additive helper for a new purpose,
    not a replacement for the existing one."""
    legs = (("L", LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), ("R", RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))
    candidates = []
    for side, hip_i, knee_i, ankle_i in legs:
        hip, knee, ankle = keypoints[hip_i], keypoints[knee_i], keypoints[ankle_i]
        if min(hip[2], knee[2], ankle[2]) < confidence_threshold:
            continue
        straight = math.hypot(hip[0] - ankle[0], hip[1] - ankle[1])
        thigh = math.hypot(hip[0] - knee[0], hip[1] - knee[1])
        shin = math.hypot(knee[0] - ankle[0], knee[1] - ankle[1])
        bent_path = thigh + shin
        if not (math.isfinite(straight) and math.isfinite(bent_path)) or bent_path < 1e-6:
            continue
        candidates.append((side, straight / bent_path))
    if not candidates:
        return None
    return max(candidates, key=lambda c: c[1])[0]


class ExtendedFeatureExtractor:
    """Extracts the extended posture/position/motion/geometry feature pool
    used by the direct feature-based Mandown scorer. Does not feed the LSTM."""

    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold

    def extract(
        self,
        keypoints: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
        prev: "ExtendedFeatureVector | None" = None,
        prev_cog_x: float | None = None,
        vertical_velocity_now: float | None = None,
        prev_vertical_velocity: float | None = None,
        expected_height_fn=None,
        ankle_x: float | None = None,
        pose_3d: "Pose3DResult | None" = None,
    ) -> ExtendedFeatureVector:
        """Extract extended features.

        Args:
            keypoints: (17, 3) array of [x, y, confidence].
            bbox: (x1, y1, x2, y2).
            frame_width/frame_height: frame dimensions in pixels.
            prev: previous frame's ExtendedFeatureVector for this track, or None.
            prev_cog_x: previous frame's normalized CoG x, for horizontal velocity.
            vertical_velocity_now: this frame's vertical velocity, as already
                computed by the base `FeatureExtractor` (delta CoG height).
            prev_vertical_velocity: previous frame's vertical velocity, so
                acceleration = vertical_velocity_now - prev_vertical_velocity.
            expected_height_fn: optional callable(bbox_bottom_y, observed_height,
                bbox_center_x) -> float returning observed/expected height ratio,
                e.g. a bound `SceneProfile.normalize_height`. Falls back to a
                neutral 1.0 (no signal) when None, invalid, or uncalibrated —
                never a fabricated value on a different scale.
            ankle_x: this person's ankle x-position (same quantity the scene
                profile's perspective fit was calibrated against — see
                `PersonDetector`/pipeline's `_ankle_position`), for the fit's
                horizontal-position term. Falls back to bbox center x when
                None (ankles not confidently visible), matching the same
                fallback `_ankle_position` itself uses during calibration.
            pose_3d: this frame's `Pose3DEstimator.estimate()` result, if
                available, computed by the caller (needs the raw frame
                pixels this method never receives). None (the default)
                means no 3D reading — `torso_angle_3d`/its evidence_quality
                then read 0.0, the same "no evidence" convention every
                other optional feature here uses.
        """
        conf = keypoints[:, 2]

        torso_angle = self._segment_angle_from_vertical(
            keypoints, (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP)
        )
        torso_angle_evidence_quality = self._segment_evidence_quality(
            keypoints, (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP)
        )
        hip_angle = self._segment_angle_from_vertical(
            keypoints, (LEFT_HIP, RIGHT_HIP), (LEFT_KNEE, RIGHT_KNEE)
        )
        hip_angle_evidence_quality = self._segment_evidence_quality(
            keypoints, (LEFT_HIP, RIGHT_HIP), (LEFT_KNEE, RIGHT_KNEE)
        )
        shoulder_angle = self._shoulder_tilt(keypoints)

        head_height = self._point_height(keypoints, [NOSE, LEFT_EYE, RIGHT_EYE], frame_height)
        knee_height = self._point_height(keypoints, [LEFT_KNEE, RIGHT_KNEE], frame_height)

        cog_x = self.cog_x(keypoints, bbox)
        x1, y1, x2, y2 = bbox
        bbox_width = max(x2 - x1, 1e-6)
        horizontal_velocity = 0.0
        if prev_cog_x is not None:
            horizontal_velocity = float(np.clip((cog_x - prev_cog_x) / bbox_width, -1.0, 1.0))

        # body_height_norm is a ratio to this track's/camera's expected
        # standing-height baseline: ~1.0 = normal, <1.0 = shorter than
        # expected (lying/crouching reduces vertical extent). Without a
        # calibrated baseline there is nothing valid to compare against, so
        # fall back to a neutral 1.0 rather than inventing a value on a
        # different scale (e.g. bbox_height/frame_height, which is not a
        # "ratio to expected" and would silently break downstream scoring).
        observed_height = max(y2 - y1, 0.0)
        bbox_center_x = ankle_x if ankle_x is not None else (x1 + x2) / 2
        body_height_norm = 1.0
        if expected_height_fn is not None:
            try:
                body_height_norm = float(expected_height_fn(y2, observed_height, bbox_center_x))
            except Exception:
                body_height_norm = 1.0
        if not math.isfinite(body_height_norm):
            body_height_norm = 1.0

        shoulder_hip_distance = self._shoulder_hip_distance_norm(keypoints, bbox)
        leg_extension_ratio, leg_evidence_quality = self._leg_extension_ratio(keypoints)
        body_axis_angle, body_axis_evidence_quality = self._body_axis_angle(keypoints)

        num_critical = int(
            sum(1 for k in _CRITICAL_KEYPOINTS if conf[k] >= self.confidence_threshold)
        )
        pose_quality = float(np.clip(np.mean(conf), 0.0, 1.0))

        angle_change = 0.0
        if prev is not None:
            angle_change = float(np.clip(abs(torso_angle - prev.torso_angle), 0.0, 1.0))

        acceleration = 0.0
        if vertical_velocity_now is not None and prev_vertical_velocity is not None:
            acceleration = float(np.clip(vertical_velocity_now - prev_vertical_velocity, -1.0, 1.0))

        return ExtendedFeatureVector(
            torso_angle=torso_angle,
            hip_angle=hip_angle,
            shoulder_angle=shoulder_angle,
            head_height=head_height,
            knee_height=knee_height,
            horizontal_velocity=horizontal_velocity,
            acceleration=acceleration,
            angle_change=angle_change,
            body_height_norm=float(np.clip(body_height_norm, 0.0, 3.0)),
            shoulder_hip_distance=shoulder_hip_distance,
            leg_extension_ratio=leg_extension_ratio,
            leg_evidence_quality=leg_evidence_quality,
            num_critical_keypoints_visible=num_critical,
            pose_quality=pose_quality,
            body_axis_angle=body_axis_angle,
            body_axis_evidence_quality=body_axis_evidence_quality,
            torso_angle_evidence_quality=torso_angle_evidence_quality,
            hip_angle_evidence_quality=hip_angle_evidence_quality,
            torso_angle_3d=pose_3d.torso_angle_3d if pose_3d is not None else 0.0,
            torso_angle_3d_evidence_quality=pose_3d.evidence_quality if pose_3d is not None else 0.0,
            body_axis_angle_3d=pose_3d.body_axis_angle_3d if pose_3d is not None else 0.0,
            body_axis_angle_3d_evidence_quality=(
                pose_3d.body_axis_angle_3d_evidence_quality if pose_3d is not None else 0.0
            ),
        )

    def _segment_angle_from_vertical(
        self, keypoints: np.ndarray, top_pair: tuple[int, int], bottom_pair: tuple[int, int]
    ) -> float:
        """Angle (0-1 normalized, 0=vertical, 1=horizontal) between the
        midpoint-to-midpoint line of two keypoint pairs and vertical."""
        top = self._midpoint(keypoints, top_pair)
        bottom = self._midpoint(keypoints, bottom_pair)
        if top is None or bottom is None:
            return 0.0
        dx = bottom[0] - top[0]
        dy = bottom[1] - top[1]
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return 0.0
        angle_rad = math.atan2(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 0.0
        return float(np.clip(math.degrees(angle_rad) / 90.0, 0.0, 1.0))

    def _segment_evidence_quality(
        self, keypoints: np.ndarray, top_pair: tuple[int, int], bottom_pair: tuple[int, int]
    ) -> float:
        """1.0 if both endpoints of a `_segment_angle_from_vertical` segment
        have at least one confidently-visible keypoint each, else 0.0 --
        without this, a fully-occluded endpoint silently defaults the angle
        to 0.0 ("vertical/upright"), which is missing evidence masquerading
        as confirmed-not-lying evidence."""
        top = self._midpoint(keypoints, top_pair)
        bottom = self._midpoint(keypoints, bottom_pair)
        return 1.0 if (top is not None and bottom is not None) else 0.0

    def _body_axis_angle(self, keypoints: np.ndarray) -> tuple[float, float]:
        """Long-baseline whole-body axis (head, or shoulder-mid if the nose
        isn't visible, to ankle-mid), angle-from-vertical (0-1). A longer
        lever arm than torso_angle/hip_angle means less angular error per
        pixel of keypoint jitter -- corroborating evidence for
        down_score/floor_proximity_score, not a leg-bend-immune replacement
        for torso_angle (ankles are just as bend-sensitive as knees, so this
        must never be allowed to unilaterally override a clearly-upright
        torso reading -- see the agreement-gated combination in
        FeatureMandownScorer).

        evidence_quality reflects BOTH endpoints, not just the ankles: if
        the top point (nose and both shoulders) is entirely missing too,
        the angle is a meaningless 0.0 default and must not be reported as
        confidently evidenced."""
        nose = keypoints[NOSE]
        has_nose = nose[2] >= self.confidence_threshold
        top_pair = (NOSE, NOSE) if has_nose else (LEFT_SHOULDER, RIGHT_SHOULDER)
        has_top = has_nose or self._midpoint(keypoints, (LEFT_SHOULDER, RIGHT_SHOULDER)) is not None
        left_ok = keypoints[LEFT_ANKLE][2] >= self.confidence_threshold
        right_ok = keypoints[RIGHT_ANKLE][2] >= self.confidence_threshold
        has_bottom = left_ok or right_ok
        if not has_top or not has_bottom:
            return 0.0, 0.0
        angle = self._segment_angle_from_vertical(keypoints, top_pair, (LEFT_ANKLE, RIGHT_ANKLE))
        evidence_quality = (float(left_ok) + float(right_ok)) / 2.0
        return angle, evidence_quality

    def _shoulder_tilt(self, keypoints: np.ndarray) -> float:
        """Angle (0-1, 0=horizontal i.e. upright shoulders, 1=vertical i.e.
        lying on side) of the shoulder-to-shoulder line relative to horizontal."""
        ls, rs = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]
        if ls[2] < self.confidence_threshold or rs[2] < self.confidence_threshold:
            return 0.0
        dx = float(rs[0] - ls[0])
        dy = float(rs[1] - ls[1])
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return 0.0
        angle_rad = math.atan2(abs(dy), abs(dx)) if (dx != 0 or dy != 0) else 0.0
        return float(np.clip(math.degrees(angle_rad) / 90.0, 0.0, 1.0))

    def _midpoint(self, keypoints: np.ndarray, pair: tuple[int, int]) -> tuple[float, float] | None:
        a, b = keypoints[pair[0]], keypoints[pair[1]]
        has_a, has_b = a[2] >= self.confidence_threshold, b[2] >= self.confidence_threshold
        if has_a and has_b:
            return float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2)
        if has_a:
            return float(a[0]), float(a[1])
        if has_b:
            return float(b[0]), float(b[1])
        return None

    def _point_height(
        self, keypoints: np.ndarray, candidates: list[int], frame_height: int
    ) -> float:
        if frame_height <= 0:
            return 0.5
        for idx in candidates:
            kp = keypoints[idx]
            if kp[2] >= self.confidence_threshold and math.isfinite(kp[1]):
                return float(np.clip(kp[1] / frame_height, 0.0, 1.0))
        return 0.5

    def cog_x(self, keypoints: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
        mid = self._midpoint(keypoints, (LEFT_HIP, RIGHT_HIP))
        if mid is not None and math.isfinite(mid[0]):
            return mid[0]
        return (bbox[0] + bbox[2]) / 2

    def _shoulder_hip_distance_norm(
        self, keypoints: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> float:
        shoulder_mid = self._midpoint(keypoints, (LEFT_SHOULDER, RIGHT_SHOULDER))
        hip_mid = self._midpoint(keypoints, (LEFT_HIP, RIGHT_HIP))
        bbox_height = max(bbox[3] - bbox[1], 1e-6)
        if shoulder_mid is None or hip_mid is None:
            return 0.0
        dist = math.hypot(shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
        if not math.isfinite(dist):
            return 0.0
        return float(np.clip(dist / bbox_height, 0.0, 2.0))

    def _leg_extension_ratio(self, keypoints: np.ndarray) -> tuple[float, float]:
        """hip-ankle straight-line distance / (hip-knee + knee-ankle) path length.

        ~1.0 for a fully extended leg (straight line hip->knee->ankle) —
        true whether the leg is extended vertically (standing) or
        horizontally (lying flat). Drops well below 1.0 when the knee is
        bent (kneeling, sitting, crouching), regardless of body/camera
        orientation, since a bent knee always makes the direct hip-ankle
        distance shorter than the two-segment path. This is what
        distinguishes a kneeling person (bent legs) from a lying person
        (extended legs) even when both have a forward-pitched torso.

        Returns (ratio, evidence_quality). Ratio defaults to a neutral 1.0
        (no bending evidence) when neither leg's three keypoints are all
        visible — never fabricates a penalty from missing data.
        evidence_quality is 0.0/0.5/1.0 for zero/one/both legs confidently
        visible, so callers can tell "confirmed extended" apart from
        "unknown, legs occluded" and avoid trusting the ratio's neutral
        default as if it were positive evidence.
        """
        legs = ((LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))
        ratios = []
        for hip_i, knee_i, ankle_i in legs:
            hip, knee, ankle = keypoints[hip_i], keypoints[knee_i], keypoints[ankle_i]
            if min(hip[2], knee[2], ankle[2]) < self.confidence_threshold:
                continue
            straight = math.hypot(hip[0] - ankle[0], hip[1] - ankle[1])
            thigh = math.hypot(hip[0] - knee[0], hip[1] - knee[1])
            shin = math.hypot(knee[0] - ankle[0], knee[1] - ankle[1])
            bent_path = thigh + shin
            if not (math.isfinite(straight) and math.isfinite(bent_path)) or bent_path < 1e-6:
                continue
            ratios.append(straight / bent_path)

        evidence_quality = len(ratios) / 2.0
        if not ratios:
            return 1.0, evidence_quality
        return float(np.clip(sum(ratios) / len(ratios), 0.0, 1.0)), evidence_quality
