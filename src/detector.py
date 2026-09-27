"""YOLO pose detection and BoT-SORT tracking wrapper."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.utils import COLOR_GREEN, COLOR_WHITE, draw_bbox, draw_skeleton

logger = logging.getLogger(__name__)


def _rotate_image(image: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate `image` about its center by `angle_deg`, expanding the
    output canvas so nothing is clipped. Returns (rotated_image,
    forward_affine_matrix) -- the matrix maps ORIGINAL-image (x,y) to
    ROTATED-image (x,y), for `_unrotate_points` to invert."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    m[0, 2] += (new_w / 2.0) - center[0]
    m[1, 2] += (new_h / 2.0) - center[1]
    rotated = cv2.warpAffine(image, m, (new_w, new_h))
    return rotated, m


def _unrotate_points(points_xy: np.ndarray, forward_matrix: np.ndarray) -> np.ndarray:
    """Map points from ROTATED-image coordinates back to the ORIGINAL
    (pre-rotation) image coordinates, inverting `_rotate_image`'s matrix."""
    inv = cv2.invertAffineTransform(forward_matrix)
    ones = np.ones((points_xy.shape[0], 1), dtype=points_xy.dtype)
    pts_h = np.hstack([points_xy, ones])
    return (inv @ pts_h.T).T


def compute_roi_crop(
    bbox: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    padding_ratio: float,
) -> tuple[int, int, int, int] | None:
    """Padded, frame-clamped integer crop box (cx1, cy1, cx2, cy2) around
    `bbox`. Returns None if the padded box degenerates (e.g. a zero-size
    input bbox). Shared by `PersonDetector.redetect_in_roi` and
    `PersonROIPoseDetector` so the crop math lives in exactly one place."""
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = (x2 - x1) * padding_ratio, (y2 - y1) * padding_ratio

    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(frame_w, int(x2 + pad_x))
    cy2 = min(frame_h, int(y2 + pad_y))

    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def select_and_map_pose(
    result, offset: tuple[int, int]
) -> tuple[tuple[float, float, float, float], np.ndarray, float] | None:
    """From a single-image pose `predict()` result on a crop, pick the
    largest detection (the crop is built tightly around one target person,
    so the largest box is almost always them) and map its bbox/keypoints
    back into full-frame coordinates via `offset`. Returns None if the
    result has nothing usable. Shared by `PersonDetector.redetect_in_roi`
    and `PersonROIPoseDetector`."""
    if result is None or result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas))

    kps = result.keypoints[idx].data.cpu().numpy()
    if kps.ndim == 3:
        kps = kps[0]
    if kps.shape[0] != 17 or kps.shape[1] < 2:
        return None
    if kps.shape[1] == 2:
        kps = np.hstack([kps, np.ones((17, 1), dtype=np.float32)])

    off_x, off_y = offset
    kps = kps.copy().astype(np.float32)
    kps[:, 0] += off_x
    kps[:, 1] += off_y

    box = boxes[idx]
    bbox_full = (
        float(box[0] + off_x), float(box[1] + off_y),
        float(box[2] + off_x), float(box[3] + off_y),
    )
    confidence = float(result.boxes.conf[idx].item())
    return bbox_full, kps, confidence


@dataclass
class DetectedPerson:
    """A single detected person in a frame."""

    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    keypoints: np.ndarray  # shape (17, 3) — [x, y, confidence]
    confidence: float
    pose_valid: bool = True  # False when the detector saw this person but pose estimation failed

    @property
    def bbox_width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def bbox_height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def bbox_center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )


@dataclass
class FrameResult:
    """Detection results for a single frame."""

    frame_num: int
    persons: list[DetectedPerson] = field(default_factory=list)

    @property
    def num_persons(self) -> int:
        return len(self.persons)

    @property
    def track_ids(self) -> list[int]:
        return [p.track_id for p in self.persons]


class PersonDetector:
    """YOLO pose detection with BoT-SORT tracking.

    This class wraps the Ultralytics YOLO pose model and provides
    a clean interface for detecting and tracking persons across frames.
    """

    def __init__(
        self,
        model_name: str = "yolo11s-pose.pt",
        device: str | None = None,
        confidence_threshold: float = 0.5,
        tracker_config: str | Path = "configs/botsort.yaml",
        imgsz: int = 960,
    ):
        """Initialize the person detector.

        Args:
            model_name: YOLO pose model name or path.
            device: Device to run on ('cuda', 'cpu', or None for auto).
            confidence_threshold: Minimum detection confidence.
            tracker_config: Path to BoT-SORT tracker configuration.
            imgsz: Inference resolution passed to the model.
        """
        self.confidence_threshold = confidence_threshold
        self.tracker_config = str(tracker_config)
        self.imgsz = imgsz

        # Auto-select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading YOLO model '{model_name}' on device '{self.device}'")
        try:
            self.model = YOLO(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model '{model_name}'. "
                f"Ensure you have internet access for first download. Error: {e}"
            ) from e

        logger.info("PersonDetector initialized successfully")

    def detect_frame(self, frame: np.ndarray, frame_num: int = 0) -> FrameResult:
        """Run detection + tracking on a single frame.

        Args:
            frame: BGR image as numpy array.
            frame_num: Frame number for the result.

        Returns:
            FrameResult with detected persons.
        """
        try:
            results = self.model.track(
                source=frame,
                tracker=self.tracker_config,
                persist=True,
                verbose=False,
                device=self.device,
                conf=self.confidence_threshold,
                imgsz=self.imgsz,
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                f"GPU OOM on frame {frame_num}. Switching to CPU permanently "
                f"and resetting tracker to avoid device mismatch."
            )
            self.device = "cpu"
            self.reset_tracker()
            results = self.model.track(
                source=frame,
                tracker=self.tracker_config,
                persist=True,
                verbose=False,
                device="cpu",
                conf=self.confidence_threshold,
                imgsz=self.imgsz,
            )

        result = results[0] if results else None
        frame_result = FrameResult(frame_num=frame_num)

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return frame_result

        boxes = result.boxes
        keypoints_data = result.keypoints

        for i in range(len(boxes)):
            # Get track ID (BoT-SORT assigns these)
            if boxes.id is not None:
                track_id = int(boxes.id[i].item())
            else:
                # No tracking ID available (first frame or lost track)
                track_id = -1

            # Get bounding box
            bbox = boxes.xyxy[i].cpu().numpy()
            bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

            # Get detection confidence
            conf = float(boxes.conf[i].item())

            # Get keypoints (17 keypoints, each with x, y, confidence)
            if keypoints_data is not None and i < len(keypoints_data):
                kps = keypoints_data[i].data.cpu().numpy()
                if kps.ndim == 3:
                    kps = kps[0]  # Remove batch dimension if present
                # Ensure shape is (17, 3)
                if kps.shape[0] != 17 or kps.shape[1] < 2:
                    logger.warning(f"Unexpected keypoint shape {kps.shape} for person {i}")
                    kps = np.zeros((17, 3), dtype=np.float32)
                elif kps.shape[1] == 2:
                    # No confidence scores, add ones
                    kps = np.hstack([kps, np.ones((17, 1), dtype=np.float32)])
            else:
                kps = np.zeros((17, 3), dtype=np.float32)

            person = DetectedPerson(
                track_id=track_id,
                bbox=bbox,
                keypoints=kps,
                confidence=conf,
            )
            frame_result.persons.append(person)

        return frame_result

    def reset_tracker(self):
        """Reset the tracker state. Call between different videos."""
        self.model.predictor = None

    def redetect_in_roi(
        self,
        frame: np.ndarray,
        last_known_bbox: tuple[float, float, float, float],
        padding_ratio: float = 0.6,
        conf_threshold: float | None = None,
    ) -> tuple[tuple[float, float, float, float], np.ndarray, float] | None:
        """Targeted pose re-acquisition for a track missing from the last
        full-frame `detect_frame()` call: crop generously around its last
        known bbox and retry the same pose model there.

        Real-footage testing found the full-frame call can lose an
        already-tracked lying person outright even at very low confidence,
        while a crop around their last known location recovers them
        confidently — cropping removes distracting background/scale
        variation the full-frame detector head struggles with. This is a
        bounded, on-demand retry (only for recently-lost tracks), not a
        second detection stage run every frame.

        Returns (bbox_in_full_frame, keypoints_in_full_frame, confidence),
        or None if nothing is found in the ROI.
        """
        h, w = frame.shape[:2]
        crop_box = compute_roi_crop(last_known_bbox, w, h, padding_ratio)
        if crop_box is None:
            return None
        cx1, cy1, cx2, cy2 = crop_box

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        conf = conf_threshold if conf_threshold is not None else self.confidence_threshold
        try:
            results = self.model.predict(crop, verbose=False, device=self.device, conf=conf)
        except torch.cuda.OutOfMemoryError:
            return None

        result = results[0] if results else None
        return select_and_map_pose(result, (cx1, cy1))

    def redetect_in_roi_with_rotation(
        self,
        frame: np.ndarray,
        last_known_bbox: tuple[float, float, float, float],
        prev_best_angle: float = 0.0,
        angle_step: float = 15.0,
        padding_ratio: float = 0.6,
        conf_threshold: float | None = None,
    ) -> tuple[tuple[float, float, float, float], np.ndarray, float, float] | None:
        """V6 Fix Set C (§4.1): narrow-range rotation search on a
        track's last known ROI, run ONLY as a fallback when
        `redetect_in_roi` (the plain, cheap straight-crop retry) has
        already failed to find anything this frame. A body oriented
        steeply off-vertical is read far more reliably by the SAME frozen
        pose model once rotated closer to its canonical upright-training
        orientation -- this is a documented technique for exactly this
        failure mode, not a new model or any retraining. Narrowing the
        search to {prev_best_angle-step, prev_best_angle, prev_best_angle
        +step} (rather than a full sweep) keeps it affordable; running it
        only on confirmed straight-crop failure keeps it rarer still.

        Returns (bbox_in_full_frame, keypoints_in_full_frame, confidence,
        winning_angle_degrees), or None if no candidate angle finds
        anything usable.
        """
        h, w = frame.shape[:2]
        crop_box = compute_roi_crop(last_known_bbox, w, h, padding_ratio)
        if crop_box is None:
            return None
        cx1, cy1, cx2, cy2 = crop_box
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        conf = conf_threshold if conf_threshold is not None else self.confidence_threshold
        candidates = {prev_best_angle - angle_step, prev_best_angle, prev_best_angle + angle_step}

        best: tuple[tuple[float, float, float, float], np.ndarray, float, float] | None = None
        best_mean_kp_conf = -1.0
        for angle in candidates:
            if angle == 0.0:
                rotated, forward_matrix = crop, None
            else:
                rotated, forward_matrix = _rotate_image(crop, angle)
            try:
                results = self.model.predict(rotated, verbose=False, device=self.device, conf=conf)
            except torch.cuda.OutOfMemoryError:
                continue
            result = results[0] if results else None
            mapped = select_and_map_pose(result, (0, 0))
            if mapped is None:
                continue
            bbox_in_rotated, kps_in_rotated, det_conf = mapped

            if forward_matrix is not None:
                kps_xy = _unrotate_points(kps_in_rotated[:, :2].astype(np.float64), forward_matrix)
                kps_crop = np.column_stack([kps_xy, kps_in_rotated[:, 2]]).astype(np.float32)
                box_corners = np.array(
                    [
                        [bbox_in_rotated[0], bbox_in_rotated[1]],
                        [bbox_in_rotated[2], bbox_in_rotated[3]],
                    ],
                    dtype=np.float64,
                )
                box_unrot = _unrotate_points(box_corners, forward_matrix)
                x1c, y1c = box_unrot[0]
                x2c, y2c = box_unrot[1]
                bbox_crop = (min(x1c, x2c), min(y1c, y2c), max(x1c, x2c), max(y1c, y2c))
            else:
                kps_crop = kps_in_rotated
                bbox_crop = bbox_in_rotated

            mean_kp_conf = float(np.mean(kps_crop[:, 2]))
            if mean_kp_conf > best_mean_kp_conf:
                kps_full = kps_crop.copy()
                kps_full[:, 0] += cx1
                kps_full[:, 1] += cy1
                bbox_full = (
                    bbox_crop[0] + cx1, bbox_crop[1] + cy1,
                    bbox_crop[2] + cx1, bbox_crop[3] + cy1,
                )
                best_mean_kp_conf = mean_kp_conf
                best = (bbox_full, kps_full, det_conf, angle)

        return best

    def draw_results(
        self,
        frame: np.ndarray,
        frame_result: FrameResult,
        draw_skeletons: bool = True,
        draw_boxes: bool = True,
        draw_ids: bool = True,
    ) -> np.ndarray:
        """Draw detection results on a frame.

        Args:
            frame: BGR image to draw on (modified in place).
            frame_result: Detection results for this frame.
            draw_skeletons: Whether to draw skeleton connections.
            draw_boxes: Whether to draw bounding boxes.
            draw_ids: Whether to draw track ID labels.

        Returns:
            The annotated frame.
        """
        for person in frame_result.persons:
            color = COLOR_GREEN

            if draw_skeletons:
                draw_skeleton(frame, person.keypoints, color=color)

            label = None
            if draw_ids and person.track_id >= 0:
                label = f"ID:{person.track_id} {person.confidence:.2f}"

            if draw_boxes:
                draw_bbox(frame, person.bbox, color=COLOR_WHITE, label=label)

        return frame


@dataclass
class TrackedBox:
    """A single tracked person bounding box, no pose — output of `LightweightPersonDetector`."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float


class LightweightPersonDetector:
    """Bbox-only person detector + BoT-SORT tracker (person class only).

    Decouples tracking from pose estimation: track IDs come from this
    detector's own boxes, independent of whether pose succeeds on the
    resulting ROI. This is the first stage of the
    detector → tracker → ROI → pose pipeline in `PersonROIPoseDetector`.
    """

    PERSON_CLASS_ID = 0  # COCO class 0

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        device: str | None = None,
        confidence_threshold: float = 0.35,
        tracker_config: str | Path = "configs/botsort.yaml",
    ):
        self.confidence_threshold = confidence_threshold
        self.tracker_config = str(tracker_config)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading person detector '{model_name}' on device '{self.device}'")
        self.model = YOLO(model_name)
        logger.info("LightweightPersonDetector initialized successfully")

    def track_frame(self, frame: np.ndarray) -> list[TrackedBox]:
        """Detect + track persons in one frame. Returns bbox + track_id only."""
        try:
            results = self.model.track(
                source=frame, tracker=self.tracker_config, persist=True, verbose=False,
                device=self.device, conf=self.confidence_threshold, classes=[self.PERSON_CLASS_ID],
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU OOM in person detector. Switching to CPU permanently.")
            self.device = "cpu"
            self.reset_tracker()
            results = self.model.track(
                source=frame, tracker=self.tracker_config, persist=True, verbose=False,
                device="cpu", conf=self.confidence_threshold, classes=[self.PERSON_CLASS_ID],
            )

        result = results[0] if results else None
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes
        out = []
        for i in range(len(boxes)):
            track_id = int(boxes.id[i].item()) if boxes.id is not None else -1
            bbox = boxes.xyxy[i].cpu().numpy()
            out.append(TrackedBox(
                track_id=track_id,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                confidence=float(boxes.conf[i].item()),
            ))
        return out

    def reset_tracker(self):
        self.model.predictor = None
