"""Two-stage detection: lightweight full-frame person detector -> per-person
ROI crop -> YOLO11-Pose on the crop.

An alternative to running YOLO11-Pose directly on the full frame — NOT the
pipeline's default. Benchmarked against the full-frame pose model on real
footage where the full-frame pose model lost a lying person outright
(man_down_zoomed_demo__2_.mp4): general-purpose COCO person detectors
(yolo11n/yolov8n/yolo11s, all tested) missed that same person entirely,
even at very low confidence — worse than the full-frame pose model, which
at least found them for a while. See MANDOWN_IMPLEMENTATION_REPORT_V2.md
for the benchmark. Kept available (and correctly implements the
detector -> tracker -> ROI -> pose architecture) for scenes where a
dedicated detector's tighter, scale-normalized crops may help (e.g. many
small/far people), but the default pipeline instead fixes the lying-person
recall gap with `PersonDetector.redetect_in_roi` (Phase 3 pose-failure
handling), which reuses the pose model itself rather than a weaker general
detector.

Tracking is decoupled from pose: track IDs come from the person detector's
own boxes (`LightweightPersonDetector`, BoT-SORT), so a track's identity
survives even on frames where the pose crop fails to yield a usable
skeleton (`DetectedPerson.pose_valid=False`, `keypoints` all zero — never
fabricated).
"""

import logging

import numpy as np
import torch
from ultralytics import YOLO

from src.detector import (
    DetectedPerson,
    FrameResult,
    LightweightPersonDetector,
    compute_roi_crop,
    select_and_map_pose,
)

logger = logging.getLogger(__name__)

_EMPTY_KEYPOINTS = np.zeros((17, 3), dtype=np.float32)


class PersonROIPoseDetector:
    """Person detector + tracker -> ROI crop -> pose. Same `detect_frame()`
    interface as `PersonDetector` so it's a drop-in replacement in the pipeline."""

    def __init__(
        self,
        person_model: str = "yolo11n.pt",
        pose_model: str = "yolo11n-pose.pt",
        device: str | None = None,
        person_confidence_threshold: float = 0.35,
        pose_confidence_threshold: float = 0.3,
        roi_padding_ratio: float = 0.15,
        tracker_config: str = "configs/botsort.yaml",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.person_detector = LightweightPersonDetector(
            model_name=person_model, device=self.device,
            confidence_threshold=person_confidence_threshold, tracker_config=tracker_config,
        )
        logger.info(f"Loading pose model '{pose_model}' on device '{self.device}'")
        self.pose_model = YOLO(pose_model)
        self.pose_confidence_threshold = pose_confidence_threshold
        self.roi_padding_ratio = roi_padding_ratio

    def detect_frame(self, frame: np.ndarray, frame_num: int = 0) -> FrameResult:
        tracked_boxes = self.person_detector.track_frame(frame)
        frame_result = FrameResult(frame_num=frame_num)
        h, w = frame.shape[:2]

        for tb in tracked_boxes:
            if tb.track_id < 0:
                continue

            # Preserve the detector's own full-frame bbox as authoritative
            # (only the keypoints come from the ROI pose pass).
            keypoints, pose_valid = self._run_pose(frame, tb.bbox, w, h)
            frame_result.persons.append(DetectedPerson(
                track_id=tb.track_id, bbox=tb.bbox, keypoints=keypoints,
                confidence=tb.confidence, pose_valid=pose_valid,
            ))

        return frame_result

    def _run_pose(
        self, frame: np.ndarray, bbox: tuple[float, float, float, float], frame_w: int, frame_h: int
    ) -> tuple[np.ndarray, bool]:
        crop_box = compute_roi_crop(bbox, frame_w, frame_h, self.roi_padding_ratio)
        if crop_box is None:
            return _EMPTY_KEYPOINTS, False
        cx1, cy1, cx2, cy2 = crop_box
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return _EMPTY_KEYPOINTS, False

        try:
            results = self.pose_model.predict(
                crop, verbose=False, device=self.device, conf=self.pose_confidence_threshold,
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU OOM in pose model. Switching to CPU permanently.")
            self.device = "cpu"
            results = self.pose_model.predict(
                crop, verbose=False, device="cpu", conf=self.pose_confidence_threshold,
            )

        result = results[0] if results else None
        mapped = select_and_map_pose(result, (cx1, cy1))
        if mapped is None:
            return _EMPTY_KEYPOINTS, False
        _bbox_full, keypoints_full, _confidence = mapped
        return keypoints_full, True

    def reset_tracker(self):
        self.person_detector.reset_tracker()
