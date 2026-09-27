"""End-to-end fall detection pipeline: video/RTSP → output video, JSON log, event screenshots."""

import json
import logging
import re
import signal
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.fall_detector import FallDetector
from src.config import MandownConfig
from src.detector import DetectedPerson, FrameResult, PersonDetector
from src.feature_extraction import (
    LEFT_ANKLE,
    RIGHT_ANKLE,
    ExtendedFeatureExtractor,
    FeatureExtractor,
    dominant_leg_side,
)
from src.feature_mandown import FeatureMandownScorer
from src.fusion import EvidenceFusion
from src.geometry import iou
from src.mandown_track_context import MandownTrackContextManager
from src.person_size_stats import is_reliable_standing_sample
from src.pose_3d import Pose3DEstimator
from src.scene_profile import SceneProfileBuilder
from src.state_machine import MandownState
from src.track_manager import TrackManager
from src.utils import (
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    draw_fall_banner,
    draw_mandown_banner,
    draw_skeleton,
    draw_status_bar,
)
from src.video_io import VideoReader, VideoWriter, format_short_timestamp, format_timestamp

logger = logging.getLogger(__name__)

# Maps the state machine's 3 states onto the existing display/event vocabulary.
_STATE_TO_LABEL = {
    MandownState.NORMAL: "normal",
    MandownState.POSSIBLE_MANDOWN: "falling",
    MandownState.MANDOWN: "fallen",
}

# State colors
STATE_COLORS = {
    "normal": COLOR_GREEN,
    "falling": COLOR_ORANGE,
    "fallen": COLOR_RED,
}

# Display labels for each classification state
STATE_LABELS = {
    "normal": "STANDING",
    "falling": "FALLING",
    "fallen": "FALLEN",
}


Box = tuple[float, float, float, float]
_iou = iou


class FallEvent:
    """Represents a detected fall/mandown event, with evidence diagnostics
    captured at the moment of highest confidence for later verification."""

    def __init__(
        self, track_id: int, start_frame: int, fps: float, confidence: float,
        bbox: Box | None = None,
    ):
        self.track_id = track_id
        self.start_frame = start_frame
        self.end_frame = start_frame
        self.fps = fps
        self.confidence = confidence
        self.state = "falling"
        self.screenshot_saved = False
        self.screenshot_path: str | None = None
        self.diagnostics: dict = {}
        self.last_bbox: Box | None = bbox
        self.duplicate_track_ids: list[int] = []

    def update(
        self, frame_num: int, confidence: float, state: str, diagnostics: dict | None = None,
        bbox: Box | None = None,
    ):
        self.end_frame = frame_num
        if confidence >= self.confidence and diagnostics:
            self.diagnostics = diagnostics
        self.confidence = max(self.confidence, confidence)
        self.state = state
        if bbox is not None:
            self.last_bbox = bbox

    def to_dict(self) -> dict:
        return {
            "type": "fall",
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "timestamp_start": format_timestamp(self.start_frame, self.fps),
            "timestamp_end": format_timestamp(self.end_frame, self.fps),
            "confidence": round(self.confidence, 3),
            # True once this track was actually confirmed MANDOWN (state
            # machine reached MANDOWN, screenshot captured) — False means it
            # only ever reached the tentative "falling"/POSSIBLE_MANDOWN
            # tier and was never confirmed. Same signal as screenshot_saved,
            # exposed under a clearer name for JSON consumers.
            "confirmed": self.screenshot_saved,
            "screenshot_path": self.screenshot_path,
            "evidence": self.diagnostics,
            "duplicate_track_ids": self.duplicate_track_ids,
        }


class FallDetectionPipeline:
    """Full pipeline: YOLO detection → tracking → features → LSTM → output.

    Supports two sources:
    - A video file, via `process_video()` — writes an annotated output video.
    - A live stream (RTSP URL or webcam index), via `run_stream()` — shows a
      real-time preview window and never writes an inference video.

    In both cases, a screenshot is saved to `events_dir` the moment a tracked
    person is classified as fully "fallen" (not just "falling"), once per event.
    """

    def __init__(
        self,
        yolo_model: str = "yolo11s-pose.pt",
        lstm_checkpoint: str = "models/checkpoints/best.pth",
        device: str | None = None,
        confidence_threshold: float = 0.40,
        cooldown_frames: int | None = None,
        tracker_config: str = "configs/botsort.yaml",
        events_dir: str | Path = "events",
        mandown_config_path: str | Path | None = None,
        imgsz: int = 1280,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing pipeline on {self.device}...")

        # YOLO detector
        self.detector = PersonDetector(
            model_name=yolo_model,
            device=self.device,
            confidence_threshold=0.5,
            tracker_config=tracker_config,
            imgsz=imgsz,
        )

        # LSTM classifier — untouched input contract (5 features, 30-frame
        # sequence) so the pretrained checkpoint's 97-98% test accuracy
        # stays valid. Treated as one evidence source, not sole authority.
        self.lstm = FallDetector()
        ckpt = torch.load(lstm_checkpoint, map_location=self.device, weights_only=False)
        self.lstm.load_state_dict(ckpt["model_state_dict"])
        self.lstm.to(self.device)
        self.lstm.eval()
        logger.info(f"LSTM loaded from {lstm_checkpoint}")

        # Config for the algorithmic core added on top of the LSTM.
        self.mandown_config = MandownConfig.from_yaml(mandown_config_path) if mandown_config_path \
            else MandownConfig.from_yaml()

        # Feature extraction — base (LSTM-compatible, unchanged) + extended
        # (independent feature-based Mandown logic, Phase 5/6).
        self.feature_extractor = FeatureExtractor()
        self.ext_feature_extractor = ExtendedFeatureExtractor()

        # Monocular 2D->3D pose lifting (V4.8) — disabled cleanly (every
        # 3D-derived feature reads as "no evidence") when the config says
        # so, or when mediapipe/the model file aren't available; never a
        # hard dependency for the rest of the pipeline.
        p3_cfg = self.mandown_config.pose_3d
        self.pose3d_estimator = (
            Pose3DEstimator(
                model_path=p3_cfg.model_path,
                min_pose_detection_confidence=p3_cfg.min_pose_detection_confidence,
                min_visibility=p3_cfg.min_visibility,
                crop_pad_ratio=p3_cfg.crop_pad_ratio,
            )
            if p3_cfg.enabled
            else None
        )
        self.track_manager = TrackManager(window_size=30, stale_threshold=90)
        self.mandown_contexts = MandownTrackContextManager(
            history_maxlen=30,
            # A context must outlive the bridging recency window with
            # margin, or a real gap near that window's edge could get the
            # old context garbage-collected just before a same-physical-
            # person reappearance could find it. Derived from the config
            # rather than a second hardcoded number, so the two can't drift
            # out of sync if recency_frames is ever retuned.
            stale_threshold=max(90, self.mandown_config.track_bridge.recency_frames * 2),
            state_machine_config=self.mandown_config.state_machine,
            track_bridge_config=self.mandown_config.track_bridge,
        )
        self.feature_scorer = FeatureMandownScorer(
            self.mandown_config.feature_mandown, self.mandown_config.kneeling
        )
        self.fusion = EvidenceFusion(self.mandown_config.fusion)

        # Camera-independent scene normalization (Phase 3/4) — (re)created
        # per run in process_video/run_stream once frame dims + camera_id
        # are known.
        self.scene_builder: SceneProfileBuilder | None = None
        self._scene_profile_cache = None
        self._scene_profile_cache_frame = -1

        # Settings
        self.confidence_threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames or self.mandown_config.state_machine.cooldown_frames
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._interrupted = False
        self._cooldowns: dict[int, int] = {}  # track_id → last alert frame
        self._active_events: dict[int, FallEvent] = {}
        self._all_events: list[FallEvent] = []
        # Remember fallen person's last bbox for occlusion persistence
        self._fallen_positions: dict[int, tuple] = {}  # track_id → (bbox, last_frame)
        # track_id → the track_id of the event it's been merged into, once
        # identified as the same physical person under a fragmented ID.
        self._duplicate_of: dict[int, int] = {}

    def _reset_state(self):
        """Clear all per-run state. Call before processing a new video/stream."""
        self._interrupted = False
        self._cooldowns.clear()
        self._active_events.clear()
        self._all_events.clear()
        self._fallen_positions.clear()
        self._duplicate_of.clear()
        self.track_manager.reset()
        self.mandown_contexts.reset()
        if self.scene_builder is not None:
            self.scene_builder.close()
        self.scene_builder = None
        self._scene_profile_cache = None
        self._scene_profile_cache_frame = -1
        self.detector.reset_tracker()

    @staticmethod
    def _derive_camera_id(source: str | Path) -> str:
        """Derive a filesystem-safe camera_id from a video path or RTSP/HTTP URL."""
        raw = str(source)
        # Strip credentials from RTSP URLs before using it as an identifier.
        raw = re.sub(r"://[^@/]+@", "://", raw)
        stem = Path(raw).stem if not raw.startswith(("rtsp://", "http://", "https://")) else raw
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "camera"

    def _ensure_scene_builder(self, camera_id: str, frame_width: int, frame_height: int) -> None:
        if self.scene_builder is not None:
            self.scene_builder.close()
        self.scene_builder = SceneProfileBuilder(
            camera_id=camera_id,
            frame_width=frame_width,
            frame_height=frame_height,
            cal=self.mandown_config.calibration,
            sample_log_dir="configs/scene_profiles",
        )
        self._scene_profile_cache = self.scene_builder.build_profile()
        self._scene_profile_cache_frame = -1

    def _current_scene_profile(self, frame_num: int):
        """Scene profile refit is O(50 samples); cache per-frame so all
        persons in a frame share one profile instead of rebuilding per-track."""
        if self._scene_profile_cache_frame != frame_num:
            self._scene_profile_cache = self.scene_builder.build_profile()
            self._scene_profile_cache_frame = frame_num
        return self._scene_profile_cache

    def _reacquire_lost_tracks(
        self, frame: np.ndarray, frame_num: int, frame_result: FrameResult, fps: float = 25.0,
    ) -> None:
        """Targeted pose re-acquisition for tracks missing from this frame's
        normal detection pass. Real-footage testing found the full-frame
        detector can lose an already-tracked lying person outright; a crop
        around their last known bbox recovers them reliably. Bounded to
        recently-lost tracks only (`hold_frames`) and capped per frame
        (`max_reacquisitions_per_frame`) so this stays a rare, targeted
        retry, not a second detection pass run every frame.

        V6 Fix Set C: two additional, progressively rarer fallbacks when
        the plain straight-crop retry above finds nothing at all --
        (§4.1) a narrow-range ROTATION SEARCH on the same ROI (a body
        oriented steeply off-vertical reads far more reliably once
        rotated closer to the pose model's canonical upright-training
        orientation -- same frozen model, no retraining), then (§4.2) a
        confidence-decayed CONSTANT-VELOCITY EXTRAPOLATION of this
        track's last known keypoints, bounded to
        `DetectionRecoveryConfig.t_gap_max_seconds` of real, continuous
        non-detection so a track is dropped for real rather than
        extrapolated indefinitely (a ghost track)."""
        cfg = self.mandown_config.pose_failure
        rot_cfg = self.mandown_config.rotation_search
        rec_cfg = self.mandown_config.detection_recovery
        active_ids = {p.track_id for p in frame_result.persons if p.track_id >= 0 and p.pose_valid}
        attempts = 0

        for tid, ctx in self.mandown_contexts.contexts.items():
            if tid in active_ids or ctx.last_known_bbox is None:
                continue
            gap = frame_num - ctx.last_seen_frame
            if gap <= 0 or gap > cfg.hold_frames:
                continue
            if attempts >= cfg.max_reacquisitions_per_frame:
                break
            attempts += 1

            recovered = self.detector.redetect_in_roi(
                frame, ctx.last_known_bbox, padding_ratio=cfg.roi_reacquisition_padding_ratio,
            )
            winning_angle = 0.0
            if recovered is None and rot_cfg.enabled:
                rotated = self.detector.redetect_in_roi_with_rotation(
                    frame, ctx.last_known_bbox,
                    prev_best_angle=ctx.last_rotation_search_angle,
                    angle_step=rot_cfg.angle_step_degrees,
                    padding_ratio=cfg.roi_reacquisition_padding_ratio,
                )
                if rotated is not None:
                    bbox_full, keypoints_full, confidence, winning_angle = rotated
                    recovered = (bbox_full, keypoints_full, confidence)

            if recovered is None:
                # Decay toward 0 -- this path wasn't triggered/didn't help
                # this frame, so don't keep biasing future searches toward
                # a stale angle from an old, resolved occlusion.
                ctx.last_rotation_search_angle *= rot_cfg.angle_decay_factor

                # §4.2 last resort: a brief TRUE detection gap (straight-crop
                # AND rotation search both failed) — extrapolate from this
                # track's own recent motion, bounded to t_gap_max_seconds.
                if rec_cfg.enabled and gap <= max(int(rec_cfg.t_gap_max_seconds * fps), 1):
                    extrapolated = ctx.extrapolated_keypoints(frame_num, rec_cfg)
                    if extrapolated is not None:
                        keypoints_full, synthetic_confidence = extrapolated
                        x_coords, y_coords = keypoints_full[:, 0], keypoints_full[:, 1]
                        bbox_full = (
                            float(np.min(x_coords)), float(np.min(y_coords)),
                            float(np.max(x_coords)), float(np.max(y_coords)),
                        )
                        if not any(_iou(bbox_full, p.bbox) > 0.4 for p in frame_result.persons):
                            frame_result.persons.append(DetectedPerson(
                                track_id=tid, bbox=bbox_full, keypoints=keypoints_full,
                                confidence=synthetic_confidence, pose_valid=True,
                            ))
                continue

            ctx.last_rotation_search_angle = winning_angle
            bbox_full, keypoints_full, confidence = recovered

            # The normal detector may have already reacquired the same real
            # person under a fresh track_id (BoT-SORT assigns a new ID once
            # its own track_buffer times out). Skip if this recovery
            # significantly overlaps an existing detection this frame —
            # otherwise the same person gets double-counted under 2 IDs.
            if any(_iou(bbox_full, p.bbox) > 0.4 for p in frame_result.persons):
                continue

            frame_result.persons.append(DetectedPerson(
                track_id=tid, bbox=bbox_full, keypoints=keypoints_full,
                confidence=confidence, pose_valid=True,
            ))

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        json_path: str | Path | None = None,
    ) -> dict:
        """Process a video file for fall detection.

        Args:
            input_path: Path to input video.
            output_path: Path for annotated output video (None to skip).
            json_path: Path for JSON event log (None to skip).

        Returns:
            Dict with events, summary, and metadata.
        """
        input_path = Path(input_path)
        self._reset_state()

        # Setup signal handler for graceful Ctrl+C
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        start_time = time.time()
        unique_tracks: set[int] = set()

        try:
            with VideoReader(input_path) as reader:
                meta = reader.metadata
                writer = None

                camera_id = self._derive_camera_id(input_path)
                self._ensure_scene_builder(camera_id, meta.width, meta.height)

                if output_path:
                    output_path = Path(output_path)
                    writer = VideoWriter(output_path, meta.width, meta.height, meta.fps)
                    writer.__enter__()

                logger.info(
                    f"Processing {input_path.name}: "
                    f"{meta.width}x{meta.height} @ {meta.fps:.1f}fps, "
                    f"{meta.total_frames} frames ({meta.duration_seconds:.1f}s)"
                )

                for frame_num, frame in tqdm(
                    reader, total=meta.total_frames, desc="Processing"
                ):
                    if self._interrupted:
                        logger.info("Interrupted by user, saving partial output...")
                        break

                    _falls_detected, active_ids = self._process_frame(
                        frame, frame_num, meta.fps, meta.width, meta.height, meta.total_frames
                    )
                    unique_tracks.update(active_ids)

                    if writer:
                        writer.write(frame)

                if writer:
                    writer.__exit__(None, None, None)

        finally:
            signal.signal(signal.SIGINT, original_handler)

        processing_time = time.time() - start_time
        scene_profile_path = self.scene_builder.save() if self.scene_builder else None

        result = {
            "metadata": {
                "input_file": str(input_path),
                "camera_id": camera_id,
                "processing_date": datetime.now().isoformat(),
                "model_version": "1.0",
                "confidence_threshold": self.confidence_threshold,
                "video_fps": meta.fps,
                "video_duration": format_timestamp(meta.total_frames, meta.fps),
                "total_frames": meta.total_frames,
                "resolution": f"{meta.width}x{meta.height}",
            },
            "events": [e.to_dict() for e in self._all_events],
            "summary": {
                "total_falls": len(self._all_events),
                "confirmed_falls": sum(1 for e in self._all_events if e.screenshot_saved),
                "persons_tracked": len(unique_tracks),
                "processing_time_seconds": round(processing_time, 1),
                "interrupted": self._interrupted,
            },
            "scene_profile": (
                self.scene_builder.build_profile().to_dict() if self.scene_builder else None
            ),
        }

        self._save_json_log(result, json_path)
        self._log_summary(processing_time, len(unique_tracks), output_path, json_path)
        if scene_profile_path:
            logger.info(f"Scene profile saved to {scene_profile_path}")

        return result

    def run_stream(
        self,
        source: str | int,
        display: bool = True,
        window_name: str = "Fall Detection - Live",
        json_path: str | Path | None = None,
    ) -> dict:
        """Run the pipeline on a live stream (e.g. an RTSP camera or webcam).

        Shows a real-time preview window (if `display`) and never writes an
        inference video — only a screenshot per confirmed fall event, saved to
        `self.events_dir`. Stop with 'q'/Esc in the preview window, or Ctrl+C.

        Args:
            source: RTSP/HTTP stream URL, or webcam index.
            display: Whether to show a live preview window.
            window_name: Title of the preview window.
            json_path: Optional path to save a JSON event log. Written
                periodically (every ~30s of stream time) as well as once the
                stream stops, so a long-running/killed stream doesn't lose
                everything since the last clean stop.

        Returns:
            Dict with events and summary, same shape as `process_video()`'s result.
        """
        self._reset_state()

        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            signal.signal(signal.SIGINT, original_handler)
            raise OSError(f"Failed to open stream: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280

        camera_id = self._derive_camera_id(source)
        self._ensure_scene_builder(camera_id, width, height)

        start_time = time.time()
        frame_num = 0
        unique_tracks: set[int] = set()

        def build_result(frame_count: int) -> dict:
            return {
                "metadata": {
                    "input_file": str(source),
                    "camera_id": camera_id,
                    "processing_date": datetime.now().isoformat(),
                    "model_version": "1.0",
                    "confidence_threshold": self.confidence_threshold,
                    "video_fps": fps,
                    "total_frames": frame_count,
                },
                "events": [e.to_dict() for e in self._all_events],
                "summary": {
                    "total_falls": len(self._all_events),
                    "confirmed_falls": sum(1 for e in self._all_events if e.screenshot_saved),
                    "persons_tracked": len(unique_tracks),
                    "processing_time_seconds": round(time.time() - start_time, 1),
                    "interrupted": self._interrupted,
                },
                "scene_profile": (
                    self.scene_builder.build_profile().to_dict() if self.scene_builder else None
                ),
            }

        logger.info(
            f"Streaming from {source} — press 'q' in the preview window "
            f"or Ctrl+C here to stop."
        )

        # A live stream may run for hours and can be killed without ever
        # reaching the "stream ended" save below — periodically re-save both
        # the scene profile and the JSON event log (with whatever falls have
        # been detected so far) so nothing is more than ~30s stale on disk,
        # and a mid-stream inspection doesn't require stopping the camera.
        # The per-sample audit log (SceneProfileBuilder.add_sample) is
        # already written in real time regardless of this.
        periodic_save_interval = max(int(fps * 30), 1)

        try:
            while not self._interrupted:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Stream read failed or ended, stopping.")
                    break

                _falls_detected, active_ids = self._process_frame(
                    frame, frame_num, fps, width, height, total_frames=None
                )
                unique_tracks.update(active_ids)

                if display:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        logger.info("Stopped by user.")
                        break

                frame_num += 1
                if frame_num % periodic_save_interval == 0:
                    self.scene_builder.save()
                    if json_path:
                        self._save_json_log(build_result(frame_num), json_path)
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            signal.signal(signal.SIGINT, original_handler)

        scene_profile_path = self.scene_builder.save() if self.scene_builder else None
        result = build_result(frame_num)

        self._save_json_log(result, json_path)
        self._log_summary(time.time() - start_time, len(unique_tracks), None, json_path)
        if scene_profile_path:
            logger.info(f"Scene profile saved to {scene_profile_path}")

        return result

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
        fps: float,
        width: int,
        height: int,
        total_frames: int | None = None,
    ) -> tuple[int, set[int]]:
        """Run detection, tracking, classification, and drawing for one frame.

        Mutates `frame` in place with all overlays, and saves an event
        screenshot the moment the fusion + temporal state machine first
        confirms a tracked person as fully MANDOWN ("fallen") in that event.

        Returns:
            Tuple of (total_falls_detected_so_far, active_track_ids_this_frame).
        """
        clean_frame = frame.copy()

        frame_result = self.detector.detect_frame(frame, frame_num)
        scene_profile = self._current_scene_profile(frame_num)
        self._reacquire_lost_tracks(frame, frame_num, frame_result, fps=fps)

        active_ids: set[int] = set()
        falls_detected = len(self._all_events)
        # Collected across the loop below, then rendered together AFTER
        # every person this frame has been processed — so a screenshot
        # captures everyone currently down, not just whichever track
        # happened to trigger the save this frame.
        pending_screenshot_events: list = []

        for person in frame_result.persons:
            if person.track_id < 0:
                continue

            tid = person.track_id
            active_ids.add(tid)

            ctx = self.mandown_contexts.get_or_create(tid, bbox=person.bbox, frame_num=frame_num)
            is_new_track = len(ctx.ext_history) == 0
            gap = 0 if is_new_track else frame_num - ctx.last_seen_frame

            if not person.pose_valid:
                # Detected (bbox) but pose estimation failed this frame — hold
                # position for possible re-acquisition, contribute no evidence.
                ctx.last_known_bbox = person.bbox
                continue

            pf_cfg = self.mandown_config.pose_failure
            ctx.state_machine.on_gap(gap, pf_cfg.gap_tolerance_frames)
            gap_multiplier = 1.0 if is_new_track else ctx.gap_confidence_multiplier(
                frame_num, pf_cfg.hold_frames, pf_cfg.min_confidence_multiplier
            )

            # Base features — unchanged, feeds the pretrained LSTM as-is.
            prev_cog = self.track_manager.get_prev_cog_height(tid)
            features = self.feature_extractor.extract(
                person.keypoints, person.bbox, height, prev_cog
            )
            self.track_manager.update(tid, features, frame_num)

            # Extended features — independent feature-based Mandown logic.
            # ankle_x/ankle_y computed before extract() (not after, as
            # before) since the perspective fit's horizontal-position term
            # needs the same ankle_x extract() feeds into it.
            prev_ext = ctx.ext_history[-1] if ctx.ext_history else None
            ankle_x, ankle_y = self._ankle_position(person.keypoints, person.bbox)
            extract_kwargs = dict(
                prev=prev_ext,
                prev_cog_x=ctx.prev_cog_x,
                vertical_velocity_now=features.vertical_velocity,
                prev_vertical_velocity=ctx.prev_vertical_velocity,
                expected_height_fn=scene_profile.normalize_height,
                ankle_x=ankle_x,
            )
            ext = self.ext_feature_extractor.extract(
                person.keypoints, person.bbox, width, height, **extract_kwargs
            )
            # 3D lifting only runs when the cheap 2D height signal already
            # looks suspicious, or isn't calibrated yet at all (never
            # unconditionally on every normal standing frame — see
            # Pose3DConfig.height_gate_threshold) — re-extract with the 3D
            # reading folded in only in that case, not a wasted first pass
            # for the common "clearly normal" frame.
            if self.pose3d_estimator is not None and (
                not scene_profile.perspective.is_valid
                or ext.body_height_norm < self.mandown_config.pose_3d.height_gate_threshold
            ):
                # V6.1 (Objective 1): which leg is confidently the
                # flat/extended one, from ordinary 2D keypoint confidence
                # alone -- computed BEFORE calling MediaPipe, so it never
                # depends on (and can't be biased by) the same 3D lifting
                # it anchors a second, hip-independent reading for.
                leg_side = dominant_leg_side(person.keypoints)
                pose_3d = self.pose3d_estimator.estimate(
                    clean_frame, person.bbox, dominant_leg_side=leg_side
                )
                if pose_3d is not None:
                    ext = self.ext_feature_extractor.extract(
                        person.keypoints, person.bbox, width, height,
                        pose_3d=pose_3d, **extract_kwargs
                    )
            cog_x = self.ext_feature_extractor.cog_x(person.keypoints, person.bbox)
            ctx.update(
                ext, cog_x, features.vertical_velocity, frame_num,
                bbox=person.bbox, keypoints=person.keypoints,
            )

            # Feed the rolling scene/size calibration from reliable,
            # standing-like samples only (never from a mid-fall detection).
            if is_reliable_standing_sample(
                person.bbox, features.keypoint_confidence, ext.num_critical_keypoints_visible,
                width, height, self.mandown_config.calibration,
            ):
                self.scene_builder.add_sample(
                    person.bbox, ankle_x, ankle_y, track_id=tid, frame_num=frame_num
                )

            floor_contact_score = scene_profile.floor_region.contact_likelihood(ankle_y)

            # V4.4: has THIS track's own hip moved down toward the floor
            # since it was last confidently upright? Computed from the
            # PRIOR baseline (before this frame updates it) — a deep-but-
            # standing bend pivots the torso around a hip that barely
            # moves, whereas a genuine fall (any orientation) brings the
            # hip down substantially. Self-relative and needs no scene
            # calibration, unlike a V4.3 attempt using the scene-wide
            # floor_region evaluated at the hip, which proved too
            # imprecise for real camera perspective and blocked genuine
            # falls (see MandownTrackContext.hip_drop_evidence).
            hdc = self.mandown_config.hip_drop
            hip_drop_evidence = ctx.hip_drop_evidence(features.cog_height, hdc)
            # torso_angle alone is NOT enough: a person fallen toward/away
            # from the camera also reads torso_angle near 0 (that's exactly
            # why angle-based cues are blind to that orientation) --
            # without also requiring body_height_norm to look genuinely
            # standing-height, a foreshortened fall would get wrongly
            # recorded as its own "standing" baseline, defeating this
            # signal for the orientation it exists to help with.
            is_confidently_upright = (
                ext.torso_angle_evidence_quality > 0.0
                and ext.torso_angle < hdc.upright_torso_angle_threshold
                and ext.body_height_norm >= hdc.upright_body_height_norm_threshold
            )
            ctx.update_standing_baseline(features.cog_height, is_confidently_upright)

            # LSTM — one evidence source, not sole authority.
            seq = self.track_manager.get_sequence(tid)
            lstm_state, lstm_conf, lstm_mandown_prob = "normal", 0.0, 0.0
            if seq is not None:
                lstm_state, lstm_conf, lstm_mandown_prob = self._classify(seq)

            # Direct feature-based scoring (independent of the LSTM).
            feature_scores = self.feature_scorer.score(
                features, ext, self.track_manager.tracks[tid].features, ctx.ext_history,
                floor_contact_score=floor_contact_score,
                hip_drop_evidence=hip_drop_evidence,
            )

            # V6 Fix Set B: purely temporal confirmation path for a
            # foreshortened/asymmetric lying posture that no single-frame
            # signal above fully resolves -- see MandownTrackContext.
            # collapse_then_stillness / MotionSignatureConfig.
            collapse_then_stillness = ctx.collapse_then_stillness(
                frame_num, fps, feature_scores.height_collapse_cue,
                is_confidently_upright, self.mandown_config.motion_signature,
            )

            # Evidence fusion. Pose-quality is discounted (never fabricated
            # up) when this frame's evidence followed a detection gap.
            fusion_result = self.fusion.combine(
                lstm_mandown_prob=lstm_mandown_prob,
                feature_confidence=feature_scores.mandown_confidence,
                persistence_score=feature_scores.persistence_score,
                pose_quality=feature_scores.pose_quality_score * gap_multiplier,
                down_score=feature_scores.down_score,
                floor_proximity_score=feature_scores.floor_proximity_score,
                height_collapse_cue=feature_scores.height_collapse_cue,
                torso_angle=ext.torso_angle,
                torso_angle_evidence_quality=ext.torso_angle_evidence_quality,
                torso_angle_3d=ext.torso_angle_3d,
                torso_angle_3d_evidence_quality=ext.torso_angle_3d_evidence_quality,
                hip_angle=ext.hip_angle,
                hip_angle_evidence_quality=ext.hip_angle_evidence_quality,
                collapse_then_stillness=collapse_then_stillness,
                body_axis_angle_3d=ext.body_axis_angle_3d,
                body_axis_angle_3d_evidence_quality=ext.body_axis_angle_3d_evidence_quality,
            )

            # Temporal state machine (hysteresis, per-track).
            step_result = ctx.state_machine.step(fusion_result.suggested_tier)
            display_state = _STATE_TO_LABEL[step_result.state]
            conf = fusion_result.fusion_score
            state = display_state  # kept for the drawing helpers below
            self.track_manager.set_classification(tid, display_state, conf)

            # Handle fall/mandown events. The screenshot for a confirmation
            # this frame is deferred (see pending_screenshot_events below)
            # until every person in the frame has been processed.
            if display_state != "normal":
                diagnostics = {
                    "camera_id": scene_profile.camera_id,
                    "lstm_state": lstm_state,
                    "lstm_confidence": round(lstm_conf, 3),
                    **fusion_result.to_dict(),
                    **feature_scores.to_dict(),
                }
                due_event = self._handle_fall_event(
                    tid, frame_num, display_state, conf, fps, bbox=person.bbox,
                    entered_mandown=step_result.entered_mandown, diagnostics=diagnostics,
                )
                if due_event is not None:
                    pending_screenshot_events.append(due_event)
                falls_detected = len(self._all_events)
            elif step_result.recovered:
                # Sustained genuine recovery to NORMAL closes this episode —
                # a later fall by the same track_id (or one merged into it
                # via dedup) starts a brand-new event/screenshot instead of
                # silently extending the old, already-alerted one.
                self._close_active_event(tid)

            # Draw on frame — scale sizes to resolution
            scale = max(height / 720, 1.0)

            color = STATE_COLORS.get(display_state, COLOR_GREEN)

            # ID/state label overlay is never drawn on screen for any state —
            # box color/presence alone conveys it; id/state remain in the
            # JSON log regardless, this only affects what's rendered.
            show_label = False
            label = f"ID:{tid} {STATE_LABELS[display_state]}"

            draw_skeleton(
                frame, person.keypoints, color=color,
                thickness=max(2, int(2 * scale)),
                point_radius=max(4, int(4 * scale)),
            )
            self._draw_label_scaled(
                frame, person.bbox, label, color, state, scale, show_label=show_label
            )

            # Save position of fallen persons for occlusion persistence
            if display_state in ("falling", "fallen"):
                self._fallen_positions[tid] = (person.bbox, frame_num)

        # Every person this frame now has an up-to-date classification —
        # safe to gather everyone CURRENTLY "fallen" for a multi-person
        # screenshot, including tracks confirmed on an EARLIER frame that
        # are still down right now, not just the one(s) that triggered a
        # NEW confirmation this exact frame.
        if pending_screenshot_events:
            fallen_persons = [
                (person.track_id, person.bbox, person.keypoints)
                for person in frame_result.persons
                if person.track_id >= 0
                and self.track_manager.get_classification(person.track_id)[0] == "fallen"
            ]
            self._save_pending_screenshots(pending_screenshot_events, clean_frame, fallen_persons)

        # Draw persistent fallen markers for occluded tracks — a brief
        # grace period so momentary detection loss doesn't flicker the
        # alert away, not an open-ended "keep showing it" window (see
        # PoseFailureConfig.occluded_marker_seconds).
        scale = max(height / 720, 1.0)
        occluded_marker_frames = fps * self.mandown_config.pose_failure.occluded_marker_seconds
        for tid, (bbox, last_frame) in list(self._fallen_positions.items()):
            if tid in active_ids:
                continue  # Still tracked, already drawn above
            frames_since = frame_num - last_frame
            if frames_since > occluded_marker_frames:
                del self._fallen_positions[tid]
                continue
            self._draw_label_scaled(
                frame, bbox, "", COLOR_RED, "fallen", scale, show_label=False,
            )

        # Draw alert banner based on current per-person state
        alert_scale = max(height / 720, 1.0)
        has_falling = False
        has_fallen = False

        for person in frame_result.persons:
            tid = person.track_id
            t_state, _ = self.track_manager.get_classification(tid)
            if t_state == "falling":
                has_falling = True
            elif t_state == "fallen":
                has_fallen = True

        # Also check occluded fallen persons
        for tid in self._fallen_positions:
            if tid not in active_ids and tid in self._active_events:
                has_fallen = True

        timestamp_str = format_short_timestamp(frame_num, fps)
        if has_falling:
            draw_fall_banner(
                frame, "Fall Alert: Falling",
                color=COLOR_ORANGE, timestamp=timestamp_str,
                font_scale=1.0 * alert_scale,
            )
        elif has_fallen:
            draw_mandown_banner(frame, "MANDOWN", color=COLOR_RED, font_scale=1.0 * alert_scale)

        # Status bar
        draw_status_bar(
            frame, frame_num, total_frames, fps, len(active_ids), falls_detected,
        )

        # Periodic cleanup
        if frame_num % 100 == 0:
            self.track_manager.cleanup(frame_num)
            self.mandown_contexts.cleanup(frame_num)

        return falls_detected, active_ids

    @staticmethod
    def _draw_label_scaled(frame, bbox, label, color, state, scale, show_label=True):
        """Draw the bounding box, and — when `show_label` — a large,
        easy-to-read ID/state label above it. The ID/label is diagnostic
        overlay only; it's never required to also appear in the box itself
        for the box's color/presence to convey the state (id/state remain
        available in the JSON log regardless of `show_label`)."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bbox_thick = max(2, int(3 * scale))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, bbox_thick)

        if not show_label:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_val = 0.7 * scale
        font_thick = max(1, int(2 * scale))
        padding = int(6 * scale)

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale_val, font_thick)

        # Position label above bbox, or below if no room
        if y1 - th - baseline - padding * 2 >= 0:
            lbl_y2 = y1
            lbl_y1 = y1 - th - baseline - padding * 2
        else:
            lbl_y1 = y1
            lbl_y2 = y1 + th + baseline + padding * 2

        lbl_x1 = x1
        lbl_x2 = x1 + tw + padding * 2

        # Background
        cv2.rectangle(frame, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)

        # Text — white for fall/fallen states, black for normal
        text_color = (255, 255, 255) if state != "normal" else (0, 0, 0)
        if y1 - th - padding * 2 >= 0:
            text_y = lbl_y2 - baseline - padding
        else:
            text_y = lbl_y1 + th + padding
        cv2.putText(
            frame, label, (lbl_x1 + padding, text_y),
            font, font_scale_val, text_color, font_thick, cv2.LINE_AA,
        )

    @staticmethod
    def _ankle_position(
        keypoints: np.ndarray, bbox: tuple[float, float, float, float], conf_threshold: float = 0.3
    ) -> tuple[float, float]:
        """Mean (x, y) of visible ankles, for scene-profile floor-direction
        and floor-region estimation. Falls back to bbox center-x / bottom-y
        when neither ankle is confidently visible."""
        la, ra = keypoints[LEFT_ANKLE], keypoints[RIGHT_ANKLE]
        pts = [
            (float(kp[0]), float(kp[1]))
            for kp in (la, ra) if kp[2] >= conf_threshold and np.isfinite(kp[0])
        ]
        if pts:
            xs, ys = zip(*pts)
            return sum(xs) / len(xs), sum(ys) / len(ys)
        return (bbox[0] + bbox[2]) / 2, bbox[3]

    def _classify(self, sequence: np.ndarray) -> tuple[str, float, float]:
        """Classify a feature sequence using the LSTM.

        Returns (predicted_class_name, predicted_class_confidence, mandown_prob),
        where mandown_prob = P(falling) + P(fallen) — the full-distribution
        signal the fusion layer needs, not just the top-1 confidence.
        """
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.lstm(x), dim=1)[0]
        class_id = int(torch.argmax(probs).item())
        confidence = float(probs[class_id].item())
        class_name = FallDetector.CLASS_NAMES.get(class_id, "normal")
        mandown_prob = float(probs[1].item() + probs[2].item())
        return class_name, confidence, mandown_prob

    def _find_duplicate_event(
        self, bbox: Box, frame_num: int, exclude_track_id: int
    ) -> "FallEvent | None":
        """Find an existing active event that likely represents the SAME
        physical person as `bbox`, fragmented under a different track_id
        (real case: man_down_c08.mp4 — near-identical head/shoulder
        keypoints, diverging leg pose estimates, two simultaneously-active
        BoT-SORT track IDs). Requires both a substantial bbox overlap and a
        recent update — a stale event from elsewhere in a long video is not
        eligible, so an unrelated later person isn't silently merged into
        an old alert."""
        cfg = self.mandown_config.event_dedup
        best: FallEvent | None = None
        best_iou = 0.0
        for other_id, event in self._active_events.items():
            if other_id == exclude_track_id or event.last_bbox is None:
                continue
            if frame_num - event.end_frame > cfg.recency_frames:
                continue
            score = _iou(bbox, event.last_bbox)
            if score >= cfg.iou_threshold and score > best_iou:
                best, best_iou = event, score
        return best

    def _close_active_event(self, track_id: int) -> None:
        """Close out this track's active event on genuine recovery, without
        touching its historical record in `_all_events` or its diagnostics.
        Also clears the cooldown so the *next* fall (a new episode, not a
        continuation) isn't throttled by the old episode's alert timer."""
        track_id = self._duplicate_of.get(track_id, track_id)
        self._active_events.pop(track_id, None)
        self._cooldowns.pop(track_id, None)

    def _handle_fall_event(
        self,
        track_id: int,
        frame_num: int,
        state: str,
        conf: float,
        fps: float,
        bbox: tuple | None = None,
        entered_mandown: bool = False,
        diagnostics: dict | None = None,
    ) -> "FallEvent | None":
        """Handle a detected fall/mandown, with cooldown logic. Returns the
        `FallEvent` if a screenshot is now due (confirmed this frame, not
        already saved) — the caller defers the actual render/save until
        every person in this frame has been processed, so a screenshot
        captures everyone currently down, not just whichever track
        happened to trigger it (see `_process_frame`).

        Deduplicates across track IDs: if `track_id` has no event yet and
        its bbox overlaps an already-active event closely enough (same
        physical person, fragmented tracking), it's merged into that event
        instead of starting a second one — see `_find_duplicate_event`."""
        if track_id not in self._duplicate_of and track_id not in self._active_events and bbox is not None:
            dup_target = self._find_duplicate_event(bbox, frame_num, exclude_track_id=track_id)
            if dup_target is not None:
                self._duplicate_of[track_id] = dup_target.track_id
                if track_id not in dup_target.duplicate_track_ids:
                    dup_target.duplicate_track_ids.append(track_id)
                logger.info(
                    f"Track {track_id} deduplicated into track {dup_target.track_id}'s "
                    f"event at frame {frame_num} (bbox overlap — same physical person)"
                )

        track_id = self._duplicate_of.get(track_id, track_id)

        # Check cooldown
        last_alert = self._cooldowns.get(track_id, -self.cooldown_frames - 1)
        if frame_num - last_alert < self.cooldown_frames:
            if track_id in self._active_events:
                event = self._active_events[track_id]
                event.update(frame_num, conf, state, diagnostics, bbox=bbox)
                if entered_mandown and not event.screenshot_saved:
                    return event
            return None

        if track_id not in self._active_events:
            # New fall/mandown event
            event = FallEvent(track_id, frame_num, fps, conf, bbox=bbox)
            event.state = state
            event.diagnostics = diagnostics or {}
            self._active_events[track_id] = event
            self._all_events.append(event)
            self._cooldowns[track_id] = frame_num
            logger.info(
                f"MANDOWN EVIDENCE: Track {track_id} at frame {frame_num} "
                f"({format_timestamp(frame_num, fps)}) state={state} conf={conf:.2f}"
            )
        else:
            event = self._active_events[track_id]
            event.update(frame_num, conf, state, diagnostics, bbox=bbox)

        if entered_mandown and not event.screenshot_saved:
            return event
        return None

    def _save_pending_screenshots(
        self,
        pending_events: list,
        clean_frame: "np.ndarray | None",
        fallen_persons: list,
    ) -> None:
        """Render and save ONE screenshot covering every person currently
        classified "fallen" this frame — not just whichever track(s)
        triggered it this frame — then mark each pending event saved,
        pointing at this same file. `fallen_persons` is
        [(track_id, bbox, keypoints), ...] for everyone currently down."""
        if not pending_events or clean_frame is None:
            return
        path = self._save_event_screenshot(clean_frame, fallen_persons)
        for event in pending_events:
            event.screenshot_saved = True
            event.screenshot_path = str(path)

    def _save_event_screenshot(self, clean_frame: np.ndarray, fallen_persons: list) -> Path:
        """Render and save a screenshot showing every currently-fallen
        person's plain red box (no ID/label overlay, see
        `_draw_label_scaled(show_label=False)`) and the MANDOWN banner.
        `fallen_persons` is [(track_id, bbox, keypoints), ...]."""
        frame = clean_frame.copy()
        scale = max(frame.shape[0] / 720, 1.0)

        for _track_id, bbox, keypoints in fallen_persons:
            if keypoints is not None:
                draw_skeleton(
                    frame, keypoints, color=COLOR_RED,
                    thickness=max(2, int(2 * scale)), point_radius=max(4, int(4 * scale)),
                )
            self._draw_label_scaled(frame, bbox, "", COLOR_RED, "fallen", scale, show_label=False)
        draw_mandown_banner(frame, "MANDOWN", color=COLOR_RED, font_scale=1.0 * scale)

        self.events_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        ids_str = "_".join(str(tid) for tid, _, _ in fallen_persons) or "unknown"
        path = self.events_dir / f"fall_track{ids_str}_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        logger.info(f"Event screenshot saved: {path} ({len(fallen_persons)} person(s))")
        return path

    def _save_json_log(self, result: dict, json_path: str | Path | None):
        if not json_path:
            return
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Event log saved to {json_path}")

    def _log_summary(
        self,
        processing_time: float,
        persons_tracked: int,
        output_path: str | Path | None,
        json_path: str | Path | None,
    ):
        confirmed = [e for e in self._all_events if e.screenshot_saved]
        tentative_count = len(self._all_events) - len(confirmed)

        logger.info(f"\n{'='*60}")
        logger.info("RESULTS")
        logger.info(f"{'='*60}")
        # "Confirmed" = the state machine actually reached MANDOWN for that
        # track (screenshot captured). A track that only ever touched the
        # tentative "falling"/POSSIBLE_MANDOWN tier and never escalated is
        # NOT a confirmed fall -- reported separately so it can't be misread
        # as one (this is exactly what caused confusion on a real run: 2
        # "falling"-only tracks, 0 ever confirmed, 0 screenshots).
        logger.info(f"Falls confirmed (MANDOWN, screenshot saved): {len(confirmed)}")
        if tentative_count:
            logger.info(
                f"Tracks flagged possible-only, never confirmed: {tentative_count} "
                f"(no screenshot for these -- see 'confirmed': false in the event log)"
            )
        logger.info(f"Persons tracked: {persons_tracked}")
        logger.info(f"Processing time: {processing_time:.1f}s")
        if output_path:
            logger.info(f"Output video: {output_path}")
        if json_path:
            logger.info(f"Event log: {json_path}")
        if confirmed:
            logger.info(f"Event screenshots saved: {len(confirmed)} in {self.events_dir}")

    def _handle_interrupt(self, signum, frame):
        self._interrupted = True
