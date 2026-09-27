#!/usr/bin/env python3
"""Benchmark candidate person detectors for the ROI-pose pipeline.

Compares the current full-frame YOLO11-Pose baseline against dedicated
bbox-only person detectors (yolo11n / yolov8n / yolo11s) on person recall
and speed, over a chosen frame range of a video. Prioritizes recall over
precision — a missed person means pose inference cannot happen at all.

Reports both "any-person recall" (at least one person detected anywhere in
frame) and, if `--target-region` is given, recall specifically for that
region (IoU > 0.1 with a detected box) — the metric that actually matters
when checking whether a detector can find a *specific* hard-to-see person
(e.g. someone lying down) rather than any bystander in the scene.

Usage:
    python scripts/benchmark_person_detector.py --video input/man_down_zoomed_demo__2_.mp4 \
        --start 490 --end 560 --target-region 590,405,960,590
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector import LightweightPersonDetector, PersonDetector  # noqa: E402


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _run(name: str, boxes_per_frame, elapsed: float, total_frames: int, target_region) -> dict:
    frames_any = sum(1 for boxes in boxes_per_frame if len(boxes) > 0)
    frames_target = None
    if target_region is not None:
        frames_target = sum(
            1 for boxes in boxes_per_frame if any(_iou(b, target_region) > 0.1 for b in boxes)
        )
    target_recall = None
    if frames_target is not None:
        target_recall = frames_target / max(total_frames, 1)
    return {
        "name": name,
        "total_frames": total_frames,
        "any_recall": frames_any / max(total_frames, 1),
        "target_recall": target_recall,
        "fps": total_frames / max(elapsed, 1e-6),
    }


def benchmark_full_frame_pose(video_path, start, end, target_region) -> dict:
    detector = PersonDetector(model_name="yolo11n-pose.pt", confidence_threshold=0.05)
    cap = cv2.VideoCapture(video_path)
    boxes_per_frame, total = [], 0
    t0 = time.time()
    for frame_num in range(end):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num < start:
            continue
        result = detector.detect_frame(frame, frame_num)
        boxes_per_frame.append([p.bbox for p in result.persons])
        total += 1
    elapsed = time.time() - t0
    cap.release()
    name = "yolo11n-pose.pt (full-frame, current baseline)"
    return _run(name, boxes_per_frame, elapsed, total, target_region)


def benchmark_bbox_detector(model_name: str, video_path, start, end, target_region) -> dict:
    detector = LightweightPersonDetector(model_name=model_name, confidence_threshold=0.05)
    cap = cv2.VideoCapture(video_path)
    boxes_per_frame, total = [], 0
    t0 = time.time()
    for frame_num in range(end):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num < start:
            continue
        tracked = detector.track_frame(frame)
        boxes_per_frame.append([t.bbox for t in tracked])
        total += 1
    elapsed = time.time() - t0
    cap.release()
    return _run(model_name, boxes_per_frame, elapsed, total, target_region)


def main():
    parser = argparse.ArgumentParser(description="Benchmark person-detector candidates")
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument(
        "--target-region", default=None,
        help="x1,y1,x2,y2 of a specific person to check recall for (e.g. a hard-to-detect person)",
    )
    parser.add_argument(
        "--candidates", nargs="+", default=["yolo11n.pt", "yolov8n.pt"],
        help="bbox-only detector model names to benchmark against the full-frame pose baseline",
    )
    args = parser.parse_args()
    target_region = None
    if args.target_region:
        target_region = tuple(float(v) for v in args.target_region.split(","))

    results = [benchmark_full_frame_pose(args.video, args.start, args.end, target_region)]
    for name in args.candidates:
        results.append(
            benchmark_bbox_detector(name, args.video, args.start, args.end, target_region)
        )

    header = f"{'Model':45s} {'AnyRecall':>10s} {'TargetRecall':>13s} {'FPS':>8s}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        target_str = f"{r['target_recall']:.1%}" if r["target_recall"] is not None else "n/a"
        print(f"{r['name']:45s} {r['any_recall']:>9.1%} {target_str:>13s} {r['fps']:>7.1f}")


if __name__ == "__main__":
    main()
