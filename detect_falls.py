#!/usr/bin/env python3
"""Fall Detection CLI — process a video file or a live RTSP/webcam stream.

Usage:
    # Video file: drop it in input/, output goes to output/ automatically
    python detect_falls.py --input video.mp4

    # Explicit paths still work
    python detect_falls.py --input video.mp4 --output result.mp4 --json-log events.json

    # Live RTSP camera: real-time preview window, no inference video is written,
    # only a screenshot per confirmed fall event (saved to events/)
    python detect_falls.py --input rtsp://user:pass@camera-ip/stream

    # Local webcam (index 0)
    python detect_falls.py --input 0
"""

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import FallDetectionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")

STREAM_PREFIXES = ("rtsp://", "rtsps://", "http://", "https://")


def _is_stream_source(source: str) -> bool:
    """True if `source` is an RTSP/HTTP stream URL or a webcam index."""
    return source.startswith(STREAM_PREFIXES) or source.isdigit()


def _resolve_input_path(raw_input: str) -> Path:
    """Resolve a video file argument, checking input/ if it's a bare filename."""
    path = Path(raw_input)
    if path.exists():
        return path
    candidate = INPUT_DIR / raw_input
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Video file not found: '{raw_input}' (also checked '{candidate}')"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Detect falls using skeleton tracking + LSTM, on a video file or live stream"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input video file (checked in input/ if not found as given), "
        "an RTSP/HTTP stream URL, or a webcam index (e.g. 0)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output annotated video path (video files only; defaults to output/<name>.mp4)",
    )
    parser.add_argument(
        "--json-log", "-j", default=None,
        help="JSON event log output path (defaults to output/<name>_events.json for video files)",
    )
    parser.add_argument(
        "--events-dir", default="events",
        help="Folder where fall event screenshots are saved",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="For live streams: don't open a preview window (still saves event screenshots)",
    )
    parser.add_argument(
        "--model", default="models/checkpoints/best.pth", help="LSTM model checkpoint"
    )
    parser.add_argument("--yolo-model", default="yolo11s-pose.pt", help="YOLO pose model")
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="YOLO inference resolution (long side, in pixels)"
    )
    parser.add_argument("--confidence", type=float, default=0.40, help="Fall confidence threshold")
    parser.add_argument("--cooldown", type=int, default=90, help="Cooldown frames between alerts")
    parser.add_argument("--device", default=None, help="Device: cuda, cpu, or auto")

    args = parser.parse_args()

    pipeline = FallDetectionPipeline(
        yolo_model=args.yolo_model,
        lstm_checkpoint=args.model,
        device=args.device,
        confidence_threshold=args.confidence,
        cooldown_frames=args.cooldown,
        events_dir=args.events_dir,
        imgsz=args.imgsz,
    )

    if _is_stream_source(args.input):
        source = int(args.input) if args.input.isdigit() else args.input
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Unlike the video-file branch below, a stream has no natural
        # filename to default from -- without this, --json-log left unset
        # means NOTHING gets saved when the stream stops (whether cleanly
        # or via Ctrl+C), so a long-running camera session's event history
        # is unrecoverable after the fact.
        camera_id = FallDetectionPipeline._derive_camera_id(source)
        json_path = args.json_log or (OUTPUT_DIR / f"{camera_id}_events.json")
        result = pipeline.run_stream(
            source=source,
            display=not args.no_display,
            json_path=json_path,
        )
    else:
        input_path = _resolve_input_path(args.input)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = args.output or (OUTPUT_DIR / f"{input_path.stem}.mp4")
        json_path = args.json_log or (OUTPUT_DIR / f"{input_path.stem}_events.json")

        result = pipeline.process_video(
            input_path=input_path,
            output_path=output_path,
            json_path=json_path,
        )

    return 0 if not result["summary"]["interrupted"] else 1


if __name__ == "__main__":
    sys.exit(main())
