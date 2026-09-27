#!/usr/bin/env python3
"""Re-run all 4 real validation videos through the current pipeline (no
annotated video output, for speed) and save JSON event logs for V3-vs-V4
comparison. Throwaway validation tool, not part of the permanent suite."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import FallDetectionPipeline  # noqa: E402

VIDEOS = [
    "mandown_2.mp4",
    "man_down_c08.mp4",
    "man_down_zoomed_demo__2_.mp4",
    "M_D_2.mp4",
]

for video in VIDEOS:
    print(f"=== {video} ===", file=sys.stderr)
    pipeline = FallDetectionPipeline()
    result = pipeline.process_video(
        input_path=f"input/{video}", output_path=None, json_path=None,
    )
    out_path = Path("output") / f"{Path(video).stem}_v4.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  events: {len(result['events'])}, time: {result['summary']['processing_time_seconds']}s")
    for e in result["events"]:
        ev = e["evidence"]
        print(f"    track={e['track_id']} frames={e['start_frame']}-{e['end_frame']} "
              f"conf={e['confidence']} tier={ev.get('suggested_tier')} screenshot="
              f"{'yes' if e.get('screenshot_path') else 'no'}")
