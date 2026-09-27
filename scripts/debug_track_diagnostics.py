#!/usr/bin/env python3
"""Per-frame diagnostic dump for one tracked person, for MANDOWN investigation.

Monkey-patches (does not modify) the real pipeline components to capture
every signal that feeds a MANDOWN decision for a specific track_id over a
frame range: raw keypoints, base+extended feature values, feature-scorer
outputs, fusion inputs/outputs, and final state.

Not part of the production pipeline or the permanent test suite -- a
throwaway investigation tool, reusable across videos/tracks.

Usage:
    python scripts/debug_track_diagnostics.py <video> <track_id> \
        [--frame-start N] [--frame-end N] [--out path.json]
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import FallDetectionPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("track_id", type=int, help="-1 to capture ALL tracks in the frame range")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=10**9)
    parser.add_argument("--out", default=None)
    parser.add_argument("--other-track-ids", type=int, nargs="*", default=[],
                         help="Additional track_ids to also capture (for fragmentation investigation)")
    args = parser.parse_args()

    targets = {args.track_id, *args.other_track_ids}
    capture_all = args.track_id == -1

    pipeline = FallDetectionPipeline()

    state = {"frame_num": -1, "tid": None}
    records: list[dict] = []

    orig_detect_frame = pipeline.detector.detect_frame

    def wrapped_detect_frame(frame, frame_num):
        state["frame_num"] = frame_num
        return orig_detect_frame(frame, frame_num)

    pipeline.detector.detect_frame = wrapped_detect_frame

    orig_get_or_create = pipeline.mandown_contexts.get_or_create

    def wrapped_get_or_create(tid):
        state["tid"] = tid
        return orig_get_or_create(tid)

    pipeline.mandown_contexts.get_or_create = wrapped_get_or_create

    def in_scope() -> bool:
        if not (args.frame_start <= state["frame_num"] <= args.frame_end):
            return False
        return capture_all or state["tid"] in targets

    orig_ext_extract = pipeline.ext_feature_extractor.extract

    def wrapped_ext_extract(keypoints, bbox, frame_width, frame_height, **kwargs):
        ext = orig_ext_extract(keypoints, bbox, frame_width, frame_height, **kwargs)
        if in_scope():
            records.append({
                "frame_num": state["frame_num"],
                "track_id": state["tid"],
                "keypoints": keypoints.tolist(),
                "bbox": list(bbox),
                "frame_width": frame_width,
                "frame_height": frame_height,
                "ext": asdict(ext),
            })
        return ext

    pipeline.ext_feature_extractor.extract = wrapped_ext_extract

    orig_score = pipeline.feature_scorer.score

    def wrapped_score(base, ext, base_history, ext_history, floor_contact_score=0.0):
        result = orig_score(base, ext, base_history, ext_history,
                             floor_contact_score=floor_contact_score)
        if records and in_scope() and records[-1]["frame_num"] == state["frame_num"]:
            records[-1]["base"] = asdict(base)
            records[-1]["floor_contact_score_input"] = floor_contact_score
            records[-1]["feature_scores"] = result.to_dict()
        return result

    pipeline.feature_scorer.score = wrapped_score

    orig_combine = pipeline.fusion.combine

    def wrapped_combine(**kwargs):
        result = orig_combine(**kwargs)
        if records and in_scope() and records[-1]["frame_num"] == state["frame_num"]:
            records[-1]["fusion_inputs"] = kwargs
            records[-1]["fusion_result"] = result.to_dict()
        return result

    pipeline.fusion.combine = wrapped_combine

    out_path = Path(args.out) if args.out else Path(f"/tmp/track_{args.track_id}_diagnostics.json")
    print(f"Processing {args.video} for track {sorted(targets)}, frames "
          f"[{args.frame_start}, {args.frame_end}]...", file=sys.stderr)
    pipeline.process_video(args.video, output_path=None, json_path=None)

    out_path.write_text(json.dumps(records, indent=2, default=str))
    print(f"Wrote {len(records)} per-frame records to {out_path}")


if __name__ == "__main__":
    main()
