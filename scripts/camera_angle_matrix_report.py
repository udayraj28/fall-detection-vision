#!/usr/bin/env python3
"""V4 Step 5/9: synthetic camera-angle validation matrix.

Honest scope: from 2D COCO keypoints alone (no 3D body model), the only
camera-condition dimensions that can be genuinely simulated -- not
fabricated -- are (a) in-image-plane camera ROLL/TILT (rotate the whole
skeleton around a center point) and (b) DISTANCE (scale the skeleton/bbox
toward the frame center). Camera ELEVATION (low/medium/high) and
ORIENTATION (front/rear/side/diagonal relative to the subject) change how
a 3D pose projects into 2D in ways a flat rotation/scale of an already-2D
skeleton cannot faithfully reproduce -- simulating those would fabricate
data, not validate anything, so they are NOT covered here. Real footage
(4 videos) already covers 3 distinct real camera setups per
MANDOWN_V3_IMPLEMENTATION_REPORT.md; this script covers roll x distance x
person-orientation, which real footage does not systematically vary.

Reports whether each combination reaches MANDOWN, POSSIBLE_MANDOWN, or
stays NORMAL. SYNTHETIC ONLY. Throwaway reporting tool.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from tests.test_already_lying_integration import (  # noqa: E402
    FRAME_H,
    FRAME_W,
    _StaticTrackSimulator,
    _bbox_for,
    diagonal_lying_keypoints,
    prone_lying_keypoints,
    side_lying_keypoints,
    standing_keypoints,
    supine_lying_keypoints,
)

PERSON_ORIENTATIONS = [
    ("supine", supine_lying_keypoints),
    ("prone", prone_lying_keypoints),
    ("side (bent knees)", side_lying_keypoints),
    ("diagonal", diagonal_lying_keypoints),
    ("standing (negative control)", standing_keypoints),
]

ROLLS_DEG = [0, 15, 30, 45]
DISTANCES = [("near", 1.4), ("medium", 1.0), ("far", 0.6)]


def transform(kps: np.ndarray, roll_deg: float, scale: float) -> np.ndarray:
    import math

    cx, cy = FRAME_W / 2, FRAME_H / 2
    rad = math.radians(roll_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    out = kps.copy()
    for i in range(kps.shape[0]):
        x, y, conf = kps[i]
        dx, dy = (x - cx) * scale, (y - cy) * scale
        out[i] = [cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a, conf]
    return out


def run(kps: np.ndarray) -> tuple[str, float]:
    sim = _StaticTrackSimulator()
    bbox = _bbox_for(kps)
    state = None
    for frame_num in range(40):
        state = sim.step(kps, bbox)
        if state.name == "MANDOWN":
            break
    ext = sim.ext_history[-1]
    scores = sim.scorer.score(sim.base_history[-1], ext, sim.base_history, sim.ext_history)
    return state.name, scores.mandown_confidence


header = f"{'orientation':<30} {'roll':>5} {'distance':>9}  {'result':<10} conf"
print(header)
print("-" * len(header))
algorithm_failures = []
for name, fn in PERSON_ORIENTATIONS:
    base_kps = fn()
    for roll in ROLLS_DEG:
        for dist_name, scale in DISTANCES:
            kps = transform(base_kps, roll, scale)
            result, conf = run(kps)
            print(f"{name:<30} {roll:>4}° {dist_name:>9}  {result:<10} {conf:.3f}")
            is_lying = name != "standing (negative control)"
            if is_lying and result == "NORMAL":
                algorithm_failures.append((name, roll, dist_name, conf))
            if not is_lying and result != "NORMAL":
                algorithm_failures.append((name, roll, dist_name, conf))

print()
print(f"Non-NORMAL-as-expected cases: {len(algorithm_failures)}")
for f in algorithm_failures:
    print(f"  {f}")
