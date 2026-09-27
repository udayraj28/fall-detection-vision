#!/usr/bin/env python3
"""V4 Step 4/8: synthetic posture validation matrix.

Runs every posture fixture through the real base+extended feature ->
scorer -> fusion -> state-machine chain (via _StaticTrackSimulator, LSTM
cold throughout) and reports down_score, floor_proximity_score,
kneeling_penalty, fusion_score, and final state for each.

SYNTHETIC ONLY -- see MANDOWN_IMPLEMENTATION_REPORT_V4.md for which of
these postures also have real-footage validation. Throwaway
reporting tool, not part of the permanent test suite.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_already_lying_integration import (  # noqa: E402
    _StaticTrackSimulator,
    _bbox_for,
    crouching_keypoints,
    deep_bend_keypoints,
    diagonal_lying_keypoints,
    kneeling_keypoints,
    lunge_keypoints,
    mild_camera_roll_standing_keypoints,
    prone_lying_keypoints,
    semi_kneeling_keypoints,
    side_lying_keypoints,
    sitting_keypoints,
    standing_keypoints,
    supine_lying_keypoints,
)

POSTURES = [
    ("standing", standing_keypoints),
    ("bending (deep, straight legs)", deep_bend_keypoints),
    ("sitting", sitting_keypoints),
    ("kneeling", kneeling_keypoints),
    ("semi-kneeling", semi_kneeling_keypoints),
    ("crouching", crouching_keypoints),
    ("lunge", lunge_keypoints),
    ("mild camera roll (standing)", mild_camera_roll_standing_keypoints),
    ("lying (supine)", supine_lying_keypoints),
    ("lying (prone)", prone_lying_keypoints),
    ("side-lying (bent knees)", side_lying_keypoints),
    ("diagonal lying", diagonal_lying_keypoints),
]

rows = []
for name, fn in POSTURES:
    sim = _StaticTrackSimulator()
    kps = fn()
    bbox = _bbox_for(kps)
    state = None
    for _ in range(40):
        state = sim.step(kps, bbox)
    ext = sim.ext_history[-1]
    scores = sim.scorer.score(
        sim.base_history[-1], ext, sim.base_history, sim.ext_history
    )
    fr = sim.fusion.combine(
        lstm_mandown_prob=0.0,
        feature_confidence=scores.mandown_confidence,
        persistence_score=scores.persistence_score,
        pose_quality=scores.pose_quality_score,
        down_score=scores.down_score,
        floor_proximity_score=scores.floor_proximity_score,
        height_collapse_cue=scores.height_collapse_cue,
    )
    rows.append((
        name, scores.down_score, scores.floor_proximity_score,
        scores.kneeling_penalty, fr.fusion_score, state.name,
    ))

header = f"{'Posture':<32} {'down':>7} {'floor_prox':>11} {'kneel_pen':>10} {'fusion':>7}  state"
print(header)
print("-" * len(header))
for name, down, floor_prox, kneel_pen, fusion, state in rows:
    print(f"{name:<32} {down:>7.3f} {floor_prox:>11.3f} {kneel_pen:>10.3f} {fusion:>7.3f}  {state}")
