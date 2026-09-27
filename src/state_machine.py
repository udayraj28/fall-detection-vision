"""Per-track temporal state machine with hysteresis.

Three explicit states: NORMAL, POSSIBLE_MANDOWN, MANDOWN. A single
anomalous frame can never trigger or clear an alert — transitions require
several consecutive frames of consistent fusion evidence, so a track
recovers back to NORMAL only after sustained normal-looking evidence too.

MANDOWN covers both a falling transition and an already-lying person —
whichever evidence (fall_motion_score or down_score/floor_proximity_score,
via the fusion score) drives the escalation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.config import StateMachineConfig


class MandownState(str, Enum):
    NORMAL = "NORMAL"
    POSSIBLE_MANDOWN = "POSSIBLE_MANDOWN"
    MANDOWN = "MANDOWN"


@dataclass
class StepResult:
    state: MandownState
    entered_mandown: bool  # True only on the frame that first confirms MANDOWN
    recovered: bool        # True only on the frame that returns to NORMAL from a non-NORMAL state


class TrackStateMachine:
    """One instance per track_id. Call `step()` once per frame with that
    frame's fusion-suggested tier ("NORMAL" | "POSSIBLE_MANDOWN" | "MANDOWN")."""

    def __init__(self, config: StateMachineConfig | None = None):
        self.config = config or StateMachineConfig()
        self.state = MandownState.NORMAL
        self._up_streak = 0
        self._down_streak = 0

    def step(self, suggested_tier: str) -> StepResult:
        c = self.config
        entered_mandown = False
        recovered = False
        is_up = suggested_tier in ("POSSIBLE_MANDOWN", "MANDOWN")

        if self.state == MandownState.NORMAL:
            self._up_streak = self._up_streak + 1 if is_up else 0
            if self._up_streak >= c.frames_to_enter_possible:
                self.state = MandownState.POSSIBLE_MANDOWN
                self._up_streak = 0
            self._down_streak = 0

        elif self.state == MandownState.POSSIBLE_MANDOWN:
            if is_up:
                self._up_streak += 1
                self._down_streak = 0
            else:
                self._down_streak += 1
                self._up_streak = 0

            if self._up_streak >= c.frames_to_confirm_mandown:
                self.state = MandownState.MANDOWN
                entered_mandown = True
                self._up_streak = 0
            elif self._down_streak >= c.frames_to_recover_normal:
                self.state = MandownState.NORMAL
                recovered = True
                self._down_streak = 0

        else:  # MANDOWN — sticky until sustained normal evidence
            if suggested_tier == "NORMAL":
                self._down_streak += 1
            else:
                self._down_streak = 0

            if self._down_streak >= c.frames_to_recover_normal:
                self.state = MandownState.NORMAL
                recovered = True
                self._down_streak = 0

        return StepResult(state=self.state, entered_mandown=entered_mandown, recovered=recovered)

    def on_gap(self, gap_frames: int, tolerance_frames: int) -> None:
        """Call once when a track resumes after `gap_frames` of no detection
        at all (before the next `step()`). Beyond `tolerance_frames`, hysteresis
        progress-toward-a-transition is discarded — otherwise a long silent gap
        could let stale pre-gap momentum finish confirming/recovering a
        transition it didn't actually earn. `self.state` itself is untouched:
        a confirmed MANDOWN stays MANDOWN through the gap, per the "temporary
        pose failure must not immediately reset Mandown" requirement."""
        if gap_frames > tolerance_frames:
            self._up_streak = 0
            self._down_streak = 0

    def reset(self) -> None:
        self.state = MandownState.NORMAL
        self._up_streak = 0
        self._down_streak = 0
