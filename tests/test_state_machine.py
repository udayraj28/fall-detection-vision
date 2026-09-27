"""Tests for the per-track temporal state machine (hysteresis)."""

from src.config import StateMachineConfig
from src.state_machine import MandownState, TrackStateMachine


def _config(**kwargs):
    defaults = dict(
        frames_to_confirm_mandown=5, frames_to_enter_possible=3,
        frames_to_recover_normal=5, cooldown_frames=90,
    )
    defaults.update(kwargs)
    return StateMachineConfig(**defaults)


class TestTrackStateMachine:
    def test_starts_normal(self):
        sm = TrackStateMachine(_config())
        assert sm.state == MandownState.NORMAL

    def test_single_spike_does_not_trigger_mandown(self):
        sm = TrackStateMachine(_config())
        result = sm.step("MANDOWN")
        assert result.state != MandownState.MANDOWN
        # Flicker back to normal immediately after.
        result2 = sm.step("NORMAL")
        assert result2.state == MandownState.NORMAL

    def test_sustained_evidence_confirms_mandown(self):
        sm = TrackStateMachine(_config())
        entered = False
        for _ in range(3):
            sm.step("POSSIBLE_MANDOWN")  # escalate to POSSIBLE_MANDOWN
        for _ in range(5):
            r = sm.step("MANDOWN")
            entered = entered or r.entered_mandown
        assert sm.state == MandownState.MANDOWN
        assert entered

    def test_entered_mandown_fires_exactly_once(self):
        sm = TrackStateMachine(_config())
        for _ in range(3):
            sm.step("POSSIBLE_MANDOWN")
        flags = [sm.step("MANDOWN").entered_mandown for _ in range(10)]
        assert flags.count(True) == 1

    def test_recovery_requires_sustained_normal(self):
        sm = TrackStateMachine(_config())
        for _ in range(3):
            sm.step("POSSIBLE_MANDOWN")
        for _ in range(5):
            sm.step("MANDOWN")
        assert sm.state == MandownState.MANDOWN

        # A single normal frame should not immediately recover.
        r = sm.step("NORMAL")
        assert r.state == MandownState.MANDOWN
        assert not r.recovered

        # Sustained normal should eventually recover.
        recovered = False
        for _ in range(10):
            r = sm.step("NORMAL")
            recovered = recovered or r.recovered
        assert sm.state == MandownState.NORMAL
        assert recovered

    def test_falling_transition_and_already_down_both_reach_mandown(self):
        """Both an active-fall pattern (possible then mandown) and an
        already-lying track that starts straight at MANDOWN-level evidence
        should end up in MANDOWN."""
        sm_fall = TrackStateMachine(_config())
        for _ in range(3):
            sm_fall.step("POSSIBLE_MANDOWN")
        for _ in range(5):
            sm_fall.step("MANDOWN")
        assert sm_fall.state == MandownState.MANDOWN

        sm_lying = TrackStateMachine(_config())
        for _ in range(3):
            sm_lying.step("MANDOWN")  # is_up counts MANDOWN too, escalates to POSSIBLE first
        for _ in range(5):
            sm_lying.step("MANDOWN")
        assert sm_lying.state == MandownState.MANDOWN

    def test_no_leakage_between_instances(self):
        sm1 = TrackStateMachine(_config())
        sm2 = TrackStateMachine(_config())
        for _ in range(3):
            sm1.step("POSSIBLE_MANDOWN")
        for _ in range(5):
            sm1.step("MANDOWN")
        assert sm1.state == MandownState.MANDOWN
        assert sm2.state == MandownState.NORMAL

    def test_reset(self):
        sm = TrackStateMachine(_config())
        for _ in range(3):
            sm.step("POSSIBLE_MANDOWN")
        sm.reset()
        assert sm.state == MandownState.NORMAL

    def test_on_gap_within_tolerance_preserves_progress(self):
        sm = TrackStateMachine(_config())
        sm.step("POSSIBLE_MANDOWN")
        sm.step("POSSIBLE_MANDOWN")
        sm.on_gap(gap_frames=2, tolerance_frames=2)  # at tolerance, no decay
        sm.step("POSSIBLE_MANDOWN")  # 3rd consecutive -> escalates
        assert sm.state == MandownState.POSSIBLE_MANDOWN

    def test_on_gap_beyond_tolerance_discards_progress(self):
        sm = TrackStateMachine(_config())
        sm.step("POSSIBLE_MANDOWN")
        sm.step("POSSIBLE_MANDOWN")
        sm.on_gap(gap_frames=10, tolerance_frames=2)  # beyond tolerance -> reset progress
        sm.step("POSSIBLE_MANDOWN")  # only the 1st consecutive frame again
        assert sm.state == MandownState.NORMAL

    def test_on_gap_never_resets_confirmed_state(self):
        """A confirmed MANDOWN must survive a gap — only hysteresis
        progress-counters decay, not the classification itself."""
        sm = TrackStateMachine(_config())
        for _ in range(3):
            sm.step("POSSIBLE_MANDOWN")
        for _ in range(5):
            sm.step("MANDOWN")
        assert sm.state == MandownState.MANDOWN

        sm.on_gap(gap_frames=20, tolerance_frames=2)
        assert sm.state == MandownState.MANDOWN
