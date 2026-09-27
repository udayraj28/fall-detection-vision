"""V4 Step 12: event-level deduplication for track-fragmented alerts.

Tests `FallDetectionPipeline._find_duplicate_event` / `_handle_fall_event`
directly against a lightweight stand-in (not a real pipeline instance —
that loads a YOLO model + LSTM checkpoint, far too heavy for a unit test)
that only carries the attributes those methods actually touch.
"""

import functools
from types import SimpleNamespace

from src.config import EventDedupConfig
from src.pipeline import FallDetectionPipeline, FallEvent


def _fake_pipeline(active_events=None, iou_threshold=0.25, recency_frames=30):
    fake = SimpleNamespace(
        mandown_config=SimpleNamespace(
            event_dedup=EventDedupConfig(
                iou_threshold=iou_threshold, recency_frames=recency_frames
            )
        ),
        _active_events=active_events or {},
        _duplicate_of={},
        _cooldowns={},
        _all_events=[],
        cooldown_frames=90,
    )
    # `_handle_fall_event` calls `self._find_duplicate_event` internally —
    # wire the real implementation bound to this stand-in, same technique
    # as calling `_handle_fall_event` itself unbound. It no longer saves a
    # screenshot itself (it just returns the `FallEvent` when one is due,
    # see `_save_pending_screenshots`), so nothing else needs wiring here.
    fake._find_duplicate_event = functools.partial(
        FallDetectionPipeline._find_duplicate_event, fake
    )
    return fake


class TestFindDuplicateEvent:
    def test_overlapping_recent_bbox_is_a_duplicate(self):
        """Real case shape: man_down_c08.mp4 tracks 1 and 2, ~0.35 IoU,
        updated every frame."""
        existing = FallEvent(track_id=1, start_frame=80, fps=25.0, confidence=0.65,
                              bbox=(552.6, 503.4, 1053.5, 722.4))
        existing.end_frame = 91
        fake = _fake_pipeline(active_events={1: existing})
        found = FallDetectionPipeline._find_duplicate_event(
            fake, bbox=(834.2, 509.5, 1056.3, 713.1), frame_num=92, exclude_track_id=2,
        )
        assert found is existing

    def test_well_separated_bbox_is_not_a_duplicate(self):
        """Two genuinely different people, far apart in frame — must NOT be
        merged, however strong the confirmed event's evidence is."""
        existing = FallEvent(track_id=1, start_frame=80, fps=25.0, confidence=0.9,
                              bbox=(0.0, 0.0, 100.0, 100.0))
        existing.end_frame = 91
        fake = _fake_pipeline(active_events={1: existing})
        found = FallDetectionPipeline._find_duplicate_event(
            fake, bbox=(900.0, 600.0, 1050.0, 750.0), frame_num=92, exclude_track_id=2,
        )
        assert found is None

    def test_stale_event_not_eligible_even_with_overlap(self):
        """An event from far earlier in a long video, not updated recently,
        must not silently absorb an unrelated later person even if their
        bbox happens to coincide."""
        existing = FallEvent(track_id=1, start_frame=80, fps=25.0, confidence=0.9,
                              bbox=(552.6, 503.4, 1053.5, 722.4))
        existing.end_frame = 91  # last updated at frame 91
        fake = _fake_pipeline(active_events={1: existing}, recency_frames=30)
        found = FallDetectionPipeline._find_duplicate_event(
            fake, bbox=(552.6, 503.4, 1053.5, 722.4), frame_num=5000, exclude_track_id=2,
        )
        assert found is None

    def test_own_track_id_excluded(self):
        existing = FallEvent(track_id=1, start_frame=80, fps=25.0, confidence=0.9,
                              bbox=(552.6, 503.4, 1053.5, 722.4))
        existing.end_frame = 91
        fake = _fake_pipeline(active_events={1: existing})
        found = FallDetectionPipeline._find_duplicate_event(
            fake, bbox=(552.6, 503.4, 1053.5, 722.4), frame_num=92, exclude_track_id=1,
        )
        assert found is None

    def test_picks_the_best_overlapping_match_among_several(self):
        low = FallEvent(track_id=1, start_frame=10, fps=25.0, confidence=0.6,
                         bbox=(0.0, 0.0, 100.0, 100.0))
        low.end_frame = 20
        high = FallEvent(track_id=2, start_frame=10, fps=25.0, confidence=0.6,
                          bbox=(10.0, 10.0, 110.0, 110.0))
        high.end_frame = 20
        fake = _fake_pipeline(active_events={1: low, 2: high})
        # Candidate bbox overlaps `high` (track 2) more closely than `low`.
        found = FallDetectionPipeline._find_duplicate_event(
            fake, bbox=(15.0, 15.0, 112.0, 112.0), frame_num=21, exclude_track_id=3,
        )
        assert found is high


class TestHandleFallEventDedup:
    """End-to-end through the real `_handle_fall_event` (still no model
    loading — only the FallEvent bookkeeping it touches)."""

    def _pipeline(self):
        fake = _fake_pipeline()
        fake.mandown_contexts = None
        return fake

    def test_fragmented_track_does_not_create_a_second_event(self):
        fake = self._pipeline()
        bbox1 = (552.6, 503.4, 1053.5, 722.4)
        bbox2 = (834.2, 509.5, 1056.3, 713.1)  # ~0.35 IoU with bbox1

        FallDetectionPipeline._handle_fall_event(
            fake, track_id=1, frame_num=80, state="fallen", conf=0.65, fps=25.0,
            bbox=bbox1, entered_mandown=True,
        )
        assert len(fake._all_events) == 1

        FallDetectionPipeline._handle_fall_event(
            fake, track_id=2, frame_num=92, state="fallen", conf=0.65, fps=25.0,
            bbox=bbox2, entered_mandown=True,
        )
        assert len(fake._all_events) == 1, "a fragmented track must not create a second event"
        assert fake._all_events[0].duplicate_track_ids == [2]

    def test_two_independent_far_apart_people_both_get_events(self):
        fake = self._pipeline()
        FallDetectionPipeline._handle_fall_event(
            fake, track_id=1, frame_num=80, state="fallen", conf=0.65, fps=25.0,
            bbox=(0.0, 0.0, 100.0, 100.0), entered_mandown=True,
        )
        FallDetectionPipeline._handle_fall_event(
            fake, track_id=2, frame_num=80, state="fallen", conf=0.7, fps=25.0,
            bbox=(900.0, 600.0, 1050.0, 750.0), entered_mandown=True,
        )
        assert len(fake._all_events) == 2, "two genuinely different people must not be merged"

    def test_screenshot_saved_only_once_across_fragmented_tracks(self):
        fake = self._pipeline()
        bbox1 = (552.6, 503.4, 1053.5, 722.4)
        bbox2 = (834.2, 509.5, 1056.3, 713.1)

        FallDetectionPipeline._handle_fall_event(
            fake, track_id=1, frame_num=80, state="fallen", conf=0.65, fps=25.0,
            bbox=bbox1, entered_mandown=False,
        )
        FallDetectionPipeline._handle_fall_event(
            fake, track_id=2, frame_num=92, state="fallen", conf=0.65, fps=25.0,
            bbox=bbox2, entered_mandown=True,
        )
        assert fake._all_events[0].screenshot_saved is False  # saving is deferred to `_save_pending_screenshots`, never called here
        # entered_mandown was requested for the duplicate track and routed to
        # the same underlying event object, not a second one.
        assert len(fake._all_events) == 1


class TestCloseActiveEventOnRecovery:
    """V4.3: a genuine recovery (TrackStateMachine reports `recovered=True`
    -- sustained NORMAL evidence, not a flicker) must close out the active
    event so the SAME track_id falling again later starts a fresh
    event/screenshot, rather than silently extending (and never
    re-screenshotting) the old, already-closed one. `_all_events` — the
    historical record — is never touched; only `_active_events`/
    `_cooldowns` bookkeeping is cleared."""

    def _pipeline(self):
        fake = _fake_pipeline()
        fake.mandown_contexts = None
        return fake

    def test_fall_recover_fall_produces_two_events(self):
        fake = self._pipeline()
        bbox = (100.0, 100.0, 200.0, 300.0)

        FallDetectionPipeline._handle_fall_event(
            fake, track_id=1, frame_num=100, state="fallen", conf=0.8, fps=25.0,
            bbox=bbox, entered_mandown=True,
        )
        assert len(fake._all_events) == 1
        assert 1 in fake._active_events

        FallDetectionPipeline._close_active_event(fake, track_id=1)
        assert 1 not in fake._active_events
        assert 1 not in fake._cooldowns
        assert len(fake._all_events) == 1, "historical record is untouched by closing"

        FallDetectionPipeline._handle_fall_event(
            fake, track_id=1, frame_num=500, state="fallen", conf=0.75, fps=25.0,
            bbox=bbox, entered_mandown=True,
        )
        assert len(fake._all_events) == 2, "a second, independent episode by the same track_id"
        assert fake._all_events[0] is not fake._all_events[1]

    def test_n_fall_recover_cycles_produce_n_events(self):
        fake = self._pipeline()
        bbox = (100.0, 100.0, 200.0, 300.0)
        n = 5
        for i in range(n):
            FallDetectionPipeline._handle_fall_event(
                fake, track_id=1, frame_num=100 * (i + 1), state="fallen", conf=0.8, fps=25.0,
                bbox=bbox, entered_mandown=True,
            )
            FallDetectionPipeline._close_active_event(fake, track_id=1)
        assert len(fake._all_events) == n

    def test_closing_with_no_active_event_is_a_harmless_noop(self):
        fake = self._pipeline()
        FallDetectionPipeline._close_active_event(fake, track_id=42)  # never had an event
        assert fake._active_events == {}

    def test_closing_resolves_through_duplicate_of_mapping(self):
        """A track merged into another (event dedup) must close the SAME
        underlying event its alias points to, not a separate, nonexistent
        entry under its own id."""
        fake = self._pipeline()
        bbox = (100.0, 100.0, 200.0, 300.0)
        FallDetectionPipeline._handle_fall_event(
            fake, track_id=1, frame_num=100, state="fallen", conf=0.8, fps=25.0,
            bbox=bbox, entered_mandown=True,
        )
        fake._duplicate_of[2] = 1  # track 2 was merged into track 1's event

        FallDetectionPipeline._close_active_event(fake, track_id=2)

        assert 1 not in fake._active_events
        assert 1 not in fake._cooldowns
