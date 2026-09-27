"""Problem 3 (multi-person MANDOWN screenshot): when several people are
classified "fallen" in the same frame, the saved event screenshot must show
ALL of their red boxes, not just whichever track happened to trigger the
save. Exercises the real `_save_event_screenshot`/`_save_pending_screenshots`
against a lightweight stand-in, same technique as `test_event_dedup.py` —
no model loading required, `_draw_label_scaled` is a staticmethod and
`_save_event_screenshot` only touches `self.events_dir`.
"""

import functools
from types import SimpleNamespace

import numpy as np

from src.pipeline import FallDetectionPipeline, FallEvent


def _fake_pipeline(tmp_path):
    fake = SimpleNamespace(events_dir=tmp_path)
    # `_draw_label_scaled` is a `@staticmethod` — accessed off the class it's
    # already a plain function, so assigning it directly (not via
    # `functools.partial(..., fake)`) calls it with no implicit `self`.
    fake._draw_label_scaled = FallDetectionPipeline._draw_label_scaled
    fake._save_event_screenshot = functools.partial(
        FallDetectionPipeline._save_event_screenshot, fake
    )
    fake._save_pending_screenshots = functools.partial(
        FallDetectionPipeline._save_pending_screenshots, fake
    )
    return fake


def _blank_frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


class TestSaveEventScreenshot:
    def test_single_person_saves_one_box(self, tmp_path):
        fake = _fake_pipeline(tmp_path)
        fallen = [(1, (100.0, 100.0, 200.0, 300.0), None)]
        path = fake._save_event_screenshot(_blank_frame(), fallen)
        assert path.exists()
        assert "track1_" in path.name

    def test_multiple_people_all_appear_in_filename_and_frame_is_drawn(self, tmp_path):
        fake = _fake_pipeline(tmp_path)
        fallen = [
            (1, (100.0, 100.0, 200.0, 300.0), None),
            (2, (900.0, 400.0, 1050.0, 700.0), None),
            (3, (500.0, 500.0, 600.0, 690.0), None),
        ]
        frame = _blank_frame()
        path = fake._save_event_screenshot(frame, fallen)
        assert path.exists()
        # All three track IDs are reflected in the saved event's filename —
        # a simple, verifiable proxy for "this screenshot covers everyone",
        # since asserting pixel content would be brittle.
        assert "track1_2_3_" in path.name

        import cv2
        saved = cv2.imread(str(path))
        assert saved is not None
        # Red (BGR: 0,0,255-ish) pixels must appear near EACH of the three
        # boxes' borders, not just the first one drawn.
        for x1, y1, x2, y2 in [(100, 100, 200, 300), (900, 400, 1050, 700), (500, 500, 600, 690)]:
            region = saved[int(y1):int(y2), int(x1):int(x2)]
            has_red = np.any(
                (region[:, :, 2].astype(int) > 150)
                & (region[:, :, 0].astype(int) < 100)
                & (region[:, :, 1].astype(int) < 100)
            )
            assert has_red, f"expected a red box drawn within bbox {(x1, y1, x2, y2)}"


class TestSavePendingScreenshots:
    def test_all_pending_events_point_at_the_same_saved_file(self, tmp_path):
        fake = _fake_pipeline(tmp_path)
        e1 = FallEvent(track_id=1, start_frame=10, fps=25.0, confidence=0.8,
                        bbox=(100.0, 100.0, 200.0, 300.0))
        e2 = FallEvent(track_id=2, start_frame=10, fps=25.0, confidence=0.75,
                        bbox=(900.0, 400.0, 1050.0, 700.0))
        fallen_persons = [
            (1, (100.0, 100.0, 200.0, 300.0), None),
            (2, (900.0, 400.0, 1050.0, 700.0), None),
        ]
        fake._save_pending_screenshots([e1, e2], _blank_frame(), fallen_persons)

        assert e1.screenshot_saved and e2.screenshot_saved
        assert e1.screenshot_path == e2.screenshot_path
        assert "track1_2_" in e1.screenshot_path

    def test_no_clean_frame_saves_nothing(self, tmp_path):
        fake = _fake_pipeline(tmp_path)
        e1 = FallEvent(track_id=1, start_frame=10, fps=25.0, confidence=0.8,
                        bbox=(100.0, 100.0, 200.0, 300.0))
        fake._save_pending_screenshots([e1], None, [(1, (100.0, 100.0, 200.0, 300.0), None)])
        assert e1.screenshot_saved is False
        assert list(tmp_path.iterdir()) == []
