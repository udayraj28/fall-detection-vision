"""Tests for the pure-logic ROI/pose-mapping helpers shared by the two-stage
detector and the pose-failure re-acquisition path. Deliberately model-free —
these test coordinate math only, not YOLO inference."""

from src.detector import compute_roi_crop
from src.pipeline import _iou


class TestComputeRoiCrop:
    def test_adds_padding_and_clamps_to_frame(self):
        crop = compute_roi_crop((100, 100, 200, 200), frame_w=640, frame_h=480, padding_ratio=0.5)
        assert crop is not None
        cx1, cy1, cx2, cy2 = crop
        assert cx1 < 100 and cy1 < 100
        assert cx2 > 200 and cy2 > 200

    def test_clamps_at_frame_edges(self):
        crop = compute_roi_crop((0, 0, 50, 50), frame_w=640, frame_h=480, padding_ratio=0.5)
        assert crop is not None
        cx1, cy1, _cx2, _cy2 = crop
        assert cx1 == 0 and cy1 == 0

    def test_clamps_at_bottom_right_edge(self):
        crop = compute_roi_crop((600, 460, 640, 480), frame_w=640, frame_h=480, padding_ratio=0.5)
        assert crop is not None
        _cx1, _cy1, cx2, cy2 = crop
        assert cx2 == 640 and cy2 == 480

    def test_degenerate_bbox_returns_none(self):
        crop = compute_roi_crop((100, 100, 100, 100), frame_w=640, frame_h=480, padding_ratio=0.5)
        assert crop is None

    def test_zero_padding_returns_original_box(self):
        crop = compute_roi_crop((100, 100, 200, 200), frame_w=640, frame_h=480, padding_ratio=0.0)
        assert crop == (100, 100, 200, 200)


class TestIou:
    def test_identical_boxes(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_no_overlap(self):
        assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_partial_overlap(self):
        iou = _iou((0, 0, 10, 10), (5, 5, 15, 15))
        # intersection 5x5=25, union 100+100-25=175
        assert abs(iou - 25 / 175) < 1e-6

    def test_zero_area_box(self):
        assert _iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0
