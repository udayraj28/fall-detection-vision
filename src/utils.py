"""Drawing helpers, color maps, and visualization utilities."""

import cv2
import numpy as np

# COCO keypoint skeleton connections (pairs of keypoint indices)
COCO_SKELETON = [
    (0, 1), (0, 2),      # nose → eyes
    (1, 3), (2, 4),      # eyes → ears
    (5, 6),               # shoulder → shoulder
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 11), (6, 12),    # torso
    (11, 12),             # hip → hip
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# COCO keypoint names (17 keypoints)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Color definitions (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE = (255, 0, 0)

# Status colors for different states
STATUS_COLORS = {
    "normal": COLOR_GREEN,
    "falling": COLOR_ORANGE,
    "fallen": COLOR_RED,
}


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color: tuple[int, int, int] = COLOR_GREEN,
    confidence_threshold: float = 0.3,
    thickness: int = 2,
    point_radius: int = 4,
) -> np.ndarray:
    """Draw skeleton on frame from COCO keypoints.

    Args:
        frame: BGR image (modified in place).
        keypoints: Array of shape (17, 3) with [x, y, confidence].
        color: BGR color for the skeleton.
        confidence_threshold: Minimum confidence to draw a keypoint/connection.
        thickness: Line thickness.
        point_radius: Radius of keypoint circles.

    Returns:
        The frame with skeleton drawn.
    """
    if keypoints is None or len(keypoints) == 0:
        return frame

    # Draw connections
    for i, j in COCO_SKELETON:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        if keypoints[i][2] < confidence_threshold or keypoints[j][2] < confidence_threshold:
            continue
        pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
        pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
        cv2.line(frame, pt1, pt2, color, thickness)

    # Draw keypoints
    for kp in keypoints:
        if kp[2] < confidence_threshold:
            continue
        center = (int(kp[0]), int(kp[1]))
        cv2.circle(frame, center, point_radius, color, -1)

    return frame


def draw_bbox(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int] = COLOR_GREEN,
    thickness: int = 2,
    label: str | None = None,
) -> np.ndarray:
    """Draw bounding box with optional label.

    Args:
        frame: BGR image (modified in place).
        bbox: (x1, y1, x2, y2) coordinates.
        color: BGR color.
        thickness: Line thickness.
        label: Optional text label above the box.

    Returns:
        The frame with bbox drawn.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
        # Draw label above bbox if room, otherwise below top edge
        label_above = y1 - text_h - baseline - 4 >= 0
        if label_above:
            cv2.rectangle(
                frame, (x1, y1 - text_h - baseline - 4), (x1 + text_w, y1), color, -1
            )
            cv2.putText(frame, label, (x1, y1 - baseline - 2), font, font_scale, COLOR_BLACK, 1)
        else:
            cv2.rectangle(
                frame, (x1, y1), (x1 + text_w, y1 + text_h + baseline + 4), color, -1
            )
            cv2.putText(
                frame, label, (x1, y1 + text_h + 2), font, font_scale, COLOR_BLACK, 1
            )

    return frame


def draw_alert(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int] | None = None,
    color: tuple[int, int, int] = COLOR_RED,
    font_scale: float = 1.2,
    thickness: int = 3,
) -> np.ndarray:
    """Draw an alert text on the frame with background.

    Args:
        frame: BGR image (modified in place).
        text: Alert text to display.
        position: (x, y) position. If None, centered at top.
        color: BGR text color.
        font_scale: Font size scale.
        thickness: Text thickness.

    Returns:
        The frame with alert drawn.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    if position is None:
        h, w = frame.shape[:2]
        x = (w - text_w) // 2
        y = text_h + 20
    else:
        x, y = position

    # Semi-transparent background (ROI-based for performance)
    padding = 10
    h_frame, w_frame = frame.shape[:2]
    rx1 = max(0, x - padding)
    ry1 = max(0, y - text_h - padding)
    rx2 = min(w_frame, x + text_w + padding)
    ry2 = min(h_frame, y + baseline + padding)
    roi = frame[ry1:ry2, rx1:rx2].copy()
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), COLOR_BLACK, -1)
    cv2.addWeighted(frame[ry1:ry2, rx1:rx2], 0.6, roi, 0.4, 0, frame[ry1:ry2, rx1:rx2])

    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
    return frame


def draw_fall_banner(
    frame: np.ndarray,
    text: str,
    color: tuple[int, int, int] = COLOR_RED,
    timestamp: str | None = None,
    font_scale: float = 1.0,
) -> np.ndarray:
    """Draw a bold, black-outlined alert label top-left with a timestamp top-right.

    Matches the look of typical CCTV/surveillance overlay alerts (e.g. a
    "Missing PPE" banner) rather than the centered semi-transparent alert box.

    Args:
        frame: BGR image (modified in place).
        text: Alert label, e.g. "Fall Alert: Falling".
        color: BGR color for the label text.
        timestamp: Optional short timestamp string drawn in the top-right corner.
        font_scale: Base font scale, typically resolution-scaled.

    Returns:
        The frame with the banner drawn.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(2, int(2 * font_scale))
    outline_thickness = thickness + 3

    x, y = 12, int(30 * font_scale) + 10
    cv2.putText(frame, text, (x, y), font, font_scale, COLOR_BLACK, outline_thickness, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    if timestamp:
        ts_font_scale = 0.6 * font_scale
        ts_thickness = max(1, int(1.5 * font_scale))
        (tw, _th), _baseline = cv2.getTextSize(timestamp, font, ts_font_scale, ts_thickness)
        tx = w - tw - 12
        ty = y
        cv2.putText(
            frame, timestamp, (tx, ty), font, ts_font_scale,
            COLOR_BLACK, ts_thickness + 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, timestamp, (tx, ty), font, ts_font_scale,
            COLOR_WHITE, ts_thickness, cv2.LINE_AA,
        )

    return frame


def draw_mandown_banner(
    frame: np.ndarray,
    text: str = "MANDOWN",
    color: tuple[int, int, int] = COLOR_RED,
    font_scale: float = 1.0,
) -> np.ndarray:
    """Draw a plain solid-black banner chip, top-left, with bold text —
    e.g. "MANDOWN". Matches common CCTV/surveillance overlay alert style
    (solid dark background behind bold colored text), deliberately minimal
    — no timestamp, duration, or other detail folded in.

    Args:
        frame: BGR image (modified in place).
        text: Banner text, e.g. "MANDOWN".
        color: BGR text color.
        font_scale: Base font scale, typically resolution-scaled.

    Returns:
        The frame with the banner drawn.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = font_scale * 2.0  # doubled — the original size read too small
    thickness = max(2, int(2 * font_scale))
    padding = int(10 * font_scale)

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x1, y1 = 0, 0
    x2, y2 = tw + padding * 2, th + baseline + padding * 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BLACK, -1)
    # Text origin is the baseline; the glyph block spans `th` (ascent) above
    # it and `baseline` (descent) below it. Box height is th+baseline+2*
    # padding, so placing the baseline at padding+th centers that block
    # exactly, with equal black margin above and below — the previous
    # `y2 - baseline - padding//2` placement left extra space on top.
    cv2.putText(
        frame, text, (x1 + padding, padding + th),
        font, font_scale, color, thickness, cv2.LINE_AA,
    )

    return frame


def draw_status_bar(
    frame: np.ndarray,
    frame_num: int,
    total_frames: int | None,
    fps: float,
    persons_tracked: int,
    falls_detected: int,
) -> np.ndarray:
    """Draw a status bar at the bottom of the frame.

    Args:
        frame: BGR image (modified in place).
        frame_num: Current frame number.
        total_frames: Total number of frames in video, or None for a live stream.
        fps: Processing FPS.
        persons_tracked: Number of persons currently tracked.
        falls_detected: Total falls detected so far.

    Returns:
        The frame with status bar drawn.
    """
    h, w = frame.shape[:2]
    bar_height = 30
    y_start = h - bar_height

    # Semi-transparent black bar (ROI-based for performance)
    roi = frame[y_start:h, 0:w].copy()
    cv2.rectangle(frame, (0, y_start), (w, h), COLOR_BLACK, -1)
    cv2.addWeighted(frame[y_start:h, 0:w], 0.7, roi, 0.3, 0, frame[y_start:h, 0:w])

    # Status text
    if total_frames:
        progress = frame_num / max(total_frames, 1) * 100
        frame_info = f"Frame: {frame_num}/{total_frames} ({progress:.0f}%)"
    else:
        frame_info = f"Frame: {frame_num} (LIVE)"
    status = (
        f"{frame_info} | "
        f"FPS: {fps:.1f} | "
        f"Persons: {persons_tracked} | "
        f"Falls: {falls_detected}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, status, (10, h - 8), font, 0.5, COLOR_WHITE, 1)

    return frame
