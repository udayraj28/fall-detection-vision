# Elderly Fall Detection System

A real-time, camera-only fall detection system that tracks people with skeleton pose estimation and combines a learned temporal classifier with a deterministic geometric scoring system to decide when someone has fallen and is down. Built for elderly care facilities, hospitals, and smart-home monitoring — no wearable device required.

---

## Why This Matters

| The Problem | The Impact |
|-------------|-----------|
| **684,000** older adults die from falls globally each year (WHO) | Falls are the **#1 cause of injury death** among people aged 65+ |
| **37.3 million** falls per year are severe enough to require medical attention | Average hospital cost per fall: **$35,000+** |
| **95% of hip fractures** are caused by falling | A person lying on the floor for **>1 hour** has a 50% mortality rate within 6 months |

The critical factor is response time. This system detects a fall and a subsequent "down" state automatically, from ordinary video, so help can be dispatched faster.

---

## How It Works

Every tracked person is scored by **two independent evidence sources** each frame, which are then fused and passed through a per-track state machine before anything is reported:

```
Video Frame
    |
    v
[1] YOLO11-Pose + BoT-SORT  -->  Detect people, 17 skeleton keypoints, persistent track IDs
    |                              (+ ROI re-detection / rotation search to recover lost tracks)
    v
    +-------------------------------+-------------------------------+
    |                                                                |
    v                                                                v
[2a] 5-feature LSTM classifier                     [2b] Deterministic "Mandown" scorer
     30-frame sliding window -->                        posture/geometry features, per-camera
     P(Normal / Falling / Fallen)                        scene + person-size calibration, optional
     (pretrained, unchanged input contract)               monocular 2D->3D pose lifting for
                                                           falls toward/away from the camera
    |                                                                |
    +-------------------------------+-------------------------------+
                                     v
                     [3] Evidence Fusion  -->  weighted combination + override rules
                                     |          (a single very strong signal can be enough;
                                     |           the two sources check each other otherwise)
                                     v
                     [4] Per-track State Machine  -->  NORMAL / POSSIBLE_MANDOWN / MANDOWN
                                     |                  (hysteresis: sustained streaks, not
                                     |                   single frames, drive transitions)
                                     v
                 Annotated Video + JSON Event Log + Event Screenshots
```

The LSTM is not the sole authority — it is one of two evidence sources that are fused together, so a confident-but-wrong classifier can be overruled by strong geometric evidence (and vice versa) before a fall is confirmed.

### Feature Visualization: Normal vs Fallen

**Standing (Normal)** — Upright posture, narrow bounding box:

![Normal State](docs/images/features_normal.png)

**Fallen** — Lying on ground, wide/flat bounding box, all indicators turn red:

![Fallen State](docs/images/features_fallen.png)

---

## Demo Results

### Elderly Person Fall (Hospital Corridor)

![Elderly Fallen](docs/images/demo_elderly_fallen.png)

*Fall detected with persistent "PERSON FALLEN" alert and duration timer. Orange skeleton and bounding box indicate fallen state.*

### Outdoor Fall (Sidewalk with Bystanders)

![Outdoor Fallen](docs/images/demo_outdoor_fallen.png)

*System correctly identifies the fallen person (ID:2, orange) while classifying standing pedestrians as NORMAL (green). Multi-person tracking works in crowded scenes.*

### Skater Fall (4K Video)

![Skating Fallen](docs/images/demo_skating_fallen.png)

*Fall detected at high confidence. Labels and skeleton scale automatically with video resolution.*

### No False Positive (Breakdancer)

![Dancing Normal](docs/images/demo_dancing_normal.png)

*Breakdancer performing on the ground is correctly classified as NORMAL — the geometric scorer and LSTM together distinguish intentional movement from an accidental fall.*

---

## Model Performance (LSTM classifier)

The 5-feature LSTM is one of the two fused evidence sources; these are its standalone metrics on held-out test data.

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.2% |
| **Macro F1** | 97.8% |
| **Macro AUC** | 0.998 |

### Per-Class Performance

![Per-Class Metrics](docs/images/per_class_metrics.png)

| Class | Precision | Recall | F1-Score | AUC |
|-------|-----------|--------|----------|-----|
| **Normal** | 98.6% | 98.7% | 98.6% | 0.998 |
| **Falling** | 99.7% | 95.3% | 97.4% | 0.998 |
| **Fallen** | 95.3% | 99.4% | 97.3% | 0.999 |

### Confusion Matrix

![Confusion Matrix](docs/images/confusion_matrix.png)

### ROC Curves

![ROC Curves](docs/images/roc_curves.png)

### Training Progress

![Training Curves](docs/images/training_curves.png)

*2-layer LSTM, 128 hidden units, dropout=0.3, LayerNorm — 201,859 parameters. Converged within 38 epochs.*

---

## Detection States

Each tracked person's state machine reports one of three tiers, displayed using the original vocabulary:

| Internal Tier | Displayed As | Description |
|----------------|--------------|-------------|
| `NORMAL` | **NORMAL** (green) | Standing, walking, sitting, or otherwise upright/active |
| `POSSIBLE_MANDOWN` | **FALLING** (red, alert) | Fusion evidence trending toward down, not yet confirmed |
| `MANDOWN` | **FALLEN** (orange, timer) | Confirmed down; state is sticky until a sustained recovery streak |

Transitions require sustained evidence over multiple frames (hysteresis), not a single frame's reading, which keeps brief occlusions or noisy detections from flipping the state.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Pose Estimation** | YOLO11-Pose (Ultralytics) | Person detection + 17 skeleton keypoints |
| **Tracking** | BoT-SORT (+ custom ROI re-detection) | Multi-person tracking, recovers tracks lost to occlusion |
| **Temporal Classifier** | LSTM (PyTorch) | Classifies 30-frame windows of 5 base features |
| **Geometric Scorer** | Custom Python (`feature_mandown.py`) | Deterministic down/fallen scoring from richer posture/geometry features |
| **3D Lifting (optional)** | MediaPipe BlazePose | Monocular 2D→3D pose lifting for falls along the camera's depth axis |
| **Evidence Fusion** | Custom Python (`fusion.py`) | Weighted combination + override rules across both evidence sources |
| **Temporal Hysteresis** | Custom Python (`state_machine.py`) | Per-track state machine (NORMAL / POSSIBLE_MANDOWN / MANDOWN) |
| **Per-Camera Calibration** | Custom Python (`scene_profile.py`, `person_size_stats.py`) | Learns expected standing size/floor region per camera at runtime |
| **Video I/O** | OpenCV | Read/write video files with annotation overlays |

---

## Dataset (LSTM training data)

| Source | Videos | Type | Content |
|--------|--------|------|---------|
| **UR Fall Detection** | 70 | Image sequences | 30 falls + 40 daily activities (indoor, Kinect) |
| **Le2i Fall Dataset** | 140 | Video files | 70 falls + 70 normal (4 room types, CCTV) |
| **Synthetic** | — | Generated | 3,000 feature sequences with realistic fall patterns |

| Split | Samples |
|-------|---------|
| Train | 14,001 |
| Validation | 1,750 |
| Test | 1,751 |
| **Total** | **17,502** |

---

## Quick Start

### Installation

```bash
git clone <repo-url>
cd fall-detection-vison

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

YOLO pose/detection weights (`yolo11n.pt`, `yolo11s-pose.pt`, etc.) and the trained LSTM checkpoint (`models/checkpoints/best.pth`) are not tracked in git — the YOLO weights are downloaded automatically by Ultralytics on first run; the LSTM checkpoint must be trained (`src/train.py`) or obtained separately.

### Run on a video file

Drop the video in `input/` and run with just its filename — the annotated video and JSON log are written to `output/` automatically:

```bash
python detect_falls.py --input your_video.mp4
# -> output/your_video.mp4, output/your_video_events.json
```

Explicit paths still work if you want to override the defaults:

```bash
python detect_falls.py \
    --input your_video.mp4 \
    --output result.mp4 \
    --json-log events.json
```

### Run on a live RTSP camera or webcam

```bash
python detect_falls.py --input rtsp://user:pass@camera-ip/stream
# or a local webcam:
python detect_falls.py --input 0
```

This opens a real-time preview window with the same skeleton/box/alert overlays as video mode, but **no inference video is written** — only a screenshot per confirmed fall event, saved to `events/`. Stop with `q`/Esc in the preview window or Ctrl+C. Use `--no-display` to run headless (events still save).

### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` / `-i` | (required) | Video file (checked in `input/` if not found as given), RTSP/HTTP stream URL, or webcam index |
| `--output` / `-o` | `output/<name>.mp4` | Output annotated video path (video files only) |
| `--json-log` / `-j` | `output/<name>_events.json` | JSON event log output path |
| `--events-dir` | `events` | Folder where fall event screenshots are saved |
| `--no-display` | off | For live streams: skip the preview window |
| `--model` | `models/checkpoints/best.pth` | LSTM model checkpoint |
| `--yolo-model` | `yolo11s-pose.pt` | YOLO pose model |
| `--imgsz` | `1280` | YOLO inference resolution (long side, pixels) |
| `--confidence` | `0.40` | Fall confidence threshold |
| `--cooldown` | `90` | Cooldown frames between alerts |
| `--device` | auto | `cuda` or `cpu` |

### Output

1. **Annotated video** (video files only) — skeletons, bounding boxes, fall alert banner, status bar
2. **JSON event log** — structured data with timestamps, confidence scores, track IDs, and the saved screenshot path
3. **Event screenshots** — saved to `events/` the moment a tracked person is classified as MANDOWN, once per event

```json
{
  "events": [
    {
      "type": "fall",
      "track_id": 1,
      "start_frame": 114,
      "timestamp_start": "00:00:04.750",
      "confidence": 0.84,
      "screenshot_path": "events/fall_track1_20260824_154735_956.jpg"
    }
  ],
  "summary": {
    "total_falls": 1,
    "persons_tracked": 5
  }
}
```

### Running the tests

```bash
pytest
```

275 tests across 17 files cover feature extraction, the LSTM-facing pipeline, the geometric scorer, fusion rules, the state machine, per-camera calibration, video I/O, and event deduplication.

---

## Project Structure

```
fall-detection-vison/
├── detect_falls.py                # CLI entry point
├── configs/
│   ├── botsort.yaml                # Tracker configuration
│   ├── mandown_config.yaml         # Thresholds/weights for calibration, scoring, fusion, state machine
│   └── scene_profiles/             # Learned per-camera calibration profiles
├── src/
│   ├── detector.py                 # YOLO pose + BoT-SORT wrapper, track re-acquisition
│   ├── feature_extraction.py       # Base 5-feature extractor (LSTM) + extended posture/geometry features
│   ├── feature_mandown.py          # Deterministic geometric down/fallen scorer
│   ├── fusion.py                   # Combines LSTM output with the geometric scorer
│   ├── state_machine.py            # Per-track hysteresis state machine
│   ├── mandown_track_context.py    # Per-track state for the geometric/fusion layer
│   ├── scene_profile.py            # Per-camera scene calibration (expected size, floor region)
│   ├── person_size_stats.py        # Rolling person-size statistics + reliability filtering
│   ├── pose_3d.py                  # Optional monocular 2D->3D pose lifting (MediaPipe)
│   ├── track_manager.py            # 30-frame sliding feature window per track (LSTM)
│   ├── pipeline.py                 # End-to-end inference pipeline
│   ├── video_io.py                 # Video read/write utilities
│   ├── dataset.py / dataset_processor.py / label_parser.py   # LSTM training data pipeline
│   ├── train.py / evaluate.py      # LSTM training loop and evaluation/plots
│   ├── config.py                   # Loads configs/mandown_config.yaml into typed configs
│   ├── geometry.py / utils.py      # Shared geometry helpers and drawing utilities
│   └── simple_tracker.py           # Lightweight IoU tracker for edge/ONNX deployment
├── models/
│   ├── fall_detector.py            # LSTM model definition
│   ├── checkpoints/                # Trained weights + training report (not tracked in git)
│   ├── exports/                    # ONNX exports
│   └── mediapipe/                  # BlazePose landmarker model for pose_3d.py
├── scripts/                        # Data prep, batch testing, ONNX export, edge inference, reports
├── tests/                          # 275 tests
├── evaluation/                     # Metrics plots & model card
└── docs/images/                    # README images
```

---

## Key Features

- **No wearable device required** — works with any standard camera
- **Multi-person tracking** — monitors multiple people simultaneously with persistent IDs
- **Dual evidence detection** — a learned classifier and a deterministic geometric scorer check each other before confirming a fall
- **Per-camera auto-calibration** — learns expected standing size and floor region at runtime, no manual setup per camera
- **Depth-axis fall coverage** — optional 2D→3D pose lifting catches falls toward/away from the camera that flat 2D angles miss
- **Persistent fall alerts** — MANDOWN state stays active with a duration timer until a sustained recovery streak is seen
- **Resolution adaptive** — overlays scale automatically from 720p to 4K
- **Low false positives** — breakdancing, yoga, sitting on the floor correctly classified as normal
- **JSON event logging** — structured output for integration with alert systems
- **Track recovery** — ROI re-detection and rotation search recover tracks BoT-SORT loses to occlusion

---

## Limitations & Future Work

### Current Limitations
- Trained primarily on indoor/surveillance-style footage — performance may vary in unusual environments
- Requires several seconds of tracking before per-camera calibration and classification are fully confident
- Very small persons (far from camera) may have noisy keypoint estimates
- No web UI yet — `streamlit`/`gradio`/`fastapi` are listed in `requirements.txt` for a planned UI, but `app/` is currently an empty stub; the CLI is the only working entry point

### Future Improvements
- Build out the web UI (Streamlit/Gradio) referenced by the `app/` stub
- Fine-tune on domain-specific data (specific camera angles, environments)
- Deploy as an edge application (an ONNX + `simple_tracker.py` path already exists in `scripts/edge_inference.py`)
- Alert integration (SMS, email, alarm system)

---

## License

This project is for educational and research purposes.

### Dataset Acknowledgments
- [UR Fall Detection Dataset](https://fenix.ur.edu.pl/mkepski/ds/uf.html) — University of Rzeszow
- [Le2i Fall Dataset](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia) — Le2i/IMVIA Lab
- YOLO11-Pose model by [Ultralytics](https://docs.ultralytics.com/)
- BlazePose (MediaPipe) by Google, used for optional 3D pose lifting
