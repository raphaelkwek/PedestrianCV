# Baseline Pedestrian Line-Crossing Counter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Detection → Tracking → Counting pipeline that counts people crossing a fixed line in UCSD pedestrian video and validates the counts against the dataset's line-crossing ground truth, producing an annotated video plus a metrics report and plot.

**Architecture:** A `pedestrian/` Python package with one focused module per pipeline stage (frame loading, GT loading, detection, tracking, counting, validation, annotation) wired together by `pipeline.py`. Each stage is a small class behind a stable interface so later roadmap modules (density, trajectory, anomaly) attach without touching siblings. A `run.py` CLI drives headless runs and artifact export; a demo notebook imports the package for the portfolio narrative.

**Tech Stack:** Python 3.13, ultralytics (YOLO), supervision (ByteTrack + LineZone + annotators), OpenCV, scipy (.mat loading), numpy, matplotlib, imageio (GIF), pytest.

---

## Resolved dataset facts (verified against the data)

- **Validation sequence:** UCSD vidf1 clips `000`–`009`, concatenated in order = exactly **2000 frames** (each clip is a directory of 200 PNGs at `ucsd_pedestrian/video/video/vidf/vidf1_33_00N.y/vidf1_33_00N_fNNN.png`, grayscale 238×158, mode L). Per the dataset readme, GT index 000 = frames 1–2000.
- **Counting line** (`ucsd_pedestrian/line_counting_gt/Line Counting Data/UCSD/lineInfo.mat`): vertical, `lx=[100,100]`, `ly=[1,158]`, theta=0, ROI = full 238×158 frame → line from (100,1) to (100,158).
- **Crossing GT** (`.../UCSD/cgt_s_all.mat`): a 1×2 cell array (two crossing directions), each a per-frame instantaneous crossing count over 2000 frames. Verified totals: **dir0 = 89**, **dir1 = 61** crossings; max 2 per frame.

These constants are used directly in tests as oracles. Paths below are relative to the repo root `/Users/raphaelkwek/Documents/GitHub/Pedestrian`.

---

## File structure

```
pedestrian/
  __init__.py
  paths.py         # dataset path helpers + default clip sequence
  data/
    __init__.py
    gt.py          # load_line(), load_crossing_gt()
    frames.py      # FrameSequence (iterate frames across clips)
  detect.py        # Detector (YOLO wrapper -> sv.Detections)
  track.py         # Tracker (ByteTrack wrapper)
  count.py         # LineCounter (supervision LineZone wrapper)
  validate.py      # calibrate_direction(), score() -> metrics; plot
  annotate.py      # Annotator (draw boxes/ids/line/tally)
  pipeline.py      # run_pipeline() wiring all stages
run.py             # CLI entry point
notebooks/
  01_baseline_demo.ipynb
tests/
  test_gt.py
  test_frames.py
  test_count.py
  test_validate.py
  test_detect_track_smoke.py
requirements.txt
pytest.ini
```

---

## Task 0: Environment & dependencies

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`

- [ ] **Step 1: Write `requirements.txt`**

```
ultralytics>=8.3.0
supervision>=0.25.0
opencv-python>=4.10.0
scipy>=1.13
numpy>=1.26
matplotlib>=3.8
imageio>=2.34
pytest>=8.0
```

- [ ] **Step 2: Write `pytest.ini`** (so `pedestrian` imports from repo root)

```ini
[pytest]
testpaths = tests
addopts = -v
```

- [ ] **Step 3: Create a virtualenv and install**

Run:
```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```
Expected: installs complete without error. (Torch is pulled in by ultralytics.)

- [ ] **Step 4: Verify imports**

Run:
```bash
.venv/bin/python -c "import ultralytics, supervision, cv2, scipy, numpy, matplotlib, imageio; print('ok')"
```
Expected: prints `ok`.

> Note: `.venv/` is already covered by `.gitignore` (`venv/`/`.venv/`). All subsequent commands use `.venv/bin/python` and `.venv/bin/pytest`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pytest.ini
git commit -m "chore: add dependencies and pytest config"
```

---

## Task 1: Dataset path helpers

**Files:**
- Create: `pedestrian/__init__.py` (empty)
- Create: `pedestrian/paths.py`
- Test: `tests/test_frames.py` (path portion)

- [ ] **Step 1: Write the failing test** (`tests/test_frames.py`)

```python
from pathlib import Path
from pedestrian.paths import DATA_ROOT, vidf1_sequence, UCSD_LINE_DIR


def test_vidf1_sequence_is_ten_existing_clip_dirs():
    clips = vidf1_sequence()
    assert len(clips) == 10
    assert [c.name for c in clips] == [f"vidf1_33_{i:03d}.y" for i in range(10)]
    for c in clips:
        assert c.is_dir(), f"missing clip dir: {c}"


def test_ucsd_line_dir_has_mat_files():
    assert (UCSD_LINE_DIR / "lineInfo.mat").exists()
    assert (UCSD_LINE_DIR / "cgt_s_all.mat").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_frames.py -k sequence_is_ten -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pedestrian'`.

- [ ] **Step 3: Write minimal implementation**

Create `pedestrian/__init__.py` (empty file).

Create `pedestrian/paths.py`:
```python
"""Dataset path helpers for the UCSD pedestrian data."""
from pathlib import Path

# Repo root = two levels up from this file (pedestrian/paths.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "ucsd_pedestrian"

VIDF_DIR = DATA_ROOT / "video" / "video" / "vidf"
UCSD_LINE_DIR = DATA_ROOT / "line_counting_gt" / "Line Counting Data" / "UCSD"


def vidf1_sequence() -> list[Path]:
    """The 10 vidf1 clip directories (000-009) = the 2000-frame GT sequence."""
    return [VIDF_DIR / f"vidf1_33_{i:03d}.y" for i in range(10)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_frames.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pedestrian/__init__.py pedestrian/paths.py tests/test_frames.py
git commit -m "feat: dataset path helpers and vidf1 sequence"
```

---

## Task 2: Frame loader

**Files:**
- Create: `pedestrian/data/__init__.py` (empty)
- Create: `pedestrian/data/frames.py`
- Test: `tests/test_frames.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_frames.py`)

```python
import numpy as np
from pedestrian.data.frames import FrameSequence
from pedestrian.paths import vidf1_sequence


def test_frame_sequence_length_and_shape():
    seq = FrameSequence(vidf1_sequence())
    assert len(seq) == 2000
    idx, frame = next(iter(seq))
    assert idx == 0
    assert frame.shape == (158, 238)
    assert frame.dtype == np.uint8


def test_frame_sequence_indices_are_contiguous():
    seq = FrameSequence(vidf1_sequence())
    indices = [idx for idx, _ in seq]
    assert indices == list(range(2000))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_frames.py -k frame_sequence -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pedestrian.data.frames'`.

- [ ] **Step 3: Write minimal implementation**

Create `pedestrian/data/__init__.py` (empty file).

Create `pedestrian/data/frames.py`:
```python
"""Iterate grayscale frames across one or more UCSD .y clip directories."""
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class FrameSequence:
    """Yields (global_frame_index, grayscale_frame) across the given clip dirs.

    Each clip dir contains PNG frames named ..._fNNN.png; frames are concatenated
    in clip order, then frame-number order within each clip.
    """

    def __init__(self, clip_dirs: list[Path]):
        self._paths: list[Path] = []
        for clip in clip_dirs:
            pngs = sorted(Path(clip).glob("*.png"))
            if not pngs:
                raise FileNotFoundError(f"no PNG frames in {clip}")
            self._paths.extend(pngs)

    def __len__(self) -> int:
        return len(self._paths)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        for idx, path in enumerate(self._paths):
            frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                raise IOError(f"failed to read {path}")
            yield idx, frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_frames.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pedestrian/data/__init__.py pedestrian/data/frames.py tests/test_frames.py
git commit -m "feat: FrameSequence loader over UCSD clip dirs"
```

---

## Task 3: Ground-truth loader

**Files:**
- Create: `pedestrian/data/gt.py`
- Test: `tests/test_gt.py`

- [ ] **Step 1: Write the failing test** (`tests/test_gt.py`)

```python
import numpy as np
from pedestrian.data.gt import load_line, load_crossing_gt
from pedestrian.paths import UCSD_LINE_DIR


def test_load_line_is_vertical_at_x100():
    start, end = load_line(UCSD_LINE_DIR / "lineInfo.mat")
    assert start[0] == 100 and end[0] == 100      # vertical line at x=100
    assert {start[1], end[1]} == {1, 158}         # spans full height


def test_load_crossing_gt_totals_match_known_values():
    dir0, dir1 = load_crossing_gt(UCSD_LINE_DIR / "cgt_s_all.mat")
    assert dir0.shape == (2000,) and dir1.shape == (2000,)
    assert int(dir0.sum()) == 89
    assert int(dir1.sum()) == 61
    assert dir0.dtype == np.int64 or dir0.dtype == np.int32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_gt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pedestrian.data.gt'`.

- [ ] **Step 3: Write minimal implementation**

Create `pedestrian/data/gt.py`:
```python
"""Load the UCSD line-counting ground truth (line geometry + per-frame crossings)."""
from pathlib import Path

import numpy as np
import scipy.io as sio


def load_line(path: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ((x1, y1), (x2, y2)) line endpoints from lineInfo.mat."""
    li = sio.loadmat(str(path))["lineInfo"][0, 0]
    lx = np.asarray(li["lx"]).ravel()
    ly = np.asarray(li["ly"]).ravel()
    start = (int(lx[0]), int(ly[0]))
    end = (int(lx[1]), int(ly[1]))
    return start, end


def load_crossing_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (dir0, dir1) per-frame crossing-count arrays from cgt_s_all.mat."""
    cell = sio.loadmat(str(path))["cgt_s_all"]
    dir0 = np.asarray(cell[0, 0]).ravel().astype(np.int64)
    dir1 = np.asarray(cell[0, 1]).ravel().astype(np.int64)
    return dir0, dir1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_gt.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pedestrian/data/gt.py tests/test_gt.py
git commit -m "feat: GT loader for line geometry and crossing counts"
```

---

## Task 4: Line counter

**Files:**
- Create: `pedestrian/count.py`
- Test: `tests/test_count.py`

- [ ] **Step 1: Write the failing test** (`tests/test_count.py`)

```python
import numpy as np
import supervision as sv
from pedestrian.count import LineCounter


def _box_track(cx: float, tracker_id: int) -> sv.Detections:
    """A single tracked box centered horizontally at cx, around y=80."""
    xyxy = np.array([[cx - 10, 70, cx + 10, 90]], dtype=float)
    det = sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    det.tracker_id = np.array([tracker_id])
    return det


def test_left_to_right_crossing_counts_once():
    lc = LineCounter(start=(100, 0), end=(100, 158))
    i1, o1 = lc.update(_box_track(80, 1))    # fully left of x=100
    i2, o2 = lc.update(_box_track(120, 1))   # fully right of x=100
    assert (i1 + i2) + (o1 + o2) == 1        # exactly one crossing
    assert {i1 + i2, o1 + o2} == {1, 0}      # in exactly one direction


def test_right_to_left_crosses_opposite_direction():
    lc_lr = LineCounter(start=(100, 0), end=(100, 158))
    lc_lr.update(_box_track(80, 1))
    in_lr, out_lr = lc_lr.update(_box_track(120, 1))

    lc_rl = LineCounter(start=(100, 0), end=(100, 158))
    lc_rl.update(_box_track(120, 2))
    in_rl, out_rl = lc_rl.update(_box_track(80, 2))

    # Whatever supervision calls "in" for L->R must be "out" for R->L.
    assert (in_lr, out_lr) == (out_rl, in_rl)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_count.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pedestrian.count'`.

- [ ] **Step 3: Write minimal implementation**

Create `pedestrian/count.py`:
```python
"""Line-crossing counter built on supervision's LineZone."""
import supervision as sv


class LineCounter:
    """Counts tracked detections crossing a line, per direction.

    `update` returns the per-frame (in_delta, out_delta). Cumulative totals are
    available via `in_count` / `out_count`.
    """

    def __init__(self, start: tuple[int, int], end: tuple[int, int]):
        self.line_zone = sv.LineZone(
            start=sv.Point(*start),
            end=sv.Point(*end),
        )

    def update(self, tracked: sv.Detections) -> tuple[int, int]:
        crossed_in, crossed_out = self.line_zone.trigger(tracked)
        return int(crossed_in.sum()), int(crossed_out.sum())

    @property
    def in_count(self) -> int:
        return self.line_zone.in_count

    @property
    def out_count(self) -> int:
        return self.line_zone.out_count
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_count.py -v`
Expected: both tests PASS.

> If `LineZone.trigger` requires `tracker_id` and raises on its absence, the test already supplies it. If supervision's default triggering anchors do not all cross for this synthetic box, widen the move (e.g. cx 60 → 140); the box half-width is 10 so 60/140 puts all four corners clearly on opposite sides.

- [ ] **Step 5: Commit**

```bash
git add pedestrian/count.py tests/test_count.py
git commit -m "feat: LineCounter wrapping supervision LineZone"
```

---

## Task 5: Validator (direction calibration + metrics)

**Files:**
- Create: `pedestrian/validate.py`
- Test: `tests/test_validate.py`

- [ ] **Step 1: Write the failing test** (`tests/test_validate.py`)

```python
import numpy as np
from pedestrian.validate import calibrate_direction, score


def test_calibrate_direction_picks_lower_error_mapping():
    # pred_in matches gt1, pred_out matches gt0 -> swapped mapping is correct.
    pred_in = np.array([0, 1, 0, 1])
    pred_out = np.array([1, 0, 0, 0])
    gt0 = np.array([1, 0, 0, 0])
    gt1 = np.array([0, 1, 0, 1])
    swapped = calibrate_direction(pred_in, pred_out, gt0, gt1)
    assert swapped is True


def test_score_zero_error_when_pred_equals_gt():
    gt0 = np.array([0, 1, 0, 2, 0])
    gt1 = np.array([1, 0, 0, 0, 1])
    m = score(pred_in=gt0, pred_out=gt1, gt0=gt0, gt1=gt1)
    assert m["dir0_abs_error"] == 0
    assert m["dir1_abs_error"] == 0
    assert m["total_abs_error"] == 0
    assert m["dir0_mae_over_time"] == 0.0
    assert m["swapped"] is False


def test_score_reports_known_offset():
    gt0 = np.array([0, 1, 1, 0])   # total 2
    gt1 = np.array([0, 0, 0, 0])   # total 0
    pred_in = np.array([0, 1, 0, 0])   # total 1 -> off by 1
    pred_out = np.array([0, 0, 0, 0])  # total 0
    m = score(pred_in=pred_in, pred_out=pred_out, gt0=gt0, gt1=gt1)
    assert m["dir0_abs_error"] == 1
    assert m["dir1_abs_error"] == 0
    assert m["total_abs_error"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pedestrian.validate'`.

- [ ] **Step 3: Write minimal implementation**

Create `pedestrian/validate.py`:
```python
"""Compare counter output against UCSD crossing GT: calibration, metrics, plot."""
from pathlib import Path

import numpy as np


def calibrate_direction(pred_in, pred_out, gt0, gt1) -> bool:
    """Return True if (pred_in,pred_out) should map to (gt1,gt0) instead of (gt0,gt1).

    Chooses the assignment minimizing total final-count error so the counter's
    in/out labels line up with the GT direction cells.
    """
    pred_in, pred_out = np.asarray(pred_in), np.asarray(pred_out)
    gt0, gt1 = np.asarray(gt0), np.asarray(gt1)
    direct = abs(pred_in.sum() - gt0.sum()) + abs(pred_out.sum() - gt1.sum())
    swap = abs(pred_in.sum() - gt1.sum()) + abs(pred_out.sum() - gt0.sum())
    return swap < direct


def score(pred_in, pred_out, gt0, gt1) -> dict:
    """Metrics for the counter vs GT, after direction calibration."""
    pred_in = np.asarray(pred_in)
    pred_out = np.asarray(pred_out)
    gt0 = np.asarray(gt0)
    gt1 = np.asarray(gt1)

    swapped = calibrate_direction(pred_in, pred_out, gt0, gt1)
    p0, p1 = (pred_out, pred_in) if swapped else (pred_in, pred_out)

    cum_p0, cum_g0 = np.cumsum(p0), np.cumsum(gt0)
    cum_p1, cum_g1 = np.cumsum(p1), np.cumsum(gt1)

    dir0_err = int(abs(p0.sum() - gt0.sum()))
    dir1_err = int(abs(p1.sum() - gt1.sum()))
    return {
        "swapped": bool(swapped),
        "dir0_pred_total": int(p0.sum()),
        "dir1_pred_total": int(p1.sum()),
        "dir0_gt_total": int(gt0.sum()),
        "dir1_gt_total": int(gt1.sum()),
        "dir0_abs_error": dir0_err,
        "dir1_abs_error": dir1_err,
        "total_abs_error": dir0_err + dir1_err,
        "dir0_mae_over_time": float(np.mean(np.abs(cum_p0 - cum_g0))),
        "dir1_mae_over_time": float(np.mean(np.abs(cum_p1 - cum_g1))),
    }


def plot_cumulative(pred_in, pred_out, gt0, gt1, out_path: Path) -> None:
    """Save a cumulative pred-vs-GT plot for both directions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    swapped = calibrate_direction(pred_in, pred_out, gt0, gt1)
    p0, p1 = (pred_out, pred_in) if swapped else (pred_in, pred_out)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, p, g, title in (
        (axes[0], p0, gt0, "Direction 0"),
        (axes[1], p1, gt1, "Direction 1"),
    ):
        ax.plot(np.cumsum(g), label="GT", linewidth=2)
        ax.plot(np.cumsum(p), label="Predicted", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("frame")
        ax.set_ylabel("cumulative crossings")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_validate.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pedestrian/validate.py tests/test_validate.py
git commit -m "feat: validator with direction calibration and metrics"
```

---

## Task 6: Detector (YOLO wrapper)

**Files:**
- Create: `pedestrian/detect.py`
- Test: `tests/test_detect_track_smoke.py`

> Network note: the first construction of `Detector` downloads the YOLO weights (~6 MB for `yolov8n.pt`). This requires internet and is cached afterward. The smoke test below is marked so it skips cleanly if weights cannot be obtained offline.

- [ ] **Step 1: Write the failing test** (`tests/test_detect_track_smoke.py`)

```python
import numpy as np
import pytest
import supervision as sv


def _make_detector():
    from pedestrian.detect import Detector
    try:
        return Detector()
    except Exception as e:  # weights unavailable offline
        pytest.skip(f"YOLO weights unavailable: {e}")


def test_detector_returns_supervision_detections():
    detector = _make_detector()
    frame = np.zeros((158, 238), dtype=np.uint8)   # blank grayscale frame
    dets = detector.detect(frame)
    assert isinstance(dets, sv.Detections)          # may be empty on a blank frame
    if len(dets) > 0:
        assert set(np.unique(dets.class_id)) <= {0}  # person class only
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_detect_track_smoke.py -k detector -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pedestrian.detect'`.

- [ ] **Step 3: Write minimal implementation**

Create `pedestrian/detect.py`:
```python
"""Person detector wrapping ultralytics YOLO; returns supervision Detections."""
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO person class


class Detector:
    """COCO-pretrained YOLO restricted to the person class.

    Swappable: later density/upscaling modules can subclass or replace this,
    keeping the `detect(frame_gray) -> sv.Detections` interface.
    """

    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.25):
        self.model = YOLO(model)
        self.conf = conf

    def detect(self, frame_gray: np.ndarray) -> sv.Detections:
        bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        result = self.model(
            bgr, conf=self.conf, classes=[PERSON_CLASS_ID], verbose=False
        )[0]
        return sv.Detections.from_ultralytics(result)
```

- [ ] **Step 4: Run test to verify it passes (or skips offline)**

Run: `.venv/bin/pytest tests/test_detect_track_smoke.py -k detector -v`
Expected: PASS (downloads weights on first run) or SKIP if offline.

- [ ] **Step 5: Commit**

```bash
git add pedestrian/detect.py tests/test_detect_track_smoke.py
git commit -m "feat: YOLO person detector returning supervision Detections"
```

---

## Task 7: Tracker (ByteTrack wrapper)

**Files:**
- Create: `pedestrian/track.py`
- Test: `tests/test_detect_track_smoke.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_detect_track_smoke.py`)

```python
def test_tracker_assigns_ids_to_detections():
    from pedestrian.track import Tracker
    tracker = Tracker()
    # Two frames of the same box; ByteTrack should assign a tracker_id.
    xyxy = np.array([[50, 50, 70, 90]], dtype=float)
    dets = sv.Detections(
        xyxy=xyxy, confidence=np.array([0.9]), class_id=np.array([0])
    )
    tracked = tracker.update(dets)
    tracked = tracker.update(dets)
    assert tracked.tracker_id is not None
    assert len(tracked.tracker_id) == len(tracked)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_detect_track_smoke.py -k tracker -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pedestrian.track'`.

- [ ] **Step 3: Write minimal implementation**

Create `pedestrian/track.py`:
```python
"""Multi-object tracker wrapping supervision's ByteTrack."""
import supervision as sv


class Tracker:
    """ByteTrack wrapper. Kalman state carries velocity (used by later modules)."""

    def __init__(self):
        self.byte_track = sv.ByteTrack()

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.byte_track.update_with_detections(detections)

    def reset(self) -> None:
        self.byte_track.reset()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_detect_track_smoke.py -k tracker -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pedestrian/track.py tests/test_detect_track_smoke.py
git commit -m "feat: ByteTrack tracker wrapper"
```

---

## Task 8: Annotator

**Files:**
- Create: `pedestrian/annotate.py`
- Test: `tests/test_detect_track_smoke.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_detect_track_smoke.py`)

```python
def test_annotator_returns_bgr_frame_same_size():
    from pedestrian.annotate import Annotator
    from pedestrian.count import LineCounter

    lc = LineCounter(start=(100, 0), end=(100, 158))
    annotator = Annotator()
    frame_bgr = np.zeros((158, 238, 3), dtype=np.uint8)
    dets = sv.Detections(
        xyxy=np.array([[40, 60, 60, 100]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    dets.tracker_id = np.array([7])
    out = annotator.draw(frame_bgr, dets, lc)
    assert out.shape == (158, 238, 3)
    assert out.dtype == np.uint8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_detect_track_smoke.py -k annotator -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pedestrian.annotate'`.

- [ ] **Step 3: Write minimal implementation**

Create `pedestrian/annotate.py`:
```python
"""Draw boxes, track IDs, the counting line, and the running tally onto frames."""
import numpy as np
import supervision as sv

from pedestrian.count import LineCounter


class Annotator:
    """Renders detections + line + tally for the output video."""

    def __init__(self):
        self.box = sv.BoxAnnotator()
        self.label = sv.LabelAnnotator()
        self.line = sv.LineZoneAnnotator()

    def draw(
        self,
        frame_bgr: np.ndarray,
        tracked: sv.Detections,
        counter: LineCounter,
    ) -> np.ndarray:
        out = frame_bgr.copy()
        out = self.box.annotate(out, tracked)
        if tracked.tracker_id is not None:
            labels = [f"#{tid}" for tid in tracked.tracker_id]
            out = self.label.annotate(out, tracked, labels=labels)
        out = self.line.annotate(out, line_counter=counter.line_zone)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_detect_track_smoke.py -k annotator -v`
Expected: PASS.

> If the installed supervision version uses a different annotate signature (e.g. `scene=`/`detections=` keywords, or `LineZoneAnnotator.annotate(frame, line_counter)` positional), adjust the calls to match the installed version's API; the test only asserts shape/dtype so it guards the contract.

- [ ] **Step 5: Commit**

```bash
git add pedestrian/annotate.py tests/test_detect_track_smoke.py
git commit -m "feat: frame annotator for boxes, ids, line, tally"
```

---

## Task 9: Pipeline wiring

**Files:**
- Create: `pedestrian/pipeline.py`
- Test: `tests/test_detect_track_smoke.py` (append — short integration over a few frames)

- [ ] **Step 1: Write the failing test** (append to `tests/test_detect_track_smoke.py`)

```python
def test_pipeline_runs_a_few_frames_and_returns_arrays():
    import pytest
    from pedestrian.pipeline import run_pipeline
    from pedestrian.paths import vidf1_sequence, UCSD_LINE_DIR

    try:
        result = run_pipeline(
            clip_dirs=vidf1_sequence(),
            line_mat=UCSD_LINE_DIR / "lineInfo.mat",
            max_frames=5,
            write_video=False,
        )
    except Exception as e:
        if "weights" in str(e).lower() or "download" in str(e).lower():
            pytest.skip(f"YOLO weights unavailable: {e}")
        raise
    assert len(result["per_frame_in"]) == 5
    assert len(result["per_frame_out"]) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_detect_track_smoke.py -k pipeline -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pedestrian.pipeline'`.

- [ ] **Step 3: Write minimal implementation**

Create `pedestrian/pipeline.py`:
```python
"""Wire detection -> tracking -> counting over a frame sequence."""
from pathlib import Path

import cv2
import numpy as np

from pedestrian.annotate import Annotator
from pedestrian.count import LineCounter
from pedestrian.data.frames import FrameSequence
from pedestrian.data.gt import load_line
from pedestrian.detect import Detector
from pedestrian.track import Tracker


def run_pipeline(
    clip_dirs: list[Path],
    line_mat: Path,
    max_frames: int | None = None,
    write_video: bool = False,
    video_path: Path | None = None,
    model: str = "yolov8n.pt",
    conf: float = 0.25,
) -> dict:
    """Run the baseline pipeline; return per-frame in/out deltas and totals."""
    start, end = load_line(line_mat)
    frames = FrameSequence(clip_dirs)
    detector = Detector(model=model, conf=conf)
    tracker = Tracker()
    counter = LineCounter(start=start, end=end)
    annotator = Annotator() if write_video else None

    writer = None
    per_frame_in: list[int] = []
    per_frame_out: list[int] = []

    for idx, frame_gray in frames:
        if max_frames is not None and idx >= max_frames:
            break
        detections = detector.detect(frame_gray)
        tracked = tracker.update(detections)
        in_delta, out_delta = counter.update(tracked)
        per_frame_in.append(in_delta)
        per_frame_out.append(out_delta)

        if write_video:
            bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            drawn = annotator.draw(bgr, tracked, counter)
            if writer is None:
                h, w = drawn.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    (w, h),
                )
            writer.write(drawn)

    if writer is not None:
        writer.release()

    return {
        "per_frame_in": np.array(per_frame_in),
        "per_frame_out": np.array(per_frame_out),
        "in_total": counter.in_count,
        "out_total": counter.out_count,
    }
```

- [ ] **Step 4: Run test to verify it passes (or skips offline)**

Run: `.venv/bin/pytest tests/test_detect_track_smoke.py -k pipeline -v`
Expected: PASS (after weights download) or SKIP if offline.

- [ ] **Step 5: Commit**

```bash
git add pedestrian/pipeline.py tests/test_detect_track_smoke.py
git commit -m "feat: pipeline wiring detect->track->count"
```

---

## Task 10: CLI entry point

**Files:**
- Create: `run.py`

- [ ] **Step 1: Write `run.py`**

```python
"""CLI: run the baseline counter on the UCSD vidf1 sequence and export artifacts."""
import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio

from pedestrian.data.gt import load_crossing_gt
from pedestrian.paths import UCSD_LINE_DIR, vidf1_sequence
from pedestrian.pipeline import run_pipeline
from pedestrian.validate import plot_cumulative, score


def export_gif(mp4_path: Path, gif_path: Path, max_frames: int = 150) -> None:
    """Write a short GIF preview from the first frames of the output MP4."""
    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if frames:
        imageio.mimsave(gif_path, frames, fps=10, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="UCSD baseline line-crossing counter")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_video = not args.no_video

    result = run_pipeline(
        clip_dirs=vidf1_sequence(),
        line_mat=UCSD_LINE_DIR / "lineInfo.mat",
        max_frames=args.max_frames,
        write_video=write_video,
        video_path=args.out_dir / "out.mp4",
        model=args.model,
        conf=args.conf,
    )

    gt0, gt1 = load_crossing_gt(UCSD_LINE_DIR / "cgt_s_all.mat")
    n = len(result["per_frame_in"])
    gt0, gt1 = gt0[:n], gt1[:n]

    metrics = score(
        pred_in=result["per_frame_in"],
        pred_out=result["per_frame_out"],
        gt0=gt0,
        gt1=gt1,
    )
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_cumulative(
        result["per_frame_in"], result["per_frame_out"], gt0, gt1,
        args.out_dir / "count_vs_gt.png",
    )
    if write_video and not args.no_gif:
        export_gif(args.out_dir / "out.mp4", args.out_dir / "preview.gif")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to {args.out_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the CLI on a small slice**

Run:
```bash
.venv/bin/python run.py --max-frames 50 --out-dir outputs
```
Expected: prints a metrics JSON block and `Artifacts written to outputs/`; creates `outputs/metrics.json`, `outputs/count_vs_gt.png`, and (unless `--no-video`) `outputs/out.mp4`. Requires internet on first run for weights.

- [ ] **Step 3: Run the full sequence**

Run:
```bash
.venv/bin/python run.py --out-dir outputs
```
Expected: processes all 2000 frames; `metrics.json` reports `dir*_gt_total` of 89 and 61 and the predicted totals / errors. (Pretrained YOLO on tiny low-res pedestrians will likely undercount — that is the honest baseline.)

- [ ] **Step 4: Commit**

```bash
git add run.py
git commit -m "feat: CLI for baseline counter with metrics and plot export"
```

---

## Task 11: Demo notebook

**Files:**
- Create: `notebooks/01_baseline_demo.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/01_baseline_demo.ipynb` with these cells (use `.venv` as the kernel, or `.venv/bin/jupyter`):

Cell 1 (markdown):
```markdown
# UCSD Pedestrian Line-Crossing Counter — Baseline Demo
Detection (YOLO) → Tracking (ByteTrack) → Line counting, validated vs UCSD GT.
```

Cell 2 (code):
```python
from pedestrian.paths import vidf1_sequence, UCSD_LINE_DIR
from pedestrian.pipeline import run_pipeline
from pedestrian.data.gt import load_crossing_gt
from pedestrian.validate import score, plot_cumulative

result = run_pipeline(
    clip_dirs=vidf1_sequence(),
    line_mat=UCSD_LINE_DIR / "lineInfo.mat",
    max_frames=300,        # keep the demo quick; drop for the full run
    write_video=False,
)
result["in_total"], result["out_total"]
```

Cell 3 (code):
```python
gt0, gt1 = load_crossing_gt(UCSD_LINE_DIR / "cgt_s_all.mat")
n = len(result["per_frame_in"])
metrics = score(result["per_frame_in"], result["per_frame_out"], gt0[:n], gt1[:n])
metrics
```

Cell 4 (code):
```python
from pathlib import Path
plot_cumulative(result["per_frame_in"], result["per_frame_out"], gt0[:n], gt1[:n],
                Path("count_vs_gt_demo.png"))
from IPython.display import Image
Image("count_vs_gt_demo.png")
```

- [ ] **Step 2: Run the notebook top to bottom**

Run:
```bash
.venv/bin/pip install jupyter
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_baseline_demo.ipynb
```
Expected: executes without error; the plot cell renders.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_baseline_demo.ipynb
git commit -m "docs: baseline demo notebook"
```

---

## Task 12: Final verification & push

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/pytest -v`
Expected: all non-network tests PASS; detector/tracker/pipeline tests PASS (weights cached) or SKIP (offline). No failures.

- [ ] **Step 2: Confirm artifacts exist**

Run: `ls -la outputs/`
Expected: `metrics.json`, `count_vs_gt.png`, `out.mp4`, `preview.gif` present.

- [ ] **Step 3: Push to GitHub**

Run: `git push origin main`
Expected: all task commits land on `https://github.com/raphaelkwek/PedestrianCV`.

---

## Self-review notes (for the implementer)

- **supervision API drift is the main risk.** `LineZone.trigger` return shape, `*Annotator.annotate` signatures, and `Detections` construction vary across supervision versions. Tasks 4, 8 include fallback notes; if a smoke test fails on an API mismatch, adapt the call to the installed version — the tests assert behavior/shape, not internal API.
- **Network dependency:** YOLO weights download on first `Detector()`. The detector/tracker/pipeline tests skip gracefully offline; the pure-logic tests (gt, frames, count, validate) are the real TDD oracles and need no network.
- **Honest baseline expectation:** pretrained YOLO on 238×158 grayscale far-field pedestrians will likely undercount vs GT totals (89 / 61). That is expected and is the baseline the roadmap's next steps (upscaling, fine-tuning, density fallback) improve on.
