# Baseline Pedestrian Line-Crossing Counter — Design

**Date:** 2026-06-29
**Status:** Approved (design); pending implementation plan
**Scope:** Step 1 of the project roadmap — a working Detection → Tracking → Counting
pipeline validated against UCSD line-crossing ground truth.

## Goal

Detect people in UCSD pedestrian video, track them across frames, count crossings of a
fixed line, and validate the counts against the dataset's line-crossing ground truth.
Produce both a visual artifact (annotated video) and a quantitative one (metrics + plot).

This baseline is the foundation the later pluggable modules attach to (density,
trajectory prediction, anomaly detection), so the pipeline stages must be cleanly
separated and independently testable.

## Non-goals (explicitly deferred)

- Improving detection accuracy on small/low-res pedestrians (upscaling, tiling,
  fine-tuning) — future improvement, designed to slot in behind the `Detector` interface.
- ReID / appearance-based occlusion handling — roadmap step 3.
- Density, trajectory, anomaly modules — later roadmap steps.

## Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Detector | COCO-pretrained YOLO (ultralytics), person class only, used as-is | Fastest honest baseline; accuracy improvements deferred behind the interface |
| Tracker | ByteTrack (via `supervision`) | Motion/IoU only, no appearance model; Kalman state carries velocity (feeds roadmap step 2 for free); ReID deferred to step 3 |
| Counting | `supervision.LineZone` | Standard, matches the project plan |
| Structure | Python package + thin demo notebook + CLI | Package = reusable/testable/importable foundation the later modules need; notebook = portfolio narrative; CLI = headless runs + artifact export |

## Dataset facts (verified locally)

- **Frames:** `ucsd_pedestrian/video/video/vidf/vidfX_33_0NN.y/` are **directories** of 200
  grayscale PNG frames each (`..._fNNN.png`, **238×158**, mode `L`). 32,930 frames total.
- **Counting line** (`line_counting_gt/Line Counting Data/UCSD/lineInfo.mat`):
  vertical line at **x=100**, spanning full height (`lx=[100,100]`, `ly=[1,158]`,
  `theta=0`, ROI = full 238×158 frame).
- **Crossing GT** (`.../UCSD/cgt_s_all.mat`): cell array of **2 directions**, each a
  per-frame crossing-event count over **2000 frames**. Cumulative-sum each to get the
  running ground-truth tally per direction.
- **Open item (resolve at implementation from `Line Counting Data/readme.pdf`):** exact
  mapping of the 2000-frame GT sequence to the `.y` clips (likely ~10 × 200-frame clips
  concatenated). Fallback: validate per-200-frame clip and document the limitation.

## Architecture

### Package layout

```
pedestrian/
  data/
    frames.py      # load a .y clip (or concatenated sequence) -> frame iterator
    gt.py          # load lineInfo.mat + cgt_s_all.mat -> line geometry + per-frame GT
  detect.py        # Detector: frame -> person detections (swappable; YOLO impl)
  track.py         # Tracker: detections -> tracked objects w/ IDs (ByteTrack)
  count.py         # LineCounter: wraps supervision LineZone -> in/out tallies
  validate.py      # compare counter tally vs GT -> metrics + plot
  annotate.py      # draw boxes/IDs/line/tally -> output video
  pipeline.py      # wires the stages together
run.py             # CLI: point at clip(s), choose outputs
notebooks/
  01_baseline_demo.ipynb   # imports pedestrian/, runs a clip, shows video + count-vs-GT plot
tests/
```

### Data flow (per frame)

```
frames.py -> detect.py -> track.py -> count.py -+-> annotate.py -> outputs/out.mp4
                                                +-> validate.py -> outputs/metrics.json + count_vs_gt.png
```

### Pluggable interfaces

Each stage is a small class with a single primary method, so later modules attach
without touching siblings:

- `Detector.detect(frame) -> Detections` — density module later wraps/falls back here.
- `Tracker.update(detections) -> TrackedDetections` — velocity extrapolation reads
  Kalman state here.
- `LineCounter.update(tracked) -> (in_count, out_count)` — anomaly/behavior logic taps
  in here.

(`Detections` / `TrackedDetections` use `supervision`'s detection containers to stay
compatible with its annotators and `LineZone`.)

## Validation method (`validate.py`)

1. **Cumsum** each GT direction cell → two ground-truth cumulative curves.
2. Build the same two cumulative curves from the counter's per-frame tallies.
3. **Direction calibration:** `LineZone`'s `in`/`out` labels depend on line-point order
   and side. Resolve the sign once by selecting the assignment that minimizes error
   against GT, then assert consistency so directions never silently swap.
4. **Metrics:**
   - **Final-count error** per direction: `|pred_total − gt_total|` and % error (headline).
   - **MAE over time** of the cumulative curve (frame-by-frame tracking quality).
   - Combined total across both directions.
5. Write `outputs/metrics.json` and `outputs/count_vs_gt.png` (cumulative pred vs GT,
   two directions, over frames).

## Output artifacts

Written to `outputs/` (gitignored):

- `out.mp4` — frames with boxes, track IDs, the vertical line at x=100, live `in:/out:`
  tally overlay.
- A short **GIF** (subset of frames) for portfolio embedding.
- `metrics.json` + `count_vs_gt.png`.

## Testing (TDD)

The GT provides oracles, enabling test-first development:

- `gt.py` — assert loader returns line geometry `(x=100, full height)` and two arrays
  whose sums equal the known GT totals.
- `count.py` — feed **synthetic tracks** (a box crossing x=100 left→right) and assert
  exactly one `in`, zero `out`; and the reverse. Nails direction logic without YOLO.
- `validate.py` — pred==GT → zero error; known offset → expected error. Pure-function.
- `detect.py` / `track.py` — light smoke tests on a few real frames (shapes/types), not
  accuracy assertions.

Detector/tracker stay swappable behind their interfaces; later modules attach without
touching these tests.

## Tech stack

Python, `ultralytics` (YOLO), `supervision` (ByteTrack + LineZone + annotators),
OpenCV (frame I/O, video write), `scipy` (`.mat` loading), `numpy`, `matplotlib`
(metrics plot), `pytest`.

## Success criteria

- Pipeline runs end-to-end on the UCSD validation sequence and produces all artifacts.
- `metrics.json` reports per-direction and combined crossing-count error vs GT.
- Direction calibration is asserted consistent (no silent in/out swap).
- Core logic (gt, count, validate) covered by passing unit tests.
- The number is an *honest baseline* — accuracy is whatever pretrained YOLO achieves;
  improvement is future work.
