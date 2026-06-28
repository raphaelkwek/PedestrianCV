# Pedestrian Line-Crossing Counter

An extensible computer-vision pipeline that detects, tracks, and counts people
crossing a line in video. Built on the UCSD Pedestrian dataset, with pluggable
modules for crowd-density estimation, trajectory prediction, and anomaly detection.

## Architecture

Core flow: `Video input → Detection → Tracking → Counting`

- **Detection** — YOLO (ultralytics), person class only
- **Tracking** — ByteTrack / DeepSORT (Kalman state includes velocity)
- **Counting** — line-crossing logic (`supervision` `LineZone`)

Pluggable modules attach to a single core stage:

1. **Density model** (→ Detection) — density-map regression fallback for overcrowding
2. **Motion prediction** (→ Tracking) — Kalman velocity extrapolation, later learned trajectories
3. **Anomaly/behavior** (→ Counting/Tracking) — loitering, wrong-direction, sudden stops

## Dataset (UCSD Pedestrian, VISAL Lab / CityU HK)

Not committed to git (~840 MB). Reproduce locally:

```bash
mkdir -p ucsd_pedestrian && cd ucsd_pedestrian
wget http://visal.cs.cityu.edu.hk/static/downloads/ucsdpeds_vidf.zip
wget http://visal.cs.cityu.edu.hk/static/downloads/ucsdpeds_gt.zip
wget http://visal.cs.cityu.edu.hk/static/downloads/LineCountingDataset.zip
unzip ucsdpeds_vidf.zip -d video
unzip ucsdpeds_gt.zip -d ground_truth
unzip LineCountingDataset.zip -d line_counting_gt
```

### Layout & formats (verified locally)

- `video/video/vidf/vidfX_33_0NN.y/` — **directories** of 200 grayscale PNG frames each
  (`..._fNNN.png`, **238×158**, mode `L`). 32,930 frames total.
- `video/segm/vidf/*.segm` — foreground/background segmentation masks.
- `ground_truth/gt/{vidd,vidf}/*_{frame_full,people_full,count_4K_roi}.mat` —
  dot annotations / crowd counts (for the density module).
- `line_counting_gt/Line Counting Data/{UCSD,Grand Central,LHI}/` —
  line-crossing GT: `lineInfo.mat` (line geometry), `cgt_s_all.mat` (crossing counts).

See `README-vids.pdf` and `README-gt.pdf` in the dataset for full annotation specs.

## Build order

1. Baseline counter (Detection + Tracking + Counting), validated vs UCSD line-crossing GT
2. Kalman velocity extrapolation
3. ReID-based occlusion handling
4. Density-map fallback for overcrowding
5. Learned trajectory prediction
6. Anomaly/behavior layer
7. Forecasting + dashboard

## Citation

Chan & Vasconcelos, UCSD Pedestrian dataset (VISAL Lab, City University of Hong Kong).
