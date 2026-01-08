# ADAS LDW PoC (Nuisance Alert Thesis)

Offline Proof-of-Concept for analyzing Lane Departure Warning (LDW) nuisance alerts. It combines precomputed lane risk, driver attention, road context, and biometrics into a single view to decide whether to warn or suppress.

This repo focuses on the PoC experience and UI. The core algorithm is based on my separate repo [adas-ldw-cli](https://github.com/ddepasquali/adas-ldw-cli) (algorithm-only pipeline). This repo contains a synced copy of the pipeline for convenience, but treat [adas-ldw-cli](https://github.com/ddepasquali/adas-ldw-cli) as the source of truth for the algorithm.

## Status and scope
- Research PoC, offline only (no real-time inference).
- UI is read-only: START loads existing data; RESET reloads the page.
- Designed around a single recorded session; alignment and thresholds are session-specific.
- No cloud services or external APIs.

## Architecture (end-to-end)
1) **Capture**: raw sensors + road/driver videos.  
2) **Process ([adas-ldw-cli](https://github.com/ddepasquali/adas-ldw-cli))**: extract features, fuse signals, generate `lane_events.csv`, render annotated videos.  
3) **Inspect (this UI)**: synchronized timeline, charts, widgets, and decision log.

## Inputs (raw)
Place raw data in `data/raw/` (or set paths via CLI):
- `scenario01-2026-01-05--16-23-30-muse-legacy.csv`
- `scenario01-2026-01-05--16-23-48-obd-legacy.csv`
- `scenario01-2026-01-05--16-23-00-phyphox-legacy.csv`
- `scenario01-2026-01-05--16-23-15-polar-full.csv`
- `muse_baseline.csv`, `polar_baseline.csv`
- `road.mp4`, `driver.mp4`

## Outputs (used by the UI)
Generated in `data/feat/` by the pipeline:
- `muse_features.csv`, `obd_features.csv`, `phyphox_features.csv`, `polar_features.csv`
- `fused_10hz.csv`
- `lane_events.csv` (should_warn, reason_code, warning_strength, confidence)
- `road_annotated.mp4`, `driver_annotated.mp4`
- `baseline_summary.json`, `run_log.txt`

The UI loads videos from `data/converted_videos/` (preferred) and falls back to `data/feat/` if needed. Use H.264 versions when possible.

## UI behavior (PoC)
- **START** only loads existing files (it does not run the algorithm).
- **God mode**: enable/disable START in code (`state.godMode` in `ui/app.js`).
- Timeline is **anchored to video duration**; charts and log use the video window.
- Road/Driver flag toggles are **UI-only**; overlays are pre-rendered in the videos.
- Charts are resampled at **10 Hz** and switchable via chip selectors.

## Chart modes (current)
- **ECG (Polar)**: HRV, Quality  
- **EEG (Muse)**: Bands, Ratios  
- **OBD**: Powertrain, Load  
- **Phyphox**: Acceleration, Gyro

## Decision logic (high level)
- **WARN** when lane risk + low attention/distracted context.
- **SUPPRESS** when attention and context reduce risk.
- Biometrics are supportive (z-score normalized), not primary triggers.

See **adas-ldw-cli** for the full algorithm and rules.

## Models (pipeline)
Available under `models/`:
- **YOLOv8n** (`models/yolov8n.pt`) for object detection.
- **YOLOP** (`models/yolop-640-640.onnx`) for lane segmentation (recommended).

Optional (supported by the pipeline, not included here):
- **UFLD** ONNX model if you prefer a lane-only approach.

Driver attention uses **MediaPipe** (face mesh, gaze/pose cues).

## Simulations and PoC constraints
- START does not trigger computation; it only reads already produced files.
- The log is derived from `lane_events.csv` (not a live detector).
- The UI does not change overlays; video annotations are fixed offline.
- Some signals can be null; smoothing is visual only.
- BPM is not present in feature outputs, so it is not exposed in the UI.

## Limitations
- Tuned for a specific session; generalization is unverified.
- Requires accurate time alignment (`anchor_obd_time`, `anchor_*_video_s`).
- Not tested on a wide range of devices/sensors.
- No streaming or multi-session support.

## Design notes
- Single-page UI with fast scanability and dense telemetry.
- Emphasis on **context** around a warning, not just the warning itself.
- Real-time UX deferred by design (future work).

## Run the pipeline (optional, local)
Requires Python 3.11:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt
```

Initialize config:
```bash
python -m src.cli init-config --config config/session.yaml
```

Run:
```bash
python -m src.cli run --in_dir data/raw --out_dir data/feat --config config/session.yaml
```

## Run the UI
Serve locally (do not open `file://`):
```bash
python -m http.server 8000
```
Open: `http://localhost:8000/ui/`

## Project structure
- `src/` pipeline (synced from adas-ldw-cli)
- `ui/` offline UI (charts, timeline, decision log)
- `config/` session config + thresholds
- `data/raw/` input data (not committed)
- `data/feat/` outputs (not committed)
- `data/converted_videos/` converted videos for UI (not committed)
- `models/` ONNX/PT models

## Author
Domenico De Pasquali  
*MSc in Interaction & Experience Design*  
*BSc in Information and Communications Technologies*

## License
This project is released under the [MIT License](https://mit-license.org).

MIT (source code). Binary distribution terms should follow the rules of the upstream model licenses.
