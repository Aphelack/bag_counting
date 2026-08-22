# Bag Counting Service

FastAPI service for detecting and counting bags on a conveyor belt from
uploaded video.

## Status

HTTP API, async job pipeline, video read/write, and overlay rendering all
work end-to-end. Detection (RTMDet via MMDetection) and tracking/counting
(IoU tracker + counting-zone crossing) are implemented — see "Detector
setup" below to enable real detections; without a configured checkpoint the
service still runs, using a no-op stub that always reports 0 bags.
Anomaly monitoring (`app/anomalies.py`) is still a stub — see "What's still
to decide".

## Run

```bash
docker compose up --build
```

API available at `http://localhost:8000` (docs at `/docs`). Uploaded and
processed videos persist in `./storage/{input,output}` on the host, so they
survive container recreation.

**GPU required for real detections.** `docker-compose.yml` requests a GPU
reservation (needs `nvidia-container-toolkit` on the host) and the image
installs a CUDA build of torch/MMDetection — see "Detector setup" below.
Without `./models/rtmdet_bag.py` + `./models/checkpoint.pth` present, the
app still starts and runs fine, just with the stub detector (0 bags always).

**Note on the Docker image itself**: the GPU/MMDetection install steps in
the `Dockerfile` reuse the exact sequence already validated working in
`training/` (see its README for why each step exists — pkg_resources vs.
setuptools 81+, the opencv-python/opencv-python-headless conflict, the
torch/numpy ABI mismatch), but the full image build has only been verified
on a GPU host, not in this project's own CPU-only dev environment. Report
back if `docker compose up --build` hits something the training env didn't.

## API

- `POST /videos` — multipart upload (`file`), returns a job with `pending` status.
- `POST /videos/{job_id}/process` — starts async processing, returns immediately.
- `GET /videos/{job_id}/status` — job status, progress, running bag count, anomalies.
- `GET /videos/{job_id}/download` — processed video (once `status == completed`).

## Architecture

```
app/
  main.py        FastAPI routes
  jobs.py         in-memory job store (thread-safe)
  processing.py   per-video pipeline: read -> detect -> track/count -> overlay -> write
  detector.py     RTMDetDetector (mmdet inference) + StubDetector fallback
  tracker.py      BagCounter: greedy IoU tracker + counting-zone crossing
  anomalies.py    AnomalyMonitor interface (still a stub)
  models.py       Job/Anomaly schemas
  config.py       storage paths, detector config/checkpoint paths
```

Processing runs on a worker thread via `asyncio.to_thread`, so the HTTP
request that starts a job returns immediately and inference never blocks
the event loop.

## Counting approach

`app/tracker.py`'s `BagCounter`:

- **Detection → tracking**: greedy IoU matching between frames, with simple
  constant-velocity prediction (advance each unmatched track by its last
  known velocity every frame it's missed). Without prediction, a track that
  goes unseen for even a few frames (brief occlusion, a confidence dip)
  sits frozen at its last position while the real bag keeps moving — by the
  time detection resumes it's out of IoU range, looks like a new object,
  and gets double-counted. Prediction keeps the track near the bag's actual
  position through short gaps instead.
- **Counting**: a rectangular zone (not a single line) placed over a clean
  stretch of the belt — past the tunnel opening, before the pile at the
  bottom-left. Each track is counted at most once, the first time its
  centroid enters the zone, keyed by track ID — that's what prevents
  double-counting the same bag across multiple frames.
- The zone's default position (`DEFAULT_ZONE_FRACTIONAL` in `tracker.py`)
  is tuned to `input.mp4`'s fixed camera angle; re-tune for a different
  camera setup.

## Detector setup

1. Train a checkpoint via `training/` (see its README): label with SAM 3,
   fine-tune RTMDet, validate with `training/notebooks/01_inspect_predictions.ipynb`.
2. Copy the config and checkpoint into `./models/`:
   ```bash
   cp training/configs/rtmdet_bag.py models/rtmdet_bag.py
   cp training/work_dirs/rtmdet_bag/best_coco_bbox_mAP_epoch_*.pth models/checkpoint.pth
   ```
3. `docker compose up --build` — `DETECTOR_CONFIG_PATH`/`DETECTOR_CHECKPOINT_PATH`
   in `docker-compose.yml` already point at those paths inside the container.

The detector loads lazily (on the first `/process` call, not at startup),
so a missing/misconfigured checkpoint fails that job with a clear error
rather than crashing the app.

## What's still to decide

- Anomaly definitions and detection method (`app/anomalies.py`).
- Job persistence beyond in-memory (only matters if job history must survive a restart — video files already persist via the volume).

These will be filled in as we settle the design.
