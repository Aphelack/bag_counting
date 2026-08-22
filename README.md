# Bag Counting Service

FastAPI service for detecting and counting bags on a conveyor belt from
uploaded video.

## Status

HTTP API, async job pipeline, video read/write, overlay rendering, batched
GPU detection, tracking/counting, and anomaly monitoring all work
end-to-end. See "Detector setup" below to enable real detections; without a
configured checkpoint the service still runs, using a no-op stub that
always reports 0 bags.

## Run

```bash
docker compose up --build
```

API available at `http://localhost:8000` (docs at `/docs`). Uploaded videos,
processed output, and job records (status/progress/bag count/anomalies)
all persist in `./storage/{input,output,jobs}` on the host, so everything
survives container recreation — including reopening a job from before a
restart via the web UI or `GET /videos/{job_id}/status`.

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

## Web UI

`http://localhost:8000/` — a single-page UI (plain HTML/CSS/JS, no build
step, no external assets, dark-mode aware) covering drag-and-drop upload,
start-processing, live status polling, bag count, and download. New
anomalies (detected while polling) surface as a dismissible banner in
addition to the running anomalies list. The "recent jobs" list is fetched
from the server (`GET /videos`), not browser storage — it reflects actual
job history and keeps working after a restart, a page reload, or from a
different browser. Source in `app/static/index.html`, served via
`StaticFiles`.

## API

- `GET /videos` — list recent jobs (`?limit=`, default 50), newest first.
- `POST /videos` — multipart upload (`file`), returns a job with `pending` status.
- `POST /videos/{job_id}/process` — starts async processing, returns immediately.
- `GET /videos/{job_id}/status` — job status, progress, running bag count, anomalies.
- `GET /videos/{job_id}/download` — processed video (once `status == completed`).

## Architecture

```
app/
  main.py        FastAPI routes + static UI mount
  static/         web UI (index.html — plain HTML/JS, no build step)
  jobs.py         JSON-file-backed job store (thread-safe, survives restarts)
  processing.py   per-video pipeline: batched detect -> track/count -> anomaly-check -> overlay -> write
  detector.py     RTMDetDetector (batched mmdet inference) + StubDetector fallback
  tracker.py      BagCounter: greedy IoU tracker + direction-aware counting-zone crossing
  anomalies.py    AnomalyMonitor + per-anomaly-type AnomalyRule classes
  models.py       Job/Anomaly schemas
  config.py       storage paths, detector config/checkpoint/batch-size settings
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
- **Direction-aware counting**: handles bags that get jostled backwards
  and then forward again on the belt. A sudden reversal is exactly what
  the constant-velocity predictor above gets wrong, so tracking can
  legitimately break mid-reversal and the same physical bag ends up as
  several separate tracks. Rather than trying to perfectly stitch that
  back together, each zone-crossing is signed by its direction relative to
  a reference direction (set from the very first crossing — the belt's
  actual forward direction isn't known in advance): +1 with the flow, -1
  against it. A bag that crosses forward/back/forward nets
  `1 + (-1) + 1 = 1`, matching the one bag that actually passed, instead of
  counting it 3 times. `BagCounter.forward_count`/`.reverse_count` expose
  the raw tallies if you want to see how often this is kicking in.
- The zone's default position (`DEFAULT_ZONE_FRACTIONAL` in `tracker.py`)
  is tuned to `input.mp4`'s fixed camera angle; re-tune for a different
  camera setup.

## Performance: batched detection

`RTMDetDetector.detect_batch()` in `app/detector.py` builds and runs a real
batched forward pass — not `mmdet.apis.inference_detector()`, which loops
over images one at a time internally even when given a list (each image
gets its own `model.test_step()` call, so passing it a list is not actual
GPU batching). Instead, each frame is preprocessed through the model's own
test pipeline, then all of them go through a single `model.test_step()`
call together, so the GPU processes the whole batch in parallel per
forward pass. `processing.py` reads/batches frames in groups of
`DETECTION_BATCH_SIZE` (config default 8; `docker-compose.yml` sets 64 for
a datacenter GPU — RTMDet-tiny is small and inference-only, so there's
likely still headroom on an A100; watch `nvidia-smi` during a run and push
it higher if so) before calling `detect_batch()`; tracking, counting,
anomaly-checking, drawing, and writing all still happen frame-by-frame in
order afterward, since those depend on frame sequence — only detection
benefits from batching.

## Anomaly monitoring

`app/anomalies.py`: each anomaly type is its own small class (`AnomalyRule`
protocol — a `check(ctx)` returning zero or more `Anomaly` records).
`AnomalyMonitor` just runs the configured rules against a per-frame
`FrameContext` and collects the results — adding a new anomaly type means
adding a new class and listing it in `AnomalyMonitor.DEFAULT_RULES`;
nothing else in the pipeline changes.

Implemented so far:

- **`ReverseDirectionRule`** — flags every reverse-direction zone crossing
  (see "Direction-aware counting" above). The count itself is already
  correct without this — it's a visibility signal, since a bag moving
  backwards on the belt (jostled, jammed, someone reaching in) is worth an
  operator's attention even when the net count stays right.
- **`DetectionGapRule`** — flags a long stretch (default: 125 frames, ~5s
  @ 25fps) with zero detections: possible belt stoppage, camera
  obstruction, or the detector losing the scene. Fires once per gap, not
  every frame it continues.

Anomalies are returned in the job's `anomalies` field (`GET
/videos/{job_id}/status`) and shown in the web UI both as a running list
and as a dismissible banner that appears when a *new* anomaly shows up
during polling.

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

- Whether more anomaly rules are worth adding (e.g. detector confidence collapse, implausible count spikes) — the `AnomalyRule` pattern makes this cheap to extend later.
