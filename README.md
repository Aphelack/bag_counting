# Bag Counting Service

FastAPI service for detecting and counting bags on a conveyor belt from
uploaded video.

## Status

This is the base scaffold: the HTTP API, async job pipeline, video
read/write, and overlay rendering all work end-to-end. Detection, tracking,
and anomaly logic are stub extension points, not yet implemented — see
"What's still to decide" below.

## Run

```bash
docker compose up --build
```

API available at `http://localhost:8000` (docs at `/docs`). Uploaded and
processed videos persist in `./storage/{input,output}` on the host, so they
survive container recreation.

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
  detector.py     BagDetector interface (StubDetector placeholder for MMDetection)
  tracker.py      BagCounter interface (cross-frame tracking + counting)
  anomalies.py    AnomalyMonitor interface
  models.py       Job/Anomaly schemas
  config.py       storage paths
```

Processing runs on a worker thread via `asyncio.to_thread`, so the HTTP
request that starts a job returns immediately and inference never blocks
the event loop.

## Building the detector

The `app/detector.py` stub is filled in by a separate offline pipeline —
labeling and training run on a GPU host, not inside this service:

```
experiments/   SAM 3 prototyping (uv env) — notebook + scripts/label_with_sam3.py
                bootstrap-labels a COCO dataset from input.mp4
training/      MMDetection (uv env) — configs/rtmdet_bag.py fine-tunes
                RTMDet-tiny on that dataset; scripts/train.py, scripts/infer.py
```

See `experiments/README.md` and `training/README.md` for the full
labeling → training → inference walkthrough.

## What's still to decide

- Wiring the trained RTMDet checkpoint into `app/detector.py` — needs the
  app's Docker image to gain GPU/CUDA inference support, which isn't
  designed yet.
- Tracking + counting-line algorithm and duplicate-count protection (`app/tracker.py`).
- Anomaly definitions and detection method (`app/anomalies.py`).
- Job persistence beyond in-memory (only matters if job history must survive a restart — video files already persist via the volume).

These will be filled in as we settle the design.
