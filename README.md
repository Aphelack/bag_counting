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

## What's still to decide

- MMDetection model/checkpoint for bag detection (`app/detector.py`).
- Tracking + counting-line algorithm and duplicate-count protection (`app/tracker.py`).
- Anomaly definitions and detection method (`app/anomalies.py`).
- Job persistence beyond in-memory (only matters if job history must survive a restart — video files already persist via the volume).

These will be filled in as we settle the design.
