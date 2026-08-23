# Bag Counting Service

FastAPI service that detects bags on a conveyor belt (MMDetection/RTMDet),
tracks and counts them, flags anomalies, and serves back the annotated
video — REST API + a small web UI.

📹 **[Demo recording](<Screencast%20from%202026-08-23%2003-38-53.webm>)**
— build, run, upload, process, download, in one take.

## Run

Needs Docker Compose v2, and an NVIDIA GPU + `nvidia-container-toolkit`
for real detections (runs without one too, just reports 0 bags every time).

```bash
git clone https://github.com/Aphelack/bag_counting.git
cd bag_counting
docker compose up --build
```

First build takes ~10 min (CUDA torch + MMDetection). Later builds are
fast unless `requirements.txt`/`Dockerfile` change.

- Web UI: http://localhost:8000/
- API docs: http://localhost:8000/docs

Checkpoint is committed in `models/`, so detection works immediately, no
training needed. Put `input.mp4` at `storage/input/input.mp4` if it's not
already there.

## Using it

Web UI: drag in a video, Upload, Start processing, watch progress/bag
count, download. "Recent jobs" is server-backed, not browser storage.

API:

```bash
JOB_ID=$(curl -s -X POST http://localhost:8000/videos \
    -F "file=@storage/input/input.mp4;type=video/mp4" | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
curl -s -X POST "http://localhost:8000/videos/$JOB_ID/process"   # returns immediately
curl -s "http://localhost:8000/videos/$JOB_ID/status"            # poll
curl -s -o result.mp4 "http://localhost:8000/videos/$JOB_ID/download"
```

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/videos` | List recent jobs |
| `POST` | `/videos` | Upload → `pending` job |
| `POST` | `/videos/{job_id}/process` | Start async processing |
| `GET`  | `/videos/{job_id}/status` | Status, progress, bag count, anomalies |
| `GET`  | `/videos/{job_id}/download` | Processed video, once `completed` |

## Architecture

```
app/
  main.py         FastAPI routes + static UI mount
  static/         web UI (index.html, plain HTML/JS, no build step)
  jobs.py         JSON-file-backed job store (survives restarts)
  processing.py   pipeline: batched detect -> track/count -> anomaly-check -> overlay -> write
  detector.py     RTMDetDetector (batched mmdet inference) + StubDetector fallback
  tracker.py      BagCounter: IoU tracker + direction-aware zone-exit counting
  anomalies.py    AnomalyMonitor + pluggable AnomalyRule classes
  models.py       Job/Anomaly pydantic schemas
  config.py       env-driven settings

models/           detector config + trained checkpoint (committed)
storage/          host volume: input/output videos + job records
experiments/      SAM 3 labeling (separate uv env, see its experiments/README.md)
training/         MMDetection training pipeline (separate uv env, see its training/README.md)
```

Each job runs as a background task (`asyncio.create_task` +
`asyncio.to_thread`) that writes progress into the job store as it goes;
every API read is just a read of that state.

## How it works

**Detection** — RTMDet-tiny, fine-tuned on one class (`bag`). No existing
labeled dataset, so `experiments/scripts/label_with_sam3.py`
bootstrap-labels one from `input.mp4` with SAM 3 (see
[`experiments/README.md`](experiments/README.md) for the prompt-tuning story). Checkpoint is
committed to the repo so the project runs standalone.

**Counting** — greedy IoU tracker with constant-velocity prediction for
occluded frames, plus a rectangular counting zone. A crossing fires on
**zone exit**, not entry, signed by exit-direction. That's a fix, not the
original design: entry-based counting flagged a bag that got pushed
backward into the zone as `-1`, then never credited it when it resumed
forward and exited normally (a track only ever counted once). Counting on
exit means arriving doesn't fire anything — only actually leaving does.
Broken tracks (occlusion during a reversal) are handled by signing every
exit against the first-ever exit direction, so forward→back→forward nets
to 1, not 3.

**Async processing** — `POST /process` schedules `asyncio.to_thread(...)`
and returns immediately; the pipeline runs on a worker thread, writing
progress every 30 frames. No Celery/broker — one video at a time is the
real scale here.

**Anomalies** — `AnomalyRule` protocol, one class per rule, run by
`AnomalyMonitor`. Two implemented: `ReverseDirectionRule` (flags a
reverse zone-exit — the count is already correct, this is just a
visibility signal) and `DetectionGapRule` (5+ seconds with zero
detections — belt stoppage or camera issue).

**Batched inference** — `mmdet.apis.inference_detector()` doesn't
actually batch on GPU even given a list (confirmed against MMDetection's
source — it loops per-image). `RTMDetDetector.detect_batch()` does a real
batched `model.test_step()` call instead. Batch size: 128 in
`docker-compose.yml` for a 40GB GPU, tune via `nvidia-smi`.

**Persistence** — videos and one JSON file per job live on the
`./storage` volume, survive container recreation. A job caught
mid-processing when the container dies reloads as `failed` instead of
hanging forever.

## Known limitations

- Counting zone position is tuned to `input.mp4`'s camera angle, no auto-calibration.
- No re-identification — an occluded bag that reappears far from its
  predicted spot is treated as new.
- Only two anomaly rules implemented (confidence-collapse, count-spikes not done).
- Job store has no pagination beyond `?limit=`.
