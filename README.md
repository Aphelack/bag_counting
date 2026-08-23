# Bag Counting Service

FastAPI service that detects bags on a conveyor belt (MMDetection/RTMDet),
tracks and counts them across frames, flags anomalies, and serves the
annotated video — via REST API and a web UI.

## Contents

[Quick start](#quick-start) · [Using the service](#using-the-service) ·
[Architecture](#architecture) · [Detection](#detection-mmdetection--rtmdet) ·
[Counting approach](#counting-approach) · [Anomaly monitoring](#anomaly-monitoring) ·
[Async processing](#asynchronous-processing) · [Batched inference](#performance-batched-gpu-inference) ·
[Docker & persistence](#docker--data-persistence) · [Key decisions](#key-technical-decisions) ·
[Limitations](#known-limitations)

## Quick start

Requires Docker Compose v2, and an NVIDIA GPU + `nvidia-container-toolkit`
for real detections (runs without one too — every job just reports 0 bags).

```bash
git clone <this-repo>
cd bag_counting
docker compose up --build
```

First build ~10 min (installs a CUDA torch + MMDetection image). Later
builds reuse Docker's cache and are fast unless `requirements.txt`/`Dockerfile` change.

- Web UI: **http://localhost:8000/**
- API docs: **http://localhost:8000/docs**

The trained checkpoint is committed in `models/`, so detection works out
of the box — no training step needed. Put `input.mp4` at
`storage/input/input.mp4` before uploading through the UI if it's not
already there.

## Using the service

**Web UI**: drag-and-drop upload → Start processing → live progress/bag
count → download. "Recent jobs" is server-backed (`GET /videos`), so it
reflects real history, not just the browser's memory.

**API**:

```bash
JOB_ID=$(curl -s -X POST http://localhost:8000/videos \
    -F "file=@storage/input/input.mp4;type=video/mp4" | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
curl -s -X POST "http://localhost:8000/videos/$JOB_ID/process"   # returns immediately
curl -s "http://localhost:8000/videos/$JOB_ID/status"            # poll
curl -s -o result.mp4 "http://localhost:8000/videos/$JOB_ID/download"
```

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/videos` | List recent jobs (`?limit=`, default 50) |
| `POST` | `/videos` | Upload a video → `pending` job |
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
  config.py       env-driven settings (paths, detector, batch size)

models/           detector config + trained checkpoint (committed)
storage/          host volume: input/output videos + job records
experiments/      SAM 3 labeling notebook/script (separate uv env)
training/         MMDetection training pipeline (separate uv env)
```

A per-job background task (`asyncio.create_task` + `asyncio.to_thread`)
runs the pipeline and writes progress into the job store; API reads are
just reads of that state.

## Detection: MMDetection / RTMDet

- **Model**: RTMDet-tiny, fine-tuned from COCO-pretrained weights on one
  class (`bag`). Best speed/accuracy trade-off in MMDetection's zoo for a
  single-class, fixed-camera scene.
- **Training data**: no existing labeled dataset, so
  `experiments/scripts/label_with_sam3.py` bootstrap-labels a COCO dataset
  straight from `input.mp4` using SAM 3 (text-prompt segmentation, prompt
  tuned by trial and error — see `experiments/README.md`). Masks → boxes →
  chronological train/val split → `training/configs/rtmdet_bag.py`.
- **mAP caveat**: train/val labels both come from the same SAM3 pass, so a
  high val mAP mostly shows RTMDet reproduced SAM3's boxes, not
  ground-truth accuracy. `training/notebooks/01_inspect_predictions.ipynb`
  checks the checkpoint on frames excluded from both splits for a less
  circular signal — see `training/README.md` for the actual numbers.
- **Checkpoint is committed** (`models/checkpoint.pth`, ~42MB, `models/rtmdet_bag.py`)
  so the repo runs standalone per the assignment's requirement, without
  needing a GPU just to reproduce it. Full labeling → training pipeline
  is still here for anyone who wants to retrain.
- **Wiring**: `get_detector()` builds `RTMDetDetector` once, lazily, as a
  process-wide singleton. Missing/misconfigured checkpoint → falls back to
  `StubDetector` (0 detections) rather than crashing; only that job fails.

## Counting approach

`app/tracker.py`'s `BagCounter`:

1. **Cross-frame tracking** — greedy IoU matching + **constant-velocity
   prediction** for unmatched tracks (advance by last known velocity
   instead of freezing). Without this, a brief occlusion drifts a track
   out of IoU range, looks like a new object, and double-counts.
2. **Counting zone, counted on exit** — a rectangular zone (tolerates a
   missed detection at the boundary, unlike a line), placed past the
   tunnel opening and before the accumulation pile. A crossing fires when
   a track **exits** the zone (not enters), signed by exit-time direction.
   *Why exit, not entry*: a real bug — a belt reversal pushed a bag
   backward into the zone, the belt stopped, then resumed forward and the
   bag exited normally. Entry-based counting flagged the backward arrival
   `-1` and (since a track counted once, ever) never credited the later
   correct exit. Exit-based counting fixes it at the root: entry is a
   non-event, only the eventual exit fires — one correct `+1`.
   - A track lost while still inside the zone is treated as exiting at
     drop time (last known velocity), so it isn't silently lost.
   - `BagCounter.finalize()` does the same for tracks still in-zone when
     the video simply ends.
3. **Direction-aware signing** — for tracks that break mid-reversal
   (constant-velocity prediction gets a sudden reversal wrong). Each exit
   is signed relative to a reference direction set by the first-ever exit:
   `+1` with the flow, `-1` against it. Forward→back→forward across broken
   track segments nets `1 + (-1) + 1 = 1`, not 3.

Custom lightweight tracker vs. ByteTrack/DeepSORT: this scene has at most
a couple of simultaneous objects and smooth motion — a full MOT tracker's
extra machinery wasn't worth the dependency weight here.

## Anomaly monitoring

`app/anomalies.py`: `AnomalyRule` protocol (`check(ctx) -> list[Anomaly]`),
one class per anomaly type, run by `AnomalyMonitor` against a per-frame
`FrameContext`. New anomaly = new class in `AnomalyMonitor.DEFAULT_RULES`.

- **`ReverseDirectionRule`** — flags every reverse zone-exit. The count is
  already correct; this is a visibility signal for an operator.
- **`DetectionGapRule`** — flags 5+ seconds (timestamp-based, not fps-tied)
  with zero detections: possible belt stoppage/camera obstruction. Fires
  once per gap.

Anomalies persist on the job (`Anomaly{frame, timestamp_sec, kind,
message}`) and show as a running list in the UI. Not implemented, but
cheap to add on this pattern: confidence-collapse, count-spike, or a
classical-CV cross-check rule.

## Asynchronous processing

```python
job = job_store.update(job_id, status=JobStatus.PROCESSING, ...)
asyncio.create_task(_run_job(job_id, job.input_path, output_path))
return job
```

`_run_job` wraps the sync pipeline in `asyncio.to_thread(...)` — the
`POST` returns once the task is *scheduled*, not completed. Progress is
written back every 30 frames for polling clients.

Chose a single in-thread worker over Celery/RQ + a broker: matches the
assignment's actual scale (one video at a time), avoids broker/orchestration
overhead for no benefit yet. Scaling out would only change how `_run_job`
dispatches work, not the API/job-store layer.

## Performance: batched GPU inference

`RTMDetDetector.detect_batch()` runs a **real** batched forward pass —
`mmdet.apis.inference_detector()` loops per-image internally even given a
list (confirmed against MMDetection's source), so it doesn't actually
batch on GPU. Instead: preprocess each frame through the model's pipeline,
then one `model.test_step()` call for the whole batch.

`processing.py` batches frames in groups of `DETECTION_BATCH_SIZE`
(default 8; `docker-compose.yml` sets 128 for a 40GB-class GPU — RTMDet-tiny
is small and inference-only, likely more headroom; watch `nvidia-smi`).
Only detection is batched — tracking/counting/anomalies/drawing/writing
stay frame-by-frame, in order, since they depend on sequence.

## Docker & data persistence

- GPU reservation in `docker-compose.yml` (needs `nvidia-container-toolkit`).
- `./storage:/data` and `./models:/models` host bind mounts.
- `Dockerfile` installs CUDA torch + MMDetection with the exact fixes
  worked out in `training/` (see its README): `pkg_resources` removed
  from `setuptools` 81+ breaking `mim`; `opencv-python`/`opencv-python-headless`
  conflict from mmcv/mmdet; a NumPy 1.x/2.x ABI mismatch with the pinned
  torch build. Each is a real error this project hit, not hypothetical.
- **Persistence**: input/output videos and one JSON file per job
  (`storage/jobs/{id}.json`) all live on the volume — survives
  `docker compose down`/recreation. A job caught mid-`processing` when the
  process dies (no way to resume) reloads as `failed` with
  `"Interrupted by service restart"` instead of hanging forever.

## Key technical decisions

| Decision | Why |
|---|---|
| RTMDet-tiny (MMDetection) | Best speed/accuracy trade-off for a single-class, fixed-camera, near-real-time scene |
| SAM 3 for bootstrap labeling | No existing dataset; hand-labeling not worth the time budget; a tuned prompt gave usable auto-labels |
| Custom IoU tracker, not ByteTrack/DeepSORT | At most a couple of simultaneous objects, smooth motion — full MOT machinery not worth the weight |
| Counting **zone**, not a line | Tolerates a missed detection at the boundary; a line loses the count outright on a miss |
| Count on zone **exit**, not entry | Entry-based counting mis-flagged a reversal-then-resume bag `-1` and never credited its later correct pass — see [Counting approach](#counting-approach) |
| Constant-velocity prediction on unmatched tracks | Without it, a few missed frames breaks IoU continuity and double-counts |
| Direction-signed crossings | Handles broken-track reversals without needing to stitch tracks back together |
| `asyncio.to_thread`, not Celery/RQ | Matches actual scale (one video at a time); avoids broker/container overhead |
| One JSON file per job, not SQLite/Postgres | Simple key-value state, no relational queries needed; smallest thing that survives recreation |
| Real batched `model.test_step()`, not `inference_detector()` | Latter doesn't actually batch on GPU despite accepting a list — matters on a 15,000-frame video |
| Trained checkpoint committed to the repo | Must run from the README alone; 42MB is well under GitHub's limit |
| Anomaly rules as pluggable classes | Assignment leaves definitions open-ended; a protocol + registry keeps adding more cheap |
| Plain HTML/CSS/JS UI, no framework | Scope doesn't justify a build pipeline; keeps image/repo simple |

## Known limitations

- Counting zone position (`DEFAULT_ZONE_FRACTIONAL`) is tuned to
  `input.mp4`'s fixed camera angle — no auto-calibration for a different setup.
- No re-identification: a bag occluded past `max_age` (10 frames) that
  reappears far from its predicted position is treated as new. Reversal
  and drop/end-of-video cases are handled (see Counting approach), but a
  large position jump elsewhere could in theory still misattribute a
  crossing's direction — not observed against `input.mp4`.
- Detection-quality anomalies (confidence collapse, count spikes) aren't
  implemented — only reverse-direction and detection-gap are.
- Job store has no pagination beyond `?limit=` — fine at this scale, would
  need a real database past a few thousand jobs.
