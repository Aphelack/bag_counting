# Bag Counting Service

A FastAPI service that processes conveyor-belt video, detects bags with an
MMDetection (RTMDet) model, tracks them across frames, counts how many
crossed the belt, flags anomalies that could affect that count, and serves
the annotated video back to the user — through both a REST API and a small
web UI. Built as a take-home assignment; this README is both the setup
guide and the technical report.

## Contents

- [Quick start](#quick-start)
- [Using the service](#using-the-service)
- [Architecture](#architecture)
- [Detection: MMDetection / RTMDet](#detection-mmdetection--rtmdet)
- [Counting approach](#counting-approach)
- [Anomaly monitoring](#anomaly-monitoring)
- [Asynchronous processing](#asynchronous-processing)
- [Performance: batched GPU inference](#performance-batched-gpu-inference)
- [Docker & data persistence](#docker--data-persistence)
- [Key technical decisions](#key-technical-decisions)
- [Known limitations](#known-limitations)

## Quick start

Requirements: Docker with Compose v2, and an NVIDIA GPU + `nvidia-container-toolkit`
on the host for real detections (the app also runs without a GPU — see
below — but then every job reports 0 bags).

```bash
git clone <this-repo>
cd bag_counting
docker compose up --build
```

First build takes a while (~10 minutes) — the image installs a full
CUDA build of torch + MMDetection, not just the lightweight FastAPI
dependencies. Subsequent builds reuse Docker's layer cache and are fast
unless `requirements.txt`/`Dockerfile` change.

Once it's up:

- Web UI: **http://localhost:8000/**
- API docs (Swagger): **http://localhost:8000/docs**

The trained detector checkpoint is committed in `models/` (see
[Detection: MMDetection / RTMDet](#detection-mmdetection--rtmdet)), so
real bag detection works immediately — no separate training step needed to
try the app. `input.mp4` (the assignment's test video) is expected at
`storage/input/input.mp4`; if it's not already there, copy it in before
uploading through the UI.

## Using the service

**Web UI** (`http://localhost:8000/`): drag-and-drop a video in, click
Upload, click "Start processing," watch the progress bar and live bag
count, download the result when it finishes. The "Recent jobs" list is
server-backed, so it reflects real job history, not just what your
browser remembers.

**API**, if you'd rather script it:

```bash
# 1. upload
JOB_ID=$(curl -s -X POST http://localhost:8000/videos \
    -F "file=@storage/input/input.mp4;type=video/mp4" | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")

# 2. start processing (returns immediately, doesn't block on inference)
curl -s -X POST "http://localhost:8000/videos/$JOB_ID/process"

# 3. poll status
curl -s "http://localhost:8000/videos/$JOB_ID/status"

# 4. download once status == "completed"
curl -s -o result.mp4 "http://localhost:8000/videos/$JOB_ID/download"
```

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/videos` | List recent jobs (`?limit=`, default 50), newest first |
| `POST` | `/videos` | Upload a video (`multipart/form-data`, field `file`) — returns a `pending` job |
| `POST` | `/videos/{job_id}/process` | Start async processing — returns immediately |
| `GET`  | `/videos/{job_id}/status` | Job status, progress, running bag count, anomalies |
| `GET`  | `/videos/{job_id}/download` | Processed video, once `status == "completed"` |

## Architecture

```
app/
  main.py         FastAPI routes + static UI mount
  static/          web UI (index.html — plain HTML/JS, no build step)
  jobs.py          JSON-file-backed job store (thread-safe, survives restarts)
  processing.py    per-video pipeline: batched detect -> track/count -> anomaly-check -> overlay -> write
  detector.py      RTMDetDetector (batched mmdet inference) + StubDetector fallback
  tracker.py       BagCounter: greedy IoU tracker + direction-aware counting-zone crossing
  anomalies.py     AnomalyMonitor + per-anomaly-type AnomalyRule classes
  models.py        Job/Anomaly pydantic schemas
  config.py        storage paths, detector config/checkpoint/batch-size settings (env-driven)

models/            detector config + trained checkpoint (committed — see below)
storage/           host-mounted volume: input/output videos + job records
experiments/        SAM 3 labeling notebook/script (separate uv env)
training/            MMDetection training pipeline (separate uv env)
```

The service itself has one moving part beyond FastAPI's request/response
cycle: a per-job background task (`asyncio.create_task` + `asyncio.to_thread`,
see [Asynchronous processing](#asynchronous-processing)) that runs the
video pipeline and writes progress back into the job store as it goes.
Everything the API returns is just a read of that job's current state.

## Detection: MMDetection / RTMDet

**Model**: RTMDet-tiny (from MMDetection), fine-tuned from COCO-pretrained
weights on a single class (`bag`). RTMDet was chosen over heavier
two-stage detectors (Faster R-CNN etc.) because the scene is a single
fixed camera angle with one object class and a real-time-ish constraint —
a small, fast one-stage detector is the right tool, and RTMDet-tiny gives
the best accuracy/speed trade-off in that class within MMDetection's own
model zoo.

**Training data**: there's no existing labeled dataset of bags on this
specific belt, and hand-labeling would have eaten most of the assignment's
time budget. Instead, `experiments/scripts/label_with_sam3.py` bootstrap-labels
a COCO dataset directly from `input.mp4` using SAM 3 (Meta's
promptable-concept-segmentation model) with a text prompt tuned by trial
and error against this footage (`"A white soft pillow positioned on an
industrial roller conveyor system."` — see `experiments/README.md` for how
that prompt was arrived at; generic prompts like `"bag"` performed
noticeably worse against this specific object). Masks are converted to
tight bounding boxes, split chronologically into train/val (to avoid
near-duplicate adjacent frames leaking across the split), and fed into
`training/configs/rtmdet_bag.py` for fine-tuning.

**A note on the resulting metrics**: train and val labels both come from
the same SAM3 pass, so a high val mAP mostly proves RTMDet learned to
reproduce SAM3's boxes — including any labeling mistakes SAM3 made — not
that it's accurate against ground truth. `training/notebooks/01_inspect_predictions.ipynb`
runs the trained checkpoint on frames deliberately excluded from both
train and val (offset from the labeling stride) specifically to get a
less circular signal. Full reasoning and the actual numbers are in
`training/README.md`.

**Why the checkpoint is committed**: `models/checkpoint.pth` (~42MB, well
under GitHub's 100MB limit) and `models/rtmdet_bag.py` are checked into
this repo rather than left as a build artifact you'd need a GPU to
reproduce — the assignment requires the project to run from the README
alone, and "train your own detector first" isn't a reasonable ask of a
grader. The full labeling → training pipeline that produced it is still
here and documented (`experiments/`, `training/`) for anyone who wants to
reproduce or retrain it.

**Detector wiring**: `app/detector.py`'s `RTMDetDetector` wraps mmdet's
inference API; `get_detector()` builds it once (lazily, on first use) and
caches it as a process-wide singleton, since loading a checkpoint onto a
GPU is expensive and shouldn't happen per-video. If `models/checkpoint.pth`
were ever missing or misconfigured, the app still starts fine and falls
back to `StubDetector` (always 0 detections) rather than crashing — a
job that hits this fails with a clear error rather than the whole service
going down.

## Counting approach

`app/tracker.py`'s `BagCounter` does three things:

**1. Cross-frame tracking** — a greedy IoU tracker: match each frame's
detections to existing tracks by highest IoU overlap (above a threshold),
spawn new tracks for anything unmatched, drop tracks unseen for too long.
On top of that, unmatched tracks get **constant-velocity prediction** —
each frame a track goes unseen, its box is advanced by its last known
velocity instead of sitting frozen. Without this, a track that misses
detection for even a few frames (a brief occlusion, a confidence dip)
falls out of IoU range of where the real bag actually is by the time
detection resumes, looks like a brand new object, and silently
double-counts the same bag. This cross-frame object continuity is exactly
what the assignment calls out as a top evaluation criterion.

**2. Counting-zone crossing, counted on exit** — a rectangular zone (not
a single line) placed over a clean stretch of belt: past the dark tunnel
opening where bags first appear (avoids partial-occlusion misses right as
an object emerges) and before the pile where bags accumulate at the
bottom-left (avoids double-counting overlapping bags there). A zone
tolerates a missed detection or two right at the boundary, where a thin
line would just lose the count outright.

A crossing fires when a track's centroid **exits** the zone (transitions
from inside to outside), not when it enters — signed by the direction of
travel at that exit. This was originally entry-based (count the first time
a track's centroid enters the zone) and got flipped after a bug report
from real footage: a bag got pushed backward into the zone by a momentary
belt reversal, the belt stopped with the bag sitting inside the zone, then
the belt resumed forward and the bag continued on and exited normally.
Entry-based counting flagged the backward arrival as a reverse crossing
(`-1`) and then, because a track was only ever counted once, permanently
lost the fact that the same bag went on to complete an ordinary forward
pass — net effect, one real bag contributed `-1` to the total instead of
the `+1` it should have. Counting on exit instead of entry fixes this at
the root: entering the zone doesn't fire anything by itself, so the
backward arrival is a non-event, and only the eventual forward exit fires
— one clean `+1`, matching the one bag that actually passed through. A
track can cross the zone boundary multiple times over its life (e.g. a
bag that completes a pass and is later genuinely pushed back through) and
each exit is independently signed, so a real re-pass in reverse correctly
nets an earlier `+1` back out to `0`.

Two things keep this exit rule from silently losing a crossing instead of
just mis-signing one:
- A track that's lost (occlusion, or it left the frame) while its last
  known position was still inside the zone is treated as exiting at the
  moment it's dropped, signed by its last known velocity — otherwise a
  track that never gets a clean "exit while still tracked" moment would
  vanish without ever registering.
- `BagCounter.finalize()`, called once after the video ends: a bag that's
  still inside the zone when the video simply runs out of frames (no more
  frames left for it to exit or age out naturally) gets the same
  last-known-velocity treatment, rather than being silently dropped.

Each track is still counted **at most once per zone visit** (entering,
dwelling, and eventually exiting counts as a single visit — the `in_zone`
transition is what fires the count, not repeated per-frame checks while
inside), keyed by track ID — this is what guarantees a given physical bag
can't be counted many times over for one pass through. The zone's default
position (`DEFAULT_ZONE_FRACTIONAL` in `tracker.py`) is tuned to
`input.mp4`'s fixed camera angle.

**3. Direction-aware counting** — handles a case the first two points
don't fully cover: a bag that gets jostled backward and then forward again
on the belt *and breaks tracking while doing it* (as opposed to the
single-continuous-track reversal case above, which exit-based counting
already handles cleanly). A sudden reversal is exactly what a
constant-velocity predictor gets wrong, so tracking can legitimately break
mid-reversal, and the same physical bag can end up as several separate
track segments, each with its own zone exit. Rather than trying to
perfectly stitch a reversal back into one track (hard, and fragile), every
zone exit is **signed** by its direction relative to a reference
direction — set from the very first exit ever recorded, since the belt's
actual forward direction isn't known in advance: `+1` with the flow, `-1`
against it. A bag that crosses forward → back → forward across broken
track segments nets `1 + (-1) + 1 = 1`, matching the one bag that actually
passed, instead of counting it 3 times. `BagCounter.forward_count` /
`.reverse_count` expose the raw tallies for visibility into how often this
triggers.

This whole approach (detector + lightweight IoU tracker + counting rule,
all custom, versus pulling in a heavier off-the-shelf tracker like
ByteTrack/DeepSORT) was a deliberate choice — the scene has at most a
couple of simultaneous objects and smooth, slow motion, so a full
multi-object tracker's extra machinery (motion models tuned for crowded
scenes, re-identification embeddings) wasn't worth the dependency weight
or complexity for what this specific footage needs.

## Anomaly monitoring

The assignment leaves anomaly definitions and detection method entirely up
to the candidate; `app/anomalies.py` treats "things that could affect
counting correctness" as the design target and picked two concrete rules,
built on an extensible pattern rather than a single hardcoded check:

- **`AnomalyRule`** — a small protocol: any class with a
  `check(ctx: FrameContext) -> list[Anomaly]` method. Each anomaly *type*
  is its own class.
- **`AnomalyMonitor`** — runs the configured rules against a per-frame
  `FrameContext` (detections, timestamp, running crossing counts) every
  frame and collects whatever they return. Adding a new anomaly type means
  adding a new class and listing it in `AnomalyMonitor.DEFAULT_RULES` —
  nothing else in the pipeline changes.

Implemented rules:

- **`ReverseDirectionRule`** — flags every reverse-direction zone crossing
  (see [Counting approach](#counting-approach)). The count itself is
  already correct without this rule firing — it's a visibility signal,
  since a bag moving backward (jostled, jammed, someone reaching into the
  frame) is worth flagging to an operator even when the net count stays
  right.
- **`DetectionGapRule`** — flags a stretch (default: 5 seconds, measured
  from frame timestamps so it isn't tied to a specific fps) with zero
  detections: a possible belt stoppage, camera obstruction, or the
  detector losing the scene entirely. Fires once per gap, not every frame
  it continues, and resets as soon as a detection reappears.

Anomalies are attached to the job (`Anomaly{frame, timestamp_sec, kind,
message}`, returned in `GET /videos/{job_id}/status` and persisted with
the rest of the job record) and shown in the web UI as a running list
under the current job.

Rules not implemented but easy to add on this pattern if useful: detector
confidence collapse (sustained low-score detections), implausible count
spikes (many new tracks in a short window), or a cross-check against a
classical CV signal (e.g. background-subtraction motion) as a second
opinion against the learned detector.

## Asynchronous processing

`POST /videos/{job_id}/process` must return before inference finishes —
the assignment states this explicitly. The mechanism:

```python
job = job_store.update(job_id, status=JobStatus.PROCESSING, ...)
asyncio.create_task(_run_job(job_id, job.input_path, output_path))
return job
```

`_run_job` wraps the actual (synchronous, CPU/GPU-bound) pipeline in
`asyncio.to_thread(...)`, which hands it off to a worker thread from
asyncio's default thread pool. The `POST` handler returns as soon as the
background task is *scheduled*, not when it completes — the HTTP request
never blocks on inference. Processing then runs in the background,
periodically writing progress (`GET /videos/{job_id}/status`) back into
the job store as it goes (every 30 processed frames), so a polling client
sees live progress rather than a single jump from `pending` to `completed`.

This is intentionally simpler than a separate task queue (Celery/RQ +
Redis/RabbitMQ): a single-process FastAPI service with an in-thread worker
is enough for the assignment's scope (one video at a time is the realistic
usage pattern here), and avoids a broker dependency and multi-container
orchestration for no real benefit at this scale. If concurrent video
processing needed to scale beyond a single machine's GPU, that's the
natural next step — nothing in the job-store/API layer would need to
change, only how `_run_job` dispatches work.

## Performance: batched GPU inference

`RTMDetDetector.detect_batch()` builds and runs a **real** batched forward
pass — not `mmdet.apis.inference_detector()`, which (confirmed against
MMDetection's own source) loops over images one at a time internally even
when given a list, giving each image its own `model.test_step()` call.
Passing it a list is a convenience, not actual GPU batching. Instead, each
frame is preprocessed through the model's own test pipeline, then all of
them go through a single `model.test_step()` call together, so the GPU
processes the whole batch in parallel per forward pass.

`processing.py` reads and batches frames from the video in groups of
`DETECTION_BATCH_SIZE` before calling `detect_batch()` — config default 8
(safe for a small/CPU-only setup), `docker-compose.yml` sets 128 for a
40GB-class datacenter GPU (RTMDet-tiny is small and this is
inference-only, `torch.no_grad()`, so there's real headroom left; watch
`nvidia-smi` during a run and push higher if utilization/memory still has
room). Everything *after* detection — tracking, counting, anomaly
checking, overlay drawing, writing — still happens frame-by-frame in
strict order, since all of that depends on frame sequence; only detection
benefits from batching.

## Docker & data persistence

Single service, defined in `docker-compose.yml`:

- Requests a GPU reservation (`deploy.resources.reservations.devices`,
  driver `nvidia`) — needs `nvidia-container-toolkit` on the host.
- Mounts `./storage:/data` and `./models:/models` as host bind mounts.
- The `Dockerfile` installs a CUDA build of torch + MMDetection on top of
  the base FastAPI dependencies, following the exact install order/fixes
  already worked out (and hit in practice) while building the training
  environment — see the comments in the `Dockerfile` and
  `training/README.md`'s setup section for the specific gotchas
  (`pkg_resources` removed from `setuptools` 81+ breaking `mim`'s CLI, the
  `opencv-python`/`opencv-python-headless` conflict mmcv/mmdet introduce,
  a NumPy 1.x/2.x ABI mismatch with the pinned torch build — each one is a
  real error this project hit and fixed, not a hypothetical).

**Persistence**: uploaded videos (`storage/input/`), processed output
(`storage/output/`), and job records (`storage/jobs/`, one JSON file per
job) all live on the host-mounted `./storage` volume — everything survives
a `docker compose down && docker compose up` or a full container
recreation, matching the assignment's requirement directly. Job records
specifically: a job's full state (status, progress, bag count, anomalies)
is written to `storage/jobs/{id}.json` on every update and reloaded on
startup, so `GET /videos/{job_id}/status` keeps working for jobs created
before a restart — including from the web UI's "Recent jobs" list, which
is server-backed rather than relying on browser storage. A job caught
mid-`processing` when the process dies has no way to ever resume (its
background task is gone), so on reload it's transitioned to `failed` with
a clear `"Interrupted by service restart"` message instead of staying
stuck showing stale progress forever.

## Key technical decisions

A condensed list of the non-obvious choices and why, for quick reference:

| Decision | Why |
|---|---|
| RTMDet-tiny (MMDetection) | Best speed/accuracy trade-off in MMDetection's model zoo for a single-class, fixed-camera, near-real-time scene |
| SAM 3 for bootstrap labeling | No existing labeled dataset for this belt; hand-labeling wasn't worth the time budget; a tuned text prompt against this footage gave usable auto-labels |
| Custom IoU tracker, not ByteTrack/DeepSORT | Scene has at most a couple of simultaneous objects and smooth motion — a full MOT tracker's extra machinery wasn't worth the dependency weight here |
| Counting **zone**, not a line | Tolerates a missed detection or two at the boundary; a single line loses the count outright on a miss |
| Count on zone **exit**, not entry | A bag pushed backward into the zone, stopped, then resumed forward and exited normally was mis-flagged `-1` on entry and — since a track only ever counted once — never got the compensating `+1` when it later exited correctly. Counting on exit instead fires exactly once, correctly signed, for that case |
| Constant-velocity prediction on unmatched tracks | Without it, a few missed frames breaks IoU continuity and silently double-counts the same bag |
| Direction-signed zone crossings | Handles bags jostled backward-then-forward without needing to perfectly stitch a broken track back together |
| `asyncio.to_thread` background task, not Celery/RQ | Matches the assignment's actual scale (one video at a time); avoids a broker + extra containers for no real benefit yet |
| One JSON file per job on the storage volume, not SQLite/Postgres | Job records are simple key-value state with no relational queries needed; a file store is the smallest thing that satisfies "must survive container recreation" |
| Real batched `model.test_step()`, not `inference_detector()` | The latter doesn't actually batch on the GPU despite accepting a list — confirmed against MMDetection's source; matters a lot on a 15,000-frame video |
| Trained checkpoint committed to the repo | Assignment requires running from the README alone; 42MB is well under GitHub's limit; avoids requiring a GPU just to grade the project |
| Anomaly rules as small pluggable classes | Assignment leaves anomaly definitions open-ended; a protocol + registry keeps adding more cheap rather than hardcoding one check |
| Plain HTML/CSS/JS UI, no framework/build step | Scope doesn't justify a frontend build pipeline; keeps the Docker image and repo simple while still covering the required upload/status/download flow |

## Known limitations

- The counting zone's position is tuned to `input.mp4`'s specific fixed
  camera angle (`DEFAULT_ZONE_FRACTIONAL` in `tracker.py`) — a different
  camera setup needs re-tuning, there's no auto-calibration.
- The IoU tracker has no re-identification: if a bag is fully occluded for
  longer than `max_age` frames (default 10) and reappears far from its
  predicted position, it's treated as a new object. Direction-aware
  counting handles the reversal case this causes, and exit-based counting
  plus `finalize()` handle a track being lost or the video ending while a
  bag is still inside the zone — but a long occlusion with a large
  position jump elsewhere on the belt could in theory still misattribute
  a crossing's direction if the jump itself crosses the zone boundary in a
  way that doesn't match the bag's real path. Not observed in testing
  against `input.mp4`, but worth flagging.
- Detection-quality anomalies (confidence collapse, implausible count
  spikes) aren't implemented — see [Anomaly monitoring](#anomaly-monitoring).
  Only reverse-direction and detection-gap are.
- Job history is a flat JSON-file store with no pagination beyond
  `?limit=`; fine at this scale, would need a real database past a few
  thousand jobs.
