import asyncio
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import processing
from app.config import settings
from app.jobs import job_store
from app.models import Job, JobStatus

app = FastAPI(title="Bag Counting Service")


@app.get("/videos", response_model=list[Job])
async def list_videos(limit: int = 50) -> list[Job]:
    return job_store.list_all(limit=limit)


@app.post("/videos", response_model=Job)
async def upload_video(file: UploadFile) -> Job:
    if not file.filename:
        raise HTTPException(400, "missing filename")

    video_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix or ".mp4"
    input_path = settings.input_dir / f"{video_id}{suffix}"

    with input_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    return job_store.create(filename=file.filename, input_path=input_path)


@app.post("/videos/{job_id}/process", response_model=Job)
async def start_processing(job_id: str) -> Job:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != JobStatus.PENDING:
        raise HTTPException(409, f"job already {job.status}")

    output_path = settings.output_dir / f"{job_id}.mp4"
    job = job_store.update(job_id, status=JobStatus.PROCESSING, output_path=output_path)
    asyncio.create_task(_run_job(job_id, job.input_path, output_path))
    return job


async def _run_job(job_id: str, input_path: Path, output_path: Path) -> None:
    try:
        await asyncio.to_thread(processing.run, job_id, input_path, output_path)
    except Exception as exc:
        job_store.update(job_id, status=JobStatus.FAILED, error=str(exc))


@app.get("/videos/{job_id}/status", response_model=Job)
async def get_status(job_id: str) -> Job:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/videos/{job_id}/download")
async def download_result(job_id: str) -> FileResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != JobStatus.COMPLETED or job.output_path is None:
        raise HTTPException(409, f"job not completed (status={job.status})")
    return FileResponse(job.output_path, media_type="video/mp4", filename=f"{job_id}.mp4")


# Mounted last so it can't shadow the API routes above — Starlette matches
# routes in registration order, and a mount at "/" would otherwise catch
# everything.
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
