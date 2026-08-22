import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.models import Job, JobStatus


class JobStore:
    """Job registry: in-memory dict (fast, thread-safe) backed by one JSON
    file per job under settings.jobs_dir, on the same host-mounted volume
    as the input/output videos — so job history (status, bag count,
    anomalies, paths) survives container recreation the same way the video
    files already do. Existing job files are loaded back into memory on
    startup.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._load_existing()

    def _job_path(self, job_id: str) -> Path:
        return settings.jobs_dir / f"{job_id}.json"

    def _load_existing(self) -> None:
        for path in settings.jobs_dir.glob("*.json"):
            try:
                job = Job.model_validate_json(path.read_text())
            except ValueError:
                continue  # skip a partially-written/corrupt file rather than fail startup

            if job.status == JobStatus.PROCESSING:
                # The background task that was running this job died with
                # the old process — nothing will ever advance it again.
                # PENDING jobs are left alone: they're still processable,
                # the input video is still on disk.
                job = job.model_copy(
                    update={
                        "status": JobStatus.FAILED,
                        "error": "Interrupted by service restart",
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                self._persist(job)

            self._jobs[job.id] = job

    def _persist(self, job: Job) -> None:
        self._job_path(job.id).write_text(job.model_dump_json())

    def create(self, filename: str, input_path: Path) -> Job:
        now = datetime.now(timezone.utc)
        job = Job(
            id=str(uuid.uuid4()),
            filename=filename,
            input_path=input_path,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._persist(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> Job:
        with self._lock:
            job = self._jobs[job_id].model_copy(
                update={**fields, "updated_at": datetime.now(timezone.utc)}
            )
            self._jobs[job_id] = job
            self._persist(job)
            return job

    def list_all(self, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]


job_store = JobStore()
