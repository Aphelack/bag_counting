import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.models import Job


class JobStore:
    """In-memory job registry.

    Deliberately simple for now: no persistence across restarts. If job
    history needs to survive a container recreation, this is the place to
    swap in a real store (SQLite/Postgres/Redis).
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

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
            return job


job_store = JobStore()
