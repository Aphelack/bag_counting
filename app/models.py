from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Anomaly(BaseModel):
    frame: int
    timestamp_sec: float
    kind: str
    message: str


class Job(BaseModel):
    id: str
    filename: str
    input_path: Path
    output_path: Path | None = None
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    bag_count: int = 0
    anomalies: list[Anomaly] = []
    error: str | None = None
    created_at: datetime
    updated_at: datetime
