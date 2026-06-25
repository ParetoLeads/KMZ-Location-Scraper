import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JobState:
    status: str = "pending"
    events: list = field(default_factory=list)
    result_excel: Optional[bytes] = None
    locations: Optional[list] = None
    filename: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    location_count: int = 0


_store: dict = {}
_lock = threading.Lock()


def create_job(job_id: str, filename: str = "") -> JobState:
    with _lock:
        state = JobState(filename=filename)
        _store[job_id] = state
        return state


def list_completed_jobs() -> list:
    """Return metadata for all complete/error jobs, newest first."""
    with _lock:
        jobs = []
        for job_id, job in _store.items():
            if job.status in ("complete", "error"):
                jobs.append({
                    "job_id": job_id,
                    "filename": job.filename,
                    "status": job.status,
                    "location_count": job.location_count,
                    "created_at": job.created_at,
                    "completed_at": job.completed_at,
                    "has_excel": job.result_excel is not None,
                })
        jobs.sort(key=lambda j: j["completed_at"] or 0, reverse=True)
        return jobs


def get_job(job_id: str) -> Optional[JobState]:
    with _lock:
        return _store.get(job_id)


def append_event(job_id: str, event: dict) -> None:
    with _lock:
        if job_id in _store:
            _store[job_id].events.append(event)
