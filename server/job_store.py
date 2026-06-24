import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JobState:
    status: str = "pending"
    events: list = field(default_factory=list)
    result_excel: Optional[bytes] = None
    locations: Optional[list] = None


_store: dict = {}
_lock = threading.Lock()


def create_job(job_id: str) -> JobState:
    with _lock:
        state = JobState()
        _store[job_id] = state
        return state


def get_job(job_id: str) -> Optional[JobState]:
    with _lock:
        return _store.get(job_id)


def append_event(job_id: str, event: dict) -> None:
    with _lock:
        if job_id in _store:
            _store[job_id].events.append(event)
