import os
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

# Paths for persistent storage — override via env vars (Railway volume mounted at /data)
RESULTS_DIR = os.getenv('RESULTS_DIR', '/data/results')
RUNS_DIR = os.getenv('RUNS_DIR', '/data/runs')


@dataclass
class JobState:
    status: str = "pending"
    events: list = field(default_factory=list)
    result_excel_path: Optional[str] = None   # on-disk path, not bytes
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


def persist_job_meta(job_id: str) -> None:
    """Write job metadata to disk so it survives container restarts."""
    try:
        os.makedirs(RUNS_DIR, exist_ok=True)
        with _lock:
            job = _store.get(job_id)
        if not job:
            return
        meta = {
            "job_id": job_id,
            "filename": job.filename,
            "status": job.status,
            "location_count": job.location_count,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "result_excel_path": job.result_excel_path,
        }
        with open(os.path.join(RUNS_DIR, f"{job_id}.json"), "w") as f:
            json.dump(meta, f)
    except Exception:
        pass  # Never let persistence failure break the main flow


def list_completed_jobs() -> list:
    """Return metadata for all complete/error jobs, newest first."""
    with _lock:
        jobs = []
        for job_id, job in _store.items():
            if job.status in ("complete", "error"):
                has_excel = (
                    job.result_excel_path is not None
                    and os.path.exists(job.result_excel_path)
                )
                jobs.append({
                    "job_id": job_id,
                    "filename": job.filename,
                    "status": job.status,
                    "location_count": job.location_count,
                    "created_at": job.created_at,
                    "completed_at": job.completed_at,
                    "has_excel": has_excel,
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


def _load_persisted_jobs() -> None:
    """On startup, reload completed job metadata from disk into the store."""
    if not os.path.isdir(RUNS_DIR):
        return
    for fname in os.listdir(RUNS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(RUNS_DIR, fname)) as f:
                meta = json.load(f)
            job_id = meta["job_id"]
            state = JobState(
                status=meta.get("status", "complete"),
                filename=meta.get("filename", ""),
                location_count=meta.get("location_count", 0),
                created_at=meta.get("created_at", 0.0),
                completed_at=meta.get("completed_at"),
                result_excel_path=meta.get("result_excel_path"),
            )
            with _lock:
                if job_id not in _store:
                    _store[job_id] = state
        except Exception:
            pass


_load_persisted_jobs()
