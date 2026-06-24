import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from server.job_store import create_job, get_job, append_event

def test_create_and_get_job():
    create_job("abc123")
    job = get_job("abc123")
    assert job is not None
    assert job.status == "pending"
    assert job.events == []
    assert job.result_excel is None

def test_append_event():
    create_job("xyz789")
    append_event("xyz789", {"type": "progress", "data": "Stage 1 done"})
    job = get_job("xyz789")
    assert len(job.events) == 1
    assert job.events[0]["data"] == "Stage 1 done"

def test_get_missing_job_returns_none():
    assert get_job("nonexistent") is None
