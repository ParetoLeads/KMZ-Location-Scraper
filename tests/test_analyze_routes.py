import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_no_file_returns_422():
    r = client.post("/api/upload")
    assert r.status_code == 422


def test_status_unknown_job_returns_404():
    r = client.get("/api/status/nonexistent-job-id")
    assert r.status_code == 404


def test_download_unknown_job_returns_404():
    r = client.get("/api/download/nonexistent-job-id")
    assert r.status_code == 404


def test_locations_unknown_job_returns_404():
    r = client.get("/api/locations/nonexistent-job-id")
    assert r.status_code == 404
