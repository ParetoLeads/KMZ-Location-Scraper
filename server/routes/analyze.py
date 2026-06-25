import os
import json
import uuid
import threading
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, FileResponse
from sse_starlette.sse import EventSourceResponse
from server.job_store import (
    create_job, get_job, append_event, list_completed_jobs,
    persist_job_meta, get_job_logs, RESULTS_DIR,
)

router = APIRouter()


def _run_analysis(job_id: str, kmz_path: str, filename: str = "", min_population: int = 10000):
    from location_analyzer import LocationAnalyzer
    from config import config

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    job = get_job(job_id)
    job.status = "running"

    def on_progress(msg: str):
        append_event(job_id, {"type": "progress", "data": str(msg)})

    on_progress(f"[API keys] OPENAI_API_KEY={'set (' + str(len(openai_key)) + ' chars)' if openai_key else 'NOT SET'} | GEMINI_API_KEY={'set (' + str(len(gemini_key)) + ' chars)' if gemini_key else 'NOT SET'}")

    try:
        analyzer = LocationAnalyzer(
            kmz_file=kmz_path,
            verbose=config.VERBOSE,
            openai_api_key=openai_key,
            gemini_api_key=gemini_key,
            ai_provider="both",
            use_gpt=config.USE_GPT,
            chunk_size=config.DEFAULT_CHUNK_SIZE,
            max_locations=config.DEFAULT_MAX_LOCATIONS,
            pause_before_gpt=False,
            enable_web_browsing=config.DEFAULT_ENABLE_WEB_BROWSING,
            primary_place_types=config.PRIMARY_PLACE_TYPES,
            additional_place_types=config.ADDITIONAL_PLACE_TYPES,
            special_place_types=config.SPECIAL_PLACE_TYPES,
            progress_callback=on_progress,
            status_callback=on_progress,
        )
        import time as _time
        locations = analyzer.run()
        if locations:
            excel_bio = analyzer.save_to_excel(locations, min_population=min_population)
            if excel_bio:
                os.makedirs(RESULTS_DIR, exist_ok=True)
                excel_path = os.path.join(RESULTS_DIR, f"{job_id}.xlsx")
                with open(excel_path, "wb") as fh:
                    fh.write(excel_bio.getvalue())
                job.result_excel_path = excel_path
        job.locations = locations or []
        job.location_count = len(job.locations)
        job.completed_at = _time.time()
        job.status = "complete"
        persist_job_meta(job_id)
        append_event(job_id, {"type": "complete", "data": json.dumps({"location_count": len(job.locations)})})
    except Exception as e:
        import time as _time
        job.completed_at = _time.time()
        append_event(job_id, {"type": "error", "data": str(e)})
        job.status = "error"
        persist_job_meta(job_id)
    finally:
        try:
            os.unlink(kmz_path)
        except OSError:
            pass


@router.post("/api/upload")
async def upload(file: UploadFile = File(...), min_population: int = Form(10000)):
    job_id = str(uuid.uuid4())
    original_name = file.filename or "unknown.kmz"
    create_job(job_id, filename=original_name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz") as f:
        f.write(await file.read())
        kmz_path = f.name
    thread = threading.Thread(target=_run_analysis, args=(job_id, kmz_path, original_name, min_population), daemon=True)
    thread.start()
    return {"job_id": job_id, "filename": original_name}


@router.get("/api/status/{job_id}")
def status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": job.status, "event_count": len(job.events)}


@router.get("/api/stream/{job_id}")
async def stream(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        import asyncio
        import time as _time
        sent = 0
        last_sent_ts = _time.monotonic()
        PING_INTERVAL = 15  # keeps Railway/nginx proxy from dropping idle SSE connections
        while True:
            events = job.events
            while sent < len(events):
                e = events[sent]
                yield {"event": e["type"], "data": e["data"]}
                sent += 1
                last_sent_ts = _time.monotonic()
            if job.status in ("complete", "error") and sent >= len(job.events):
                break
            if _time.monotonic() - last_sent_ts >= PING_INTERVAL:
                yield {"event": "ping", "data": "keep-alive"}
                last_sent_ts = _time.monotonic()
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@router.get("/api/download/{job_id}")
def download(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    path = job.result_excel_path
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found — container may have restarted before the volume was mounted")
    return FileResponse(
        path=path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"results_{job_id[:8]}.xlsx",
    )


@router.get("/api/locations/{job_id}")
def locations(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"locations": job.locations or []}


@router.get("/api/logs/{job_id}")
def logs(job_id: str):
    """Return full log lines for a job — works even after container restart."""
    job = get_job(job_id)
    if job is None:
        from server.job_store import RUNS_DIR
        import os as _os
        if not _os.path.exists(_os.path.join(RUNS_DIR, f"{job_id}.json")):
            raise HTTPException(status_code=404, detail="Job not found")
    log_lines = get_job_logs(job_id)
    meta = {}
    if job:
        duration = None
        if job.created_at and job.completed_at:
            duration = round(job.completed_at - job.created_at, 1)
        meta = {
            "job_id": job_id,
            "filename": job.filename,
            "status": job.status,
            "location_count": job.location_count,
            "duration_seconds": duration,
        }
    return {**meta, "log_count": len(log_lines), "logs": log_lines}


@router.get("/api/runs")
def runs():
    """Return metadata for all completed runs (newest first)."""
    return {"runs": list_completed_jobs()}
