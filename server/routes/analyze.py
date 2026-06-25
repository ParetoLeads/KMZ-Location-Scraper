import os
import json
import uuid
import threading
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from sse_starlette.sse import EventSourceResponse
from server.job_store import create_job, get_job, append_event

router = APIRouter()


def _run_analysis(job_id: str, kmz_path: str):
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
        locations = analyzer.run()
        if locations:
            excel_bio = analyzer.save_to_excel(locations)
            job.result_excel = excel_bio.getvalue() if excel_bio else None
        else:
            job.result_excel = None
        job.locations = locations or []
        job.status = "complete"
        append_event(job_id, {"type": "complete", "data": json.dumps({"location_count": len(job.locations)})})
    except Exception as e:
        append_event(job_id, {"type": "error", "data": str(e)})
        job.status = "error"
    finally:
        try:
            os.unlink(kmz_path)
        except OSError:
            pass


@router.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    create_job(job_id)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz") as f:
        f.write(await file.read())
        kmz_path = f.name
    thread = threading.Thread(target=_run_analysis, args=(job_id, kmz_path), daemon=True)
    thread.start()
    return {"job_id": job_id}


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
        PING_INTERVAL = 15  # seconds — keeps Railway/nginx proxy from dropping idle SSE connections
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
    if job is None or job.result_excel is None:
        raise HTTPException(status_code=404, detail="Result not ready or job not found")
    return Response(
        content=job.result_excel,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="results_{job_id[:8]}.xlsx"'},
    )


@router.get("/api/locations/{job_id}")
def locations(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"locations": job.locations or []}
