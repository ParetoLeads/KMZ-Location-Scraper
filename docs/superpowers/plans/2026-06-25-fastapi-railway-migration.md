# FastAPI + Railway Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit app with a FastAPI backend + vanilla HTML/JS frontend, containerized for Railway deployment. Eliminates Streamlit Cloud's 60-second timeout constraint, removes the state machine chunking, and gives a proper web app interface.

**Architecture:** FastAPI serves a single-page HTML app. File upload returns a `job_id`. Processing runs in a background thread (sync, no asyncio refactor needed). Progress streams via Server-Sent Events. Map uses Leaflet.js (no API key). All existing `location_analyzer.py` logic is reused unchanged.

**Tech Stack:** FastAPI, uvicorn, python-multipart, sse-starlette, Leaflet.js, Vanilla JS, Docker, Railway

## Global Constraints

- `location_analyzer.py`, `config.py`, `utils/` must not be modified in this plan — they are used as-is
- The performance optimizations from `2026-06-25-performance-optimization.md` should be applied first
- Railway requires a `Dockerfile` or will auto-detect via Nixpacks — we use `Dockerfile` for explicit control
- All secrets (OPENAI_API_KEY, GEMINI_API_KEY) are set as Railway environment variables — never hardcoded
- The `/api/download/{job_id}` endpoint must stream the Excel bytes directly — no disk writes of results
- `RAILWAY_PUBLIC_DOMAIN` env var is set automatically by Railway — use it in health check responses

---

## File Structure

- Create: `server/main.py` — FastAPI app, mounts static files, registers routers
- Create: `server/job_store.py` — In-memory job state (dict of job_id → JobState dataclass)
- Create: `server/routes/analyze.py` — `/api/upload`, `/api/stream/{job_id}`, `/api/download/{job_id}`, `/api/status/{job_id}`
- Create: `server/routes/health.py` — `/health` endpoint
- Create: `static/index.html` — Single-page app (upload form, progress, map, results table, download)
- Create: `static/app.js` — EventSource SSE, fetch calls, Leaflet map, table rendering
- Create: `Dockerfile` — Python 3.12-slim, installs requirements, runs uvicorn
- Create: `requirements-server.txt` — FastAPI-specific additions (fastapi, uvicorn, python-multipart, sse-starlette)
- Modify: `requirements.txt` — no Streamlit dependency needed for Railway build (keep for Streamlit Cloud compatibility; `requirements-server.txt` supplements it)

---

### Task 1: Job Store

**Files:**
- Create: `server/__init__.py` (empty)
- Create: `server/job_store.py`
- Create: `tests/test_job_store.py`

**Interfaces:**
- Produces:
  - `create_job(job_id: str) -> JobState`
  - `get_job(job_id: str) -> Optional[JobState]`
  - `append_event(job_id: str, event: dict) -> None`
  - `JobState.status: str` — `"pending"` | `"running"` | `"complete"` | `"error"`
  - `JobState.events: list[dict]` — each dict has `{"type": str, "data": any}`
  - `JobState.result_excel: Optional[bytes]`
  - `JobState.locations: Optional[list[dict]]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_job_store.py`:

```python
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
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `python -m pytest tests/test_job_store.py -v`
Expected: `FAILED` — `ModuleNotFoundError: No module named 'server'`

- [ ] **Step 3: Implement job_store.py**

Create `server/__init__.py` (empty file).

Create `server/job_store.py`:

```python
import threading
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class JobState:
    status: str = "pending"
    events: list = field(default_factory=list)
    result_excel: Optional[bytes] = None
    locations: Optional[list] = None

_store: dict[str, JobState] = {}
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
```

- [ ] **Step 4: Run tests to confirm PASS**

Run: `python -m pytest tests/test_job_store.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add server/__init__.py server/job_store.py tests/test_job_store.py
git commit -m "feat: add in-memory job store for FastAPI background tasks"
```

---

### Task 2: FastAPI routes — upload, stream, status, download

**Files:**
- Create: `server/routes/__init__.py` (empty)
- Create: `server/routes/analyze.py`
- Create: `server/routes/health.py`
- Create: `tests/test_analyze_routes.py`

**Interfaces:**
- Consumes: `create_job`, `get_job`, `append_event` from [[job-store]]
- Produces:
  - `POST /api/upload` → `{"job_id": str}`
  - `GET /api/stream/{job_id}` → SSE stream of `{"type": "progress"|"complete"|"error", "data": ...}`
  - `GET /api/status/{job_id}` → `{"status": str, "event_count": int}`
  - `GET /api/download/{job_id}` → Excel file bytes
  - `GET /health` → `{"status": "ok"}`

- [ ] **Step 1: Write failing tests**

Create `tests/test_analyze_routes.py`:

```python
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
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `python -m pytest tests/test_analyze_routes.py -v`
Expected: `FAILED` — `ModuleNotFoundError: No module named 'server.main'`

- [ ] **Step 3: Create `requirements-server.txt`**

```
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9
sse-starlette>=2.1.0
```

- [ ] **Step 4: Install server dependencies**

Run: `pip install -r requirements-server.txt`
Expected: packages install without error

- [ ] **Step 5: Implement health route**

Create `server/routes/__init__.py` (empty).

Create `server/routes/health.py`:

```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 6: Implement analyze routes**

Create `server/routes/analyze.py`:

```python
import os
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
        append_event(job_id, {"type": "progress", "data": msg})

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
        analyzer.run()
        excel_bytes = analyzer.save_to_excel()
        job.result_excel = excel_bytes.getvalue() if hasattr(excel_bytes, "getvalue") else excel_bytes
        job.locations = analyzer.locations if hasattr(analyzer, "locations") else []
        job.status = "complete"
        append_event(job_id, {"type": "complete", "data": {"location_count": len(job.locations or [])}})
    except Exception as e:
        job.status = "error"
        append_event(job_id, {"type": "error", "data": str(e)})
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
        sent = 0
        import asyncio
        while True:
            events = job.events
            while sent < len(events):
                e = events[sent]
                yield {"event": e["type"], "data": e["data"]}
                sent += 1
            if job.status in ("complete", "error"):
                break
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
```

- [ ] **Step 7: Create `server/main.py`**

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from server.routes.analyze import router as analyze_router
from server.routes.health import router as health_router

app = FastAPI(title="KMZ Location Scraper")
app.include_router(health_router)
app.include_router(analyze_router)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

- [ ] **Step 8: Run tests to confirm PASS**

Run: `python -m pytest tests/test_analyze_routes.py -v`
Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git add server/ requirements-server.txt tests/test_analyze_routes.py
git commit -m "feat: add FastAPI routes for upload, stream, status, download"
```

---

### Task 3: Frontend (HTML + JS)

**Files:**
- Create: `static/index.html`
- Create: `static/app.js`

**Interfaces:**
- Consumes: `POST /api/upload`, `GET /api/stream/{job_id}`, `GET /api/download/{job_id}`
- Produces: working single-page app — file upload → live progress log → map + table → Excel download

- [ ] **Step 1: Create `static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>KMZ Location Scraper</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #f5f7fa; color: #222; }
    header { background: #1f77b4; color: white; padding: 1.2rem 2rem; }
    header h1 { font-size: 1.6rem; }
    header p { font-size: 0.85rem; opacity: 0.85; margin-top: 0.2rem; }
    main { max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }
    .card { background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-bottom: 1.5rem; }
    #upload-area { border: 2px dashed #ccc; border-radius: 8px; padding: 2rem; text-align: center; cursor: pointer; transition: border-color .2s; }
    #upload-area.drag-over { border-color: #1f77b4; background: #eef5fc; }
    #file-input { display: none; }
    #start-btn { margin-top: 1rem; background: #1f77b4; color: white; border: none; padding: 0.7rem 2rem; border-radius: 6px; font-size: 1rem; cursor: pointer; }
    #start-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    #progress-section { display: none; }
    #log { background: #111; color: #0f0; font-family: monospace; font-size: 0.78rem; height: 200px; overflow-y: auto; padding: 0.8rem; border-radius: 6px; }
    #map { height: 400px; border-radius: 8px; }
    #results-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    #results-table th { background: #1f77b4; color: white; padding: 0.5rem; text-align: left; }
    #results-table td { padding: 0.4rem 0.5rem; border-bottom: 1px solid #eee; }
    #download-btn { display: none; background: #FF5733; color: white; border: none; padding: 0.7rem 2rem; border-radius: 6px; font-size: 1rem; cursor: pointer; margin-top: 1rem; }
    .stats { display: flex; gap: 1rem; margin-bottom: 1rem; }
    .stat { background: #f0f4f8; border-radius: 6px; padding: 0.7rem 1rem; flex: 1; text-align: center; }
    .stat strong { display: block; font-size: 1.6rem; color: #1f77b4; }
  </style>
</head>
<body>
  <header>
    <h1>🗺️ KMZ Location Scraper</h1>
    <p>Extract locations from KMZ files and estimate populations using OpenStreetMap and AI</p>
  </header>
  <main>
    <div class="card">
      <div id="upload-area">
        <p>Drag &amp; drop a .kmz file here, or click to select</p>
        <input type="file" id="file-input" accept=".kmz" />
        <br/>
        <button id="start-btn" disabled>🚀 Start Analysis</button>
      </div>
    </div>
    <div class="card" id="progress-section">
      <h2>Progress</h2>
      <div id="log"></div>
    </div>
    <div class="card" id="results-section" style="display:none">
      <div class="stats">
        <div class="stat"><strong id="stat-total">—</strong>Total</div>
        <div class="stat"><strong id="stat-with-pop">—</strong>With Population</div>
        <div class="stat"><strong id="stat-over-10k">—</strong>Population &gt; 10K</div>
      </div>
      <div id="map"></div>
      <button id="download-btn">📥 Download Excel</button>
      <div style="overflow-x:auto; margin-top:1rem;">
        <table id="results-table">
          <thead><tr>
            <th>Name</th><th>Type</th><th>Lat</th><th>Lon</th>
            <th>GPT Pop</th><th>Gemini Pop</th><th>Combined Pop</th><th>Confidence</th>
          </tr></thead>
          <tbody id="results-body"></tbody>
        </table>
      </div>
    </div>
  </main>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `static/app.js`**

```javascript
let currentJobId = null;
let map = null;

const fileInput = document.getElementById("file-input");
const startBtn = document.getElementById("start-btn");
const uploadArea = document.getElementById("upload-area");
const progressSection = document.getElementById("progress-section");
const logEl = document.getElementById("log");
const resultsSection = document.getElementById("results-section");
const downloadBtn = document.getElementById("download-btn");

// Drag and drop
uploadArea.addEventListener("click", () => fileInput.click());
uploadArea.addEventListener("dragover", e => { e.preventDefault(); uploadArea.classList.add("drag-over"); });
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("drag-over"));
uploadArea.addEventListener("drop", e => {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");
  fileInput.files = e.dataTransfer.files;
  startBtn.disabled = false;
});
fileInput.addEventListener("change", () => { startBtn.disabled = !fileInput.files.length; });

function log(msg) {
  const line = document.createElement("div");
  line.textContent = msg;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

startBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  startBtn.disabled = true;
  progressSection.style.display = "block";
  logEl.innerHTML = "";
  log("Uploading file...");

  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: form });
  if (!res.ok) { log("Upload failed: " + await res.text()); return; }
  const { job_id } = await res.json();
  currentJobId = job_id;
  log("Job started: " + job_id);
  streamProgress(job_id);
});

function streamProgress(jobId) {
  const es = new EventSource(`/api/stream/${jobId}`);
  es.addEventListener("progress", e => log(e.data));
  es.addEventListener("complete", e => {
    const info = JSON.parse(e.data);
    log(`✅ Complete! ${info.location_count} locations found.`);
    es.close();
    fetchAndShowResults(jobId);
  });
  es.addEventListener("error", e => {
    log("❌ Error: " + (e.data || "Unknown error"));
    es.close();
  });
  es.onerror = () => { log("SSE connection lost."); es.close(); };
}

async function fetchAndShowResults(jobId) {
  const res = await fetch(`/api/status/${jobId}`);
  const data = await res.json();
  // We don't have a /api/locations endpoint — locations come from the job in memory
  // Show download button
  downloadBtn.style.display = "inline-block";
  downloadBtn.onclick = () => {
    window.location.href = `/api/download/${jobId}`;
  };
  resultsSection.style.display = "block";
  // Note: to show map/table we need a /api/locations/{job_id} endpoint (added in Task 4)
}
```

- [ ] **Step 3: Create the static directory if it doesn't exist**

Run: `mkdir -p static`

- [ ] **Step 4: Manual smoke test — start the server**

Run: `pip install -r requirements-server.txt && uvicorn server.main:app --reload --port 8000`

Expected: server starts, `http://localhost:8000` shows the upload UI.

- [ ] **Step 5: Commit**

```bash
git add static/
git commit -m "feat: add HTML/JS frontend for FastAPI app"
```

---

### Task 4: Locations endpoint + map and table rendering

**Files:**
- Modify: `server/routes/analyze.py` — add `/api/locations/{job_id}`
- Modify: `static/app.js` — fetch locations, render Leaflet map + results table

**Interfaces:**
- Produces: `GET /api/locations/{job_id}` → `{"locations": list[dict]}`

- [ ] **Step 1: Add `GET /api/locations/{job_id}` to `server/routes/analyze.py`**

After the `download` route in `server/routes/analyze.py`, add:

```python
@router.get("/api/locations/{job_id}")
def locations(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"locations": job.locations or []}
```

- [ ] **Step 2: Update `fetchAndShowResults` in `static/app.js`**

Replace the `fetchAndShowResults` function:

```javascript
async function fetchAndShowResults(jobId) {
  downloadBtn.style.display = "inline-block";
  downloadBtn.onclick = () => { window.location.href = `/api/download/${jobId}`; };
  resultsSection.style.display = "block";

  const res = await fetch(`/api/locations/${jobId}`);
  const { locations } = await res.json();

  // Stats
  const withPop = locations.filter(l => l.combined_population).length;
  const over10k = locations.filter(l => (l.combined_population || 0) > 10000).length;
  document.getElementById("stat-total").textContent = locations.length;
  document.getElementById("stat-with-pop").textContent = withPop;
  document.getElementById("stat-over-10k").textContent = over10k;

  // Map
  if (map) { map.remove(); map = null; }
  const locs = locations.filter(l => l.latitude && l.longitude);
  if (locs.length) {
    const center = [
      locs.reduce((s, l) => s + l.latitude, 0) / locs.length,
      locs.reduce((s, l) => s + l.longitude, 0) / locs.length,
    ];
    map = L.map("map").setView(center, 9);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap contributors"
    }).addTo(map);
    locs.forEach(l => {
      L.circleMarker([l.latitude, l.longitude], { radius: 6, color: "#FF5733", fillOpacity: 0.8 })
        .bindPopup(`<b>${l.name}</b><br/>${l.type || ""}<br/>Pop: ${l.combined_population || "—"}`)
        .addTo(map);
    });
  }

  // Table
  const tbody = document.getElementById("results-body");
  tbody.innerHTML = "";
  locations.forEach(l => {
    const tr = document.createElement("tr");
    const fmt = n => n ? Number(n).toLocaleString() : "—";
    tr.innerHTML = `
      <td>${l.name || "—"}</td>
      <td>${l.type || "—"}</td>
      <td>${l.latitude ? l.latitude.toFixed(4) : "—"}</td>
      <td>${l.longitude ? l.longitude.toFixed(4) : "—"}</td>
      <td>${fmt(l.gpt_population)}</td>
      <td>${fmt(l.gemini_population)}</td>
      <td>${fmt(l.combined_population)}</td>
      <td>${l.combined_confidence || "—"}</td>
    `;
    tbody.appendChild(tr);
  });
}
```

- [ ] **Step 3: Manual test — full run**

Run: `uvicorn server.main:app --reload --port 8000`

Upload `Boston Area.kmz`. Verify:
1. Progress log fills with messages
2. SSE `complete` event fires
3. Map shows location pins
4. Table populates with population data
5. "Download Excel" downloads a valid `.xlsx`

- [ ] **Step 4: Commit**

```bash
git add server/routes/analyze.py static/app.js
git commit -m "feat: add locations endpoint, render Leaflet map and results table"
```

---

### Task 5: Dockerfile + Railway config

**Files:**
- Create: `Dockerfile`
- Create: `railway.toml`

**Interfaces:**
- Produces: Docker image that runs `uvicorn server.main:app --host 0.0.0.0 --port $PORT`

- [ ] **Step 1: Create `Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-server.txt

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

- [ ] **Step 2: Create `railway.toml`**

```toml
[build]
builder = "dockerfile"

[deploy]
startCommand = "uvicorn server.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 30
restartPolicyType = "on-failure"
```

- [ ] **Step 3: Test Docker build locally**

Run:
```bash
docker build -t kmz-scraper .
docker run -p 8080:8080 -e PORT=8080 -e OPENAI_API_KEY=test -e GEMINI_API_KEY=test kmz-scraper
```

Expected: server starts, `http://localhost:8080/health` returns `{"status":"ok"}`

- [ ] **Step 4: Commit and push**

```bash
git add Dockerfile railway.toml requirements-server.txt
git commit -m "feat: add Dockerfile and railway.toml for Railway deployment"
git push origin main
```

- [ ] **Step 5: Deploy to Railway**

**Prerequisite:** Railway MCP must be connected (user confirmed they will do this before executing this task).

Via Railway MCP or Railway dashboard:
1. Create new project → "Deploy from GitHub repo" → select `ParetoLeads/kmz-location-scraper`
2. Set environment variables: `OPENAI_API_KEY`, `GEMINI_API_KEY`
3. Verify Railway picks up `railway.toml` and builds from `Dockerfile`
4. Visit the generated `.railway.app` domain → confirm upload UI loads and `/health` returns 200
