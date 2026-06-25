let currentJobId = null;
let currentFilename = null;
let map = null;
let allLocations = [];
let timerInterval = null;
let jobStartTime = null;

// ── Previous Runs (localStorage) ───────────────────────────────
const RUNS_KEY = "kmz_runs";

function loadLocalRuns() {
  try { return JSON.parse(localStorage.getItem(RUNS_KEY) || "[]"); } catch { return []; }
}

function saveLocalRun(run) {
  const runs = loadLocalRuns();
  const idx = runs.findIndex(r => r.job_id === run.job_id);
  if (idx >= 0) runs[idx] = { ...runs[idx], ...run };
  else runs.unshift(run);
  // Keep at most 50 runs
  if (runs.length > 50) runs.splice(50);
  try { localStorage.setItem(RUNS_KEY, JSON.stringify(runs)); } catch {}
}

function fmtDate(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) +
    " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

async function renderRunsTable() {
  const container = document.getElementById("runs-content");
  let localRuns = loadLocalRuns();
  if (!localRuns.length) {
    container.innerHTML = '<p class="runs-empty">No previous runs yet.</p>';
    return;
  }

  // Check which job IDs the server still has (for download availability)
  let serverIds = new Set();
  try {
    const res = await fetch("/api/runs");
    if (res.ok) {
      const data = await res.json();
      (data.runs || []).forEach(r => serverIds.add(r.job_id));
      // Merge server location counts into local runs
      const serverMap = {};
      (data.runs || []).forEach(r => { serverMap[r.job_id] = r; });
      localRuns = localRuns.map(r => serverMap[r.job_id] ? { ...r, ...serverMap[r.job_id] } : r);
    }
  } catch {}

  const rows = localRuns.map(r => {
    const name = (r.filename || "unknown.kmz").replace(/\.kmz$/i, "");
    const ts = fmtDate(r.completed_at);
    const count = r.location_count != null ? r.location_count + " locations" : "—";
    const canDownload = serverIds.has(r.job_id) && r.has_excel !== false;
    const dlBtn = canDownload
      ? `<button class="btn-dl" onclick="window.location.href='/api/download/${r.job_id}'">⬇ Excel</button>`
      : `<span class="run-expired">Expired</span>`;
    return `<tr>
      <td class="run-name">${esc(name)}</td>
      <td class="run-ts">${ts}</td>
      <td>${count}</td>
      <td>${dlBtn}</td>
    </tr>`;
  }).join("");

  container.innerHTML = `<table id="runs-table">
    <thead><tr><th>File</th><th>Completed</th><th>Locations</th><th></th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function esc(s) { return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

const fileInput    = document.getElementById("file-input");
const startBtn     = document.getElementById("start-btn");
const uploadArea   = document.getElementById("upload-area");
const fileNameEl   = document.getElementById("file-name");
const progressSection = document.getElementById("progress-section");
const logEl        = document.getElementById("log");
const resultsSection = document.getElementById("results-section");
const downloadBtn  = document.getElementById("download-btn");
const searchInput  = document.getElementById("search-input");

// ── Stage tracker ──────────────────────────────────────────────
const STAGE_MAP = {
  "Stage 1": 1, "Parsing KMZ": 1,
  "Stage 2": 2, "Finding OSM": 2,
  "Stage 3": 3, "Retrieving administrative": 3,
  "Stage 4": 4, "Population estimation": 4,
  "Excel export": 5,
};

function detectStage(msg) {
  for (const [key, num] of Object.entries(STAGE_MAP)) {
    if (msg.includes(key)) return num;
  }
  return null;
}

function setStage(num, done) {
  const el = document.getElementById("stage-" + num);
  if (!el) return;
  el.classList.remove("active", "done");
  el.classList.add(done ? "done" : "active");
  // Mark all prior stages done
  if (done) {
    for (let i = 1; i < num; i++) {
      const prev = document.getElementById("stage-" + i);
      if (prev) { prev.classList.remove("active"); prev.classList.add("done"); }
    }
  }
}

// ── File selection ──────────────────────────────────────────────
uploadArea.addEventListener("click", () => fileInput.click());
uploadArea.addEventListener("dragover", e => { e.preventDefault(); uploadArea.classList.add("drag-over"); });
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("drag-over"));
uploadArea.addEventListener("drop", e => {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    onFileSelected();
  }
});
fileInput.addEventListener("change", onFileSelected);

function onFileSelected() {
  const f = fileInput.files[0];
  startBtn.disabled = !f;
  if (f) {
    fileNameEl.textContent = "📄 " + f.name;
    fileNameEl.style.display = "inline-block";
  }
}

// ── Timer ──────────────────────────────────────────────────────
function fmtSecs(s) {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${String(sec).padStart(2,'0')}s`;
  return `${sec}s`;
}

function startTimer() {
  jobStartTime = Date.now();
  document.getElementById("timer-elapsed").textContent = "0:00";
  document.getElementById("timer-remaining").textContent = "—";
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    const elapsed = (Date.now() - jobStartTime) / 1000;
    const s = Math.floor(elapsed);
    const m = Math.floor(s / 60);
    document.getElementById("timer-elapsed").textContent =
      m > 0 ? `${m}m ${String(s % 60).padStart(2,'0')}s` : `${s}s`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
  document.getElementById("timer-pulse").style.animation = "none";
  document.getElementById("timer-pulse").style.background = "#16A34A";
}

// ── Log ────────────────────────────────────────────────────────
function log(msg) {
  const line = document.createElement("div");
  if (msg.includes("CHECKPOINT"))      line.className = "ln-checkpoint";
  else if (msg.includes("Warning"))    line.className = "ln-warn";
  else if (msg.includes("❌") || msg.includes("Error") || msg.includes("error"))
                                       line.className = "ln-err";
  else if (msg.includes("✅") || msg.includes("Complete")) line.className = "ln-ok";
  line.textContent = msg;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

// ── Upload + stream ────────────────────────────────────────────
startBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  startBtn.disabled = true;
  resultsSection.style.display = "none";
  progressSection.style.display = "block";
  logEl.innerHTML = "";

  // Reset stage tracker
  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById("stage-" + i);
    if (el) el.classList.remove("active", "done");
  }

  log("Uploading " + file.name + "…");

  const form = new FormData();
  form.append("file", file);
  let res;
  try {
    res = await fetch("/api/upload", { method: "POST", body: form });
  } catch (err) {
    log("❌ Upload failed: " + err.message);
    startBtn.disabled = false;
    return;
  }
  if (!res.ok) {
    log("❌ Upload error: " + await res.text());
    startBtn.disabled = false;
    return;
  }
  const data = await res.json();
  currentJobId = data.job_id;
  currentFilename = data.filename || file.name;
  // Save stub to localStorage so it appears immediately
  saveLocalRun({ job_id: data.job_id, filename: currentFilename, completed_at: null, location_count: null });
  renderRunsTable();
  startTimer();
  streamProgress(data.job_id);
});

function streamProgress(jobId) {
  const es = new EventSource("/api/stream/" + jobId);

  es.addEventListener("progress", e => {
    const msg = e.data;
    log(msg);

    // Drive stage tracker from CHECKPOINT messages
    if (msg.includes("CHECKPOINT")) {
      const stageNum = detectStage(msg);
      if (stageNum) {
        const isDone = msg.includes("completed") || msg.includes("successfully");
        setStage(stageNum, isDone);
      }
    }

    // Update estimated remaining from log line "Estimated remaining time: Xm Ys"
    const estMatch = msg.match(/Estimated remaining time:\s*(.+)/);
    if (estMatch) {
      document.getElementById("timer-remaining").textContent = estMatch[1].trim();
    }
  });

  es.addEventListener("complete", e => {
    let info = {};
    try { info = JSON.parse(e.data); } catch (_) {}
    for (let i = 1; i <= 5; i++) setStage(i, true);
    stopTimer();
    const elapsed = jobStartTime ? fmtSecs((Date.now() - jobStartTime) / 1000) : "";
    document.getElementById("timer-remaining").textContent = "done";
    log("✅ Complete — " + (info.location_count || 0) + " locations found" + (elapsed ? " in " + elapsed : "") + ".");
    // Record completed run
    saveLocalRun({
      job_id: jobId,
      filename: currentFilename || jobId + ".kmz",
      completed_at: Math.floor(Date.now() / 1000),
      location_count: info.location_count || 0,
      has_excel: (info.location_count || 0) > 0,
    });
    renderRunsTable();
    es.close();
    fetchAndShowResults(jobId);
    startBtn.disabled = false;
  });

  es.addEventListener("error", e => {
    stopTimer();
    log("❌ Error: " + (e.data || "processing failed"));
    es.close();
    startBtn.disabled = false;
  });

  es.onerror = () => es.close();
}

// ── Results ────────────────────────────────────────────────────
async function fetchAndShowResults(jobId) {
  downloadBtn.style.display = "inline-flex";
  downloadBtn.onclick = () => { window.location.href = "/api/download/" + jobId; };

  let data;
  try {
    const res = await fetch("/api/locations/" + jobId);
    data = await res.json();
  } catch (err) {
    log("Could not load location data: " + err.message);
    return;
  }

  allLocations = data.locations || [];
  renderResults(allLocations);
  resultsSection.style.display = "block";
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderResults(locs) {
  const withPop  = locs.filter(l => l.combined_population).length;
  const over10k  = locs.filter(l => (l.combined_population || 0) > 10000).length;
  document.getElementById("stat-total").textContent   = locs.length;
  document.getElementById("stat-with-pop").textContent = withPop;
  document.getElementById("stat-over-10k").textContent = over10k;
  renderMap(locs);
  renderTable(locs);
}

function renderMap(locs) {
  if (map) { map.remove(); map = null; }
  const valid = locs.filter(l => l.latitude && l.longitude);
  if (!valid.length) return;

  const lat = valid.reduce((s, l) => s + l.latitude, 0) / valid.length;
  const lon = valid.reduce((s, l) => s + l.longitude, 0) / valid.length;
  map = L.map("map").setView([lat, lon], 9);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a>"
  }).addTo(map);

  valid.forEach(l => {
    const h   = l.admin_hierarchy || {};
    const pop = l.combined_population ? Number(l.combined_population).toLocaleString() : "—";
    const loc = [h.level_8_name, h.level_6_name, h.level_4_name].filter(Boolean).join(", ") || "";
    L.circleMarker([l.latitude, l.longitude], {
      radius: 6, color: "#F26522", fillColor: "#F26522", fillOpacity: 0.75, weight: 1.5
    }).bindPopup(
      "<b>" + (l.name || "—") + "</b>" +
      (loc ? "<br/><span style='color:#6B7280;font-size:0.85em'>" + loc + "</span>" : "") +
      "<br/>Pop: " + pop
    ).addTo(map);
  });
}

function renderTable(locs) {
  const tbody = document.getElementById("results-body");
  tbody.innerHTML = "";

  const fmt = n => (n != null && n !== "" && !isNaN(n)) ? Number(n).toLocaleString() : "—";

  locs.forEach(l => {
    const h    = l.admin_hierarchy || {};
    const conf = (l.combined_confidence || "").trim();
    const badgeClass = conf === "High" ? "b-high" : conf === "Medium" ? "b-medium" : conf === "Low" ? "b-low" : "";
    const typeClass = "b-type";
    const names = Array.isArray(l.local_names) ? l.local_names.join(", ") : (l.local_names || "");

    const tr = document.createElement("tr");
    tr.innerHTML =
      "<td><strong>" + esc(l.name) + "</strong></td>" +
      "<td><span class='badge " + typeClass + "'>" + esc(l.type) + "</span></td>" +
      "<td>" + esc(h.level_8_name || "—") + "</td>" +
      "<td>" + esc(h.level_6_name || "—") + "</td>" +
      "<td>" + esc(h.level_4_name || "—") + "</td>" +
      "<td><strong>" + fmt(l.combined_population) + "</strong></td>" +
      "<td>" + (badgeClass ? "<span class='badge " + badgeClass + "'>" + esc(conf) + "</span>" : "—") + "</td>" +
      "<td class='local-names'>" + esc(names) + "</td>";
    tbody.appendChild(tr);
  });
}

// ── Search ─────────────────────────────────────────────────────
searchInput.addEventListener("input", () => {
  const term = searchInput.value.toLowerCase();
  renderTable(term ? allLocations.filter(l => (l.name || "").toLowerCase().includes(term)) : allLocations);
});

// ── Init ───────────────────────────────────────────────────────
renderRunsTable();
