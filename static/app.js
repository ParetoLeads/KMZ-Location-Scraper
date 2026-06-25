let currentJobId = null;
let map = null;
let allLocations = [];

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
  const { job_id } = await res.json();
  currentJobId = job_id;
  streamProgress(job_id);
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
  });

  es.addEventListener("complete", e => {
    let info = {};
    try { info = JSON.parse(e.data); } catch (_) {}
    // Mark all stages done
    for (let i = 1; i <= 5; i++) setStage(i, true);
    log("✅ Complete — " + (info.location_count || 0) + " locations found.");
    es.close();
    fetchAndShowResults(jobId);
    startBtn.disabled = false;
  });

  es.addEventListener("error", e => {
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
  const esc = s => (s || "").replace(/</g, "&lt;").replace(/>/g, "&gt;");

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
