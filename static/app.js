let currentJobId = null;
let map = null;
let allLocations = [];

const fileInput = document.getElementById("file-input");
const startBtn = document.getElementById("start-btn");
const uploadArea = document.getElementById("upload-area");
const progressSection = document.getElementById("progress-section");
const logEl = document.getElementById("log");
const resultsSection = document.getElementById("results-section");
const downloadBtn = document.getElementById("download-btn");
const searchInput = document.getElementById("search-input");

// File selection
uploadArea.addEventListener("click", () => fileInput.click());
uploadArea.addEventListener("dragover", e => { e.preventDefault(); uploadArea.classList.add("drag-over"); });
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("drag-over"));
uploadArea.addEventListener("drop", e => {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    startBtn.disabled = false;
  }
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
  resultsSection.style.display = "none";
  progressSection.style.display = "block";
  logEl.innerHTML = "";
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
  log("Job started: " + job_id);
  streamProgress(job_id);
});

function streamProgress(jobId) {
  const es = new EventSource("/api/stream/" + jobId);

  es.addEventListener("progress", e => log(e.data));

  es.addEventListener("complete", e => {
    let info = {};
    try { info = JSON.parse(e.data); } catch (_) {}
    log("✅ Complete! " + (info.location_count || 0) + " locations found.");
    es.close();
    fetchAndShowResults(jobId);
    startBtn.disabled = false;
  });

  es.addEventListener("error", e => {
    log("❌ Processing error: " + (e.data || "unknown"));
    es.close();
    startBtn.disabled = false;
  });

  es.onerror = () => {
    // SSE connection drop after completion is normal; ignore if job is done
    es.close();
  };
}

async function fetchAndShowResults(jobId) {
  downloadBtn.style.display = "inline-block";
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
}

function renderResults(locs) {
  const withPop = locs.filter(l => l.combined_population).length;
  const over10k = locs.filter(l => (l.combined_population || 0) > 10000).length;
  document.getElementById("stat-total").textContent = locs.length;
  document.getElementById("stat-with-pop").textContent = withPop;
  document.getElementById("stat-over-10k").textContent = over10k;

  renderMap(locs);
  renderTable(locs);
}

function renderMap(locs) {
  if (map) { map.remove(); map = null; }
  const valid = locs.filter(l => l.latitude && l.longitude);
  if (!valid.length) return;

  const centerLat = valid.reduce((s, l) => s + l.latitude, 0) / valid.length;
  const centerLon = valid.reduce((s, l) => s + l.longitude, 0) / valid.length;

  map = L.map("map").setView([centerLat, centerLon], 9);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
  }).addTo(map);

  valid.forEach(l => {
    const pop = l.combined_population ? Number(l.combined_population).toLocaleString() : "—";
    L.circleMarker([l.latitude, l.longitude], {
      radius: 6,
      color: "#FF5733",
      fillColor: "#FF5733",
      fillOpacity: 0.75,
      weight: 1.5
    })
      .bindPopup("<b>" + (l.name || "—") + "</b><br/>" + (l.type || "") + "<br/>Pop: " + pop)
      .addTo(map);
  });
}

function renderTable(locs) {
  const tbody = document.getElementById("results-body");
  tbody.innerHTML = "";
  const fmt = n => (n != null && n !== "" && !isNaN(n)) ? Number(n).toLocaleString() : "—";
  locs.forEach(l => {
    const tr = document.createElement("tr");
    tr.innerHTML =
      "<td>" + (l.name || "—") + "</td>" +
      "<td>" + (l.type || "—") + "</td>" +
      "<td>" + (l.latitude != null ? l.latitude.toFixed(4) : "—") + "</td>" +
      "<td>" + (l.longitude != null ? l.longitude.toFixed(4) : "—") + "</td>" +
      "<td>" + fmt(l.gpt_population) + "</td>" +
      "<td>" + fmt(l.gemini_population) + "</td>" +
      "<td>" + fmt(l.combined_population) + "</td>" +
      "<td>" + (l.combined_confidence || "—") + "</td>";
    tbody.appendChild(tr);
  });
}

// Live search filter
searchInput.addEventListener("input", () => {
  const term = searchInput.value.toLowerCase();
  const filtered = term
    ? allLocations.filter(l => (l.name || "").toLowerCase().includes(term))
    : allLocations;
  renderTable(filtered);
});
