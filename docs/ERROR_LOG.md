# Overpass API Error Log

Track recurring errors, root-cause analysis, and fix attempts so we don't repeat failed approaches.

---

## Error: HTTP 406 from Overpass API (persistent, Stage 2)

**Symptom**: Stage 2 (Finding OSM Locations) always returns 0 locations. App completes all stages but exports empty Excel.

**HTTP error**: `406 Not Acceptable` from `https://overpass-api.de/api/interpreter`

**Log pattern**:
```
primary query: Primary timeout, trying fallback https://overpass-api.de/api/interpreter
primary query: DIAGNOSTIC: HTTPError | msg=406 Client Error: Not Acceptable for url: https://overpass-api.de/api/interpreter
primary query: HTTP error 406. Waiting 5s before retry (attempt 1/3)...
... (repeats 3 times, all identical pattern)
Error querying OpenStreetMap for primary locations: Failed after 3 attempts. HTTP error 406
```

Same pattern repeats for `additional` and `special` queries.

---

### Fix Attempt 1 — FAILED
**Commit**: `fix: correct Overpass API POST encoding (data dict instead of raw string)`  
**What changed**: Changed POST body from raw string to `data={"data": query}` dict.  
**Why it failed**: POST encoding was not the cause of 406. The server was already receiving valid form data.

### Fix Attempt 2 — FAILED
**Commit**: `feat: add second Overpass fallback mirror for resilience`  
**What changed**: Added `https://overpass.openstreetmap.ru/api/interpreter` to `OVERPASS_FALLBACK_URLS`.  
**Why it failed**: The fallback loop has a bug — it only `continue`s on `Timeout`, but `break`s on **any** non-timeout response including HTTP 406. So `openstreetmap.ru` is never reached when `overpass-api.de` returns 406.

### Fix Attempt 3 — FAILED
**Commit**: `fix: strip leading whitespace from Overpass queries before sending`  
**What changed**: Called `.strip()` on query string before POST.  
**Why it failed**: Whitespace was not the cause of 406. Overpass QL is whitespace-tolerant.

---

### Fix Attempt 4 — IN PROGRESS
**Root cause identified**: The fallback loop in `_execute_query` (location_analyzer.py ~line 521) uses `break` after any non-timeout POST response. When `overpass-api.de` returns 406, the loop exits immediately — `openstreetmap.ru` is never tried. The 406 then gets re-raised, retried 3 times, and all 3 attempts fail identically.

**Fix**:
- Change fallback loop to only `break` on a 2xx response
- On non-2xx (e.g. 406), log and continue to the next fallback URL
- Apply same fix to hierarchy query (`_execute_hierarchy_query`), which only falls back to one URL anyway
- Improve User-Agent header to include contact info (Overpass policy)

**Files changed**: `location_analyzer.py`
