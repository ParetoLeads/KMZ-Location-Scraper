# Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut end-to-end processing time for a 100-location run by ~2–3 minutes through three targeted changes: combining OSM discovery queries, parallelizing GPT+Gemini calls, and increasing hierarchy batch size.

**Architecture:** All changes are in `location_analyzer.py` and `config.py`. No new files. No changes to `app.py` or the state machine. Safe to deploy to the existing Streamlit app.

**Tech Stack:** Python `concurrent.futures.ThreadPoolExecutor` (stdlib, no new dependency)

## Global Constraints

- Do not change the Overpass query format (`[out:json]`) — required by all mirrors
- Do not remove the fallback loop in `_find_osm_locations` — see `docs/ERROR_LOG.md`
- `HIERARCHY_FIRST_BATCH_SIZE` and `DEFAULT_BATCH_SIZE` live in `config.py`
- All changes must leave the existing state machine in `app.py` unchanged
- Bump `APP_VERSION` in `config.py` after all tasks complete

---

## File Structure

- Modify: `location_analyzer.py` — Task 1 (combined OSM query), Task 2 (parallel AI)
- Modify: `config.py` — Task 3 (hierarchy batch size, version bump)
- Modify: `tests/test_overpass_fix.py` — Task 1 tests
- Create: `tests/test_performance_opts.py` — Tasks 2 and 3 tests

---

### Task 1: Combine Primary + Additional OSM Queries

**Problem:** `run_kmz_and_osm_only` fires three sequential Overpass requests with 2-second delays between them. Primary and additional use the same tag (`place=`) with different values — they can be merged into one request, saving one round trip and a 2-second delay.

**Files:**
- Modify: `location_analyzer.py:395-413` (query builder methods), `location_analyzer.py:1572-1600` (`run_kmz_and_osm_only`)
- Modify: `tests/test_overpass_fix.py`

**Interfaces:**
- Produces: `_create_combined_place_query(min_lat, min_lon, max_lat, max_lon) -> str`
- `run_kmz_and_osm_only` calls `_find_osm_locations(..., "combined_place", ...)` instead of separate primary + additional calls

- [ ] **Step 1: Write failing test**

Add to `tests/test_overpass_fix.py`:

```python
from location_analyzer import LocationAnalyzer
import os

def _make_analyzer():
    kmz_path = "Boston Area.kmz"
    if not os.path.exists(kmz_path):
        import pytest
        pytest.skip("Boston Area.kmz not present")
    return LocationAnalyzer(
        kmz_file=kmz_path,
        openai_api_key="",
        gemini_api_key="",
    )

def test_combined_place_query_contains_all_types():
    a = _make_analyzer()
    q = a._create_combined_place_query(42.0, -71.5, 42.5, -71.0)
    all_types = a.primary_place_types + a.additional_place_types
    for t in all_types:
        assert t in q, f"Expected place type '{t}' in combined query"

def test_combined_place_query_valid_overpass_syntax():
    a = _make_analyzer()
    q = a._create_combined_place_query(42.0, -71.5, 42.5, -71.0)
    assert "[out:json]" in q
    assert "out body" in q
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `python -m pytest tests/test_overpass_fix.py::test_combined_place_query_contains_all_types -v`
Expected: `FAILED` with `AttributeError: '_create_combined_place_query'`

- [ ] **Step 3: Add `_create_combined_place_query` method**

In `location_analyzer.py`, after the `_create_special_osm_query` method (around line 413), add:

```python
def _create_combined_place_query(self, min_lat, min_lon, max_lat, max_lon):
    """Merge primary + additional place queries into one Overpass request."""
    combined_pattern = "|".join(self.primary_place_types + self.additional_place_types)
    return self._create_osm_query(min_lat, min_lon, max_lat, max_lon, combined_pattern)
```

- [ ] **Step 4: Run tests to confirm PASS**

Run: `python -m pytest tests/test_overpass_fix.py::test_combined_place_query_contains_all_types tests/test_overpass_fix.py::test_combined_place_query_valid_overpass_syntax -v`
Expected: both PASS

- [ ] **Step 5: Update `_find_osm_locations` to support `"combined_place"` query type**

In `location_analyzer.py`, find `_find_osm_locations` (around line 500). In the `if query_type == "primary":` block, add a branch for `"combined_place"`:

```python
if query_type == "primary":
    query = self._create_primary_osm_query(min_lat, min_lon, max_lat, max_lon)
elif query_type == "additional":
    query = self._create_additional_osm_query(min_lat, min_lon, max_lat, max_lon)
elif query_type == "combined_place":
    query = self._create_combined_place_query(min_lat, min_lon, max_lat, max_lon)
elif query_type == "special":
    query = self._create_special_osm_query(min_lat, min_lon, max_lat, max_lon)
else:
    raise ValueError(f"Invalid query_type: {query_type}")
```

Also update the cache key call below it — `cache_osm_query` and `set_osm_query_cache` use `query_type` as the cache key, which already works for `"combined_place"` since it's just a string key. No change needed there.

- [ ] **Step 6: Update `run_kmz_and_osm_only` to use combined query**

In `location_analyzer.py`, find `run_kmz_and_osm_only` (around line 1572). Replace the three sequential calls:

**Old code (lines ~1580–1588):**
```python
primary_locations = self._find_osm_locations(self.polygon_points, "primary", self.primary_place_types)
time.sleep(config.OSM_QUERY_DELAY)
additional_places = []
if self.additional_place_types:
    additional_places = self._find_osm_locations(self.polygon_points, "additional", self.additional_place_types)
    time.sleep(config.OSM_QUERY_DELAY)
special_locations = []
if self.special_place_types:
    special_locations = self._find_osm_locations(self.polygon_points, "special", self.special_place_types)
administrative_areas = []
all_locations = primary_locations + additional_places + special_locations + administrative_areas
```

**New code:**
```python
combined_place_types = self.primary_place_types + self.additional_place_types
place_locations = self._find_osm_locations(
    self.polygon_points, "combined_place", combined_place_types
)
time.sleep(config.OSM_QUERY_DELAY)
special_locations = []
if self.special_place_types:
    special_locations = self._find_osm_locations(self.polygon_points, "special", self.special_place_types)
all_locations = place_locations + special_locations
```

Also apply the same change to the `run()` method (around line 1614) so the CLI path gets the same optimization.

- [ ] **Step 7: Commit**

```bash
git add location_analyzer.py tests/test_overpass_fix.py
git commit -m "perf: combine primary+additional OSM queries into one request"
```

---

### Task 2: Parallelize GPT and Gemini Population Calls

**Problem:** In `estimate_populations_single_batch`, GPT and Gemini are called sequentially (lines ~1357–1360), even though they use independent HTTP clients. Running them with a thread pool cuts per-batch time roughly in half (~8–15s → ~5–10s per batch).

**Files:**
- Modify: `location_analyzer.py:1326–1374` (`estimate_populations_single_batch`)
- Create: `tests/test_performance_opts.py`

**Interfaces:**
- Produces: same `gpt_results_map` and `gemini_results_map` dicts — no interface change to callers

- [ ] **Step 1: Write failing test**

Create `tests/test_performance_opts.py`:

```python
import time
import concurrent.futures
from unittest.mock import patch, MagicMock
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from location_analyzer import LocationAnalyzer

KMZ = "Boston Area.kmz"

def _make_analyzer():
    if not os.path.exists(KMZ):
        import pytest; pytest.skip("Boston Area.kmz not present")
    return LocationAnalyzer(kmz_file=KMZ, openai_api_key="fake", gemini_api_key="fake")

def test_gpt_and_gemini_called_concurrently():
    """Both AI calls must overlap in time (not strictly sequential)."""
    a = _make_analyzer()
    a.use_gpt = True
    a.use_openai = True
    a.use_gemini_flag = True
    a._ai_clients_initialized = True

    call_times = {}

    def fake_gpt(batch, start):
        call_times["gpt_start"] = time.time()
        time.sleep(0.3)
        call_times["gpt_end"] = time.time()
        return {}

    def fake_gemini(batch, start):
        call_times["gemini_start"] = time.time()
        time.sleep(0.3)
        call_times["gemini_end"] = time.time()
        return {}

    with patch.object(a, "_get_gpt_populations_batch", side_effect=fake_gpt), \
         patch.object(a, "_get_gemini_populations_batch", side_effect=fake_gemini):
        locs = [{"name": "Test", "latitude": 42.0, "longitude": -71.0}]
        a.estimate_populations_single_batch(locs, 0)

    # Concurrent: both should have started before either finished
    assert call_times["gemini_start"] < call_times["gpt_end"], \
        "Gemini must start before GPT finishes (they should overlap)"
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `python -m pytest tests/test_performance_opts.py::test_gpt_and_gemini_called_concurrently -v`
Expected: FAIL — `AssertionError: Gemini must start before GPT finishes`

- [ ] **Step 3: Add `import concurrent.futures` and update `estimate_populations_single_batch`**

At the top of `location_analyzer.py`, add after the existing imports:
```python
import concurrent.futures
```

In `estimate_populations_single_batch`, replace the sequential block (lines ~1357–1365):

**Old:**
```python
if use_openai and use_gemini:
    # Sequential: GPT first, then Gemini. Ensures Gemini rate-limit spacing (e.g. 5 RPM) is enforceable between batches.
    gpt_results_map = self._get_gpt_populations_batch(batch_locations, start_index)
    gemini_results_map = self._get_gemini_populations_batch(batch_locations, start_index)
else:
    if use_openai:
        gpt_results_map = self._get_gpt_populations_batch(batch_locations, start_index)
    if use_gemini:
        gemini_results_map = self._get_gemini_populations_batch(batch_locations, start_index)
```

**New:**
```python
if use_openai and use_gemini:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        gpt_future = executor.submit(self._get_gpt_populations_batch, batch_locations, start_index)
        gemini_future = executor.submit(self._get_gemini_populations_batch, batch_locations, start_index)
        gpt_results_map = gpt_future.result()
        gemini_results_map = gemini_future.result()
else:
    if use_openai:
        gpt_results_map = self._get_gpt_populations_batch(batch_locations, start_index)
    if use_gemini:
        gemini_results_map = self._get_gemini_populations_batch(batch_locations, start_index)
```

- [ ] **Step 4: Run test to confirm PASS**

Run: `python -m pytest tests/test_performance_opts.py::test_gpt_and_gemini_called_concurrently -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add location_analyzer.py tests/test_performance_opts.py
git commit -m "perf: run GPT and Gemini population calls in parallel"
```

---

### Task 3: Increase Hierarchy Batch Size and Bump Version

**Problem:** `HIERARCHY_FIRST_BATCH_SIZE=3` and `DEFAULT_BATCH_SIZE=5` mean a 100-location run does ~20 Overpass round-trips for hierarchy. Raising to 5 and 15 respectively cuts that to ~7 round-trips without meaningfully increasing per-request latency (Overpass handles multi-location queries efficiently).

**Files:**
- Modify: `config.py:48–49`
- Modify: `config.py:13` (version bump)
- Modify: `tests/test_performance_opts.py`

**Interfaces:**
- Consumes: `config.DEFAULT_BATCH_SIZE`, `config.HIERARCHY_FIRST_BATCH_SIZE`
- Produces: no interface change — `app.py` reads these values from config

- [ ] **Step 1: Write failing test**

Add to `tests/test_performance_opts.py`:

```python
from config import config as app_config

def test_hierarchy_batch_size_sufficient():
    """Batch size must be large enough to avoid excessive round-trips for typical use."""
    assert app_config.DEFAULT_BATCH_SIZE >= 15, \
        f"DEFAULT_BATCH_SIZE={app_config.DEFAULT_BATCH_SIZE} is too small; expect >=15"
    assert app_config.HIERARCHY_FIRST_BATCH_SIZE >= 5, \
        f"HIERARCHY_FIRST_BATCH_SIZE={app_config.HIERARCHY_FIRST_BATCH_SIZE} is too small; expect >=5"
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `python -m pytest tests/test_performance_opts.py::test_hierarchy_batch_size_sufficient -v`
Expected: FAIL — `AssertionError: DEFAULT_BATCH_SIZE=5 is too small`

- [ ] **Step 3: Update config.py**

In `config.py`, change lines 48–49:

**Old:**
```python
DEFAULT_BATCH_SIZE: int = 5   # For hierarchy (Overpass); larger batches = fewer round-trips
HIERARCHY_FIRST_BATCH_SIZE: int = 3   # First hierarchy batch size
```

**New:**
```python
DEFAULT_BATCH_SIZE: int = 15   # For hierarchy (Overpass); larger batches = fewer round-trips
HIERARCHY_FIRST_BATCH_SIZE: int = 5   # First hierarchy batch size
```

Also bump the version in `config.py` line 13:
```python
APP_VERSION: str = "1.0.5"
```

- [ ] **Step 4: Run test to confirm PASS**

Run: `python -m pytest tests/test_performance_opts.py::test_hierarchy_batch_size_sufficient -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS (or previously-skipped tests still skip)

- [ ] **Step 6: Commit and push**

```bash
git add config.py tests/test_performance_opts.py
git commit -m "perf: increase hierarchy batch sizes, bump version to v1.0.5"
git push origin main
```
