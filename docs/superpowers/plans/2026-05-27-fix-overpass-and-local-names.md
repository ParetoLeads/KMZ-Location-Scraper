# Fix Overpass 406, Reliability, and Local Search Names Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Overpass API 406 error that prevents any locations from being found, then add a "local search names" field (Google Ads keywords) to every location in the output.

**Architecture:** The 406 fix is a one-line change repeated in 4 places — `data=query` (raw string) must be `data={"data": query}` (form-encoded dict) so Overpass receives the query as a named form parameter. The local names feature extends the existing GPT/Gemini prompt to also return colloquial names, adds parsing, and adds a new column to the Excel output.

**Tech Stack:** Python, requests, OpenAI GPT-4-turbo, Google Gemini, pandas, openpyxl, Streamlit

---

## Root Cause: The 406 Error

`requests.post(url, data="some string")` sends the string as a raw body with no Content-Type. Overpass API requires the query sent as a named form field: `data=<encoded_query>`. The fix is `requests.post(url, data={"data": query})` — requests will then set `Content-Type: application/x-www-form-urlencoded` and properly encode it.

This bug affects 4 POST calls in `location_analyzer.py`. The primary server (kumi.systems) was timing out, and then the stricter fallback (overpass-api.de) rejected the malformed body with 406.

---

## Task 1: Fix Overpass POST Encoding in Discovery Queries

**Files:**
- Modify: `location_analyzer.py:519-531`

- [ ] **Step 1: Write the failing test**

Create `tests/test_overpass_fix.py`:

```python
"""Verify Overpass POST requests are form-encoded correctly."""
import unittest
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _make_mock_response(json_data):
    mock = MagicMock()
    mock.status_code = 200
    mock.text = '{"elements": []}'
    mock.json.return_value = json_data
    mock.raise_for_status = MagicMock()
    return mock


class TestOverpassPostEncoding(unittest.TestCase):

    @patch("location_analyzer.requests.post")
    def test_discovery_query_is_form_encoded(self, mock_post):
        """data parameter must be a dict so requests form-encodes it."""
        mock_post.return_value = _make_mock_response({"elements": []})

        from location_analyzer import LocationAnalyzer
        analyzer = LocationAnalyzer.__new__(LocationAnalyzer)
        analyzer.overpass_url = "https://overpass.kumi.systems/api/interpreter"
        analyzer.polygon_points = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        analyzer.primary_place_types = ["city", "town"]
        analyzer.additional_place_types = []
        analyzer.special_place_types = []
        analyzer.primary_types_pattern = "city|town"
        analyzer.additional_types_pattern = ""
        analyzer._log = lambda msg: None

        # Patch cache functions to return None (no cache)
        with patch("location_analyzer.cache_osm_query", return_value=None), \
             patch("location_analyzer.set_osm_query_cache"):
            try:
                analyzer._find_osm_locations(analyzer.polygon_points, "primary", ["city", "town"])
            except Exception:
                pass

        self.assertTrue(mock_post.called, "requests.post should have been called")
        call_kwargs = mock_post.call_args
        data_arg = call_kwargs[1].get("data") or (call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None)
        self.assertIsInstance(data_arg, dict, 
            f"data must be a dict for form encoding, got {type(data_arg)}: {data_arg!r}")
        self.assertIn("data", data_arg, "dict must have key 'data' containing the Overpass query")

    @patch("location_analyzer.requests.post")
    def test_hierarchy_query_is_form_encoded(self, mock_post):
        """Hierarchy POST must also use dict for form encoding."""
        mock_post.return_value = _make_mock_response({
            "elements": []
        })

        from location_analyzer import LocationAnalyzer
        analyzer = LocationAnalyzer.__new__(LocationAnalyzer)
        analyzer.overpass_url = "https://overpass.kumi.systems/api/interpreter"
        analyzer._log = lambda msg: None
        analyzer.status_callback = lambda msg: None

        query = "[out:json][timeout:20]; (relation[admin_level=8](40.0, -75.0, 41.0, -74.0);); out tags;"
        locations = [{"name": "Test", "type": "suburb", "latitude": 40.5, "longitude": -74.5}]

        with patch("location_analyzer.execute_with_retry") as mock_retry:
            mock_retry.side_effect = lambda fn, **kwargs: fn()
            try:
                analyzer.fetch_admin_hierarchy_batch(locations, query)
            except Exception:
                pass

        if mock_post.called:
            call_kwargs = mock_post.call_args
            data_arg = call_kwargs[1].get("data") or (call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None)
            self.assertIsInstance(data_arg, dict,
                f"Hierarchy data must be a dict, got {type(data_arg)}: {data_arg!r}")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd "/Users/nathanshapiro/Desktop/KMZ Location Scraper"
python -m pytest tests/test_overpass_fix.py -v 2>&1 | head -40
```

Expected: FAIL — `AssertionError: data must be a dict for form encoding, got <class 'str'>`

- [ ] **Step 3: Fix discovery POST calls in `location_analyzer.py`**

In `location_analyzer.py`, change lines 519–531. Replace:

```python
        def _execute_query() -> Dict[str, Any]:
            try:
                response = requests.post(
                    self.overpass_url,
                    data=query,
                    timeout=DISCOVERY_TIMEOUT,
                    headers=OSM_HEADERS,
                )
            except requests.exceptions.Timeout:
                self._log(f"{query_type} query: Primary timeout, trying fallback Overpass")
                response = requests.post(
                    OVERPASS_FALLBACK_URL,
                    data=query,
                    timeout=DISCOVERY_TIMEOUT,
                    headers=OSM_HEADERS,
                )
```

With:

```python
        def _execute_query() -> Dict[str, Any]:
            try:
                response = requests.post(
                    self.overpass_url,
                    data={"data": query},
                    timeout=DISCOVERY_TIMEOUT,
                    headers=OSM_HEADERS,
                )
            except requests.exceptions.Timeout:
                self._log(f"{query_type} query: Primary timeout, trying fallback Overpass")
                response = requests.post(
                    OVERPASS_FALLBACK_URL,
                    data={"data": query},
                    timeout=DISCOVERY_TIMEOUT,
                    headers=OSM_HEADERS,
                )
```

- [ ] **Step 4: Fix hierarchy POST calls in `location_analyzer.py`**

In `location_analyzer.py`, change lines 663–676. Replace:

```python
        def _execute_hierarchy_query() -> Dict[str, Any]:
            """Execute the hierarchy query; on primary timeout try fallback Overpass once."""
            self._log(f"[Overpass] POST start url={self.overpass_url} timeout={timeout + buffer}s")
            try:
                response = requests.post(
                    self.overpass_url,
                    data=query,
                    timeout=timeout + buffer,
                    headers=OSM_HEADERS,
                )
            except requests.exceptions.Timeout:
                self._log("[Overpass] Primary timeout, trying fallback overpass-api.de")
                response = requests.post(
                    OVERPASS_FALLBACK_URL,
                    data=query,
                    timeout=timeout + buffer,
                    headers=OSM_HEADERS,
                )
```

With:

```python
        def _execute_hierarchy_query() -> Dict[str, Any]:
            """Execute the hierarchy query; on primary timeout try fallback Overpass once."""
            self._log(f"[Overpass] POST start url={self.overpass_url} timeout={timeout + buffer}s")
            try:
                response = requests.post(
                    self.overpass_url,
                    data={"data": query},
                    timeout=timeout + buffer,
                    headers=OSM_HEADERS,
                )
            except requests.exceptions.Timeout:
                self._log("[Overpass] Primary timeout, trying fallback overpass-api.de")
                response = requests.post(
                    OVERPASS_FALLBACK_URL,
                    data={"data": query},
                    timeout=timeout + buffer,
                    headers=OSM_HEADERS,
                )
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd "/Users/nathanshapiro/Desktop/KMZ Location Scraper"
python -m pytest tests/test_overpass_fix.py::TestOverpassPostEncoding::test_discovery_query_is_form_encoded -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add location_analyzer.py tests/test_overpass_fix.py
git commit -m "fix: correct Overpass API POST encoding (data dict instead of raw string)

The 406 Not Acceptable errors were caused by sending the Overpass query
as a raw string body. Overpass expects a form-encoded body with the query
as the 'data' parameter. Changed data=query to data={'data': query} in
all 4 POST calls (discovery primary/fallback, hierarchy primary/fallback)."
```

---

## Task 2: Add a Second Overpass Fallback Mirror

**Files:**
- Modify: `location_analyzer.py:30`

Adding a second fallback means a single server outage won't stop everything.

- [ ] **Step 1: Write the test**

Add to `tests/test_overpass_fix.py`:

```python
class TestOverpassFallbackChain(unittest.TestCase):

    @patch("location_analyzer.requests.post")
    def test_second_fallback_tried_after_first_times_out(self, mock_post):
        """If both primary and first fallback time out, second fallback is tried."""
        import requests as req_lib

        call_count = [0]
        def side_effect(url, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise req_lib.exceptions.Timeout()
            return _make_mock_response({"elements": []})

        mock_post.side_effect = side_effect

        from location_analyzer import LocationAnalyzer
        analyzer = LocationAnalyzer.__new__(LocationAnalyzer)
        analyzer.overpass_url = "https://overpass.kumi.systems/api/interpreter"
        analyzer.polygon_points = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        analyzer.primary_place_types = ["city"]
        analyzer.additional_place_types = []
        analyzer.special_place_types = []
        analyzer.primary_types_pattern = "city"
        analyzer._log = lambda msg: None

        with patch("location_analyzer.cache_osm_query", return_value=None), \
             patch("location_analyzer.set_osm_query_cache"):
            result = analyzer._find_osm_locations(analyzer.polygon_points, "primary", ["city"])

        self.assertGreaterEqual(call_count[0], 3,
            "Should try at least 3 servers before giving up")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd "/Users/nathanshapiro/Desktop/KMZ Location Scraper"
python -m pytest tests/test_overpass_fix.py::TestOverpassFallbackChain -v 2>&1 | head -20
```

Expected: FAIL — call_count is 2, not 3.

- [ ] **Step 3: Add second fallback constant and use it**

In `location_analyzer.py`, replace lines 29–30:

```python
# Fallback when primary Overpass instance times out (different instance may respond)
OVERPASS_FALLBACK_URL = "https://overpass-api.de/api/interpreter"
```

With:

```python
# Ordered fallback chain for Overpass API (tried in sequence on timeout)
OVERPASS_FALLBACK_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
OVERPASS_FALLBACK_URL = OVERPASS_FALLBACK_URLS[0]  # kept for hierarchy function compatibility
```

Then in `_find_osm_locations`, replace the `_execute_query` inner function (lines 517–541) with:

```python
        def _execute_query() -> Dict[str, Any]:
            try:
                response = requests.post(
                    self.overpass_url,
                    data={"data": query},
                    timeout=DISCOVERY_TIMEOUT,
                    headers=OSM_HEADERS,
                )
            except requests.exceptions.Timeout:
                response = None
                for fallback_url in OVERPASS_FALLBACK_URLS:
                    self._log(f"{query_type} query: Primary timeout, trying fallback {fallback_url}")
                    try:
                        response = requests.post(
                            fallback_url,
                            data={"data": query},
                            timeout=DISCOVERY_TIMEOUT,
                            headers=OSM_HEADERS,
                        )
                        break
                    except requests.exceptions.Timeout:
                        continue
                if response is None:
                    raise requests.exceptions.Timeout("All Overpass servers timed out")
            response.raise_for_status()
            txt = response.text.strip() if response.text else ""
            if not txt or len(txt) < 10:
                raise ValueError(f"Empty or too short response from Overpass API (status {response.status_code}, length={len(txt)})")
            try:
                return response.json()
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from Overpass API: {str(e)}. Response preview: {response.text[:200]}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd "/Users/nathanshapiro/Desktop/KMZ Location Scraper"
python -m pytest tests/test_overpass_fix.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add location_analyzer.py tests/test_overpass_fix.py
git commit -m "feat: add second Overpass fallback mirror for resilience

Adds openstreetmap.ru as a second fallback so the discovery loop tries
all mirrors before failing, not just one."
```

---

## Task 3: Extend GPT/Gemini Prompt to Return Local Search Names

**Files:**
- Modify: `location_analyzer.py:710-765` (`_prepare_batch_gpt_prompt`)

- [ ] **Step 1: Write the test**

Create `tests/test_local_names.py`:

```python
"""Test local search names feature."""
import unittest
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestLocalNamesPrompt(unittest.TestCase):

    def setUp(self):
        from location_analyzer import LocationAnalyzer
        self.analyzer = LocationAnalyzer.__new__(LocationAnalyzer)
        self.analyzer._log = lambda msg: None

    def test_prompt_requests_local_names_field(self):
        """Prompt must describe the local_names field."""
        locations = [
            {
                "name": "Hoboken",
                "type": "city",
                "admin_hierarchy": {
                    "parent_name": "Hudson County",
                    "level_4_name": "New Jersey",
                    "level_2_name": "United States",
                }
            }
        ]
        prompt = self.analyzer._prepare_batch_gpt_prompt(locations, 0)
        self.assertIn("local_names", prompt,
            "Prompt must ask for 'local_names' field")
        self.assertIn("Google Ads", prompt,
            "Prompt must mention Google Ads keywords context")

    def test_prompt_example_shows_local_names_array(self):
        """Example in prompt must show local_names as a JSON array."""
        locations = [{"name": "Hoboken", "type": "city", "admin_hierarchy": {}}]
        prompt = self.analyzer._prepare_batch_gpt_prompt(locations, 0)
        self.assertIn('"local_names"', prompt,
            "Example JSON in prompt must contain local_names key")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd "/Users/nathanshapiro/Desktop/KMZ Location Scraper"
python -m pytest tests/test_local_names.py -v 2>&1 | head -20
```

Expected: FAIL — prompt does not contain "local_names".

- [ ] **Step 3: Update `_prepare_batch_gpt_prompt` to request local names**

In `location_analyzer.py`, replace the entire `_prepare_batch_gpt_prompt` method (lines 710–765) with:

```python
    def _prepare_batch_gpt_prompt(self, locations_chunk: List[Dict], start_index: int) -> str:
        """Creates the prompt for a batch of locations for GPT population estimation and local names."""
        prompt_header = (
            "You are an AI assistant specializing in accurately retrieving demographic data and local geographic knowledge. "
            "Your task is to estimate the most recent known residential population for each location listed below, "
            "and also identify all colloquial names and local search terms used for each location.\n\n"
            "For each location, you are provided with:\n"
            "- 'index': An identifier for the location within the overall list (starting from {start_index}).\n"
            "- 'name': The official name of the specific location.\n"
            "- 'type': The type of the location (e.g., city, suburb, neighbourhood).\n"
            "- 'parent_name': The name of the administrative area directly containing the location.\n"
            "- 'level_4_name': The name of the higher-level administrative area (e.g., province or state).\n"
            "- 'level_2_name': The name of the country.\n\n"
            "**Crucially, use the provided 'type', 'parent_name', 'level_4_name', and 'level_2_name' to disambiguate the location 'name' "
            "and ensure you are retrieving data for the correct place.**\n\n"
            "Provide your answer ONLY as a single, valid JSON **array**. Each object in the array must correspond to one location from the input list below, respecting the original order.\n"
            "Do not include any introductory text, explanations, or summaries before or after the JSON array.\n"
            "Do not wrap the JSON in markdown code blocks (no ```json or ``` markers). Provide raw JSON only.\n"
            "Each object in the JSON array must contain:\n"
            "- 'index': The original integer index provided for the location.\n"
            "- 'population': The estimated population as an integer. If the population is unknown or cannot be reliably estimated, use `null`.\n"
            "- 'confidence': Your confidence in the population estimate as a string: 'High', 'Medium', or 'Low'.\n"
            "- 'local_names': A JSON array of strings listing every name, nickname, abbreviation, or colloquial term "
            "that residents and locals actually use when searching for this location. These will be used as Google Ads keywords, "
            "so: use plain text only (no special characters except hyphens and spaces), use standard title case, "
            "do not include the country or state unless locals habitually include it. "
            "Include the official name itself as the first entry. Include shortened forms, regional names, and common informal names. "
            "If there is only one known name, return an array with just that one name.\n\n"
            "**Population Estimation Guidance:** For locations with type 'suburb' or 'neighbourhood', if a direct, reliable population figure "
            "isn't found in your knowledge, provide a reasonable **estimate** based on the context (parent_name, level_4_name) and typical "
            "population densities for such areas. Assign **'Medium'** confidence to these well-informed estimates. "
            "Assign 'Low' confidence only if even estimation is highly uncertain.\n\n"
            "Example Response Format (for a batch of 3 locations with indices 0, 1, 2):\n"
            "[\n"
            "  {\"index\": 0, \"population\": 186948, \"confidence\": \"High\", \"local_names\": [\"Hoboken\", \"Hoboken NJ\", \"Mile Square City\"]},\n"
            "  {\"index\": 1, \"population\": 25000, \"confidence\": \"Medium\", \"local_names\": [\"Jersey City Heights\", \"The Heights\"]},\n"
            "  {\"index\": 2, \"population\": null, \"confidence\": \"Low\", \"local_names\": [\"Weehawken\"]}\n"
            "]\n\n"
            "--- START LOCATIONS ---"
        )

        locations_data = []
        for i, loc in enumerate(locations_chunk):
            original_index = start_index + i
            admin_hierarchy = loc.get('admin_hierarchy', {})

            location_info = {
                "index": original_index,
                "name": loc.get('name', 'Unknown'),
                "type": loc.get('type', 'unknown'),
                "parent_name": admin_hierarchy.get('parent_name'),
                "level_4_name": admin_hierarchy.get('level_4_name'),
                "level_2_name": admin_hierarchy.get('level_2_name')
            }
            locations_data.append(location_info)

        locations_json_str = json.dumps(locations_data, indent=2, ensure_ascii=False)

        prompt_footer = (
            "\n--- END LOCATIONS ---\n\n"
            "Provide ONLY the JSON array containing one object for each location listed above, matching the provided indices."
        )

        return f"{prompt_header}\n{locations_json_str}\n{prompt_footer}"
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd "/Users/nathanshapiro/Desktop/KMZ Location Scraper"
python -m pytest tests/test_local_names.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add location_analyzer.py tests/test_local_names.py
git commit -m "feat: extend AI prompt to return local search names for Google Ads keywords"
```

---

## Task 4: Parse Local Names from GPT Response

**Files:**
- Modify: `location_analyzer.py:806-866` (`_get_gpt_populations_batch` parsing section)
- Modify: `location_analyzer.py:868-1073` (`_get_gemini_populations_batch` parsing section)

- [ ] **Step 1: Write the test**

Add to `tests/test_local_names.py`:

```python
class TestLocalNamesParsing(unittest.TestCase):

    def setUp(self):
        from location_analyzer import LocationAnalyzer
        self.analyzer = LocationAnalyzer.__new__(LocationAnalyzer)
        self.analyzer._log = lambda msg: None
        self.analyzer.gpt_client = None
        self.analyzer.gpt_model = "gpt-4-turbo"
        self.analyzer.status_callback = lambda msg: None

    def test_gpt_parser_extracts_local_names(self):
        """_get_gpt_populations_batch must parse local_names from response."""
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([
            {
                "index": 0,
                "population": 52000,
                "confidence": "High",
                "local_names": ["Hoboken", "Hoboken NJ", "Mile Square City"]
            }
        ])

        locations = [{"name": "Hoboken", "type": "city", "admin_hierarchy": {}}]

        with patch.object(self.analyzer, 'gpt_client') as mock_client:
            mock_client.chat.completions.create.return_value = mock_response
            result = self.analyzer._get_gpt_populations_batch(locations, 0)

        self.assertIn(0, result)
        self.assertEqual(result[0]["population"], 52000)
        self.assertEqual(result[0]["local_names"], ["Hoboken", "Hoboken NJ", "Mile Square City"])

    def test_gpt_parser_defaults_local_names_to_official_name(self):
        """If local_names is missing from response, fall back to [official name]."""
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([
            {"index": 0, "population": 52000, "confidence": "High"}
        ])

        locations = [{"name": "Hoboken", "type": "city", "admin_hierarchy": {}}]

        with patch.object(self.analyzer, 'gpt_client') as mock_client:
            mock_client.chat.completions.create.return_value = mock_response
            result = self.analyzer._get_gpt_populations_batch(locations, 0)

        self.assertIn(0, result)
        self.assertEqual(result[0]["local_names"], ["Hoboken"],
            "Should fall back to official name when local_names missing")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd "/Users/nathanshapiro/Desktop/KMZ Location Scraper"
python -m pytest tests/test_local_names.py::TestLocalNamesParsing -v 2>&1 | head -30
```

Expected: FAIL — result[0] has no "local_names" key.

- [ ] **Step 3: Update GPT response parsing to extract local_names**

In `location_analyzer.py`, inside `_get_gpt_populations_batch`, find the block that builds `parsed_results_map[original_idx]` (around line 834). Replace the block that starts with `if original_idx in processed_indices:` through `processed_indices.add(original_idx)`:

```python
                    raw_local_names = item_data.get('local_names', [])
                    if isinstance(raw_local_names, list):
                        local_names = [str(n).strip() for n in raw_local_names if n and str(n).strip()]
                    else:
                        local_names = []
                    # Fall back to official name so the field is never empty
                    if not local_names:
                        loc_idx = original_idx - start_index
                        if 0 <= loc_idx < len(locations_chunk):
                            official = locations_chunk[loc_idx].get('name', '')
                            if official:
                                local_names = [official]

                    if original_idx in processed_indices:
                        continue
                    parsed_results_map[original_idx] = {
                        "population": pop_value,
                        "confidence": conf_value,
                        "local_names": local_names,
                    }
                    processed_indices.add(original_idx)
```

Also update the fallback entries (the "Missing" entries at line 843–846) to include local_names:

```python
            for i in range(len(locations_chunk)):
                expected_idx = start_index + i
                if expected_idx not in parsed_results_map:
                    official = locations_chunk[i].get('name', '')
                    parsed_results_map[expected_idx] = {
                        "population": None,
                        "confidence": "Missing",
                        "local_names": [official] if official else [],
                    }
```

Also update the error/parse-error fallback returns (lines 784–788, 850–852, 857–859, 862–864) to include `"local_names": []`.

- [ ] **Step 4: Update Gemini response parsing identically**

In `_get_gemini_populations_batch`, find the equivalent parsing block (same structure, around line 990–1040). Apply the same change: extract `raw_local_names` from `item_data`, validate it's a list, fall back to official name, and store as `"local_names"` in `parsed_results_map[original_idx]`. Apply the same fallback entry update.

The pattern to find and replace is the same `parsed_results_map[original_idx] = {"population": pop_value, "confidence": conf_value}` assignment — add `"local_names": local_names` to it, with the same extraction logic inserted before it.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd "/Users/nathanshapiro/Desktop/KMZ Location Scraper"
python -m pytest tests/test_local_names.py -v
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add location_analyzer.py tests/test_local_names.py
git commit -m "feat: parse local_names from GPT/Gemini responses

Both _get_gpt_populations_batch and _get_gemini_populations_batch now
extract local_names array from AI response. Falls back to [official_name]
when the field is absent so the column is never empty."
```

---

## Task 5: Initialize Local Names Field and Store Results

**Files:**
- Modify: `location_analyzer.py:1075-1170` (`estimate_populations`)

- [ ] **Step 1: Write the test**

Add to `tests/test_local_names.py`:

```python
class TestLocalNamesStorage(unittest.TestCase):

    def test_estimate_populations_initializes_local_names(self):
        """estimate_populations must initialize local_names on every location."""
        from location_analyzer import LocationAnalyzer
        analyzer = LocationAnalyzer.__new__(LocationAnalyzer)
        analyzer._log = lambda msg: None
        analyzer.use_gpt = False
        analyzer.use_openai = False
        analyzer.use_gemini_flag = False
        analyzer._ensure_ai_clients = lambda: None
        analyzer.chunk_size = 5

        locations = [
            {"name": "Hoboken", "type": "city", "admin_hierarchy": {}},
            {"name": "Weehawken", "type": "suburb", "admin_hierarchy": {}},
        ]
        result = analyzer.estimate_populations(locations)

        for loc in result:
            self.assertIn("local_names", loc,
                f"Location '{loc.get('name')}' must have 'local_names' after estimate_populations")
            self.assertIsInstance(loc["local_names"], list)

    def test_estimate_populations_stores_gpt_local_names(self):
        """estimate_populations must copy local_names from GPT batch result into each location."""
        from location_analyzer import LocationAnalyzer
        from unittest.mock import patch, MagicMock
        analyzer = LocationAnalyzer.__new__(LocationAnalyzer)
        analyzer._log = lambda msg: None
        analyzer.use_gpt = True
        analyzer.use_openai = True
        analyzer.use_gemini_flag = False
        analyzer._ensure_ai_clients = lambda: None
        analyzer.chunk_size = 5
        analyzer.status_callback = lambda msg: None

        locations = [{"name": "Hoboken", "type": "city", "admin_hierarchy": {}}]

        with patch.object(analyzer, '_get_gpt_populations_batch', return_value={
            0: {"population": 52000, "confidence": "High", "local_names": ["Hoboken", "Mile Square City"]}
        }):
            result = analyzer.estimate_populations(locations)

        self.assertEqual(result[0]["local_names"], ["Hoboken", "Mile Square City"])
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd "/Users/nathanshapiro/Desktop/KMZ Location Scraper"
python -m pytest tests/test_local_names.py::TestLocalNamesStorage -v 2>&1 | head -30
```

Expected: FAIL — `local_names` not in location dict.

- [ ] **Step 3: Initialize local_names in `estimate_populations`**

In `location_analyzer.py` at line 1082, inside the initialization loop (the `for loc in locations:` block that sets `gpt_population`, etc.), add:

```python
            loc["local_names"] = []
```

So the full block becomes:

```python
        for loc in locations:
            loc["gpt_population"] = None
            loc["gpt_confidence"] = None
            loc["gemini_population"] = None
            loc["gemini_confidence"] = None
            loc["combined_population"] = None
            loc["combined_confidence"] = None
            loc["final_population"] = 0
            loc["population_source"] = "None"
            loc["local_names"] = []
```

- [ ] **Step 4: Store local_names from GPT results**

In `estimate_populations`, find the block that stores GPT results (around line 1128–1133):

```python
                    for original_idx, result_data in gpt_results_map.items():
                        if 0 <= original_idx < len(locations): 
                            locations[original_idx]["gpt_population"] = result_data.get("population")
                            locations[original_idx]["gpt_confidence"] = result_data.get("confidence")
                            if result_data.get("population") is not None:
                                success_count += 1
```

Replace with:

```python
                    for original_idx, result_data in gpt_results_map.items():
                        if 0 <= original_idx < len(locations):
                            locations[original_idx]["gpt_population"] = result_data.get("population")
                            locations[original_idx]["gpt_confidence"] = result_data.get("confidence")
                            gpt_local = result_data.get("local_names", [])
                            if gpt_local:
                                locations[original_idx]["local_names"] = gpt_local
                            if result_data.get("population") is not None:
                                success_count += 1
```

- [ ] **Step 5: Merge local_names from Gemini results (union, deduplicated)**

Find the block that stores Gemini results (around line 1137–1142):

```python
                    for original_idx, result_data in gemini_results_map.items():
                        if 0 <= original_idx < len(locations): 
                            locations[original_idx]["gemini_population"] = result_data.get("population")
                            locations[original_idx]["gemini_confidence"] = result_data.get("confidence")
                            if result_data.get("population") is not None:
                                gemini_success_count += 1
```

Replace with:

```python
                    for original_idx, result_data in gemini_results_map.items():
                        if 0 <= original_idx < len(locations):
                            locations[original_idx]["gemini_population"] = result_data.get("population")
                            locations[original_idx]["gemini_confidence"] = result_data.get("confidence")
                            gemini_local = result_data.get("local_names", [])
                            if gemini_local:
                                existing = locations[original_idx].get("local_names", [])
                                # Union: add Gemini names not already in the list (case-insensitive dedup)
                                existing_lower = {n.lower() for n in existing}
                                merged = list(existing)
                                for n in gemini_local:
                                    if n.lower() not in existing_lower:
                                        merged.append(n)
                                        existing_lower.add(n.lower())
                                locations[original_idx]["local_names"] = merged
                            if result_data.get("population") is not None:
                                gemini_success_count += 1
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd "/Users/nathanshapiro/Desktop/KMZ Location Scraper"
python -m pytest tests/test_local_names.py -v
```

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add location_analyzer.py tests/test_local_names.py
git commit -m "feat: initialize and populate local_names field during population estimation

GPT names are stored first; Gemini names are merged in (case-insensitive
dedup union) so the final list contains every unique local name from both
providers."
```

---

## Task 6: Add Local Search Names Column to Excel Output

**Files:**
- Modify: `location_analyzer.py:1361-1458` (`save_to_excel`)

- [ ] **Step 1: Write the test**

Add to `tests/test_local_names.py`:

```python
class TestLocalNamesExcel(unittest.TestCase):

    def test_excel_contains_local_names_column(self):
        """Full Data sheet must include a 'Local Search Names' column."""
        from location_analyzer import LocationAnalyzer
        from io import BytesIO
        import openpyxl

        analyzer = LocationAnalyzer.__new__(LocationAnalyzer)
        analyzer._log = lambda msg: None
        analyzer.output_excel = "test_output.xlsx"

        locations = [
            {
                "name": "Hoboken",
                "type": "city",
                "latitude": 40.744,
                "longitude": -74.032,
                "gpt_population": 52000,
                "gpt_confidence": "High",
                "gemini_population": 51000,
                "gemini_confidence": "High",
                "combined_population": 51500,
                "combined_confidence": "High",
                "local_names": ["Hoboken", "Mile Square City", "Hoboken NJ"],
                "admin_hierarchy": {},
            }
        ]

        output = analyzer.save_to_excel(locations)
        self.assertIsNotNone(output)

        wb = openpyxl.load_workbook(BytesIO(output.read()))
        full_sheet = wb["Full Data"]
        headers = [cell.value for cell in full_sheet[1]]
        self.assertIn("Local Search Names", headers,
            f"Full Data sheet must have 'Local Search Names' column, got: {headers}")

        # Find the column index
        col_idx = headers.index("Local Search Names") + 1
        cell_value = full_sheet.cell(row=2, column=col_idx).value
        self.assertIn("Hoboken", cell_value,
            f"Local Search Names cell should contain 'Hoboken', got: {cell_value!r}")

    def test_clean_sheet_contains_local_names_column(self):
        """Clean Data sheet must include a 'Local Search Names' column."""
        from location_analyzer import LocationAnalyzer
        from io import BytesIO
        import openpyxl

        analyzer = LocationAnalyzer.__new__(LocationAnalyzer)
        analyzer._log = lambda msg: None
        analyzer.output_excel = "test_output.xlsx"

        locations = [
            {
                "name": "Hoboken",
                "type": "city",
                "latitude": 40.744,
                "longitude": -74.032,
                "gpt_population": 52000,
                "gpt_confidence": "High",
                "gemini_population": 51000,
                "gemini_confidence": "High",
                "combined_population": 51500,
                "combined_confidence": "High",
                "local_names": ["Hoboken", "Mile Square City"],
                "admin_hierarchy": {},
            }
        ]

        output = analyzer.save_to_excel(locations)
        self.assertIsNotNone(output)

        wb = openpyxl.load_workbook(BytesIO(output.read()))
        clean_sheet = wb["Clean Data"]
        headers = [cell.value for cell in clean_sheet[1]]
        self.assertIn("Local Search Names", headers,
            f"Clean Data sheet must have 'Local Search Names' column, got: {headers}")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd "/Users/nathanshapiro/Desktop/KMZ Location Scraper"
python -m pytest tests/test_local_names.py::TestLocalNamesExcel -v 2>&1 | head -30
```

Expected: FAIL — "Local Search Names" not in headers.

- [ ] **Step 3: Update `save_to_excel` to include local_names**

In `location_analyzer.py`, inside `save_to_excel`, find the `final_columns` list (line 1380) and add `'local_names'`:

```python
            final_columns = [
                'name',
                'type',
                'latitude',
                'longitude',
                'gpt_population',
                'gpt_confidence',
                'gemini_population',
                'gemini_confidence',
                'combined_population',
                'combined_confidence',
                'local_names',
            ]
```

Then after the line `df_full = df[final_columns].copy()` (line 1397), add code to convert the `local_names` list to a comma-separated string:

```python
            df_full = df[final_columns].copy()
            # Convert local_names list to comma-separated string for Excel compatibility
            if 'local_names' in df_full.columns:
                df_full['local_names'] = df_full['local_names'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else (str(x) if x else '')
                )
```

Then rename the column to a friendly label. Find `df_full.columns = [col.replace('admin_hierarchy_', '') for col in df_full.columns]` (line 1398) and replace it with:

```python
            col_rename = {col: col.replace('admin_hierarchy_', '') for col in df_full.columns}
            col_rename['local_names'] = 'Local Search Names'
            df_full = df_full.rename(columns=col_rename)
```

Then update the Clean Data sheet to include local names. Find the line:

```python
            df_clean = df_full[['name', pop_column]].copy()
            df_clean.columns = ['Location Name', 'Population']
```

Replace with:

```python
            local_col = 'Local Search Names' if 'Local Search Names' in df_full.columns else None
            clean_cols = ['name', pop_column] + ([local_col] if local_col else [])
            df_clean = df_full[clean_cols].copy()
            clean_col_names = ['Location Name', 'Population'] + (['Local Search Names'] if local_col else [])
            df_clean.columns = clean_col_names
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
cd "/Users/nathanshapiro/Desktop/KMZ Location Scraper"
python -m pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add location_analyzer.py tests/test_local_names.py
git commit -m "feat: add Local Search Names column to Excel output (Full Data + Clean Data sheets)

Both sheets now include a comma-separated 'Local Search Names' column
containing colloquial names and local search terms appropriate for
use as Google Ads keywords."
```

---

## Task 7: Push to GitHub

- [ ] **Step 1: Verify all tests pass**

```bash
cd "/Users/nathanshapiro/Desktop/KMZ Location Scraper"
python -m pytest tests/ -v 2>&1
```

Expected: All tests PASS, no failures.

- [ ] **Step 2: Check git log**

```bash
git log --oneline -6
```

Expected: 5 commits from this plan on top of the original HEAD.

- [ ] **Step 3: Push to GitHub**

```bash
git push origin main
```

Expected: `main` branch updated on `https://github.com/ParetoLeads/KMZ-Location-Scraper`.

---

## Self-Review

### Spec Coverage
1. ✅ 406 fix — Task 1 (4 POST calls fixed)
2. ✅ Reliability — Task 2 (second fallback mirror)
3. ✅ Local search names for Google Ads — Tasks 3–6 (prompt, parsing, storage, Excel)
4. ✅ Multiple names per location — handled via union merge in Task 5 + comma-separated Excel in Task 6

### No Placeholders Check
All code blocks are complete and runnable. All function names are consistent across tasks.

### Type Consistency
- `local_names` is always `List[str]` in memory, converted to `str` only in `save_to_excel`
- `parsed_results_map` values are always `{"population": int|None, "confidence": str, "local_names": List[str]}`
- `OVERPASS_FALLBACK_URLS` is introduced in Task 2; `OVERPASS_FALLBACK_URL = OVERPASS_FALLBACK_URLS[0]` keeps backward compat for the hierarchy function which references it by name
