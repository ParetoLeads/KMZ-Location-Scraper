# KMZ Location Scraper — Project Context for Claude

## What this app does
A Streamlit app that:
1. Accepts a KMZ boundary file (e.g., "Boston Area.kmz")
2. Queries the Overpass API (OpenStreetMap) to find named places inside the boundary
3. Retrieves administrative hierarchy (city/county/state) for each location
4. Calls GPT or Gemini to estimate population and generate local search names for Google Ads
5. Exports results to Excel

## Architecture
- `app.py` — Streamlit UI, state machine (idle → hierarchy → population → excel)
- `location_analyzer.py` — all heavy logic (KMZ parsing, OSM queries, AI calls)
- `config.py` — all constants and env vars
- `utils/retry_handler.py` — `execute_with_retry()` used for all API calls
- `utils/cache.py` — optional in-memory caching for OSM queries
- `utils/exceptions.py` — custom exception types

## Overpass API configuration
- Primary URL: `config.OSM_OVERPASS_URL` (default: `https://overpass.kumi.systems/api/interpreter`)
- Fallback list: `OVERPASS_FALLBACK_URLS` in `location_analyzer.py` (tried in order on timeout)
- The fallback loop must only `break` on 2xx responses. Non-2xx (e.g., 406) must `continue` to the next URL.

## Known recurring bug — READ BEFORE TOUCHING OVERPASS CODE
See `docs/ERROR_LOG.md` for full history. Short version:
- HTTP 406 from overpass-api.de has been misdiagnosed 3 times
- Root cause: fallback loop breaks on 406 instead of trying the next mirror
- Any change to Overpass request logic must verify the fallback loop continues past error responses

## Key constraints
- The app runs on Streamlit Cloud (shared IP) so Overpass rate limits hit frequently
- All Overpass queries use `[out:json]` format — don't change this
- Population and local_names come from AI (GPT/Gemini) — Overpass only finds the locations
- Stage 2 (OSM) returning 0 locations means ALL downstream stages produce empty output
- Excel is not exported if locations = 0

## Testing
- Upload any small KMZ file (Boston Area.kmz in repo root works)
- Success = Stage 2 logs "Total unique OSM locations found: N" where N > 0
- Failure = all stages complete but "No locations to save" in Stage 5

## Running locally
```
pip install -r requirements.txt
streamlit run app.py
```
