import streamlit as st
import pandas as pd
import tempfile
import os
import pydeck as pdk
from location_analyzer import LocationAnalyzer
from io import BytesIO
from config import config
from utils.validators import validate_kmz_file, validate_file_size, validate_api_key
from utils.exceptions import ValidationError, KMZParseError
from utils.progress_tracker import ProgressTracker, ProgressUI, create_progress_callback
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="KMZ Location Scraper",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Light mode styling
st.markdown("""
    <style>
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4 !important;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #333 !important;
        margin-bottom: 2rem;
    }
    /* Orange download button */
    .stDownloadButton > button {
        background-color: #FF5733 !important;
        color: white !important;
        border-color: #FF5733 !important;
        font-weight: bold !important;
    }
    .stDownloadButton > button:hover {
        background-color: #E64A2E !important;
        border-color: #E64A2E !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🗺️ KMZ Location Scraper</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header" style="text-align: center;">Extract locations from KMZ files and estimate populations using OpenStreetMap and GPT</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 0.85rem; margin-top: -0.5rem;">Developed with 💡 by Paretoleads.com</p>', unsafe_allow_html=True)

# Configuration is now managed by config module
# Always use both AI providers
ai_provider = "Both (Compare Results)"

# Main content area
uploaded_file = st.file_uploader(
    f"Upload KMZ File (Max {config.MAX_FILE_SIZE_MB}MB)",
    type=['kmz'],
    help=f"Select a KMZ file containing the boundary polygon. Maximum file size: {config.MAX_FILE_SIZE_MB}MB",
    accept_multiple_files=False
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'excel_data' not in st.session_state:
    st.session_state.excel_data = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'progress_messages' not in st.session_state:
    st.session_state.progress_messages = []
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []
if 'polygon_points' not in st.session_state:
    st.session_state.polygon_points = None
# Processing state for handling timeouts
if 'processing_state' not in st.session_state:
    st.session_state.processing_state = None  # Will store: stage, locations, analyzer_config, etc.
if 'processing_stage' not in st.session_state:
    st.session_state.processing_stage = None  # 'kmz', 'osm', 'hierarchy', 'population', 'excel', 'complete'
if 'processing_locations' not in st.session_state:
    st.session_state.processing_locations = None  # List of locations being processed
if 'processing_config' not in st.session_state:
    st.session_state.processing_config = None  # Analyzer configuration
if 'hierarchy_batch_index' not in st.session_state:
    st.session_state.hierarchy_batch_index = 0
if 'population_batch_index' not in st.session_state:
    st.session_state.population_batch_index = 0
if 'tmp_kmz_path' not in st.session_state:
    st.session_state.tmp_kmz_path = None
if 'diagnostic_run_id' not in st.session_state:
    st.session_state.diagnostic_run_id = 0
if '_commit_done' not in st.session_state:
    st.session_state._commit_done = set()  # (stage, idx) for which we did the quick commit run
if 'rerun_count' not in st.session_state:
    st.session_state.rerun_count = 0

# Log every rerun at the top level
st.session_state.rerun_count += 1
st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] ========================================")
st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [APP-START] RERUN #{st.session_state.rerun_count}")
st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [APP-START] Session state snapshot:")
st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [APP-START]   processing={st.session_state.processing}")
st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [APP-START]   processing_stage={st.session_state.processing_stage}")
st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [APP-START]   hierarchy_batch_index={st.session_state.hierarchy_batch_index}")
st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [APP-START]   population_batch_index={st.session_state.population_batch_index}")
st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [APP-START]   locations_count={len(st.session_state.processing_locations) if st.session_state.processing_locations else 0}")
st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [APP-START]   results={'exists' if st.session_state.results else 'None'}")
st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] ========================================")

# Helper function to save processing state
def save_processing_state(stage, locations, config_dict, hierarchy_idx=0, population_idx=0):
    """Save current processing state to session state."""
    st.session_state.processing_stage = stage
    st.session_state.processing_locations = locations
    st.session_state.processing_config = config_dict
    st.session_state.hierarchy_batch_index = hierarchy_idx
    st.session_state.population_batch_index = population_idx
    # Force rerun to continue processing
    st.rerun()

# Helper function to clear processing state
def clear_processing_state():
    """Clear all processing state. Does not clear results or excel_data."""
    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CLEAR] clear_processing_state() called")
    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CLEAR] Before clear: processing={st.session_state.processing}, stage={st.session_state.processing_stage}")
    st.session_state.processing = False
    st.session_state.processing_stage = None
    st.session_state.processing_locations = None
    st.session_state.processing_config = None
    st.session_state.hierarchy_batch_index = 0
    st.session_state.population_batch_index = 0
    st.session_state._commit_done = set()
    if st.session_state.tmp_kmz_path and os.path.exists(st.session_state.tmp_kmz_path):
        try:
            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CLEAR] Deleting temp KMZ file: {st.session_state.tmp_kmz_path}")
            os.unlink(st.session_state.tmp_kmz_path)
        except Exception as e:
            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CLEAR] Failed to delete temp file: {str(e)}")
            pass
    st.session_state.tmp_kmz_path = None
    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CLEAR] After clear: processing={st.session_state.processing}, stage={st.session_state.processing_stage}")
    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CLEAR] clear_processing_state() completed")


def create_analyzer_from_state(progress_callback, status_callback, skip_ai_status_log: bool = False):
    """Create LocationAnalyzer from processing_config and config. Used on each rerun."""
    cfg = st.session_state.processing_config
    return LocationAnalyzer(
        kmz_file=cfg["kmz_file"],
        verbose=config.VERBOSE,
        openai_api_key=cfg["openai_api_key"],
        gemini_api_key=cfg["gemini_api_key"],
        ai_provider="both",
        use_gpt=config.USE_GPT,
        chunk_size=config.DEFAULT_CHUNK_SIZE,
        max_locations=config.DEFAULT_MAX_LOCATIONS,
        pause_before_gpt=False,
        enable_web_browsing=config.DEFAULT_ENABLE_WEB_BROWSING,
        primary_place_types=config.PRIMARY_PLACE_TYPES,
        additional_place_types=config.ADDITIONAL_PLACE_TYPES,
        special_place_types=config.SPECIAL_PLACE_TYPES,
        progress_callback=progress_callback,
        status_callback=status_callback,
        skip_ai_status_log=skip_ai_status_log,
    )

# Process button
st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [APP] Checking if file uploaded: {uploaded_file is not None}")
if uploaded_file is not None:
    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [APP] File uploaded: {uploaded_file.name}, size={uploaded_file.size} bytes")
    # Validate file size
    try:
        validate_file_size(uploaded_file.size)
        st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [APP] File size validation passed")
    except ValidationError as e:
        st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [APP] ERROR: File size validation failed: {str(e)}")
        st.error(f"❌ {str(e)}")
    else:
        st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [APP] Checking button state: processing={st.session_state.processing}")
        if st.button("🚀 Start Analysis", type="primary", disabled=st.session_state.processing):
            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [BUTTON] ========================================")
            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [BUTTON] START ANALYSIS BUTTON CLICKED!")
            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [BUTTON] ========================================")
            st.session_state.processing = True
            st.session_state.results = None
            st.session_state.excel_data = None
            # Reset processing state for new run
            st.session_state.processing_stage = None
            st.session_state.processing_locations = None
            st.session_state.hierarchy_batch_index = 0
            st.session_state.population_batch_index = 0
            st.session_state.diagnostic_run_id = 0
            st.session_state._logged_ai = False
            st.session_state._commit_done = set()
            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [BUTTON] Session state reset complete")

            # Create temporary file for KMZ
            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [BUTTON] Creating temporary KMZ file...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.kmz') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_kmz_path = tmp_file.name
            st.session_state.tmp_kmz_path = tmp_kmz_path
            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [BUTTON] Temporary KMZ file created: {tmp_kmz_path}")
            
            # Validate KMZ file
            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [BUTTON] Validating KMZ file structure...")
            kmz_valid = True
            try:
                validate_kmz_file(tmp_kmz_path)
                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [BUTTON] KMZ file validation passed")
            except (ValidationError, KMZParseError) as e:
                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [BUTTON] ERROR: KMZ validation failed: {str(e)}")
                st.error(f"❌ Invalid KMZ file: {str(e)}")
                st.session_state.processing = False
                if os.path.exists(tmp_kmz_path):
                    os.unlink(tmp_kmz_path)
                kmz_valid = False
            
            if kmz_valid:
                try:
                    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] KMZ validation passed, starting API key validation")
                    # Always use both providers
                    provider_value = "both"
                    api_keys_valid = True
                    
                    # Get OpenAI API key
                    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] Checking OpenAI API key...")
                    openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
                    if not openai_api_key:
                        st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] ERROR: OpenAI API key not found")
                        st.error("⚠️ OpenAI API key not found in secrets. Please add OPENAI_API_KEY to your Streamlit secrets.")
                        st.session_state.processing = False
                        if os.path.exists(tmp_kmz_path):
                            os.unlink(tmp_kmz_path)
                        api_keys_valid = False
                    else:
                        st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] OpenAI API key found, validating...")
                        try:
                            validate_api_key(openai_api_key)
                            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] OpenAI API key validated successfully")
                        except ValidationError as e:
                            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] ERROR: OpenAI API key validation failed: {str(e)}")
                            st.error(f"⚠️ {str(e)}")
                            st.session_state.processing = False
                            if os.path.exists(tmp_kmz_path):
                                os.unlink(tmp_kmz_path)
                            api_keys_valid = False
                    
                    # Get Gemini API key
                    if api_keys_valid:
                        st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] Checking Gemini API key...")
                        gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
                        if not gemini_api_key:
                            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] ERROR: Gemini API key not found")
                            st.error("⚠️ Gemini API key not found in secrets. Please add GEMINI_API_KEY to your Streamlit secrets.")
                            st.session_state.processing = False
                            if os.path.exists(tmp_kmz_path):
                                os.unlink(tmp_kmz_path)
                            api_keys_valid = False
                        else:
                            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] Gemini API key found")
                    
                    # Check if at least one API key is available
                    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] Checking if at least one API key available... api_keys_valid={api_keys_valid}, openai={bool(openai_api_key)}, gemini={bool(gemini_api_key)}")
                    if api_keys_valid and not openai_api_key and not gemini_api_key:
                        st.error("⚠️ No API keys found. Please configure at least one API key in Streamlit secrets.")
                        st.session_state.processing = False
                        if os.path.exists(tmp_kmz_path):
                            os.unlink(tmp_kmz_path)
                        api_keys_valid = False
                    
                    # If we have valid keys, run initial KMZ+OSM (chunked state machine starts after this)
                    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] API keys validated, starting initial KMZ+OSM processing")
                    if api_keys_valid and (openai_api_key or gemini_api_key):
                        st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] Creating progress UI containers...")
                        progress_container = st.container()
                        with progress_container:
                            stage_text = st.empty()
                            main_progress = st.progress(0)
                            progress_metrics = st.empty()
                            status_text = st.empty()
                        progress_messages = list(st.session_state.progress_messages or [])
                        status_messages = list(st.session_state.status_messages or [])
                        progress_tracker = ProgressTracker()
                        progress_ui = ProgressUI(
                            stage_text_container=stage_text,
                            progress_bar=main_progress,
                            metrics_container=progress_metrics,
                            status_text_container=status_text,
                        )

                        def progress_callback_with_save(msg: str) -> None:
                            progress_messages.append(msg)
                            st.session_state.progress_messages = progress_messages.copy()
                            progress_tracker.update_from_message(msg)
                            progress_ui.update(progress_tracker, msg)

                        def status_callback(msg: str) -> None:
                            status_messages.append(msg)
                            progress_messages.append(msg)
                            st.session_state.status_messages = status_messages.copy()
                            st.session_state.progress_messages = progress_messages.copy()

                        try:
                            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] Creating LocationAnalyzer instance...")
                            analyzer = LocationAnalyzer(
                                kmz_file=tmp_kmz_path,
                                verbose=config.VERBOSE,
                                openai_api_key=openai_api_key,
                                gemini_api_key=gemini_api_key,
                                ai_provider=provider_value,
                                use_gpt=config.USE_GPT,
                                chunk_size=config.DEFAULT_CHUNK_SIZE,
                                max_locations=config.DEFAULT_MAX_LOCATIONS,
                                pause_before_gpt=False,
                                enable_web_browsing=config.DEFAULT_ENABLE_WEB_BROWSING,
                                primary_place_types=config.PRIMARY_PLACE_TYPES,
                                additional_place_types=config.ADDITIONAL_PLACE_TYPES,
                                special_place_types=config.SPECIAL_PLACE_TYPES,
                                progress_callback=progress_callback_with_save,
                                status_callback=status_callback,
                            )
                            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] LocationAnalyzer created, calling run_kmz_and_osm_only()...")
                            polygon_points, locations = analyzer.run_kmz_and_osm_only()
                            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] run_kmz_and_osm_only() returned: polygon_points={polygon_points is not None}, locations={locations is not None}, locations_count={len(locations) if locations else 0}")
                            if locations is None:
                                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] ERROR: locations is None, KMZ/OSM failed")
                                st.error("❌ KMZ parsing or OSM lookup failed. Check the log below.")
                                st.session_state.progress_messages = progress_messages
                                st.session_state.status_messages = status_messages
                                clear_processing_state()
                            else:
                                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] KMZ+OSM success! Found {len(locations)} locations")
                                estimated_time = analyzer._estimate_processing_time(len(locations), 0, 0)
                                analyzer._log(f"Estimated remaining time: {estimated_time}")
                                analyzer._log("CHECKPOINT: Stage 3 - Retrieving administrative boundaries started")
                                analyzer._log("\n--- Retrieving Administrative Boundaries ---")
                                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] Saving state to session_state...")
                                st.session_state.polygon_points = polygon_points
                                st.session_state.processing_locations = locations
                                st.session_state.processing_stage = "hierarchy"
                                st.session_state.hierarchy_batch_index = 0
                                st.session_state.processing_config = {
                                    "kmz_file": tmp_kmz_path,
                                    "openai_api_key": openai_api_key,
                                    "gemini_api_key": gemini_api_key,
                                }
                                st.session_state.progress_messages = progress_messages
                                st.session_state.status_messages = status_messages
                                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] State saved. Session state snapshot: processing={st.session_state.processing}, stage={st.session_state.processing_stage}, hierarchy_idx={st.session_state.hierarchy_batch_index}, locations_count={len(st.session_state.processing_locations)}")
                                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] Calling st.rerun() to enter state machine...")
                                st.rerun()
                        except Exception as e:
                            import traceback as tb
                            full_tb = tb.format_exc()
                            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] ========== EXCEPTION IN INITIAL PROCESSING ==========")
                            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] EXCEPTION type={type(e).__name__}")
                            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] EXCEPTION message={str(e)}")
                            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] EXCEPTION traceback:")
                            st.session_state.progress_messages.append(full_tb)
                            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] ========== END EXCEPTION ==========")
                            st.error(f"❌ Error during analysis: {str(e)}")
                            st.session_state.progress_messages = progress_messages
                            st.session_state.status_messages = status_messages
                            with st.expander("Error Details"):
                                st.code(full_tb)
                            clear_processing_state()
                except Exception as e:
                    import traceback as tb
                    full_tb = tb.format_exc()
                    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [BUTTON] ========== OUTER EXCEPTION ==========")
                    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [BUTTON] EXCEPTION type={type(e).__name__}")
                    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [BUTTON] EXCEPTION message={str(e)}")
                    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [BUTTON] EXCEPTION traceback:")
                    st.session_state.progress_messages.append(full_tb)
                    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [BUTTON] ========== END EXCEPTION ==========")
                    st.error(f"❌ Error: {str(e)}")
                    with st.expander("Error Details"):
                        st.code(full_tb)
                    st.session_state.processing = False
                    if st.session_state.tmp_kmz_path and os.path.exists(st.session_state.tmp_kmz_path):
                        try:
                            os.unlink(st.session_state.tmp_kmz_path)
                        except Exception:
                            pass
                    st.session_state.tmp_kmz_path = None

            # State machine: run one chunk per rerun when processing is active
            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CHECK] ========== RERUN CHECKPOINT ==========")
            st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CHECK] State machine entry check: processing={st.session_state.processing}, processing_stage={st.session_state.processing_stage}")
            if st.session_state.processing and st.session_state.processing_stage is not None:
                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [SM-ENTER] ✓ CONDITION MET - Entering state machine")
                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [SM-ENTER] Current state: Stage={st.session_state.processing_stage}, hierarchy_idx={st.session_state.hierarchy_batch_index}, population_idx={st.session_state.population_batch_index}, locations_count={len(st.session_state.processing_locations) if st.session_state.processing_locations else 0}")
                progress_container = st.container()
                with progress_container:
                    stage_text = st.empty()
                    main_progress = st.progress(0)
                    progress_metrics = st.empty()
                    status_text = st.empty()
                progress_messages = list(st.session_state.progress_messages or [])
                status_messages = list(st.session_state.status_messages or [])
                progress_tracker = ProgressTracker()
                progress_ui = ProgressUI(
                    stage_text_container=stage_text,
                    progress_bar=main_progress,
                    metrics_container=progress_metrics,
                    status_text_container=status_text,
                )

                def progress_cb(msg: str) -> None:
                    progress_messages.append(msg)
                    st.session_state.progress_messages = progress_messages.copy()
                    progress_tracker.update_from_message(msg)
                    progress_ui.update(progress_tracker, msg)

                def status_cb(msg: str) -> None:
                    status_messages.append(msg)
                    progress_messages.append(msg)
                    st.session_state.status_messages = status_messages.copy()
                    st.session_state.progress_messages = progress_messages.copy()

                if st.button("Cancel", key="cancel_processing"):
                    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CANCEL] Cancel button clicked - clearing processing state")
                    clear_processing_state()
                    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CANCEL] Processing state cleared - calling st.rerun()")
                    st.rerun()

                st.session_state.diagnostic_run_id = st.session_state.diagnostic_run_id + 1
                run_id = st.session_state.diagnostic_run_id
                locations = st.session_state.processing_locations or []
                batch_size = config.DEFAULT_BATCH_SIZE
                chunk_size = config.DEFAULT_CHUNK_SIZE
                stage = st.session_state.processing_stage
                first_batch_size = config.HIERARCHY_FIRST_BATCH_SIZE
                _ts = datetime.now().strftime("%H:%M:%S")
                progress_cb(f"[{_ts}] [SM] ENTER run_id={run_id} stage={stage} hierarchy_idx={st.session_state.hierarchy_batch_index} population_idx={st.session_state.population_batch_index} locations={len(locations)}")
                if stage == "hierarchy":
                    idx = st.session_state.hierarchy_batch_index
                    n = len(locations) if locations else 0
                    total_batches = (1 + max(0, (n - first_batch_size + batch_size - 1) // batch_size)) if n else 0
                elif stage == "population":
                    idx = st.session_state.population_batch_index
                    total_batches = (len(locations) + chunk_size - 1) // chunk_size if locations else 0
                else:
                    idx = 0
                    total_batches = 0

                # Two-phase: commit run (log RUN, save, rerun) so state persists even if work run is killed
                commit_key = (stage, idx)
                commit_done = st.session_state.get("_commit_done") or set()
                in_commit_done = commit_key in commit_done
                progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] commit_key=({stage},{idx}) commit_done_size={len(commit_done)} in_commit_done={in_commit_done} doing_commit_run={not in_commit_done}")
                if commit_key not in commit_done:
                    ts = datetime.now().strftime("%H:%M:%S")
                    progress_cb(f"[{ts}] [SM] COMMIT: writing RUN and CALL_START, then st.rerun()")
                    progress_cb(f"[{ts}] RUN #{run_id} stage={stage} batch {idx + 1}/{total_batches}")
                    progress_cb(f"[{ts}] CALL_START next batch {idx + 1}/{total_batches}")
                    st.session_state.progress_messages = progress_messages.copy()
                    st.session_state.status_messages = status_messages.copy()
                    st.session_state._commit_done = set(commit_done) | {commit_key}
                    progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] COMMIT: state saved, calling st.rerun()")
                    st.session_state.progress_messages = progress_messages.copy()
                    st.session_state.status_messages = status_messages.copy()
                    st.rerun()

                progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] WORK: checking processing_config and kmz file...")
                cfg = st.session_state.processing_config
                if not cfg:
                    progress_cb("[ERROR] Processing config missing. Please start the analysis again.")
                    st.session_state.progress_messages = progress_messages.copy()
                    st.session_state.status_messages = status_messages.copy()
                    st.error("Processing config missing. Click Cancel and start the analysis again.")
                    st.rerun()
                kmz_path = cfg.get("kmz_file")
                if not kmz_path or not os.path.exists(kmz_path):
                    progress_cb(f"[ERROR] KMZ file not found: {kmz_path}. Temp file may have been lost.")
                    st.session_state.progress_messages = progress_messages.copy()
                    st.session_state.status_messages = status_messages.copy()
                    st.error("KMZ temp file was lost (e.g. app restarted). Please upload the file again and click Start Analysis.")
                    st.rerun()

                try:
                    progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] WORK: creating analyzer (skip_ai_log={st.session_state.get('_logged_ai', False)})...")
                    skip_ai_log = st.session_state.get("_logged_ai", False)
                    analyzer = create_analyzer_from_state(progress_cb, status_cb, skip_ai_status_log=skip_ai_log)
                    st.session_state._logged_ai = True
                    locations = st.session_state.processing_locations
                    progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] WORK: analyzer created, entering stage branch (stage={st.session_state.processing_stage})")

                    if st.session_state.processing_stage == "hierarchy":
                        idx = st.session_state.hierarchy_batch_index
                        n_loc = len(locations) if locations else 0
                        total_batches = (1 + max(0, (n_loc - first_batch_size + batch_size - 1) // batch_size)) if n_loc else 0
                        if idx == 0:
                            batch_start = 0
                            batch_end = first_batch_size
                        else:
                            batch_start = first_batch_size + (idx - 1) * batch_size
                            batch_end = batch_start + batch_size
                        batch = locations[batch_start:batch_end]
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] hierarchy: idx={idx} batch_start={batch_start} batch_end={batch_end} len(batch)={len(batch)} total_batches={total_batches}")
                        if batch:
                            hierarchy_timeout = getattr(config, "HIERARCHY_QUERY_TIMEOUT", 20)
                            hierarchy_retries = getattr(config, "HIERARCHY_MAX_RETRIES_CHUNKED", 2)
                            progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] hierarchy: calling fetch_admin_hierarchy_batch timeout={hierarchy_timeout}s retries={hierarchy_retries} ...")
                            analyzer._log(f"Retrieving hierarchy batch {idx + 1}/{total_batches} ({len(batch)} locations)...")
                            analyzer._log(f"CALL_START Overpass hierarchy batch {idx + 1}/{total_batches} ({len(batch)} locations)")
                            analyzer.fetch_admin_hierarchy_batch(
                                batch,
                                timeout_sec=hierarchy_timeout,
                                max_retry_attempts=hierarchy_retries,
                            )
                            progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] hierarchy: fetch_admin_hierarchy_batch returned ok")
                            analyzer._log(f"CALL_END Overpass hierarchy batch {idx + 1}/{total_batches}")
                            estimated_time = analyzer._estimate_processing_time(len(locations), idx + 1, 0)
                            analyzer._log(f"Estimated remaining time: {estimated_time}")
                            time.sleep(config.HIERARCHY_BATCH_DELAY)
                        st.session_state.hierarchy_batch_index = idx + 1
                        if idx == 0:
                            processed_after_this_batch = first_batch_size
                        else:
                            processed_after_this_batch = first_batch_size + idx * batch_size
                        if processed_after_this_batch >= len(locations):
                            progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] hierarchy: ALL BATCHES COMPLETE! processed_after_this_batch={processed_after_this_batch}, total_locations={len(locations)}")
                            analyzer._log("Finished retrieving administrative boundaries.")
                            analyzer._log("CHECKPOINT: Stage 3 - Retrieving administrative boundaries completed")
                            max_loc = config.DEFAULT_MAX_LOCATIONS
                            if max_loc > 0 and len(locations) > max_loc:
                                progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] hierarchy: Limiting locations from {len(locations)} to {max_loc}")
                                analyzer._log(f"Limiting results to {max_loc} locations (out of {len(locations)} found)")
                                st.session_state.processing_locations = locations[:max_loc]
                                locations = st.session_state.processing_locations
                            progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] hierarchy: TRANSITIONING TO POPULATION STAGE")
                            analyzer._log("CHECKPOINT: Stage 4 - Population estimation started")
                            analyzer._log("\n--- Starting Population Estimation (GPT) ---")
                            st.session_state.processing_stage = "population"
                            st.session_state.population_batch_index = 0
                            progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] hierarchy: State updated. NEW stage={st.session_state.processing_stage}, population_idx={st.session_state.population_batch_index}")
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] hierarchy: saving state and st.rerun()")
                        st.session_state.progress_messages = progress_messages
                        st.session_state.status_messages = status_messages
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] hierarchy: State saved. About to call st.rerun()")
                        st.rerun()

                    elif st.session_state.processing_stage == "population":
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] population: ENTERED population stage")
                        idx = st.session_state.population_batch_index
                        num_batches = (len(locations) + chunk_size - 1) // chunk_size
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] population: Processing batch {idx + 1}/{num_batches}, locations={len(locations)}, chunk_size={chunk_size}")
                        analyzer._log(f"Calculating population for batch {idx + 1}/{num_batches}...")
                        analyzer._log(f"CALL_START Population batch {idx + 1}/{num_batches}")
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] population: Calling estimate_populations_single_batch(locations, {idx})...")
                        analyzer.estimate_populations_single_batch(locations, idx)
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] population: estimate_populations_single_batch() completed")
                        analyzer._log(f"CALL_END Population batch {idx + 1}/{num_batches}")
                        if idx < num_batches - 1:
                            progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] population: Sleeping for {config.GPT_BATCH_DELAY}s before next batch...")
                            time.sleep(config.GPT_BATCH_DELAY)
                        st.session_state.population_batch_index = idx + 1
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] population: Updated population_batch_index to {st.session_state.population_batch_index}")
                        if (idx + 1) * chunk_size >= len(locations):
                            progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] population: ALL POPULATION BATCHES COMPLETE! Transitioning to excel stage")
                            analyzer._log("CHECKPOINT: Stage 4 - Population estimation completed")
                            analyzer.calculate_combined_populations(locations)
                            st.session_state.processing_stage = "excel"
                            progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] population: Stage updated to '{st.session_state.processing_stage}'")
                        st.session_state.progress_messages = progress_messages
                        st.session_state.status_messages = status_messages
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] population: State saved, calling st.rerun()")
                        st.rerun()

                    elif st.session_state.processing_stage == "excel":
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] excel: ENTERED excel stage")
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] excel: Calling save_to_excel({len(locations)} locations)...")
                        excel_data = analyzer.save_to_excel(locations)
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] excel: save_to_excel() completed. excel_data={'SUCCESS' if excel_data else 'FAILED'}")
                        st.session_state.results = locations
                        st.session_state.excel_data = excel_data
                        st.session_state.progress_messages = progress_messages
                        st.session_state.status_messages = status_messages
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] excel: Marking progress as complete and clearing state...")
                        progress_ui.mark_complete()
                        clear_processing_state()
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] excel: Processing state cleared. processing={st.session_state.processing}, processing_stage={st.session_state.processing_stage}")
                        if excel_data:
                            st.success(f"✅ Successfully processed {len(locations)} locations!")
                            progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] excel: SUCCESS - Displaying success message")
                        else:
                            st.warning("⚠️ Excel export failed, but results are available below.")
                            progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] excel: WARNING - Excel export failed")
                        progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] excel: Calling st.rerun() to display results...")
                        st.rerun()

                except Exception as e:
                    import traceback as tb
                    err_msg = str(e)
                    progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] ========== EXCEPTION CAUGHT ==========")
                    progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] EXCEPTION type={type(e).__name__}")
                    progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] EXCEPTION message={err_msg[:500]}")
                    progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] EXCEPTION Current stage={st.session_state.processing_stage}, hierarchy_idx={st.session_state.hierarchy_batch_index}, population_idx={st.session_state.population_batch_index}")
                    progress_cb(f"[ERROR] {err_msg}")
                    full_traceback = tb.format_exc()
                    progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] EXCEPTION Full traceback:")
                    progress_cb(full_traceback)
                    progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] ========== END EXCEPTION ==========")
                    st.session_state.progress_messages = progress_messages.copy()
                    st.session_state.status_messages = status_messages.copy()
                    st.error(f"Error during processing: {err_msg}")
                    if "504" in err_msg or "Gateway Timeout" in err_msg or "overpass" in err_msg.lower():
                        st.info("Overpass often returns 504 when busy. Set env `OVERPASS_URL` or `OSM_OVERPASS_URL` to another public Overpass instance and restart the app.")
                    with st.expander("Error details"):
                        st.code(full_traceback)
                    progress_cb(f"[{datetime.now().strftime('%H:%M:%S')}] [SM] EXCEPTION: Calling st.rerun() after error...")
                    st.rerun()
            else:
                # State machine NOT entered - log why
                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CHECK] ✗ STATE MACHINE NOT ENTERED")
                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CHECK] Reason analysis:")
                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CHECK]   - processing={st.session_state.processing} (expected: True)")
                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CHECK]   - processing_stage={st.session_state.processing_stage} (expected: not None)")
                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CHECK]   - processing_locations={'exists' if st.session_state.processing_locations else 'None'}")
                st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CHECK]   - processing_config={'exists' if st.session_state.processing_config else 'None'}")
                if not st.session_state.processing:
                    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CHECK] ⚠️ CRITICAL: processing flag is False - processing was stopped or never started")
                if st.session_state.processing_stage is None:
                    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [CHECK] ⚠️ CRITICAL: processing_stage is None - stage was not set or was cleared")

# Display results
st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [DISPLAY] Checking if results should be displayed: results={'exists' if st.session_state.results is not None else 'None'}")
if st.session_state.results is not None:
    st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [DISPLAY] ✓ Displaying results for {len(st.session_state.results)} locations")
    st.divider()
    st.header("📊 Results")
    
    # Convert results to DataFrame
    df = pd.json_normalize(st.session_state.results, sep='_')
    
    # Select and reorder columns for display
    display_columns = [
        'name',
        'type',
        'latitude',
        'longitude',
        'gpt_population',
        'gpt_confidence',
        'gemini_population',
        'gemini_confidence',
        'combined_population',
        'combined_confidence'
    ]
    
    # Ensure all columns exist
    for col in display_columns:
        if col not in df.columns:
            df[col] = None
    
    # Select available columns
    available_columns = [col for col in display_columns if col in df.columns]
    df_display = df[available_columns].copy()
    
    # Clean column names
    df_display.columns = [col.replace('admin_hierarchy_', '').replace('_', ' ').title() for col in df_display.columns]
    
    # Format numeric columns
    population_columns = ['Gpt Population', 'Gemini Population', 'Combined Population']
    for col in population_columns:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) and isinstance(x, (int, float)) and x > 0 else "-")
    
    if 'Latitude' in df_display.columns:
        df_display['Latitude'] = df_display['Latitude'].apply(lambda x: f"{x:.6f}" if pd.notna(x) and isinstance(x, (int, float)) else "-")
    if 'Longitude' in df_display.columns:
        df_display['Longitude'] = df_display['Longitude'].apply(lambda x: f"{x:.6f}" if pd.notna(x) and isinstance(x, (int, float)) else "-")
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Locations", len(df_display))
    
    with col2:
        # Use combined_population if available, otherwise fallback to individual results
        pop_col = 'combined_population' if 'combined_population' in df.columns else ('gpt_population' if 'gpt_population' in df.columns else 'gemini_population')
        if pop_col in df.columns:
            identified_count = df[pop_col].notna().sum()
        else:
            identified_count = 0
        st.metric("Identified", identified_count)
    
    with col3:
        # Un-identified: locations without population
        pop_col = 'combined_population' if 'combined_population' in df.columns else ('gpt_population' if 'gpt_population' in df.columns else 'gemini_population')
        if pop_col in df.columns:
            unidentified_count = df[pop_col].isna().sum()
        else:
            unidentified_count = len(df_display)
        st.metric("Un-identified", unidentified_count)
    
    with col4:
        # Use combined_population if available, otherwise fallback
        pop_col = 'combined_population' if 'combined_population' in df.columns else ('gpt_population' if 'gpt_population' in df.columns else 'gemini_population')
        if pop_col in df.columns:
            clean_count = ((df[pop_col].notna()) & (df[pop_col] > 10000)).sum()
        else:
            clean_count = 0
        st.metric("Population > 10K", clean_count)
    
    # Download button
    if st.session_state.excel_data is not None:
        st.download_button(
            label="📥 Download Excel File",
            data=st.session_state.excel_data,
            file_name=f"{uploaded_file.name.replace('.kmz', '')}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch'
        )
    
    # Display map with locations and polygon boundary
    st.subheader("🗺️ Location Map")
    
    # Prepare location data for map
    map_columns = ['latitude', 'longitude', 'name']
    if 'type' in df.columns:
        map_columns.append('type')
    map_df = df[map_columns].dropna(subset=['latitude', 'longitude', 'name'])
    
    if len(map_df) > 0:
        # Calculate center of locations
        center_lat = map_df['latitude'].mean()
        center_lon = map_df['longitude'].mean()
        
        # Create layers
        layers = []
        
        # Add polygon layer if we have polygon points
        if st.session_state.polygon_points:
            # Convert polygon points to the format pydeck expects (list of [lon, lat])
            polygon_coords = [[pt[0], pt[1]] for pt in st.session_state.polygon_points]
            # Close the polygon
            if polygon_coords[0] != polygon_coords[-1]:
                polygon_coords.append(polygon_coords[0])
            
            polygon_layer = pdk.Layer(
                "PolygonLayer",
                data=[{"polygon": polygon_coords}],
                get_polygon="polygon",
                get_fill_color=[31, 119, 180, 50],  # Blue with transparency
                get_line_color=[31, 119, 180, 255],  # Solid blue outline
                line_width_min_pixels=2,
                pickable=False,
            )
            layers.append(polygon_layer)
        
        # Add scatter plot layer for locations with larger, more visible markers
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["longitude", "latitude"],
            get_color=[255, 87, 51, 220],  # Orange-red with higher opacity
            get_radius=500,
            radius_min_pixels=5,
            radius_max_pixels=15,
            pickable=True,
        )
        layers.append(scatter_layer)
        
        # Add TextLayer to display location names on the map
        text_layer = pdk.Layer(
            "TextLayer",
            data=map_df,
            get_position=["longitude", "latitude"],
            get_text="name",
            get_color=[0, 0, 0, 255],  # Black text
            get_size=12,
            get_angle=0,
            get_text_anchor="middle",
            get_alignment_baseline="center",
            pickable=True,
        )
        layers.append(text_layer)
        
        # Create the deck with HTML tooltips
        # Use OpenStreetMap tiles when map_style is None (no API key required)
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=config.MAP_DEFAULT_ZOOM,
                pitch=config.MAP_DEFAULT_PITCH,
            ),
            map_style=config.MAP_STYLE,  # None defaults to OpenStreetMap tiles
            tooltip={
                "html": "<b>{name}</b>" + ("<br/>Type: {type}" if 'type' in map_columns else "") + "<br/>Coordinates: {latitude:.4f}, {longitude:.4f}",
                "style": {
                    "backgroundColor": "#1A1A1A",
                    "color": "white",
                    "fontSize": "12px",
                    "padding": "8px"
                }
            }
        )
        
        st.pydeck_chart(deck, use_container_width=True)
    
    # Add search/filter
    st.subheader("Location Data")
    search_term = st.text_input("Search locations", placeholder="Type to filter by name...")
    
    if search_term:
        df_display = df_display[df_display['Name'].str.contains(search_term, case=False, na=False)]
    
    # Display table with sorting
    st.dataframe(
        df_display,
        width='stretch',
        height=400,
        hide_index=True
    )

# Debug section for testing API connections
st.divider()
with st.expander("🔧 Debug: Test API Connections", expanded=False):
    gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
    
    if not gemini_api_key:
        st.error("⚠️ Gemini API key not found in secrets.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("List Available Models")
            if st.button("List Gemini Models"):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_api_key)
                    
                    models = genai.list_models()
                    available_models = []
                    for model in models:
                        if 'generateContent' in model.supported_generation_methods:
                            available_models.append(model.name.replace('models/', ''))
                    
                    if available_models:
                        st.success(f"✅ Found {len(available_models)} available model(s):")
                        for model_name in available_models:
                            st.code(model_name, language=None)
                    else:
                        st.warning("⚠️ No models found with generateContent support.")
                except Exception as e:
                    import traceback
                    error_traceback = traceback.format_exc()
                    st.error(f"❌ Failed to list models: {str(e)}")
                    with st.expander("Full Error Details"):
                        st.code(error_traceback, language=None)
        
        with col2:
            st.subheader("Test Gemini API Connection")
            if st.button("Test Gemini API"):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_api_key)
                    model = genai.GenerativeModel(config.GEMINI_MODEL)
                    
                    # Simple test prompt
                    test_prompt = "What is 2+2? Respond with just the number."
                    response = model.generate_content(test_prompt)
                    
                    if response and hasattr(response, 'text') and response.text:
                        st.success(f"✅ Gemini API connection successful!")
                        st.info(f"Model: {config.GEMINI_MODEL}")
                        st.info(f"Test response: {response.text}")
                    else:
                        st.error(f"❌ Gemini API returned invalid response: {response}")
                except Exception as e:
                    import traceback
                    error_traceback = traceback.format_exc()
                    st.error(f"❌ Gemini API test failed: {str(e)}")
                    st.info(f"💡 Current model: {config.GEMINI_MODEL}")
                    st.info("💡 Try clicking 'List Gemini Models' to see available models.")
                    with st.expander("Full Error Details"):
                        st.code(error_traceback, language=None)

# Collapsible log section at bottom (closed by default)
st.divider()
with st.expander("📋 Processing Log (click to expand)", expanded=False):
    all_messages = st.session_state.progress_messages + st.session_state.status_messages
    if all_messages:
        # Parse log messages to extract errors, checkpoints, and stages
        errors = []
        warnings = []
        checkpoints = []
        stages_completed = []
        stages_started = []
        last_checkpoint = None
        start_time = None
        end_time = None
        
        for msg in all_messages:
            # Extract timestamps
            if msg.startswith('[') and ']' in msg:
                timestamp = msg[1:msg.index(']')]
                if start_time is None:
                    start_time = timestamp
                end_time = timestamp
            
            # Identify errors
            if 'ERROR:' in msg.upper() or 'ERROR' in msg.upper():
                errors.append(msg)
            
            # Identify warnings
            if 'WARNING' in msg.upper() or '⚠️' in msg:
                warnings.append(msg)
            
            # Identify checkpoints
            if 'CHECKPOINT:' in msg.upper():
                checkpoints.append(msg)
                last_checkpoint = msg
                # Extract stage completion
                if 'completed' in msg.lower():
                    if 'Stage 1' in msg:
                        stages_completed.append('Stage 1: KMZ Parsing')
                    elif 'Stage 2' in msg:
                        stages_completed.append('Stage 2: Finding OSM Locations')
                    elif 'Stage 3' in msg:
                        stages_completed.append('Stage 3: Administrative Boundaries')
                    elif 'Stage 4' in msg:
                        stages_completed.append('Stage 4: Population Estimation')
                    elif 'All processing stages completed' in msg:
                        stages_completed.append('All Stages')
                    elif 'Excel export completed' in msg:
                        stages_completed.append('Excel Export')
                elif 'started' in msg.lower():
                    if 'Stage 1' in msg:
                        stages_started.append('Stage 1: KMZ Parsing')
                    elif 'Stage 2' in msg:
                        stages_started.append('Stage 2: Finding OSM Locations')
                    elif 'Stage 3' in msg:
                        stages_started.append('Stage 3: Administrative Boundaries')
                    elif 'Stage 4' in msg:
                        stages_started.append('Stage 4: Population Estimation')
        
        # Calculate execution time if we have timestamps
        execution_time = None
        if start_time and end_time:
            try:
                from datetime import datetime, timedelta
                start_dt = datetime.strptime(start_time, "%H:%M:%S")
                end_dt = datetime.strptime(end_time, "%H:%M:%S")
                # Handle day rollover
                if end_dt < start_dt:
                    end_dt = end_dt + timedelta(days=1)
                delta = end_dt - start_dt
                execution_time = f"{delta.total_seconds():.1f} seconds"
            except Exception as e:
                # If time calculation fails, just show the start and end times
                execution_time = f"{start_time} - {end_time}"
        
        # Display Summary Section
        st.markdown("### 📊 Log Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if errors:
                st.error(f"❌ **Errors:** {len(errors)}")
            else:
                st.success("✅ **Errors:** 0")
        
        with col2:
            if warnings:
                st.warning(f"⚠️ **Warnings:** {len(warnings)}")
            else:
                st.info("ℹ️ **Warnings:** 0")
        
        with col3:
            if execution_time:
                st.info(f"⏱️ **Execution Time:** {execution_time}")
            else:
                st.info("⏱️ **Execution Time:** N/A")
        
        # Show last checkpoint
        if last_checkpoint:
            st.info(f"📍 **Last Checkpoint:** {last_checkpoint}")
        
        # Show stage completion status
        if stages_completed or stages_started:
            st.markdown("#### Stage Status")
            all_stages = ['Stage 1: KMZ Parsing', 'Stage 2: Finding OSM Locations', 
                         'Stage 3: Administrative Boundaries', 'Stage 4: Population Estimation', 
                         'Excel Export']
            for stage in all_stages:
                if stage in stages_completed:
                    st.success(f"✅ {stage}")
                elif stage in stages_started:
                    st.warning(f"⏳ {stage} (started but not completed)")
                else:
                    st.text(f"⚪ {stage} (not started)")
        
        # Diagnostics: last exception and last position (RUN / CALL_START / CALL_END, or last [SM] line)
        st.markdown("#### Diagnostics")
        last_diagnostic = None
        last_position = None
        last_sm = None
        last_check = None
        last_button = None
        last_init = None
        for msg in reversed(all_messages):
            if "DIAGNOSTIC:" in msg and last_diagnostic is None:
                last_diagnostic = msg
            if ("RUN " in msg or "CALL_START" in msg or "CALL_END" in msg) and last_position is None:
                last_position = msg
            if "[SM]" in msg and last_sm is None:
                last_sm = msg
            if "[CHECK]" in msg and last_check is None:
                last_check = msg
            if "[BUTTON]" in msg and last_button is None:
                last_button = msg
            if "[INIT]" in msg and last_init is None:
                last_init = msg
        
        if last_diagnostic:
            st.warning(f"**Last exception:** {last_diagnostic}")
        else:
            st.text("Last exception: none")
        
        if last_position:
            st.info(f"**Last position (workflow):** {last_position}")
        elif last_sm:
            st.info(f"**Last position (state machine):** {last_sm}")
        elif last_check:
            st.info(f"**Last position (check):** {last_check}")
        elif last_init:
            st.info(f"**Last position (init):** {last_init}")
        elif last_button:
            st.info(f"**Last position (button):** {last_button}")
        else:
            st.text("Last position: (no position markers found)")
        
        # Execution flow summary
        st.markdown("#### Execution Flow Summary")
        flow_stages = []
        for msg in all_messages:
            if "BUTTON CLICKED" in msg:
                flow_stages.append("🔵 Button clicked")
            elif "[INIT] KMZ+OSM success" in msg:
                flow_stages.append("✅ KMZ+OSM completed")
            elif "TRANSITIONING TO POPULATION" in msg:
                flow_stages.append("🔄 Hierarchy → Population transition")
            elif "[SM] population: ENTERED population stage" in msg:
                flow_stages.append("✅ Population stage entered")
            elif "ALL POPULATION BATCHES COMPLETE" in msg:
                flow_stages.append("✅ Population batches completed")
            elif "Transitioning to excel stage" in msg:
                flow_stages.append("🔄 Population → Excel transition")
            elif "[SM] excel: ENTERED excel stage" in msg:
                flow_stages.append("✅ Excel stage entered")
            elif "STATE MACHINE NOT ENTERED" in msg:
                flow_stages.append("❌ State machine NOT entered (critical issue)")
        
        if flow_stages:
            for stage in flow_stages:
                st.text(stage)
        else:
            st.text("No major flow stages detected yet")
        
        # Show error summary if there are errors
        if errors:
            st.markdown("#### ❌ Error Summary")
            with st.expander("View Errors", expanded=True):
                for i, error in enumerate(errors[:10], 1):  # Show first 10 errors
                    st.error(f"**Error {i}:**\n```\n{error}\n```")
                if len(errors) > 10:
                    st.caption(f"... and {len(errors) - 10} more errors")
        
        st.divider()
        st.markdown("### 📝 Full Log")
        
        # Display log with syntax highlighting for errors
        log_text = "\n".join(all_messages)
        
        # Create HTML with error highlighting
        log_lines = log_text.split('\n')
        highlighted_log = []
        for line in log_lines:
            if 'ERROR:' in line.upper() or 'ERROR' in line.upper():
                highlighted_log.append(f'<span style="color: red; font-weight: bold;">{line}</span>')
            elif 'WARNING' in line.upper() or '⚠️' in line:
                highlighted_log.append(f'<span style="color: orange; font-weight: bold;">{line}</span>')
            elif 'CHECKPOINT:' in line.upper():
                highlighted_log.append(f'<span style="color: blue; font-weight: bold;">{line}</span>')
            elif '[SM]' in line or '[Overpass]' in line:
                highlighted_log.append(f'<span style="color: #0066cc; font-weight: bold;">{line}</span>')
            elif 'DIAGNOSTIC:' in line or 'RUN #' in line or 'CALL_START' in line or 'CALL_END' in line:
                highlighted_log.append(f'<span style="color: #006600; font-weight: bold;">{line}</span>')
            else:
                highlighted_log.append(line)
        
        st.markdown(f'<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 0.85em;">{"<br>".join(highlighted_log)}</pre>', unsafe_allow_html=True)
        
        st.caption("💡 Tip: Copy this log if you need to report any issues. Errors are highlighted in red, warnings in orange, and checkpoints in blue.")
    else:
        st.info("Log will appear here once processing starts.")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #888; padding: 2rem;'>
        <p>Built with Streamlit | Uses OpenStreetMap and OpenAI GPT</p>
    </div>
""", unsafe_allow_html=True)
