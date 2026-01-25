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

# Page configuration
st.set_page_config(
    page_title="Location Scraper | Paretoleads",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# PARETOLEADS BRANDED STYLES - Light Theme
# ============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Root variables */
    :root {
        --primary-orange: #FF6B00;
        --primary-orange-hover: #E55F00;
        --primary-orange-light: #FFF4ED;
        --dark-text: #1A1A1A;
        --secondary-text: #6B7280;
        --light-bg: #FFFFFF;
        --card-bg: #F8F9FA;
        --border-color: #E5E7EB;
        --success-green: #10B981;
        --warning-yellow: #F59E0B;
    }

    /* Global styles */
    .stApp {
        background-color: var(--light-bg) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header styles */
    .main-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0 2rem 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }

    .logo-section {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .logo-text {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--dark-text);
        margin: 0;
        letter-spacing: -0.5px;
    }

    .logo-text span {
        color: var(--primary-orange);
    }

    .app-badge {
        background: var(--primary-orange-light);
        color: var(--primary-orange);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section titles */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--dark-text);
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .section-subtitle {
        font-size: 0.9rem;
        color: var(--secondary-text);
        margin: -0.5rem 0 1.5rem 0;
    }

    /* Card styles */
    .info-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--dark-text);
        line-height: 1.2;
    }

    .metric-value.orange {
        color: var(--primary-orange);
    }

    .metric-label {
        font-size: 0.8rem;
        color: var(--secondary-text);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }

    /* Buttons */
    .stButton > button {
        background: var(--primary-orange) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }

    .stButton > button:hover {
        background: var(--primary-orange-hover) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255, 107, 0, 0.3) !important;
    }

    .stButton > button:disabled {
        background: var(--border-color) !important;
        color: var(--secondary-text) !important;
    }

    /* Download button special styling */
    .stDownloadButton > button {
        background: var(--primary-orange) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stDownloadButton > button:hover {
        background: var(--primary-orange-hover) !important;
    }

    /* File uploader - Style it to look better */
    .stFileUploader {
        background: var(--card-bg) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    .stFileUploader:hover {
        border-color: var(--primary-orange) !important;
        background: var(--primary-orange-light) !important;
    }

    .stFileUploader > div {
        background: transparent !important;
    }

    .stFileUploader label {
        color: var(--dark-text) !important;
    }

    /* Style the file uploader button */
    .stFileUploader button {
        background: var(--primary-orange) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: var(--primary-orange) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--card-bg);
        border-radius: 10px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        color: var(--secondary-text);
    }

    .stTabs [aria-selected="true"] {
        background: white !important;
        color: var(--dark-text) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Data table */
    .stDataFrame {
        border: 1px solid var(--border-color);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Text input */
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 1px solid var(--border-color) !important;
        padding: 0.75rem !important;
        color: var(--dark-text) !important;
        background: white !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary-orange) !important;
        box-shadow: 0 0 0 2px var(--primary-orange-light) !important;
    }

    /* Expander (for logs) - FIXED: Dark background with light text */
    .streamlit-expanderHeader {
        background: #2D2D2D !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
        color: #FFFFFF !important;
    }

    .streamlit-expanderContent {
        background: #1E1E1E !important;
        border: 1px solid #3D3D3D !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* Code block inside expander */
    .streamlit-expanderContent pre {
        background: #1E1E1E !important;
        color: #E0E0E0 !important;
    }

    .streamlit-expanderContent code {
        background: #1E1E1E !important;
        color: #E0E0E0 !important;
    }

    /* Alerts */
    .stSuccess {
        background: #ECFDF5 !important;
        border: 1px solid #10B981 !important;
        color: #065F46 !important;
    }

    .stError {
        background: #FEF2F2 !important;
        border: 1px solid #EF4444 !important;
        color: #991B1B !important;
    }

    /* Map container */
    .map-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: var(--secondary-text);
        font-size: 0.8rem;
        border-top: 1px solid var(--border-color);
        margin-top: 3rem;
    }

    .footer a {
        color: var(--primary-orange);
        text-decoration: none;
    }

    /* Hide default Streamlit elements we're replacing */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--dark-text) !important;
    }

    /* File info card */
    .file-info {
        background: #ECFDF5;
        border: 1px solid #10B981;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin-top: 1rem;
    }

    .file-info-name {
        font-weight: 600;
        color: #065F46;
        margin: 0;
    }

    .file-info-size {
        font-size: 0.85rem;
        color: #047857;
        margin: 0.25rem 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="main-header">
    <div class="logo-section">
        <div>
            <p class="logo-text">pareto<span>leads</span></p>
        </div>
        <span class="app-badge">Location Scraper</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# UPLOAD SECTION
# ============================================================================
st.markdown('<p class="section-title">📁 Upload KMZ File</p>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Upload a KMZ file containing the boundary polygon to analyze locations within the area.</p>', unsafe_allow_html=True)

# File uploader - using native Streamlit uploader (no custom overlay that blocks clicks)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        f"📍 Drop your KMZ file here (Max {config.MAX_FILE_SIZE_MB}MB)",
        type=['kmz'],
        help=f"Select a KMZ file containing the boundary polygon. Maximum file size: {config.MAX_FILE_SIZE_MB}MB",
        accept_multiple_files=False
    )

    # Show file info if uploaded
    if uploaded_file is not None:
        st.markdown(f"""
        <div class="file-info">
            <p class="file-info-name">✅ {uploaded_file.name}</p>
            <p class="file-info-size">{uploaded_file.size / 1024:.1f} KB • Ready to analyze</p>
        </div>
        """, unsafe_allow_html=True)

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

# ============================================================================
# PROCESS BUTTON & ANALYSIS
# ============================================================================
if uploaded_file is not None:
    # Validate file size
    try:
        validate_file_size(uploaded_file.size)
    except ValidationError as e:
        st.error(f"❌ {str(e)}")
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Analysis", type="primary", disabled=st.session_state.processing, use_container_width=True):
                st.session_state.processing = True
                st.session_state.results = None
                st.session_state.excel_data = None

                # Create temporary file for KMZ
                with tempfile.NamedTemporaryFile(delete=False, suffix='.kmz') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_kmz_path = tmp_file.name

                # Validate KMZ file
                kmz_valid = True
                try:
                    validate_kmz_file(tmp_kmz_path)
                except (ValidationError, KMZParseError) as e:
                    st.error(f"❌ Invalid KMZ file: {str(e)}")
                    st.session_state.processing = False
                    if os.path.exists(tmp_kmz_path):
                        os.unlink(tmp_kmz_path)
                    kmz_valid = False

                if kmz_valid:
                    try:
                        # Get and validate API key from secrets
                        api_key = st.secrets.get("OPENAI_API_KEY", "")
                        try:
                            validate_api_key(api_key)
                        except ValidationError as e:
                            st.error(f"⚠️ {str(e)}")
                            st.session_state.processing = False
                        else:
                            # Progress UI
                            progress_container = st.container()
                            with progress_container:
                                st.markdown("---")
                                st.markdown("**⏳ Processing...**")
                                stage_text = st.empty()
                                main_progress = st.progress(0)
                                progress_metrics = st.empty()
                                status_text = st.empty()

                            # Progress tracking
                            progress_messages = []
                            status_messages = []

                            progress_tracker = ProgressTracker()
                            progress_ui = ProgressUI(
                                stage_text_container=stage_text,
                                progress_bar=main_progress,
                                metrics_container=progress_metrics,
                                status_text_container=status_text
                            )

                            progress_callback = create_progress_callback(
                                tracker=progress_tracker,
                                ui=progress_ui,
                                progress_messages=progress_messages
                            )

                            def status_callback(msg: str) -> None:
                                status_messages.append(msg)
                                progress_messages.append(msg)

                            try:
                                analyzer = LocationAnalyzer(
                                    kmz_file=tmp_kmz_path,
                                    verbose=config.VERBOSE,
                                    openai_api_key=api_key,
                                    use_gpt=config.USE_GPT,
                                    chunk_size=config.DEFAULT_CHUNK_SIZE,
                                    max_locations=config.DEFAULT_MAX_LOCATIONS,
                                    pause_before_gpt=False,
                                    enable_web_browsing=config.DEFAULT_ENABLE_WEB_BROWSING,
                                    primary_place_types=config.PRIMARY_PLACE_TYPES,
                                    additional_place_types=config.ADDITIONAL_PLACE_TYPES,
                                    special_place_types=config.SPECIAL_PLACE_TYPES,
                                    progress_callback=progress_callback,
                                    status_callback=status_callback
                                )

                                results = analyzer.run()

                                if results:
                                    progress_ui.mark_complete()
                                    excel_data = analyzer.save_to_excel(results)

                                    if excel_data:
                                        st.session_state.results = results
                                        st.session_state.excel_data = excel_data
                                        st.session_state.progress_messages = progress_messages
                                        st.session_state.status_messages = status_messages
                                        st.session_state.polygon_points = analyzer.polygon_points
                                        st.success(f"✅ Successfully processed {len(results)} locations!")
                                    else:
                                        st.warning("⚠️ Analysis completed but Excel export failed.")
                                        st.session_state.results = results
                                        st.session_state.progress_messages = progress_messages
                                        st.session_state.status_messages = status_messages
                                else:
                                    st.error("❌ Analysis failed. Check the log below for details.")
                                    st.session_state.progress_messages = progress_messages
                                    st.session_state.status_messages = status_messages
                                    main_progress.progress(0)

                            except Exception as e:
                                st.error(f"❌ Error during analysis: {str(e)}")
                                st.session_state.progress_messages = progress_messages
                                st.session_state.status_messages = status_messages
                                import traceback
                                error_traceback = traceback.format_exc()
                                progress_messages.append(f"ERROR: {str(e)}")
                                progress_messages.append(error_traceback)
                                main_progress.progress(0)

                            finally:
                                if 'tmp_kmz_path' in locals() and os.path.exists(tmp_kmz_path):
                                    os.unlink(tmp_kmz_path)
                                st.session_state.processing = False

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        st.session_state.processing = False
                        if 'tmp_kmz_path' in locals() and os.path.exists(tmp_kmz_path):
                            os.unlink(tmp_kmz_path)

# ============================================================================
# RESULTS SECTION
# ============================================================================
if st.session_state.results is not None:
    st.markdown("---")

    # Results header with download button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="section-title">📊 Analysis Results</p>', unsafe_allow_html=True)
    with col2:
        if st.session_state.excel_data is not None:
            st.download_button(
                label="📥 Download Excel",
                data=st.session_state.excel_data,
                file_name=f"{uploaded_file.name.replace('.kmz', '')}_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    # Convert results to DataFrame
    df = pd.json_normalize(st.session_state.results, sep='_')

    # ========== METRICS ==========
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value orange">{len(df)}</div>
            <div class="metric-label">Total Locations</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        gpt_count = df['gpt_population'].notna().sum() if 'gpt_population' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{gpt_count}</div>
            <div class="metric-label">GPT Data</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        pop_count = ((df['gpt_population'].notna()) & (df['gpt_population'] > 0)).sum() if 'gpt_population' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{pop_count}</div>
            <div class="metric-label">With Population</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        clean_count = ((df['gpt_population'].notna()) & (df['gpt_population'] > 10000)).sum() if 'gpt_population' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{clean_count}</div>
            <div class="metric-label">Pop. > 10K</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========== TABS: Map / Data ==========
    tab1, tab2 = st.tabs(["🗺️ Map View", "📋 Data Table"])

    with tab1:
        # Prepare map data
        map_df = df[['latitude', 'longitude', 'name', 'type']].dropna(subset=['latitude', 'longitude'])

        if len(map_df) > 0:
            # Calculate center
            center_lat = map_df['latitude'].mean()
            center_lon = map_df['longitude'].mean()

            # Calculate zoom based on polygon extent
            if st.session_state.polygon_points:
                lons = [p[0] for p in st.session_state.polygon_points]
                lats = [p[1] for p in st.session_state.polygon_points]
                lon_range = max(lons) - min(lons)
                lat_range = max(lats) - min(lats)
                max_range = max(lon_range, lat_range)
                # Approximate zoom level
                if max_range > 5:
                    zoom = 6
                elif max_range > 2:
                    zoom = 7
                elif max_range > 1:
                    zoom = 8
                elif max_range > 0.5:
                    zoom = 9
                else:
                    zoom = 10
            else:
                zoom = 9

            layers = []

            # Polygon boundary layer (KMZ border)
            if st.session_state.polygon_points:
                polygon_coords = [[pt[0], pt[1]] for pt in st.session_state.polygon_points]
                if polygon_coords[0] != polygon_coords[-1]:
                    polygon_coords.append(polygon_coords[0])

                # Polygon fill
                polygon_layer = pdk.Layer(
                    "PolygonLayer",
                    data=[{"polygon": polygon_coords}],
                    get_polygon="polygon",
                    get_fill_color=[255, 107, 0, 30],  # Orange with low opacity
                    get_line_color=[255, 107, 0, 200],  # Orange border
                    line_width_min_pixels=3,
                    pickable=False,
                )
                layers.append(polygon_layer)

            # Location markers
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["longitude", "latitude"],
                get_color=[255, 107, 0, 220],  # Paretoleads orange
                get_radius=500,
                radius_min_pixels=5,
                radius_max_pixels=15,
                pickable=True,
            )
            layers.append(scatter_layer)

            # Create deck with OpenStreetMap tiles (light style)
            deck = pdk.Deck(
                layers=layers,
                initial_view_state=pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=zoom,
                    pitch=0,
                ),
                map_style="mapbox://styles/mapbox/light-v10",  # Light map style
                tooltip={"html": "<b>{name}</b><br/>Type: {type}", "style": {"backgroundColor": "#1A1A1A", "color": "white"}}
            )

            st.pydeck_chart(deck, use_container_width=True)
        else:
            st.info("No location data available for map display.")

    with tab2:
        # Search filter
        search_term = st.text_input("🔍 Search locations", placeholder="Type to filter by name...")

        # Prepare display dataframe
        display_columns = ['name', 'type', 'latitude', 'longitude', 'gpt_population', 'gpt_confidence']
        for col in display_columns:
            if col not in df.columns:
                df[col] = None

        df_display = df[display_columns].copy()
        df_display.columns = ['Name', 'Type', 'Latitude', 'Longitude', 'Population', 'Confidence']

        # Format columns
        if 'Population' in df_display.columns:
            df_display['Population'] = df_display['Population'].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "-")
        if 'Latitude' in df_display.columns:
            df_display['Latitude'] = df_display['Latitude'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
        if 'Longitude' in df_display.columns:
            df_display['Longitude'] = df_display['Longitude'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

        # Apply search filter
        if search_term:
            df_display = df_display[df_display['Name'].str.contains(search_term, case=False, na=False)]

        # Display table
        st.dataframe(
            df_display,
            use_container_width=True,
            height=450,
            hide_index=True
        )

# ============================================================================
# LOG SECTION (Discreet, at bottom) - Dark themed for visibility
# ============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("📋 Processing Log", expanded=False):
    all_messages = st.session_state.progress_messages + st.session_state.status_messages
    if all_messages:
        # Show last 50 messages to keep it manageable
        log_text = "\n".join(all_messages[-50:])
        st.code(log_text, language="text")
        if len(all_messages) > 50:
            st.caption(f"Showing last 50 of {len(all_messages)} messages")
    else:
        st.markdown("*Log will appear here once processing starts.*")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    <p>Built by <a href="https://paretoleads.com" target="_blank">Paretoleads</a> • Powered by OpenStreetMap & OpenAI</p>
</div>
""", unsafe_allow_html=True)
