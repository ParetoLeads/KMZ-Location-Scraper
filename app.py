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
# STYLES - Minimal CSS, working WITH Streamlit's theme
# ============================================================================
# Typography Scale (1.25 ratio, 16px minimum):
# - Metric Values: 36px / 800
# - Logo: 28px / 700
# - Section Title: 22px / 600
# - Body/Labels/Buttons: 18px / 400-600
# - Minimum (captions): 16px / 400

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container spacing */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* ==================== CUSTOM HTML ELEMENTS ==================== */

    /* Header */
    .main-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 16px 0 32px 0;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 32px;
    }

    .logo-text {
        font-size: 28px;
        font-weight: 700;
        color: #1A1A1A;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .logo-text span {
        color: #FF6B00;
    }

    .app-badge {
        background: #FFF4ED;
        color: #FF6B00;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 16px;
        font-weight: 600;
    }

    /* Section titles */
    .section-title {
        font-size: 22px;
        font-weight: 600;
        color: #1A1A1A;
        margin: 0 0 8px 0;
    }

    .section-subtitle {
        font-size: 16px;
        color: #6B7280;
        margin: 0 0 24px 0;
    }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 36px;
        font-weight: 800;
        color: #1A1A1A;
        line-height: 1.2;
    }

    .metric-value.orange {
        color: #FF6B00;
    }

    .metric-label {
        font-size: 16px;
        color: #6B7280;
        margin-top: 4px;
    }

    /* File upload success card */
    .file-success {
        background: #ECFDF5;
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 16px;
    }

    .file-success-name {
        font-size: 18px;
        font-weight: 600;
        color: #065F46;
        margin: 0;
    }

    .file-success-size {
        font-size: 16px;
        color: #047857;
        margin: 8px 0 0 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 32px 0 16px 0;
        color: #6B7280;
        font-size: 16px;
        border-top: 1px solid #E5E7EB;
        margin-top: 48px;
    }

    .footer a {
        color: #FF6B00;
        text-decoration: none;
    }

    /* ==================== STREAMLIT NATIVE ELEMENTS ==================== */
    /* Only override what's necessary, let theme handle the rest */

    /* Buttons - orange theme */
    .stButton > button {
        background: #FF6B00 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        background: #E55F00 !important;
    }

    .stDownloadButton > button {
        background: #FF6B00 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    /* File uploader - just add orange dashed border */
    [data-testid="stFileUploader"] > div > div {
        border: 2px dashed #FF6B00 !important;
        border-radius: 12px !important;
        min-height: 200px !important;
    }

    /* File uploader button */
    [data-testid="stFileUploader"] button {
        background: #FF6B00 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: #FF6B00 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #F5F5F5;
        border-radius: 10px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
    }

    /* Text input */
    .stTextInput input {
        font-size: 16px !important;
        padding: 12px !important;
        border-radius: 8px !important;
    }

    /* Expander (Processing Log) - dark theme for contrast */
    [data-testid="stExpander"] {
        background: #2D2D2D !important;
        border-radius: 8px !important;
        border: none !important;
    }

    [data-testid="stExpander"] summary {
        color: #FFFFFF !important;
        font-size: 16px !important;
    }

    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {
        color: #FFFFFF !important;
    }

    [data-testid="stExpander"] svg {
        stroke: #FFFFFF !important;
    }

    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: #1E1E1E !important;
    }

    [data-testid="stExpander"] pre,
    [data-testid="stExpander"] code {
        background: #1E1E1E !important;
        color: #E0E0E0 !important;
        font-size: 14px !important;
    }

    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span {
        color: #E0E0E0 !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        font-size: 16px !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="main-header">
    <p class="logo-text">pareto<span>leads</span></p>
    <span class="app-badge">Location Scraper</span>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# UPLOAD SECTION
# ============================================================================
st.markdown('<p class="section-title">Upload KMZ File</p>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Upload a KMZ file containing the boundary polygon to analyze locations within the area.</p>', unsafe_allow_html=True)

# File uploader - native Streamlit, styled via CSS
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        f"Drop your KMZ file here or click to browse (Max {config.MAX_FILE_SIZE_MB}MB)",
        type=['kmz'],
        help=f"Select a KMZ file containing the boundary polygon. Maximum file size: {config.MAX_FILE_SIZE_MB}MB"
    )

    # Show success message when file uploaded
    if uploaded_file is not None:
        st.markdown(f"""
        <div class="file-success">
            <p class="file-success-name">✓ {uploaded_file.name}</p>
            <p class="file-success-size">{uploaded_file.size / 1024:.1f} KB • Ready to analyze</p>
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
        st.error(f"File too large: {str(e)}")
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Start Analysis", type="primary", disabled=st.session_state.processing, use_container_width=True):
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
                    st.error(f"Invalid KMZ file: {str(e)}")
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
                            st.error(f"API Key Error: {str(e)}")
                            st.session_state.processing = False
                        else:
                            # Progress UI
                            progress_container = st.container()
                            with progress_container:
                                st.divider()
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
                                        st.success(f"Successfully processed {len(results)} locations!")
                                    else:
                                        st.warning("Analysis completed but Excel export failed.")
                                        st.session_state.results = results
                                        st.session_state.progress_messages = progress_messages
                                        st.session_state.status_messages = status_messages
                                else:
                                    st.error("Analysis failed. Check the log below for details.")
                                    st.session_state.progress_messages = progress_messages
                                    st.session_state.status_messages = status_messages
                                    main_progress.progress(0)

                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
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
                        st.error(f"Error: {str(e)}")
                        st.session_state.processing = False
                        if 'tmp_kmz_path' in locals() and os.path.exists(tmp_kmz_path):
                            os.unlink(tmp_kmz_path)

# ============================================================================
# RESULTS SECTION
# ============================================================================
if st.session_state.results is not None:
    st.divider()

    # Results header with download button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="section-title">Analysis Results</p>', unsafe_allow_html=True)
    with col2:
        if st.session_state.excel_data is not None:
            st.download_button(
                label="Download Excel",
                data=st.session_state.excel_data,
                file_name=f"{uploaded_file.name.replace('.kmz', '')}_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    # Convert results to DataFrame
    df = pd.json_normalize(st.session_state.results, sep='_')

    # Metrics row
    st.markdown("<br>", unsafe_allow_html=True)
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

    # Tabs: Map / Data
    tab1, tab2 = st.tabs(["Map View", "Data Table"])

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

                polygon_layer = pdk.Layer(
                    "PolygonLayer",
                    data=[{"polygon": polygon_coords}],
                    get_polygon="polygon",
                    get_fill_color=[255, 107, 0, 30],
                    get_line_color=[255, 107, 0, 200],
                    line_width_min_pixels=3,
                    pickable=False,
                )
                layers.append(polygon_layer)

            # Location markers
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["longitude", "latitude"],
                get_color=[255, 107, 0, 220],
                get_radius=500,
                radius_min_pixels=5,
                radius_max_pixels=15,
                pickable=True,
            )
            layers.append(scatter_layer)

            # Create map
            deck = pdk.Deck(
                layers=layers,
                initial_view_state=pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=zoom,
                    pitch=0,
                ),
                map_style="mapbox://styles/mapbox/light-v10",
                tooltip={"html": "<b>{name}</b><br/>Type: {type}", "style": {"backgroundColor": "#1A1A1A", "color": "white"}}
            )

            st.pydeck_chart(deck, use_container_width=True)
        else:
            st.info("No location data available for map display.")

    with tab2:
        # Search filter
        search_term = st.text_input("Search locations", placeholder="Type to filter by name...")

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
# PROCESSING LOG (at bottom, collapsed)
# ============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("Processing Log", expanded=False):
    all_messages = st.session_state.progress_messages + st.session_state.status_messages
    if all_messages:
        log_text = "\n".join(all_messages[-50:])
        st.code(log_text, language="text")
        if len(all_messages) > 50:
            st.caption(f"Showing last 50 of {len(all_messages)} messages")
    else:
        st.write("Log will appear here once processing starts.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    Built by <a href="https://paretoleads.com" target="_blank">Paretoleads</a> • Powered by OpenStreetMap & OpenAI
</div>
""", unsafe_allow_html=True)
