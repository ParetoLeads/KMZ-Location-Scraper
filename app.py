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

# Process button
if uploaded_file is not None:
    # Validate file size
    try:
        validate_file_size(uploaded_file.size)
    except ValidationError as e:
        st.error(f"❌ {str(e)}")
    else:
        if st.button("🚀 Start Analysis", type="primary", disabled=st.session_state.processing):
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
                    # Always use both providers
                    provider_value = "both"
                    api_keys_valid = True
                    
                    # Get OpenAI API key
                    openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
                    if not openai_api_key:
                        st.error("⚠️ OpenAI API key not found in secrets. Please add OPENAI_API_KEY to your Streamlit secrets.")
                        st.session_state.processing = False
                        if os.path.exists(tmp_kmz_path):
                            os.unlink(tmp_kmz_path)
                        api_keys_valid = False
                    else:
                        try:
                            validate_api_key(openai_api_key)
                        except ValidationError as e:
                            st.error(f"⚠️ {str(e)}")
                            st.session_state.processing = False
                            if os.path.exists(tmp_kmz_path):
                                os.unlink(tmp_kmz_path)
                            api_keys_valid = False
                    
                    # Get Gemini API key
                    if api_keys_valid:
                        gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
                        if not gemini_api_key:
                            st.error("⚠️ Gemini API key not found in secrets. Please add GEMINI_API_KEY to your Streamlit secrets.")
                            st.session_state.processing = False
                            if os.path.exists(tmp_kmz_path):
                                os.unlink(tmp_kmz_path)
                            api_keys_valid = False
                    
                    # Check if at least one API key is available
                    if api_keys_valid and not openai_api_key and not gemini_api_key:
                        st.error("⚠️ No API keys found. Please configure at least one API key in Streamlit secrets.")
                        st.session_state.processing = False
                        if os.path.exists(tmp_kmz_path):
                            os.unlink(tmp_kmz_path)
                        api_keys_valid = False
                    
                    # If we have valid keys, proceed
                    if api_keys_valid and (openai_api_key or gemini_api_key):
                        # Create progress containers
                        progress_container = st.container()
                        status_container = st.container()
                        
                        with progress_container:
                            # Enhanced progress display
                            stage_text = st.empty()
                            main_progress = st.progress(0)
                            progress_metrics = st.empty()
                            status_text = st.empty()
                        
                        # Progress tracking using new ProgressTracker
                        progress_messages = []
                        status_messages = []
                        
                        # Initialize progress tracker and UI
                        progress_tracker = ProgressTracker()
                        progress_ui = ProgressUI(
                            stage_text_container=stage_text,
                            progress_bar=main_progress,
                            metrics_container=progress_metrics,
                            status_text_container=status_text
                        )
                        
                        # Create callbacks
                        progress_callback = create_progress_callback(
                            tracker=progress_tracker,
                            ui=progress_ui,
                            progress_messages=progress_messages
                        )
                        
                        def status_callback(msg: str) -> None:
                            status_messages.append(msg)
                            progress_messages.append(msg)  # Also add to progress messages for log
                        
                        # Initialize analyzer
                        try:
                            analyzer = LocationAnalyzer(
                                kmz_file=tmp_kmz_path,
                                verbose=config.VERBOSE,
                                openai_api_key=openai_api_key,
                                gemini_api_key=gemini_api_key,
                                ai_provider=provider_value,
                                use_gpt=config.USE_GPT,
                                chunk_size=config.DEFAULT_CHUNK_SIZE,
                                max_locations=config.DEFAULT_MAX_LOCATIONS,
                                pause_before_gpt=False,  # Not used in Streamlit
                                enable_web_browsing=config.DEFAULT_ENABLE_WEB_BROWSING,
                                primary_place_types=config.PRIMARY_PLACE_TYPES,
                                additional_place_types=config.ADDITIONAL_PLACE_TYPES,
                                special_place_types=config.SPECIAL_PLACE_TYPES,
                                progress_callback=progress_callback,
                                status_callback=status_callback
                            )
                            
                            # Run analysis
                            results = analyzer.run()
                            
                            if results:
                                # Update progress to 100%
                                progress_ui.mark_complete()
                                
                                # Save results FIRST, before Excel export (in case Excel fails)
                                st.session_state.results = results
                                st.session_state.progress_messages = progress_messages
                                st.session_state.status_messages = status_messages
                                st.session_state.polygon_points = analyzer.polygon_points
                                
                                # Then attempt Excel export (non-blocking)
                                try:
                                    excel_data = analyzer.save_to_excel(results)
                                    if excel_data:
                                        st.session_state.excel_data = excel_data
                                        st.success(f"✅ Successfully processed {len(results)} locations!")
                                    else:
                                        st.warning("⚠️ Excel export failed, but results are available below.")
                                        st.session_state.excel_data = None
                                except Exception as excel_error:
                                    import traceback
                                    error_traceback = traceback.format_exc()
                                    st.warning(f"⚠️ Excel export error: {str(excel_error)}. Results are still available below.")
                                    # Log the error
                                    status_messages.append(f"Excel export error: {str(excel_error)}")
                                    status_messages.append(error_traceback)
                                    st.session_state.excel_data = None
                            else:
                                st.error("❌ Analysis failed. Check the status messages above.")
                                st.session_state.progress_messages = progress_messages
                                st.session_state.status_messages = status_messages
                                main_progress.progress(0)
                        
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                            st.session_state.progress_messages = progress_messages
                            st.session_state.status_messages = status_messages
                            import traceback
                            error_traceback = traceback.format_exc()
                            with st.expander("Error Details"):
                                st.code(error_traceback)
                            progress_messages.append(f"ERROR: {str(e)}")
                            progress_messages.append(error_traceback)
                            main_progress.progress(0)
                        
                        finally:
                            # Clean up temp file
                            if 'tmp_kmz_path' in locals() and os.path.exists(tmp_kmz_path):
                                os.unlink(tmp_kmz_path)
                            st.session_state.processing = False
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                    st.session_state.processing = False
                    if 'tmp_kmz_path' in locals() and os.path.exists(tmp_kmz_path):
                        os.unlink(tmp_kmz_path)

# Display results
if st.session_state.results is not None:
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
