"""
Configuration module for KMZ Location Scraper.

Centralizes all configuration constants, settings, and environment variable support.
"""
import os
from typing import List


class Config:
    """Centralized configuration for the application."""
    
    # Place Types
    PRIMARY_PLACE_TYPES: List[str] = [
        'city', 'town', 'district', 'county', 'municipality', 'borough', 'suburb'
    ]
    ADDITIONAL_PLACE_TYPES: List[str] = [
        'neighbourhood', 'village', 'locality'
    ]
    SPECIAL_PLACE_TYPES: List[str] = [
        'commercial_area'
    ]
    
    @property
    def all_place_types(self) -> List[str]:
        """Get all place types combined."""
        return self.PRIMARY_PLACE_TYPES + self.ADDITIONAL_PLACE_TYPES + self.SPECIAL_PLACE_TYPES
    
    # API Settings
    # Overpass API base URL. Set OVERPASS_URL or OSM_OVERPASS_URL to an alternative (e.g. https://overpass.private.coffee/api/interpreter) to reduce 504 Gateway Timeout errors.
    OSM_OVERPASS_URL: str = os.getenv(
        'OVERPASS_URL',
        os.getenv('OSM_OVERPASS_URL', 'http://overpass-api.de/api/interpreter')
    )
    OSM_API_TIMEOUT: int = int(os.getenv('OSM_API_TIMEOUT', '60'))
    OSM_API_TIMEOUT_BUFFER: int = 10  # Additional seconds for timeout buffer
    
    # Retry Settings
    MAX_RETRY_ATTEMPTS: int = int(os.getenv('MAX_RETRY_ATTEMPTS', '12'))
    RETRY_DELAY_BASE: int = 5  # Base delay in seconds (increments by attempt number)
    DEFAULT_MAX_RETRIES: int = 3  # Legacy default (kept for compatibility)
    
    # Processing Settings
    DEFAULT_CHUNK_SIZE: int = int(os.getenv('DEFAULT_CHUNK_SIZE', '25'))
    DEFAULT_BATCH_SIZE: int = 3   # For hierarchy (Overpass); small so each rerun finishes within Streamlit timeout
    HIERARCHY_FIRST_BATCH_SIZE: int = 1   # First hierarchy batch size (minimal so first work run completes within timeout)
    HIERARCHY_QUERY_TIMEOUT: int = 15   # Short timeout so work run finishes before platform/gateway timeout
    HIERARCHY_MAX_RETRIES_CHUNKED: int = 1   # One retry per batch; keeps run under ~30s
    DEFAULT_MAX_LOCATIONS: int = int(os.getenv('DEFAULT_MAX_LOCATIONS', '0'))  # 0 = no limit
    DEFAULT_PAUSE_BEFORE_GPT: bool = False
    DEFAULT_ENABLE_WEB_BROWSING: bool = os.getenv('ENABLE_WEB_BROWSING', 'True').lower() == 'true'
    
    # Geometry Settings
    POLYGON_BUFFER_KM: float = float(os.getenv('POLYGON_BUFFER_KM', '2.0'))
    BOUNDING_BOX_BUFFER: float = 0.04  # Degrees buffer for bounding box
    
    # Delay Settings (in seconds)
    OSM_QUERY_DELAY: int = int(os.getenv('OSM_QUERY_DELAY', '6'))  # Between primary/additional/special to avoid 504 from Overpass
    HIERARCHY_BATCH_DELAY: int = int(os.getenv('HIERARCHY_BATCH_DELAY', '3'))  # Between hierarchy batches
    GPT_BATCH_DELAY: int = int(os.getenv('GPT_BATCH_DELAY', '2'))  # Between GPT batches
    
    # GPT Settings
    USE_GPT: bool = os.getenv('USE_GPT', 'True').lower() == 'true'
    GPT_MODEL: str = os.getenv('GPT_MODEL', 'gpt-4-turbo')
    GPT_TEMPERATURE: float = 0.1
    
    # Gemini Settings
    USE_GEMINI: bool = os.getenv('USE_GEMINI', 'True').lower() == 'true'
    GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')  # Updated to gemini-2.5-pro
    GEMINI_TEMPERATURE: float = float(os.getenv('GEMINI_TEMPERATURE', '0.2'))
    GEMINI_TOP_P: float = float(os.getenv('GEMINI_TOP_P', '0.95'))
    GEMINI_TOP_K: int = int(os.getenv('GEMINI_TOP_K', '40'))
    GEMINI_MAX_OUTPUT_TOKENS: int = int(os.getenv('GEMINI_MAX_OUTPUT_TOKENS', '4096'))
    
    # UI Settings
    MAX_FILE_SIZE_MB: int = int(os.getenv('MAX_FILE_SIZE_MB', '1'))
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # Map Settings
    MAP_DEFAULT_ZOOM: int = 9
    MAP_DEFAULT_PITCH: int = 0
    MAP_STYLE: str = None  # None uses OpenStreetMap tiles (free, no API key required)
    
    # Progress Settings
    TOTAL_STAGES: int = 5  # KMZ, OSM, Hierarchy, GPT, Excel
    PROGRESS_KMZ_OSM_PERCENT: int = 10  # Percentage for KMZ/OSM parsing
    PROGRESS_EXCEL_PERCENT: int = 10  # Percentage for Excel export
    PROGRESS_BATCH_PERCENT: int = 80  # Percentage for batch processing
    
    # Administrative Hierarchy Levels
    ADMIN_LEVELS: List[int] = [8, 6, 4, 2]
    
    # Population Filter
    MIN_POPULATION_FOR_CLEAN_DATA: int = 10000
    
    # Verbose Output
    VERBOSE: bool = os.getenv('VERBOSE', 'False').lower() == 'true'
    
    # Caching (optional)
    ENABLE_CACHE: bool = os.getenv('ENABLE_CACHE', 'False').lower() == 'true'
    CACHE_TTL_SECONDS: int = int(os.getenv('CACHE_TTL_SECONDS', '3600'))  # 1 hour default
    
    # Parallelization (optional)
    ENABLE_PARALLEL_OSM_QUERIES: bool = os.getenv('ENABLE_PARALLEL_OSM_QUERIES', 'False').lower() == 'true'
    MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', '3'))  # For thread pool


# Global config instance
config = Config()
