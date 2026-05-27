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


if __name__ == "__main__":
    unittest.main()
