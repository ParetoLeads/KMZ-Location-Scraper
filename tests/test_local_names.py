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


if __name__ == "__main__":
    unittest.main()
