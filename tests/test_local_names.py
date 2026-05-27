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


if __name__ == "__main__":
    unittest.main()
