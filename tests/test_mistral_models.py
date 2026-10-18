"""Unit tests for Mistral model integration and summariser routing."""

import os
import unittest
from unittest.mock import MagicMock, patch

import config
from src.summariser import LegalSummariser


class TestMistralModels(unittest.TestCase):

    def test_mistral_model_definitions(self):
        """Verify MISTRAL_MODELS contains exactly mistral-large-4 and mistral-small-2603."""
        expected_models = {
            "mistral-large-4": "mistral-large-4",
            "mistral-small-2603": "mistral-small-2603",
        }
        self.assertEqual(config.MISTRAL_MODELS, expected_models)

        # Verify display annotations in ALL_MODELS
        self.assertEqual(
            config.ALL_MODELS["mistral-large-4"],
            "mistral-large-4 (gpt-5 alternative)",
        )
        self.assertEqual(
            config.ALL_MODELS["mistral-small-2603"],
            "mistral-small-2603 (gpt-5-mini alternative)",
        )

    def test_is_mistral_model(self):
        """Verify detection of Mistral cloud models and local models."""
        self.assertTrue(config.is_mistral_model("mistral-large-4"))
        self.assertTrue(config.is_mistral_model("mistral-small-2603"))

        # Ollama local models or non-existent models
        self.assertFalse(config.is_mistral_model("gemma2:9b"))
        self.assertFalse(config.is_mistral_model("llama3.1:8b"))
        self.assertFalse(config.is_mistral_model("mistral:7b"))
        self.assertFalse(config.is_mistral_model(""))

    def test_resolve_model_name(self):
        """Verify model name resolution."""
        self.assertEqual(config.resolve_model_name("mistral-large-4"), "mistral-large-4")
        self.assertEqual(config.resolve_model_name("mistral-small-2603"), "mistral-small-2603")

    def test_missing_api_key_error_handling(self):
        """Verify helpful error message when MISTRAL_API_KEY is unset."""
        with patch.object(config, "MISTRAL_API_KEY", ""):
            with patch.dict(os.environ, {"MISTRAL_API_KEY": ""}):
                summariser = LegalSummariser()
                res = summariser._summarise_with_mistral(
                    model_name="mistral-large-4",
                    system_instruction="System prompt",
                    user_prompt="User query",
                    total_docs=3,
                )
                self.assertEqual(res["status"], "error")
                self.assertIn("Mistral API Key Missing", res["summary"])

    def test_mistral_rest_api_success_mock(self):
        """Verify REST API response parsing when SDK is bypassed."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is a synthesised legal summary of EU acts.",
                    }
                }
            ]
        }

        with patch.object(config, "MISTRAL_API_KEY", "dummy_key"):
            with patch("requests.post", return_value=mock_response):
                with patch("mistralai.client.Mistral", side_effect=ImportError("mocked import error")):
                    summariser = LegalSummariser()
                    res = summariser._summarise_with_mistral(
                        model_name="mistral-large-4",
                        system_instruction="System prompt",
                        user_prompt="User prompt",
                        total_docs=2,
                    )
                    self.assertEqual(res["status"], "success")
                    self.assertEqual(res["summary"], "This is a synthesised legal summary of EU acts.")
                    self.assertEqual(res["model_used"], "mistral-large-4 (gpt-5 alternative)")
                    self.assertEqual(res["docs_analyzed"], 2)


if __name__ == "__main__":
    unittest.main()
