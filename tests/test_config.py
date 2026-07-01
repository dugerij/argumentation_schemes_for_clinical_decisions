import unittest
import os
from unittest.mock import patch

from helpers.config import parse_optional_int, startup_check


class ConfigParsingTests(unittest.TestCase):
    def test_parse_optional_int_accepts_all_aliases(self):
        for value in ("all", "none", "null", "unlimited"):
            self.assertIsNone(parse_optional_int(value, default=25))

    def test_parse_optional_int_accepts_integers_and_defaults(self):
        self.assertEqual(parse_optional_int("42", default=25), 42)
        self.assertEqual(parse_optional_int("", default=25), 25)

    def test_startup_check_defaults_to_ollama_models(self):
        with patch.dict(
            "os.environ",
            {
                "INPUT_BASE_DIR": "data/evidence/mimic_discharge_subset",
                "OUTPUT_BASE_DIR": "output",
                "UMLS_ENABLED": "false",
            },
            clear=True,
        ):
            startup_check()
            self.assertEqual(os.environ["GENERATION_MODEL_PROVIDER"], "ollama")
            self.assertEqual(os.environ["RAG_EMBEDDING_MODEL_PROVIDER"], "ollama")

    def test_startup_check_rejects_non_ollama_provider(self):
        with patch.dict(
            "os.environ",
            {
                "INPUT_BASE_DIR": "data/evidence/mimic_discharge_subset",
                "OUTPUT_BASE_DIR": "output",
                "UMLS_ENABLED": "false",
                "GENERATION_MODEL_PROVIDER": "openai",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                startup_check()


if __name__ == "__main__":
    unittest.main()
