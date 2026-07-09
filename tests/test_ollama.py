import unittest
from unittest.mock import patch

from helpers.ollama import DEFAULT_OLLAMA_ENDPOINT, ollama_headers, ollama_llm_endpoint


class OllamaConfigTests(unittest.TestCase):
    def test_ollama_headers_empty_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(ollama_headers(), {})

    def test_ollama_headers_use_bearer_authorization_by_default(self):
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "secret-token"}, clear=True):
            self.assertEqual(ollama_headers(), {"Authorization": "Bearer secret-token"})

    def test_ollama_headers_support_custom_header_and_no_scheme(self):
        with patch.dict(
            "os.environ",
            {
                "OLLAMA_API_KEY": "secret-token",
                "OLLAMA_AUTH_HEADER": "X-API-Key",
                "OLLAMA_AUTH_SCHEME": "",
            },
            clear=True,
        ):
            self.assertEqual(ollama_headers(), {"X-API-Key": "secret-token"})

    def test_ollama_llm_endpoint_defaults_to_localhost(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(ollama_llm_endpoint(), DEFAULT_OLLAMA_ENDPOINT)


if __name__ == "__main__":
    unittest.main()
