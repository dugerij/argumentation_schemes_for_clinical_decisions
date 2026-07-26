import unittest
from unittest.mock import patch

from clinical_cds.ollama import (
    DEFAULT_OLLAMA_ENDPOINT,
    ollama_chat,
    ollama_headers,
    ollama_llm_endpoint,
)
from clinical_cds.model import OllamaDiagnosticModel


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

    @patch("clinical_cds.ollama.ollama_post")
    def test_ollama_chat_can_disable_hidden_reasoning(self, mock_post):
        mock_post.return_value = {"message": {"content": '{"answer": "Asthma"}'}}

        response = ollama_chat(
            model="local-model",
            messages=[{"role": "user", "content": "Return JSON."}],
            format={"type": "object"},
            think=False,
        )

        self.assertEqual(response, '{"answer": "Asthma"}')
        payload = mock_post.call_args.args[1]
        self.assertIs(payload["think"], False)

    def test_model_exposes_reproducible_decoding_configuration(self):
        model = OllamaDiagnosticModel(
            model_name="local-model",
            seed=23,
            context_window=4096,
            max_output_tokens=768,
        )

        self.assertEqual(
            model.decoding_config,
            {
                "temperature": 0,
                "seed": 23,
                "context_window": 4096,
                "max_output_tokens": 768,
                "think": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
