import unittest
from unittest.mock import patch

from clinical_cds.ollama import (
    DEFAULT_OLLAMA_ENDPOINT,
    ollama_chat,
    ollama_headers,
    ollama_llm_endpoint,
)
from clinical_cds.model import (
    OUTPUT_SCHEMA,
    OllamaDiagnosticModel,
    OpenAICompatibleDiagnosticModel,
)


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
                "top_p": 1,
                "top_k": -1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "seed": 23,
                "context_window": 4096,
                "max_output_tokens": 768,
                "think": False,
            },
        )

    @patch("clinical_cds.model.requests.post")
    def test_openai_compatible_model_uses_structured_output(self, mock_post):
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": '{"answer":"Asthma"}'}}]
        }
        model = OpenAICompatibleDiagnosticModel(
            model_name="local-medgemma",
            base_url="http://127.0.0.1:8000/v1",
        )

        response = model.complete(
            "System",
            "User",
            output_schema={"type": "object"},
        )

        self.assertEqual(response, '{"answer":"Asthma"}')
        mock_post.return_value.raise_for_status.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(
            payload["structured_outputs"],
            {"json": {"type": "object"}},
        )
        self.assertEqual(
            {key: payload[key] for key in (
                "temperature", "top_p", "top_k", "presence_penalty",
                "frequency_penalty", "n", "stream", "seed",
            )},
            {
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "n": 1,
                "stream": False,
                "seed": 17,
            },
        )

    @patch("clinical_cds.model.requests.post")
    def test_openai_compatible_model_uses_vllm_schema_subset(self, mock_post):
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": '{"values":["S1"]}'}}]
        }
        model = OpenAICompatibleDiagnosticModel(
            model_name="local-medgemma",
            base_url="http://127.0.0.1:8000/v1",
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "values": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"type": "string", "enum": ["S1", "S2"]},
                }
            },
            "required": ["values"],
        }

        model.complete("System", "User", output_schema=schema)

        transported = mock_post.call_args.kwargs["json"]["structured_outputs"]["json"]
        self.assertNotIn("uniqueItems", transported["properties"]["values"])
        self.assertIs(transported["additionalProperties"], False)
        self.assertEqual(
            transported["properties"]["values"]["items"]["enum"],
            ["S1", "S2"],
        )
        self.assertIs(schema["properties"]["values"]["uniqueItems"], True)

    def test_standard_output_schema_is_closed_at_every_object_boundary(self):
        self.assertIs(OUTPUT_SCHEMA["additionalProperties"], False)
        observation = OUTPUT_SCHEMA["properties"]["observations"]["items"]
        self.assertIs(observation["additionalProperties"], False)

    def test_openai_compatible_model_rejects_remote_direct_data(self):
        model = OpenAICompatibleDiagnosticModel(
            model_name="local-medgemma",
            base_url="https://example.invalid/v1",
        )

        with self.assertRaisesRegex(ValueError, "locally controlled"):
            model.assert_data_boundary("direct")


if __name__ == "__main__":
    unittest.main()
