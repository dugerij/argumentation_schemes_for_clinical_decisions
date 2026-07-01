import unittest
from unittest.mock import Mock, patch

from requests import HTTPError, Response

from retrieval.concepts.umls import UMLSClient, UMLSConfig, parse_source_vocabularies


def make_response(status_code: int, payload: dict | None = None) -> Response:
    response = Response()
    response.status_code = status_code
    response._content = b'{"result": {"results": []}}'
    if payload is not None:
        import json

        response._content = json.dumps(payload).encode("utf-8")
    response.url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    return response


class UMLSConfigTests(unittest.TestCase):
    def test_parse_source_vocabularies_strips_inline_comments(self):
        self.assertEqual(
            parse_source_vocabularies("ICD10CM # diagnosis,SNOMEDCT_US,RXNORM"),
            ("ICD10CM", "SNOMEDCT_US", "RXNORM"),
        )


class UMLSClientTests(unittest.TestCase):
    def test_retryable_http_error_returns_empty_matches_after_retries(self):
        client = UMLSClient(
            UMLSConfig(
                api_key="test-key",
                source_vocabularies=("ICD10CM",),
                max_retries=1,
                retry_backoff_seconds=0,
            )
        )

        with patch("retrieval.concepts.umls.requests.get", return_value=make_response(502)) as get:
            self.assertEqual(client.search("DS-11 subject id"), [])

        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args.kwargs["params"]["sabs"], "ICD10CM")

    def test_non_retryable_http_error_is_raised(self):
        client = UMLSClient(UMLSConfig(api_key="test-key", max_retries=1, retry_backoff_seconds=0))

        with patch("retrieval.concepts.umls.requests.get", return_value=make_response(401)):
            with self.assertRaises(HTTPError):
                client.search("hypertension")

    def test_success_after_retry_returns_matches(self):
        client = UMLSClient(
            UMLSConfig(
                api_key="test-key",
                source_vocabularies=("SNOMEDCT_US",),
                max_retries=1,
                retry_backoff_seconds=0,
            )
        )
        payload = {
            "result": {
                "results": [
                    {
                        "ui": "C0020538",
                        "name": "Hypertensive disorder",
                        "rootSource": "SNOMEDCT_US",
                        "semanticType": "Disease or Syndrome",
                    }
                ]
            }
        }

        with patch(
            "retrieval.concepts.umls.requests.get",
            Mock(side_effect=[make_response(502), make_response(200, payload)]),
        ):
            matches = client.search("hypertension")

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].cui, "C0020538")


if __name__ == "__main__":
    unittest.main()
