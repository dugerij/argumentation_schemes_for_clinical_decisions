from unittest.mock import patch

import pytest

from clinical_cds.model import OllamaDiagnosticModel, is_loopback_endpoint


def test_loopback_endpoint_detection():
    assert is_loopback_endpoint("http://127.0.0.1:11434")
    assert is_loopback_endpoint("http://localhost:11434")
    assert not is_loopback_endpoint("https://models.example.org")


def test_direct_rejects_cloud_named_model():
    model = OllamaDiagnosticModel(model_name="example:cloud")

    with patch(
        "clinical_cds.model.ollama_llm_endpoint",
        return_value="http://127.0.0.1:11434",
    ), pytest.raises(ValueError, match="local model"):
        model.assert_data_boundary("direct")


def test_dataset_outside_the_protected_set_allows_remote_model_boundary():
    # Only "direct" and "submitted_patient" carry patient data today and are
    # restricted to a local model; any other dataset label is unaffected.
    model = OllamaDiagnosticModel(model_name="example:cloud")
    model.assert_data_boundary("unrestricted_dataset")
