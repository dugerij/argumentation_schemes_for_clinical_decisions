import json

import pytest


@pytest.fixture
def direct_root(tmp_path):
    root = tmp_path / "direct"
    graph_dir = root / "Diagnosis_flowchart"
    sample_dir = root / "Finished" / "Hypertension"
    graph_dir.mkdir(parents=True)
    sample_dir.mkdir(parents=True)

    graph = {
        "diagnostic": {
            "Suspected Hypertension": {
                "Hypertension": [],
            }
        },
        "knowledge": {
            "Suspected Hypertension": {
                "Symptoms": "Headache; Dizziness",
                "Risk Factors": "Older age; Family history",
            },
            "Hypertension": "Repeated blood pressure at or above 140/90 mmHg",
        },
    }
    (graph_dir / "Hypertension.json").write_text(
        json.dumps(graph),
        encoding="utf-8",
    )

    case = {
        "input1": "Headache",
        "input2": "Persistent headache for two weeks.",
        "input3": "No prior diagnosis.",
        "input4": "A parent had hypertension.",
        "input5": "Blood pressure is 170/100 mmHg.",
        "input6": "Repeat blood pressure remains elevated.",
        "Hypertension$Intermedia_1": {
            "Repeated elevation supports hypertension.$Cause_1": {
                "Blood pressure is 170/100 mmHg.$Input5": {},
            },
            "Symptoms support suspected hypertension.$Cause_2": {
                "Headache$Input1": {},
            },
        },
    }
    (sample_dir / "case-one.json").write_text(
        json.dumps(case),
        encoding="utf-8",
    )
    return root
