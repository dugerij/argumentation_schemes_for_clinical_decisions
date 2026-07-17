# Clinical Argumentation Pipeline

This repository builds a persistent graph from note-like clinical case data, retrieves graph evidence for a target case, and uses an argumentation loop to produce:

- a recommendation or answer
- an evidence bundle
- an argumentation trace

The active pipeline is:

1. Materialize a case graph from a supported dataset adapter.
2. Retrieve the target case plus similar precedent cases from the graph.
3. Pass the retrieved evidence into the generator, verifier, and reasoner agents.
4. Return a final answer with a saved trace in the framework logs.

The repository currently includes one adapter for `mimic_ext_cds`, which is useful as a compact demonstration dataset. The graph path is not restricted to that dataset in principle; the adapter is simply the first implemented input format.

## Supported Tasks

- `diagnosis`
- `triage`
- `specialty_referral`

## Input Model

The materializer expects a note-like case dataset where each case can be represented as:

- case identifier
- history or presenting complaint
- patient information
- vital signs
- tests
- medication history
- task labels for evaluation

For the current `mimic_ext_cds` adapter, those fields are read from the PhysioNet archive and merged by `stay_id`.

## Core Files

- [main.py](/Users/oluwatosinoso/Library/CloudStorage/OneDrive-hull.ac.uk/argumentation_schemes/main.py)
- [api/recommendation.py](/Users/oluwatosinoso/Library/CloudStorage/OneDrive-hull.ac.uk/argumentation_schemes/api/recommendation.py)
- [retrieval/cds_graph.py](/Users/oluwatosinoso/Library/CloudStorage/OneDrive-hull.ac.uk/argumentation_schemes/retrieval/cds_graph.py)
- [retrieval/cds_query.py](/Users/oluwatosinoso/Library/CloudStorage/OneDrive-hull.ac.uk/argumentation_schemes/retrieval/cds_query.py)
- [eval/case_eval.py](/Users/oluwatosinoso/Library/CloudStorage/OneDrive-hull.ac.uk/argumentation_schemes/eval/case_eval.py)
- [argumentation/agents.py](/Users/oluwatosinoso/Library/CloudStorage/OneDrive-hull.ac.uk/argumentation_schemes/argumentation/agents.py)

## Environment

The question-answering path requires local Ollama-backed models because the argumentation agents call:

- generator model
- verifier model
- reasoner model

Set:

- `OUTPUT_BASE_DIR`
- `GENERATOR_MODEL`
- `VERIFIER_MODEL`
- `REASONER_MODEL`

If those model variables are omitted, the existing defaults in the repo config are used.

## Build The Graph

Example using the CDS archive:

```bash
OUTPUT_BASE_DIR=output/mimic_ext_cds \
./venv/bin/python main.py materialize-graph \
  --dataset-format mimic_ext_cds \
  --input-path "data/mimic-iv-ext-clinical-decision-support-for-referral-triage-and-diagnosis-1.0.2 (1).zip"
```

This writes:

- `cds_case_graph.pkl.gz`
- `cds_case_graph_manifest.json`

into `OUTPUT_BASE_DIR`.

## Answer One Question

```bash
OUTPUT_BASE_DIR=output/mimic_ext_cds \
./venv/bin/python main.py answer-question \
  --task diagnosis \
  --case-id 30000153
```

Optional flags:

- `--question` to override the default task prompt
- `--clinical-goal` to append a goal line
- `--top-k-cases` to control precedent retrieval depth
- `--max-rounds` to control the argumentation loop
- `--dry-run` to inspect retrieval without model calls

The default task prompt asks for short exam-style reasoning and ends with:

```text
Answer: <label>
```

## Evaluate

```bash
OUTPUT_BASE_DIR=output/mimic_ext_cds \
./venv/bin/python main.py evaluate \
  --task diagnosis \
  --sample-size 5
```

This runs question answering over graph-backed cases and reports:

- expected answer
- predicted answer
- exact-match correctness
- aggregate accuracy

## API

Start the API:

```bash
OUTPUT_BASE_DIR=output/mimic_ext_cds \
./venv/bin/python main.py serve-api
```

Main endpoint:

- `POST /recommend`

Request body:

```json
{
  "case_id": 30000153,
  "task": "diagnosis",
  "max_rounds": 3,
  "top_k_cases": 5
}
```

Useful read endpoints:

- `GET /health`
- `GET /runs`
- `GET /events`
- `GET /eval-records`
- `GET /recommendations`

## Retrieval Design

The graph stores one merged case record per case id and a token-to-case index for fast candidate lookup.

For a target case, retrieval returns:

1. the target case evidence
2. similar precedent cases with observed labels for the chosen task

The target case does not include its own answer label in the retrieved evidence. The label appears only in precedent cases and in evaluation metadata.

## Argumentation Output

The argumentation layer produces:

- generator arguments
- verifier critiques
- reasoner summary

Run artifacts are written to:

- `output/logs/framework/events.jsonl`
- `output/logs/framework/eval_records.jsonl`

## Tests

Focused tests for the active path:

```bash
./venv/bin/pytest tests/test_main.py tests/test_recommendation.py tests/test_materialized_graph.py -q
```
