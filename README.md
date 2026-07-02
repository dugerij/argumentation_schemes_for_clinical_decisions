# Argumentation Schemes for Clinical Reasoning

Clinical reasoning pipeline built around UMLS-guided knowledge graphs, evidence-grounded retrieval, structured argumentation, and Abstract Argumentation Framework adjudication.

Supporting documentation:

- [Local Data Sources](docs/data_sources.md)
- [Argumentation Framework Comparison](docs/framework_comparison.md)

Target data:

- MIMIC-IV notes after approval
- UMLS/MEDCIN-normalized clinical concepts

The included MedQA files are lightweight pipeline-check inputs.

## Overview

Main stages:

1. normalize clinical entities with UMLS/MEDCIN
2. build a LlamaIndex property graph from evidence documents
3. retrieve grounded passages for downstream reasoning
4. instantiate argumentation schemes and attacks
5. resolve conflicts under grounded AAF semantics

Default clinical vocabulary priority:

```text
ICD10CM, SNOMEDCT_US, RXNORM, ATC, CPT, HCPCS, LNC, MEDCIN, MSH
```

## Repository Layout

- `ingest/`: normalized loading for MIMIC-IV notes.
- `retrieval/`: concept normalization, LlamaIndex indexing, query helpers, and graph visualization.
- `argumentation/`: agents, schemes, critical questions, and AAF semantics.
- `eval/`: MedQA checks, MIMIC-IV-Ext-ITR evaluation code, metrics, rubrics, and record-pulling tools.
- `helpers/`: shared environment validation, JSONL logging, and record utilities.
- `api/`: local read-only API for events, evaluation records, and recommendation traces.

## Setup

Use a standard Python environment:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Main runtime paths:

```text
INPUT_BASE_DIR=data/evidence/mimic_discharge_subset
OUTPUT_BASE_DIR=output
```

This repository uses credentialed MIMIC data. The main expected sources are:

- `MIMIC-IV-Note`: request access through PhysioNet, complete the required credentialed-access course/training, and download the note release after approval. This repo expects the discharge-note CSV at `data/mimic_iv_note/discharge.csv`.
- `MIMIC-IV-Ext Clinical Decision Support for Referral, Triage, and Diagnosis`: request the dataset from PhysioNet under credentialed access, complete the required course/training, and download the approved release. This repo defaults to `data/mimic-iv-ext-clinical-decision-support-for-referral-triage-and-diagnosis-1.0.2/` and expects files such as `initial_assessment_info.csv` and `clinical_data.csv.zip` inside that directory.

Baseline UMLS settings:

```bash
UMLS_ENABLED=true
UMLS_API_KEY=<your-umls-api-key>
```

This repository is Ollama-only. If command-line `tqdm` bars do not appear during index builds, force them with:

```bash
SHOW_PROGRESS=true
```

## Quick Start

### Prepare Evidence

```bash
python make_index.py extract-mimic-discharge --csv-path data/mimic_iv_note/discharge.csv --limit 25 --max-chars 6000
```

or:

```bash
python make_index.py extract-mimic-ext-cardiovascular --dataset-dir data/mimic-iv-ext-clinical-decision-support-for-referral-triage-and-diagnosis-1.0.2 --limit 100 --max-chars 4000
```

Create a matched domain subset of discharge notes and MedQA questions:

```bash
python make_domain_subset.py \
  --domain renal_metabolic \
  --notes-csv-path data/mimic_iv_note/discharge.csv \
  --notes-output-dir data/evidence/renal_metabolic_discharge_subset \
  --questions-output-path data/eval/renal_metabolic_medqa.jsonl \
  --note-limit all \
  --question-limit all
```

This command uses a fast keyword prefilter plus UMLS confirmation by default and requires `UMLS_API_KEY`. It writes all matching notes, all matching questions, and selection metadata for the chosen clinical domain. Use `--matcher umls` for UMLS-only matching or `--matcher keyword` for term-based matching.

### Build the Knowledge Graph

UMLS-first graph:

```text
UMLS_ENABLED=true
```

Command:

```bash
python make_index.py build
```

UMLS + schema-guided graph:

```text
UMLS_ENABLED=true
```

Command:

```bash
python make_index.py build-schema
```

Use `build` for the UMLS-first graph and `build-schema` for the UMLS + schema-guided graph. Use `INDEX_SCHEMA_GUIDED` only when other entrypoints should default to schema-guided behavior.

Inspect graph outputs with the visualization helpers and notebooks, especially [`umls_schema_comparison.ipynb`](/Users/oluwatosinoso/Library/CloudStorage/OneDrive-hull.ac.uk/argumentation_schemes/umls_schema_comparison.ipynb:1). Main schema-build tuning knobs: `INDEX_LLM_REQUEST_TIMEOUT`, `INDEX_SCHEMA_NUM_WORKERS`, `INDEX_SCHEMA_MAX_TRIPLETS_PER_CHUNK`.

### Run Retrieval Workflows

To run retrieval or benchmarks on a domain subset, point `INPUT_BASE_DIR` at the subset folder and use the matching question file where applicable.

Pipeline check:

```bash
python main.py pipeline-check \
  --scenario "A patient with chronic kidney disease and hypertension needs blood pressure management." \
  --clinical-goal "reduce blood pressure while avoiding renal harm" \
  --dry-run
```

Embedding benchmark:

```bash
python main.py benchmark-embeddings \
  --generation-model gemma4 \
  --embedding-model qwen3-embedding:0.6b \
  --embedding-model embeddinggemma:latest \
  --embedding-model all-minilm:latest \
  --sample-size 5
```

Generation benchmark:

```bash
python main.py benchmark-models \
  --generation-model gemma4 \
  --generation-model qwen3.5:9b \
  --generation-model medgemma1.5 \
  --embedding-model qwen3-embedding:0.6b \
  --sample-size 5
```

API entrypoint:

```bash
python main.py serve-api --reload
```

### Introduce Argumentation Frameworks

The graph is the evidence layer. The argumentation layer turns that evidence into explicit support and attack structures:

1. retrieve grounded evidence
2. generate candidate recommendations and objections
3. instantiate schemes and critical questions
4. build `AAF = <Ar, R>`
5. resolve attacks under grounded semantics

Main code:

- `argumentation/agents.py`
- `argumentation/schemes.py`
- `argumentation/critical_questions.py`
- `argumentation/aaf.py`

Background notes: [docs/framework_comparison.md](/Users/oluwatosinoso/Library/CloudStorage/OneDrive-hull.ac.uk/argumentation_schemes/docs/framework_comparison.md:1)

## Logs and Records

LlamaIndex writes its own internal index files under the configured output directory. The framework writes structured JSONL logs under:

- `output/logs/framework/events.jsonl`: step-level events, durations, failures, and previews.
- `output/logs/framework/eval_records.jsonl`: one pullable record per evaluated question.

Inspect records from the command line with:

```bash
python -m eval.pull_records --run-id medqa_smoke
python -m eval.pull_records --limit 5
```

## API

Start the local read-only API with:

```bash
uvicorn api.app:app --reload
```

Useful endpoints:

- `GET /health`
- `POST /recommend`
- `GET /runs`
- `GET /events`
- `GET /events?run_id=medqa_smoke`
- `GET /eval-records`
- `GET /eval-records?run_id=medqa_smoke`
- `GET /recommendations`

FastAPI docs:

```text
http://127.0.0.1:8000/docs
```

Sample request:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": "A patient with chronic kidney disease and hypertension needs blood pressure management.",
    "clinical_goal": "reduce blood pressure while avoiding renal harm",
    "patient_id": "example-patient",
    "max_rounds": 3
  }'
```

For request logging without model calls:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"scenario": "Test clinical scenario for API logging.", "dry_run": true}'
```

## Notebook Guidance

Main notebooks:

- `ollama_connection_model_access_test.ipynb`: reachability plus minimal per-model access checks against a remote Ollama host.
- `ollama_test.ipynb`: full-flow Ollama-backed property-graph test across generation and embedding combinations.
- `ollama_benchmarks.ipynb`: combined embedding benchmark plus generation benchmark, reusing the shared embedding index cache between sections.
- `umls_schema_comparison.ipynb`: comparison notebook for UMLS-first versus schema-guided graph construction and retrieval behavior.
- `pipeline_check.ipynb`: end-to-end index and recommendation workflow check.
