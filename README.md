# Argumentation Schemes for Clinical Reasoning

Clinical reasoning pipeline for diagnosis identification from MIMIC-IV discharge notes.

Supporting documentation:

- [Local Data Sources](docs/data_sources.md)
- [Argumentation Framework Comparison](docs/framework_comparison.md)

Workflow:

1. download `MIMIC-IV-Note`
2. extract the full discharge-note corpus into local `.txt` evidence files
3. build the index
4. run retrieval and reasoning

## Repository Layout

- `ingest/`: normalized loading for MIMIC-IV notes.
- `retrieval/`: concept normalization, LlamaIndex indexing, query helpers, and graph visualization.
- `argumentation/`: agents, schemes, critical questions, and AAF semantics.
- `eval/`: MedQA checks, metrics, rubrics, and record-pulling tools.
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
INPUT_BASE_DIR=data/evidence/mimic_discharge_full
OUTPUT_BASE_DIR=output
```

The required dataset is `MIMIC-IV-Note`. The repo can read either a plain CSV or a gzipped CSV. It checks common local paths automatically, including `data/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv.gz`.

## Download MIMIC-IV-Note

PhysioNet page: `https://physionet.org/content/mimic-iv-note/2.2/`

Download the `discharge` table and place it under `data/`. The repo can read either a plain CSV or the extracted PhysioNet gzip layout. For example:

```text
data/mimic_iv_note/discharge.csv
```

or:

```text
data/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv.gz
```

Default UMLS settings:

```bash
UMLS_ENABLED=true
UMLS_BACKEND=local
UMLS_LOCAL_DB_PATH=output/cache/umls_local.sqlite3
UMLS_GUIDANCE_NUM_WORKERS=1
```

Build the local SQLite lookup before indexing:

```bash
export UMLS_API_KEY=<your-umls-api-key>
python scripts/download_umls.py
python -m retrieval.concepts.local_umls build --meta-dir data/umls
```

`python scripts/download_umls.py` uses the UTS Release API to resolve the current UMLS archive, downloads it through the UTS Download API, and extracts it under `data/umls`. Set `UMLS_API_KEY` in your environment first.
The builder will auto-discover a nested `META` directory under the path you pass, so `data/umls`, `data/umls/META`, and versioned layouts such as `data/umls/2026AA/META` all work.

If you prefer the remote UMLS API instead, set:

```bash
UMLS_BACKEND=api
UMLS_API_KEY=<your-umls-api-key>
```

This repository is Ollama-only. If command-line `tqdm` bars do not appear during index builds, force them with:

```bash
SHOW_PROGRESS=true
```

## Quick Start

### 1. Extract Notes

Place the MIMIC-IV-Note discharge file in the repo, or keep the extracted PhysioNet folder as-is, then extract the full discharge-note corpus into plain text evidence files:

```bash
python make_index.py extract-mimic-discharge --limit all --max-chars all
```

For quicker iteration, start with a smaller subset:

```bash
python make_index.py extract-mimic-discharge --limit 1000 --max-chars all
```

### 2. Build The Index

```bash
python make_index.py build
```

The build path uses the extracted full-note corpus for retrieval and applies UMLS concept normalization during graph construction. Local UMLS lookup is the default because it is practical for large corpora.

### 3. Optional Schema-Guided Build

Use the schema-guided graph only if you specifically want that build mode:

```text
UMLS_ENABLED=true
UMLS_BACKEND=local
```

Command:

```bash
python make_index.py build-schema
```

`build` and `build-schema` are alternatives, not a sequence.

`build-schema` is substantially slower than `build`. On the full discharge-note corpus, start with a smaller extracted subset unless you already know you need the schema-guided graph.
Schema guidance now caches per-document UMLS results on disk, so restarting the same `build-schema` command reuses completed unchanged notes instead of recomputing phase 1 from zero.

Inspect graph outputs with the visualization helpers and notebooks, especially [`umls_schema_comparison.ipynb`](/Users/oluwatosinoso/Library/CloudStorage/OneDrive-hull.ac.uk/argumentation_schemes/umls_schema_comparison.ipynb:1). Main schema-build tuning knobs: `UMLS_HINT_LIMIT`, `UMLS_GUIDANCE_NUM_WORKERS`, `INDEX_LLM_REQUEST_TIMEOUT`, `INDEX_SCHEMA_NUM_WORKERS`, `INDEX_SCHEMA_MAX_TRIPLETS_PER_CHUNK`.

### Run Retrieval Workflows

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
python main.py serve-api --reload
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
