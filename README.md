# Argumentation Schemes for Clinical Reasoning

Clinical reasoning pipeline for graph-backed recommendation experiments over MIMIC-IV discharge notes.

Supporting documentation:

- [Local Data Sources](docs/data_sources.md)
- [Argumentation Framework Comparison](docs/framework_comparison.md)

Workflow:

1. download `MIMIC-IV-Note`
2. extract or reuse a discharge-note `.txt` reservoir
3. build or resume the index
4. run retrieval, recommendation, and benchmarks

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

Commands below assume the repo-local interpreter `./venv/bin/python`. If you use another environment, replace it with `python`.

Main runtime paths:

```text
INPUT_BASE_DIR=data/evidence/mimic_discharge_subset
OUTPUT_BASE_DIR=output
```

The current repo-local extracted note reservoir is `data/evidence/mimic_discharge_subset`. For repeatable experiments, prefer explicit `--input-dir` and `--output-dir` values instead of relying on defaults.

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
./venv/bin/python scripts/download_umls.py
./venv/bin/python -m retrieval.concepts.local_umls build --meta-dir data/umls
```

`./venv/bin/python scripts/download_umls.py` uses the UTS Release API to resolve the current UMLS archive, downloads it through the UTS Download API, and extracts it under `data/umls`. Set `UMLS_API_KEY` in your environment first.
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

### 1. Reuse Or Extract The Source Note Reservoir

If `data/evidence/mimic_discharge_subset` already exists, you can skip this step and reuse it directly.

To build the extracted note reservoir from the MIMIC discharge CSV:

```bash
./venv/bin/python make_index.py extract-mimic-discharge \
  --input-dir data/evidence/mimic_discharge_subset \
  --limit all \
  --max-chars all
```

For quicker iteration, start with a smaller extracted source set:

```bash
./venv/bin/python make_index.py extract-mimic-discharge \
  --input-dir data/evidence/mimic_discharge_subset_small \
  --limit 1000 \
  --max-chars all
```

If you want to enforce a hard `4000`-character extraction cap in a fresh source directory, do it here:

```bash
./venv/bin/python make_index.py extract-mimic-discharge \
  --input-dir data/evidence/mimic_discharge_4000 \
  --limit all \
  --max-chars 4000
```

The `4000`-character cap only matters during extraction and other commands that re-read the source CSV. If you are indexing an already extracted `.txt` reservoir, that cap is not re-applied at index time.

Resume behavior:

- Extraction is rebuild-only in the current implementation.
- Rerunning `extract-mimic-discharge` starts again from the source CSV and rewrites the target note files.

### 2. Build Or Resume The Index

Use a dedicated output directory per build. Keep the directory if you want to resume. Delete it only when you intentionally want a clean restart.

Deterministic UMLS graph build:

```bash
UMLS_ENABLED=true \
INDEX_EMBED_KG_NODES=false \
INDEX_UMLS_CANDIDATE_LIMIT=120 \
INDEX_UMLS_MAX_CONCEPTS_PER_CHUNK=5 \
./venv/bin/python make_index.py build \
  --input-dir data/evidence/mimic_discharge_subset \
  --output-dir output/mimic_umls
```

Hybrid graph build with schema LLM enrichment enabled:

```bash
INDEX_SCHEMA_LLM_ENRICH=true \
UMLS_HINT_LIMIT=40 \
INDEX_SCHEMA_MAX_TRIPLETS_PER_CHUNK=4 \
./venv/bin/python make_index.py build-schema \
  --input-dir data/evidence/mimic_discharge_subset \
  --output-dir output/mimic_schema
```

The default build path now creates a hybrid clinical graph:

- note chunk nodes
- normalized clinical entity nodes from local UMLS plus rule-based extraction
- lab test and lab result nodes
- medication action edges such as `HELD`, `RESTARTED`, `CONTINUED`, and `ADMINISTERED`
- `MENTIONS`, `HAS_RESULT`, `TREATS`, and `CONTRAINDICATES` edges where supported by the chunk text

`build` uses deterministic extraction only. `build-schema` uses the same deterministic base graph and adds schema-guided LLM relation extraction only when `INDEX_SCHEMA_LLM_ENRICH=true`.

Resume behavior:

- `build` and `build-schema` resume when you rerun them with the same `--input-dir` and `--output-dir` and keep the existing output directory.
- Completed batches are persisted before they are marked done, so reruns continue from the last durable completed batch.
- If you change graph-shaping settings such as `INDEX_SCHEMA_LLM_ENRICH`, `INDEX_EMBED_KG_NODES`, `UMLS_HINT_LIMIT`, or `INDEX_SCHEMA_MAX_TRIPLETS_PER_CHUNK`, delete the old output directory and rebuild to avoid a mixed index.

If you need a clean deterministic rebuild, clear only the persisted index artifacts and keep the UMLS cache:

```bash
rm -f output/mimic_umls/default__vector_store.json \
  output/mimic_umls/docstore.json \
  output/mimic_umls/graph_store.json \
  output/mimic_umls/image__vector_store.json \
  output/mimic_umls/index_checkpoints.sqlite \
  output/mimic_umls/index_store.json \
  output/mimic_umls/property_graph_store.json
```

Keep:

- `output/cache/umls_local.sqlite3`
- `output/schema_guidance.sqlite` if you may want to switch back to a schema-guided build later

Inspect graph outputs with the visualization helpers and notebooks, especially [`umls_schema_comparison.ipynb`](/Users/oluwatosinoso/Library/CloudStorage/OneDrive-hull.ac.uk/argumentation_schemes/umls_schema_comparison.ipynb:1). Main indexing tuning knobs: `INDEX_EMBED_KG_NODES`, `INDEX_HYBRID_CANDIDATE_LIMIT`, `INDEX_SCHEMA_LLM_ENRICH`, `UMLS_HINT_LIMIT`, `UMLS_GUIDANCE_NUM_WORKERS`, `INDEX_LLM_REQUEST_TIMEOUT`, `INDEX_EMBED_BATCH_SIZE`, `INDEX_SCHEMA_MAX_TRIPLETS_PER_CHUNK`, `INDEX_UMLS_CANDIDATE_LIMIT`, `INDEX_UMLS_MAX_CONCEPTS_PER_CHUNK`.

### 3. Run Retrieval And Reasoning

Single dry recommendation run:

```bash
INPUT_BASE_DIR=data/evidence/mimic_discharge_subset \
OUTPUT_BASE_DIR=output/mimic_umls \
./venv/bin/python main.py recommend \
  --scenario "A patient with chronic kidney disease and hypertension needs blood pressure management." \
  --dry-run
```

Embedding benchmark:

```bash
./venv/bin/python main.py benchmark-embeddings \
  --generation-model gemma4 \
  --embedding-model qwen3-embedding:0.6b \
  --embedding-model embeddinggemma:latest \
  --embedding-model all-minilm:latest \
  --sample-size 5
```

Generation benchmark:

```bash
./venv/bin/python main.py benchmark-models \
  --generation-model gemma4 \
  --generation-model qwen3.5:9b \
  --generation-model medgemma1.5 \
  --embedding-model qwen3-embedding:0.6b \
  --sample-size 5
```

API entrypoint:

```bash
./venv/bin/python main.py serve-api --reload
```

This binds to `127.0.0.1:8000` by default. Use `--host 0.0.0.0` only if you intentionally want LAN access.

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

## Extraction Notes

- The current graph path is hybrid: deterministic clinical extraction first, local UMLS normalization second, optional schema-guided LLM relation enrichment last.
- UMLS hints are now stored as metadata for the schema LLM path instead of being prepended into source text, which keeps chunk embeddings and retrieval text cleaner.
- KG node embedding is disabled by default to reduce indexing cost when the graph contains many extracted entities and results. Enable it only when you explicitly want vector search over KG nodes.

## Logs and Records

LlamaIndex writes its own internal index files under the configured output directory. The framework writes structured JSONL logs under:

- `output/logs/framework/events.jsonl`: step-level events, durations, failures, and previews.
- `output/logs/framework/eval_records.jsonl`: one pullable record per evaluated question.

Inspect records from the command line with:

```bash
./venv/bin/python -m eval.pull_records --run-id medqa_smoke
./venv/bin/python -m eval.pull_records --limit 5
```

## API

Start the local read-only API with:

```bash
./venv/bin/python main.py serve-api --reload
```

This uses `127.0.0.1:8000` by default. For LAN exposure, pass `--host 0.0.0.0`.

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
    "scenario": "A patient with chronic kidney disease and hypertension needs blood pressure management."
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
