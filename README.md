# Argumentation Schemes for Clinical Reasoning

This repository builds an auditable clinical reasoning pipeline that combines evidence-grounded retrieval, structured medical argumentation, and formal Abstract Argumentation Framework adjudication.

Supporting documentation:

- [Local Data Sources](docs/data_sources.md)
- [Argumentation Framework Comparison](docs/framework_comparison.md)

The target corpus is:

- MIMIC-IV notes after approval
- UMLS/MEDCIN-normalized clinical concepts

The current MedQA files are smoke-test inputs only. They are useful for checking that the pipeline runs, but they are not the final evaluation corpus.

## Pipeline

1. **UMLS Clinical Normalization**
   UMLS/MEDCIN is the first step for all medical indexing. It identifies clinical entities, filters text toward medically relevant mentions, and seeds relationship hints before graph construction. The default vocabulary priority covers diagnoses, medications, procedures, therapies, labs, and broad clinical findings:

   ```text
   ICD10CM, SNOMEDCT_US, RXNORM, ATC, CPT, HCPCS, LNC, MEDCIN, MSH
   ```

2. **Evidence Graph Construction**
   LlamaIndex builds a property graph from UMLS-annotated evidence sources and retrieves supporting passages for downstream reasoning. The goal is to ground reasoning in inspectable evidence rather than statistical associations.

3. **Structured Argument Generation**
   Multi-agent LLM components generate candidate clinical recommendations and instantiate argumentation schemes such as goal-oriented practical reasoning, contraindication checks, adverse-effect checks, and history/failure checks.

4. **Formal Adjudication**
   Arguments and conflicts are represented as `AAF = <Ar, R>`, then resolved with grounded semantics.

5. **Evaluation and Auditability**
   Future evaluation will use MIMIC-IV and MIMIC-IV-Ext-ITR open-ended questions, measuring clinical correctness, grounding, safety, reasoning quality, and auditability.

## Repository Layout

- `ingest/`: normalized loading for MIMIC-IV notes.
- `entity_extraction/`: UMLS/MEDCIN schemas, vocabulary priorities, lookup client, and extraction helpers.
- `rag/`: LlamaIndex indexing and retrieval helpers.
- `argumentation/`: agents, schemes, critical questions, and AAF semantics.
- `eval/`: MedQA smoke test, MIMIC-IV-Ext-ITR placeholder, metrics, rubrics, and record-pulling tools.
- `helpers/`: shared environment validation, JSONL logging, and record utilities.
- `api/`: local read-only API for events, evaluation records, and recommendation traces.
- `prompts/`: prompt templates kept for reference and future extractor experiments.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy and edit the environment template:

```bash
cp .env.example .env
```

The cleaned env file is organized into:

- Ollama model names and endpoint
- LlamaIndex storage/indexing
- smoke/evaluation controls
- UMLS configuration

By default, indexing points to an extracted MIMIC discharge-note subset:

```text
INPUT_BASE_DIR=data/evidence/mimic_discharge_subset
OUTPUT_BASE_DIR=output
```

Place extracted MIMIC note `.txt` files in that input folder. The folder is ignored by git because it may contain licensed clinical material.

Create the note subset explicitly before building the index:

```bash
python make_index.py extract-mimic-discharge --csv-path data/mimic_iv_note/discharge.csv --limit 25 --max-chars 6000
```

For medical indexing, keep `UMLS_ENABLED=true`. That makes the build path run the UMLS prepass before graph construction, even when schema-guided extraction is off.

If you want the index build to use a different model from the argumentation agents, set `INDEX_LLM_MODEL` and `INDEX_EMBEDDING_MODEL`.

The repo is now Ollama-only. The current experiment sweeps are:

```text
generation_model_sweep = ["gemma4", "qwen3.5:9b", "medgemma1.5"]
embedding_model_sweep = ["qwen3-embedding:0.6b", "embeddinggemma:latest", "all-minilm:latest"]
```

For UMLS lookup:

```bash
UMLS_ENABLED=true
UMLS_API_KEY=<your-umls-api-key>
```

## Common Commands

Run the current smoke test:

```bash
python main.py
```

Run the explicit smoke-eval subcommand:

```bash
python main.py smoke-eval
```

Build the LlamaIndex property graph explicitly:

```bash
python make_index.py build
```

Build the schema-guided graph with UMLS-based entity normalization and relation hints:

```bash
python make_index.py build-schema
```

Set `INDEX_SCHEMA_GUIDED=true` to make the smoke test and API/index helpers use the same mode by default.
For Ollama-backed builds, `INDEX_LLM_REQUEST_TIMEOUT`, `INDEX_SCHEMA_NUM_WORKERS`, and `INDEX_SCHEMA_MAX_TRIPLETS_PER_CHUNK` control how aggressively the schema extractor calls the model. If you see `ReadTimeout` on large chapters, raise the timeout first and keep workers low.

If you want the safer baseline for notebook work or large note subsets, keep schema-guided extraction off and rely on the UMLS-first graph build:

```text
UMLS_ENABLED=true
INDEX_SCHEMA_GUIDED=false
```

That still focuses the graph on medical entities and relationships, but avoids the slowest extractor path.

Extract a small MIMIC discharge subset into the input folder:

```bash
python make_index.py extract-mimic-discharge --csv-path data/mimic_iv_note/discharge.csv --limit 25 --max-chars 6000
```

Look up UMLS concepts:

```bash
python -m entity_extraction.lookup "heart failure" metformin "renal replacement therapy"
```

Run the offline vocabulary smoke test:

```bash
python -m entity_extraction.smoke_test
```

Run the end-to-end pipeline check:

```bash
python main.py pipeline-check \
  --scenario "A patient with chronic kidney disease and hypertension needs blood pressure management." \
  --clinical-goal "reduce blood pressure while avoiding renal harm" \
  --dry-run
```

Benchmark embedding models against the same note subset and question sample:

```bash
python main.py benchmark-embeddings \
  --generation-model gemma4 \
  --embedding-model qwen3-embedding:0.6b \
  --embedding-model embeddinggemma:latest \
  --embedding-model all-minilm:latest \
  --sample-size 5
```

Benchmark generation models while holding the embedding fixed:

```bash
python main.py benchmark-models \
  --generation-model gemma4 \
  --generation-model qwen3.5:9b \
  --generation-model medgemma1.5 \
  --embedding-model qwen3-embedding:0.6b \
  --sample-size 5
```

Start the local API through the unified entrypoint:

```bash
python main.py serve-api --reload
```

## Logs and Records

LlamaIndex writes its own internal index files under the configured output directory. The framework writes structured JSONL logs under:

- `logs/framework/events.jsonl`: step-level events, durations, failures, and previews.
- `logs/framework/eval_records.jsonl`: one pullable record per evaluated question.

Pull records from the command line:

```bash
python -m eval.pull_records --run-id medqa_smoke
python -m eval.pull_records --limit 5
```

These records are intended to support later analysis of what worked, what failed, and why.

## Results API

Start the local read-only API:

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

FastAPI docs are available at:

```text
http://127.0.0.1:8000/docs
```

Submit a clinical scenario:

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

Use `dry_run` to test request logging without model calls:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"scenario": "Test clinical scenario for API logging.", "dry_run": true}'
```

The API writes recommendation traces into the same JSONL record structure used by evaluation. Historical recommendation records can be inspected through `/recommendations`.

## Notebook Guidance

Use these notebooks for model selection and integration checks:

- `ollama_test.ipynb`: full generation x embedding sweep on the Ollama-backed property-graph workflow.
- `ollama_embedding_benchmark.ipynb`: fixed generation model, compare embedding models.
- `ollama_model_benchmark.ipynb`: fixed embedding model, compare generation models.
- `ollama_connection_model_access_test.ipynb`: reachability plus minimal per-model access checks against a remote Ollama host.
- `pipeline_check.ipynb`: end-to-end index plus recommendation smoke test.

## Notes

- Do not treat MedQA exact match as the final evaluation metric. It is only a smoke-test signal.
- Do not index all MIMIC-IV notes blindly. Filter and normalize first, then build evidence and patient-specific retrieval layers deliberately.
- Keep UMLS/MEDCIN as a preprocessing and normalization layer. LlamaIndex should consume enriched text/metadata rather than replacing clinical concept extraction.
