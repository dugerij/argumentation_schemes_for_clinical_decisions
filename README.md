# Argumentation Schemes for Clinical Reasoning

This repository builds an auditable clinical reasoning pipeline that combines evidence-grounded retrieval, structured medical argumentation, and formal Abstract Argumentation Framework adjudication.

The long-term target corpus is:

- medical textbooks
- NICE/WHO guidelines
- MIMIC-IV notes after approval
- UMLS/MEDCIN-normalized clinical concepts

The current MedQA/textbook files are smoke-test inputs only. They are useful for checking that the pipeline runs, but they are not the final evaluation corpus.

## Pipeline

1. **Evidence Graph Construction**
   GraphRAG extracts clinically relevant entities and relationships from verified evidence sources. The goal is to ground reasoning in inspectable evidence rather than statistical associations.

2. **Clinical Concept Normalization**
   UMLS/MEDCIN maps mentions to clinical concepts. The default vocabulary priority covers diagnoses, medications, procedures, therapies, labs, and broad clinical findings:

   ```text
   ICD10CM, SNOMEDCT_US, RXNORM, ATC, CPT, HCPCS, LNC, MEDCIN, MSH
   ```

3. **Structured Argument Generation**
   Multi-agent LLM components generate candidate clinical recommendations and instantiate argumentation schemes such as goal-oriented practical reasoning, contraindication checks, adverse-effect checks, and history/failure checks.

4. **Formal Adjudication**
   Arguments and conflicts are represented as `AAF = <Ar, R>`, then resolved with grounded semantics.

5. **Evaluation and Auditability**
   Future evaluation will use MIMIC-IV and MIMIC-IV-Ext-ITR open-ended questions, measuring clinical correctness, grounding, safety, reasoning quality, and auditability.

## Repository Layout

- `ingest/`: normalized loading for textbooks, guidelines, and future MIMIC-IV notes.
- `entity_extraction/`: UMLS/MEDCIN schemas, vocabulary priorities, lookup client, and extraction helpers.
- `rag/`: GraphRAG indexing and retrieval helpers.
- `argumentation/`: agents, schemes, critical questions, and AAF semantics.
- `eval/`: MedQA smoke test, MIMIC-IV-Ext-ITR placeholder, metrics, rubrics, and record-pulling tools.
- `helpers/`: shared environment validation, JSONL logging, and record utilities.
- `api/`: local read-only API for events, evaluation records, and recommendation traces.
- `prompts/`: GraphRAG prompts customized for clinical evidence extraction.

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

- LLM providers
- model names
- provider credentials
- GraphRAG storage/indexing
- smoke/evaluation controls
- UMLS configuration

For local Ollama, provider keys can remain blank. For UMLS lookup:

```bash
UMLS_ENABLED=true
UMLS_API_KEY=<your-umls-api-key>
```

## Common Commands

Run the current smoke test:

```bash
python main.py
```

Build a GraphRAG index explicitly:

```bash
python make_index.py build
```

Generate clinically tuned GraphRAG prompts:

```bash
python make_index.py prompt-tune-clinical
```

Look up UMLS concepts:

```bash
python -m entity_extraction.lookup "heart failure" metformin "renal replacement therapy"
```

Run the offline vocabulary smoke test:

```bash
python -m entity_extraction.smoke_test
```

## Logs and Records

GraphRAG writes its own logs under `logs/`. The framework writes structured JSONL logs under:

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

## Notes

- Do not treat MedQA exact match as the final evaluation metric. It is only a smoke-test signal.
- Do not index all MIMIC-IV notes blindly. Filter and normalize first, then build evidence and patient-specific retrieval layers deliberately.
- Keep UMLS/MEDCIN as a preprocessing and normalization layer. GraphRAG should consume enriched text/metadata rather than replacing clinical concept extraction.
