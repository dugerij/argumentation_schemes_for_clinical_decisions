# Diagnostic Graph Argumentation

This repository evaluates how guideline retrieval, diagnostic graph structure,
structured LLM-agent arguments, and symbolic argument resolution affect
diagnostic recommendation performance. The study compares five controlled
conditions using the same cases, decoding settings, and evidence inventory.

## Study Design

| Condition | Guideline evidence | Graph topology | Structured agents | Symbolic resolution |
| --- | --- | --- | --- | --- |
| `direct` |  |  |  |  |
| `flat_rag` | Yes |  |  |  |
| `graph_rag` | Yes | Yes |  |  |
| `structured_argument` | Yes | Yes | Yes |  |
| `symbolic_argument` | Yes | Yes | Yes | Yes |

Every condition receives the same submitted clinical state.
`flat_rag`, `graph_rag`, `structured_argument`, and `symbolic_argument` receive
the same retrieved guideline premises. The graph conditions additionally
receive diagnostic paths and typed support edges.

The two argument conditions share exactly one reasoner-agent call and one
verifier-agent call per case. `structured_argument` retains the reasoner's
preferred diagnosis. `symbolic_argument` applies deterministic resolution to
the same generated and verified argument graph. This isolates symbolic
resolution without giving that condition an additional LLM call.

Retrieval uses two-stage BM25 ranking. It first ranks leaf-diagnosis routes
using all premises on each route, then selects the strongest premises from the
highest-ranked routes. Case labels remain isolated from retrieval and prompt
construction.

The implementation is contained in `clinical_cds/`.

## Argumentation Method

The static DiReCT diagnostic graph and the patient-specific argument graph have
separate roles. The diagnostic graph supplies guideline premises and diagnostic
paths. The argument graph is constructed for each submitted patient state.

The argumentation pipeline is:

1. The reasoner agent proposes up to three candidate diagnoses with at most two
   strong, non-duplicate evidence arguments per diagnosis.
2. The verifier agent evaluates every argument using scheme-specific critical
   questions and may introduce rebuttals or undercutters.
3. A deterministic graph builder validates evidence identifiers and confirms
   that every cited knowledge fact lies on a diagnostic path containing the
   argument conclusion. It then constructs support and attack relations and
   adds one abductive `argument_from_best_explanation` node per candidate
   diagnosis.
4. The symbolic resolver applies grounded attack labelling and explicit scheme
   priorities. A diagnosis is returned only when it is the unique undefeated
   best explanation with accepted clinical-sign or diagnostic-criterion
   support; otherwise the resolver abstains.

The reasoner may instantiate:

- `argument_from_clinical_sign`
- `argument_from_diagnostic_criterion`
- `argument_from_risk_factor`
- `argument_from_guideline_authority`

Risk factors and guideline authority can strengthen an explanation but cannot
establish a diagnosis without accepted clinical-sign or diagnostic-criterion
support. The verifier can instantiate
`argument_from_negative_evidence` and
`argument_from_alternative_explanation`. Failed critical questions become
explicit undercutting arguments.

Support priorities are fixed before evaluation:

| Supporting scheme | Priority |
| --- | ---: |
| Diagnostic criterion | 4 |
| Clinical sign | 3 |
| Guideline authority | 2 |
| Risk factor | 1 |

Distinct grounded arguments contribute once per scheme and evidence set.
Rebuttals and undercutters are resolved before these priorities are compared.
Tied undefeated candidates produce abstention rather than an arbitrary
diagnosis.

The primary experiment uses the same local model in both agent roles. Different
local models can be supplied with `--reasoner-model` and `--verifier-model` for
a separately reported sensitivity experiment.

## Datasets

### MIMIC-IV-Ext-DiReCT

DiReCT is the primary dataset. The local release contains:

- 511 physician-annotated diagnostic cases
- six clinical sections per case
- 5,109 annotated observation nodes
- 24 supplied diagnostic guideline graphs
- 25 represented disease families

The loader creates a deterministic partition:

- development: 88 cases
- test: 423 cases

The release includes gastritis cases and 24 guideline files. Dataset auditing
records graph coverage, directory/conclusion differences, intermediate
conclusions, and graph-consistent case membership.

### MedQA

MedQA provides an external multiple-choice generalization evaluation. The
adapter selects questions that explicitly request a diagnosis:

- US development: 145 questions, including 1 gold diagnosis covered by the
  DiReCT graph inventory
- US test: 157 questions, including 3 gold diagnoses covered by the DiReCT
  graph inventory

MedQA is therefore an out-of-domain stress test of the complete system rather
than a powered test of graph-retrieval benefit. Results are stratified by
DiReCT graph coverage so that graph-conditioned outputs are interpreted against
the available guideline inventory.

### UMLS

Local UMLS normalization is available for diagnosis equivalence and retrieval
query expansion. The normalizer maps diagnostic terms to CUIs, adds preferred
terms and synonyms to BM25 indexing, and caches lookups in SQLite. Normalization
is performed lazily for guideline premises and selected case terms.

## Data Placement

Licensed data is ignored by Git and remains in the local `data/` directory.

```text
data/
  mimic_iv_ext_direct/
    raw/
      mimic-iv-ext-direct-1.0.0.zip
    unpacked/
      Finished/
      Diagnosis_flowchart/
  medqa/
    data_clean/questions/US/
      dev.jsonl
      test.jsonl
  umls/
    META/
      MRCONSO.RRF
      MRSTY.RRF
```

## Installation

Python 3.12 is the tested environment.

```bash
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and select a locally installed Ollama model:

```dotenv
DIAGNOSTIC_MODEL=medgemma1.5:latest
OLLAMA_ENDPOINT=http://127.0.0.1:11434
OLLAMA_LLM_ENDPOINT=http://127.0.0.1:11434
```

The experiment runner uses temperature zero, a fixed seed, schema-constrained
JSON, and disabled hidden reasoning so the output budget is reserved for the
auditable response. DiReCT and submitted patient states use loopback Ollama
endpoints by default.

## Prepare Data

Extract the DiReCT ZIP and its two nested RAR archives:

```bash
./venv/bin/python -m clinical_cds prepare-direct
```

Extraction uses `bsdtar` or `unar`. Existing `unpacked/Finished` and
`unpacked/Diagnosis_flowchart` directories are reused.

UMLS users can place an existing release under `data/umls/` or retrieve a
licensed release through the NLM API:

```bash
UMLS_API_KEY=your_api_key ./venv/bin/python scripts/download_umls.py
```

Build the local terminology index:

```bash
./venv/bin/python -m clinical_cds build-umls \
  --meta-dir data/umls/META \
  --db-path output/cache/umls_local.sqlite3
```

Pass `--umls-db output/cache/umls_local.sqlite3` to audit, experiment,
evaluation, perturbation, or diagnosis commands to enable UMLS normalization.

## Audit Data

```bash
./venv/bin/python -m clinical_cds audit-direct \
  --output output/audits/direct.json

./venv/bin/python -m clinical_cds audit-medqa
```

The audit commands validate corpus structure and produce aggregate statistics.
For a terminology-normalized retrieval audit:

```bash
./venv/bin/python -m clinical_cds audit-direct \
  --umls-db output/cache/umls_local.sqlite3 \
  --output output/audits/direct_umls.json
```

## Run Experiments

Start with a five-case development run:

```bash
./venv/bin/python -m clinical_cds run-experiment \
  --dataset direct \
  --partition development \
  --modes all \
  --limit 5 \
  --model medgemma1.5:latest \
  --run-name direct_smoke
```

Run the held-out DiReCT comparison:

```bash
./venv/bin/python -m clinical_cds run-experiment \
  --dataset direct \
  --partition test \
  --modes all \
  --model medgemma1.5:latest \
  --run-name direct_test
```

The `all` mode executes five prediction conditions but uses five model calls per
case: one call for each of the three non-agent conditions and one shared
reasoner-verifier exchange for both argument conditions.

While an experiment runs, a progress bar on standard error reports the current
case and stage (`reasoner`, `verifier`, `direct`, `flat_rag`, or `graph_rag`),
completed stages, elapsed time, throughput, and estimated time remaining.
Pass `--no-progress` when redirecting output or running without an interactive
progress display.

Run the graph-consistent sensitivity cohort:

```bash
./venv/bin/python -m clinical_cds run-experiment \
  --dataset direct \
  --partition test \
  --strict-direct \
  --modes all \
  --model medgemma1.5:latest \
  --run-name direct_test_strict
```

Run the terminology-normalization sensitivity comparison:

```bash
./venv/bin/python -m clinical_cds run-experiment \
  --dataset direct \
  --partition test \
  --modes all \
  --model medgemma1.5:latest \
  --umls-db output/cache/umls_local.sqlite3 \
  --run-name direct_test_umls
```

Run the MedQA external evaluation:

```bash
./venv/bin/python -m clinical_cds run-experiment \
  --dataset medqa \
  --medqa-split test \
  --modes all \
  --model medgemma1.5:latest \
  --run-name medqa_test
```

Identical prompts are served from the response cache using the model identifier,
prompt version, condition, and prompt hash.

## Perturbation Analysis

The section-removal analysis deletes the clinical section containing the
largest number of annotated observations and reruns the selected condition.

```bash
./venv/bin/python -m clinical_cds run-perturbations \
  --partition test \
  --mode symbolic_argument \
  --model medgemma1.5:latest \
  --umls-db output/cache/umls_local.sqlite3 \
  --run-name direct_section_removal
```

It reports paired accuracy, answer-change rate, abstention, and stale-citation
rate.

## Diagnose A Patient State

Patient input is a JSON object containing structured clinical sections such as
symptoms, history, examination findings, and test results.

```json
{
  "case_id": "local-case-001",
  "sections": {
    "chief_complaint": "Chest pain",
    "history_of_present_illness": "Central pressure for two hours",
    "past_medical_history": "Hypertension",
    "physical_exam": "Blood pressure 174/102 mmHg",
    "pertinent_results": "Troponin elevated"
  }
}
```

```bash
./venv/bin/python -m clinical_cds diagnose \
  --patient examples/patient.example.json \
  --mode symbolic_argument \
  --model medgemma1.5:latest \
  --umls-db output/cache/umls_local.sqlite3
```

The command returns a diagnosis, concise reasoning, evidence citations, and
abstention status.

## Visualize an Argument Trace

Render the core reasoner, verifier, and symbolic-resolution steps for a
completed case:

```bash
./venv/bin/python -m clinical_cds plot-trace \
  --traces output/experiments/direct_test/argument_traces.jsonl \
  --case-id <case-id> \
  --output output/notebook_artifacts/argument_trace.png
```

If `--case-id` is omitted, the first complete trace is selected. The plot
shows candidate arguments and cited evidence, verifier decisions and attacks,
symbolic scores, accepted support, and the final selected diagnosis or
abstention.

## Outputs

Each experiment creates:

```text
output/experiments/<run_id>/
  manifest.json
  predictions.jsonl
  argument_traces.jsonl
  evaluation/
    case_metrics.csv
    mode_summary.csv
    paired_comparisons.csv
    ablation_metrics.png
    argument_quality.png
    accuracy_progression.png
```

The manifest records the model, prompt version, per-role decoding
configuration, retrieval configuration, terminology normalizer, agent-role
models, symbolic resolver, case count, and condition set.
`predictions.jsonl` contains case-level predictions.
`argument_traces.jsonl` contains the model proposals, verifier judgements,
formal argument graph, accepted/rejected/undecided labels, and deterministic
human-readable resolution trace. CSV summaries and plots contain aggregate,
text-free evaluation data.

Reported metrics include:

- normalized diagnosis accuracy
- hierarchy-aware diagnosis score
- coverage, abstention, and selective accuracy
- citation validity
- DiReCT observation precision, recall, and F1
- gold-diagnosis retrieval coverage
- argument schema validity
- patient/guideline evidence validity
- verifier review coverage
- symbolic decision-trace fidelity
- symbolic resolution change rate
- uncached model latency
- paired bootstrap confidence intervals
- exact McNemar tests against the direct baseline

## Notebooks

Launch Jupyter from the repository root:

```bash
./venv/bin/jupyter lab
```

Run the notebooks in numerical order:

1. `notebooks/01_direct_dataset_audit.ipynb` reports corpus, annotation, graph,
   and quality-control statistics.
2. `notebooks/02_retrieval_and_umls_analysis.ipynb` reports lexical retrieval,
   optional UMLS sensitivity, and MedQA graph coverage.
3. `notebooks/03_ablation_results.ipynb` produces the main model results,
   baseline comparisons, incremental retrieval/graph/structured/symbolic
   effects, explanation-quality measures, confidence intervals, tests, and
   figures.
4. `notebooks/04_robustness_and_error_analysis.ipynb` produces error profiles,
   paired corrections and regressions, a selected argument-trace figure, and
   section-removal results.

Notebooks `01` and `02` can run after data preparation and do not call Ollama.
Run the CLI experiments before notebook `03`, and run the perturbation command
before the section-removal portion of notebook `04`. The standard run names
shown above are detected automatically. For other names, set `EXPERIMENT_DIR`
or `PERTURBATION_DIR`; multiple labeled paths can be supplied with
`EXPERIMENT_DIRS` or `PERTURBATION_DIRS`.
Set `TRACE_CASE_ID` to select a case for the argument-trace figure in notebook
`04`; otherwise the first complete trace is used.

Generated tables and figures are written under
`output/notebook_artifacts/`. See `notebooks/README.md` for configuration
details.

## Tests

```bash
./venv/bin/pytest -q
```

Pytest discovery is scoped to `tests/`, keeping licensed data, UMLS releases,
and generated output outside test collection.

## Evaluation Context

- DiReCT observation-grounding scores measure agreement with the released
  physician annotations. The annotation protocol permitted plausible
  observations to complete diagnostic chains.
- Results are reported for the full DiReCT cohort and a graph-consistent
  sensitivity cohort.
- MedQA results include graph-coverage strata derived from the DiReCT guideline
  inventory.
- The evaluation is retrospective and benchmark-based, with diagnosis accuracy,
  evidence grounding, retrieval coverage, selective performance, and latency as
  study endpoints.
