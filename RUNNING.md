# Running the comparison

This covers the four-condition comparison (Direct, Flat RAG, Graph RAG, and
evidence-grounded argumentation), both locally and as a remote Hugging Face
Jobs run. The runtime prepares and executes a sealed package; it never
submits, retries, or resubmits a paid job by itself.

## Prerequisites

Create the local environment and authenticate the Hugging Face CLI:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/hf auth login
```

The local `output/` tree must contain the recovered GraphRAG index,
controlled corpus tables, and query files referenced by `hf_job/config.py`.

UMLS must be available at `data/umls/<version>/META`, including
`MRCONSO.RRF` and `MRSTY.RRF`. The derived database is expected at
`output/cache/umls_local.sqlite3`. Package preparation stages only a
compact study subset; it does not upload the raw release or full local
index.

Before preparing any remote package, run:

```bash
venv/bin/pytest -q
```

## Run identifiers and isolation

Choose a new `RUN_ID` for every job. The identifier must contain 3–80
lowercase letters, numbers, dots, underscores, or hyphens. It names the
local package, remote prefixes, scratch directory, and output target.

Prepared packages are written beneath `.hf-runs/$RUN_ID/`:

```text
source/   executable source snapshot
input/    immutable queries, retrieval data, and terminology subset
output/   remote output placeholder and binding metadata
```

Inspect the generated manifests and hashes before uploading.

## Scopes

`RUN_SCOPE` selects the case set (see `hf_job/config.py`):

| Scope | Case count | Typical timeout |
|---|---|---|
| `validation` | 5 | 2h |
| `development` | 88 | 4h |

`RUN_CASE_LIMIT` and `RUN_SAMPLE_SEED` (preparation-time only, not job
environment variables) let you bound `development` to a smaller,
reproducibly-sampled subset for a faster check before committing to the
full case count. The sealed manifest records their effective values.

## Preparing and submitting a job

```bash
export RUN_SCOPE=validation
export RUN_PHASE=comparison
export RUN_ID="run-$(date -u +%Y%m%dt%H%M%Sz)"
export HF_BUCKET="Dugerij/jobs-artifacts"
export HF_IMAGE="vllm/vllm-openai:v0.18.1"

venv/bin/python -m hf_job.prepare
```

Upload each prepared directory to its isolated private prefix:

```bash
venv/bin/hf buckets sync ".hf-runs/$RUN_ID/source" \
  "hf://buckets/$HF_BUCKET/$RUN_ID/source"

venv/bin/hf buckets sync ".hf-runs/$RUN_ID/input" \
  "hf://buckets/$HF_BUCKET/$RUN_ID/input"

venv/bin/hf buckets sync ".hf-runs/$RUN_ID/output" \
  "hf://buckets/$HF_BUCKET/$RUN_ID/output"
```

Submit one detached job:

```bash
venv/bin/hf jobs run --detach \
  --name "argumentation-$RUN_ID" \
  --flavor a100-large \
  --timeout 2h \
  --secrets HF_TOKEN \
  --env "RUN_SCOPE=$RUN_SCOPE" \
  --env "RUN_PHASE=$RUN_PHASE" \
  --env "RUN_ID=$RUN_ID" \
  --env PROJECT_ROOT=/workspace/project \
  --env PYTHONPATH=/workspace/project \
  --env QUERY_PACKAGE_ROOT=/workspace/query \
  --env RUNTIME_SCRATCH_ROOT="/tmp/argumentation-schemes/$RUN_ID" \
  --env OUTPUT_ROOT=/outputs \
  --env VLLM_IPC_ROOT=/tmp/vli \
  --volume "hf://buckets/$HF_BUCKET/$RUN_ID/source:/workspace/project:ro" \
  --volume "hf://buckets/$HF_BUCKET/$RUN_ID/input:/workspace/query:ro" \
  --volume "hf://buckets/$HF_BUCKET/$RUN_ID/output:/outputs:rw" \
  "$HF_IMAGE" \
  python3 /workspace/project/hf_job/run.py
```

Use `--timeout 4h` for the `development` scope.

Monitor the returned job ID:

```bash
venv/bin/hf jobs inspect <JOB_ID>
venv/bin/hf jobs logs <JOB_ID>
venv/bin/hf jobs ls
```

Download completed output:

```bash
venv/bin/hf buckets sync \
  "hf://buckets/$HF_BUCKET/$RUN_ID/output" \
  "output/hf-downloads/$RUN_ID"
```

Results land under `output/hf-downloads/$RUN_ID/runs/$RUN_ID/`.

## Retrieval-only diagnostic

The retrieval phase audits all selected GraphRAG bundles and stops before
model generation. Use it to check candidate-family coverage, provenance,
and budget utilisation without paying for the complete comparison.

```bash
export RUN_SCOPE=development
export RUN_PHASE=retrieval
export RUN_ID="retrieval-$(date -u +%Y%m%dt%H%M%Sz)"

venv/bin/python -m hf_job.prepare
```

Upload and submit normally. Set `RUN_CASE_LIMIT`/`RUN_SAMPLE_SEED` before
preparation for a bounded check. After download, inspect
`graphrag_retrieval_audit.json` and `retrieval_only_summary.json` beneath
`runs/$RUN_ID/`.

## Resuming post-hoc evaluation

Use the evaluation phase only when all predictions and traces were frozen
but the blinded family evaluation did not complete. It reruns the
evaluator without retrieval or model inference.

```bash
export RUN_SCOPE=development
export RUN_PHASE=evaluation
export RUN_CASE_LIMIT=12
export RUN_SAMPLE_SEED=bounded-development-v1
export RESUME_COMPARISON_ROOT="output/hf-downloads/PRIOR_RUN/runs/PRIOR_RUN/comparison-development"
export RUN_ID="evaluation-resume-$(date -u +%Y%m%dt%H%M%Sz)"

venv/bin/python -m hf_job.prepare
```

Inspect, upload, and submit the new sealed package. The terminal manifest
and `evaluation_resume_audit.json` must report `phase=evaluation` and
`medgemma_invocation_count=0`.

## Interpreting a completed run

Confirm the following before using a result:

- the terminal manifest reports `status=completed`;
- staged input and runtime hashes match the prepared package;
- every case has Direct, Flat RAG, Graph RAG, and argumentation predictions;
- argument traces and adjudications are present for every case;
- prompt-boundary and citation-provenance audits pass;
- execution errors, diagnostic coverage, abstentions, exact accuracy, and
  blinded family accuracy are reported separately.

An abstention is not an execution failure. Selective accuracy must always
be read together with diagnostic coverage.

For the argumentation method, also verify that every `protected_incumbent`
decision:

- occurs only after an inconclusive direct differential;
- names an active Graph RAG family candidate;
- uses the active graph candidate through either strict (patient+knowledge)
  evidence overlap, or knowledge-only overlap with family-label
  compatibility when strict overlap is not available;
- returns the parent family rather than unsupported subtype specificity;
  and
- survived independent adversarial verification.

The post-hoc family judge is evaluation-only. Its output must never be fed
back into candidate activation, diagnosis generation, or resolution.

### Failure analysis for argumentation abstentions

Produce a breakdown of why the argumentation method abstained or fell
back:

```bash
python3 -m hf_job.failure_profile \
  output/hf-downloads/$RUN_ID/runs/$RUN_ID/comparison-development \
  --markdown
```

The script writes:

- `argumentation_failure_profile.json` (case-level buckets and diagnostics)
- `argumentation_failure_profile.md` (human-readable summary)

Useful buckets:

- `direct_declined_no_protected`: direct method returned
  `none_of_supplied_candidates` but no protected incumbent was chosen.
- `direct_declined_no_supported_family`: no direct candidate id was
  accepted and the protected-graph family could not be mapped to the
  active candidates.
- `execution_failure`: the trace or prediction stage ended with an
  execution error.

Each failure row also carries `graph_shape` with:

- `graph_candidates`: family IDs matching the Graph RAG label,
- `active_graph_candidates`: which of those candidates were active,
- `active_candidate_evidence_overlap`: overlap of graph citations with
  each active candidate's evidence IDs,
- `graph_citation_namespaces`: citation token namespace mix (`knowledge`,
  `section_or_patient`, etc.).

Use that block first when a failure is not obvious from resolver logic. It
tells you whether the block is:

- a label-shape miss (`active_graph_candidates` is empty),
- a citation-shape miss (namespaces are empty or missing knowledge
  overlap),
- or a protection-policy miss (active label is matched but no owned pair
  overlap survives).

Rows where `direct_decision` is `none_of_supplied_candidates` and
`resolution_reason` is empty are where the direct comparison and the
protected-incumbent path are the decision bottleneck.
