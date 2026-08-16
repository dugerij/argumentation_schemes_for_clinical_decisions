# Evidence-Grounded Clinical Argumentation

This repository compares four diagnostic approaches on de-identified clinical
cases:

1. Direct model inference
2. Flat retrieval-augmented generation
3. Graph retrieval-augmented generation
4. Evidence-grounded clinical argumentation

## Research question

Does explicit, provenance-bound argumentation improve diagnostic decisions
over direct inference and retrieval alone? The comparison holds the clinical
cases, model family, and knowledge boundary constant while changing how
retrieved evidence is used.

The primary quantitative outputs are exact-label accuracy, blinded clinical
family accuracy, diagnostic coverage, abstention rate, and structural
validity. Argumentation traces provide a separate qualitative account of
evidence use and decision changes.

## Method

Each case is processed through the following pipeline:

```text
Clinical case
    ├── Direct inference
    ├── Flat retrieval ──> Flat RAG
    └── Graph retrieval ─┬─> Graph RAG
                         └─> Argumentation
                               1. Build eight family evidence profiles
                               2. Activate up to four families
                               3. Select a family, then an evidenced subtype
                               4. Independently test the proposal
                               5. Validate any attack
                               6. Resolve or use a protected incumbent
                               7. Grade frozen outputs with an independent LLM
```

The argumentation model can select only server-defined candidates and cite
only server-owned patient/knowledge evidence pairs. The verifier can
challenge a proposal for citation failure, wrong subject or encounter,
diagnostic insufficiency, explicit contradiction, a better-supported
alternative, unsupported specificity, or material unexplained evidence.

A competing diagnosis does not defeat the proposal merely because it has
support. A switch requires evidence that directly falsifies a necessary
proposition of the current diagnosis and independently establishes the
alternative. Otherwise the deterministic resolver maintains the diagnosis,
falls back to its parent family, or returns an abstention with the leading
differential.

If the direct comparison is inconclusive, the resolver may retain the
existing Graph RAG answer only as an abstention backstop. The answer must
map uniquely to an active candidate through cited, candidate-owned
patient/knowledge evidence; it is reduced to the family label and must
survive the same independent attack. A supported direct argument always
takes precedence over this incumbent.

After all predictions are frozen, a separate pinned judge model grades
whether the reference and evaluated diagnoses preserve the same core disease
identity. It has no diagnostic or resolver authority and cannot alter
predictions.

## Evidence controls

- Patient findings retain section, polarity, temporality, quantities, and
  units.
- Knowledge passages are reduced to atomic warrants before binding.
- UMLS supports terminology identity and synonym normalisation, not
  diagnostic authority.
- Risk factors, manifestations, guideline authority, and subtype features do
  not independently establish a parent diagnosis.
- A derived decisive-anchor view marks only existing, sourced diagnostic-test
  or numeric-threshold claims. It preserves the original graph node, wording,
  diagnostic path, and source IDs; it excludes symptoms, signs, risk factors,
  and generic guidance. Anchor coverage is audited before the view is allowed
  to influence resolution.
- Missing support is not treated as contradiction.
- Every cited identifier must belong to the immutable case or retrieval
  bundle.
- Prompts use bounded JSON schemas and fixed completion limits, with
  reasoning/evidence fields ordered before the fields they justify so a
  verdict is generated after, not before, its own supporting citations.
- Gold labels are unavailable to retrieval, generation, verification, and
  resolution; they are used only during evaluation.

## Repository layout

| Path | Purpose |
|---|---|
| `clinical_cds/` | Clinical cases, retrieval integration, argumentation, evaluation, and reporting |
| `graphrag_runtime/` | GraphRAG corpus, retrieval, provenance, and prompt-boundary controls |
| `job/` | Reproducible Hugging Face package preparation and remote runtime |
| `tests/` | Unit, boundary, provenance, retrieval, and evaluation tests |
| `examples/` | Example input records |
| `RUNNING.md` | How to run the comparison, locally or as a remote job |

Generated outputs, local caches, licensed terminology data, downloaded model
artifacts, and staged remote packages are excluded from version control.

## Local setup

Create a virtual environment and install the pinned requirements:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

The complete test suite is:

```bash
venv/bin/pytest -q
```

Tests cover typed clinical bindings, deterministic argument resolution,
retrieval provenance, GraphRAG prompt construction, structured-output
limits, privacy-safe presentation, evaluation, and remote package
preparation.

## Data and models

The comparison uses de-identified MIMIC-derived diagnostic cases, a local
UMLS installation, and provenance-backed diagnostic knowledge graphs. These
resources are not committed to the repository; see
[RUNNING.md](RUNNING.md) for expected locations and preparation checks.

Remote comparison runs use the pinned model and container configuration
defined in `job/`. Runtime code never submits, retries, or resubmits a
paid job by itself.

## Running a comparison

See [RUNNING.md](RUNNING.md) for the full setup, preparation, submission,
and result-collection sequence.
