# Evidence-Grounded Clinical Argumentation

Code for my MSc project on whether making a model argue for a diagnosis — and cite the
guideline criteria it relied on — produces decisions a clinician could actually check.

Four conditions run over the same de-identified clinical cases:

1. **Direct** — ask the model, nothing retrieved.
2. **Basic retrieval** — retrieve guideline text, then ask.
3. **Graph retrieval** — retrieve over diagnostic knowledge graphs, then ask.
4. **Argumentation** — the same graph retrieval, plus a generator, a verifier and a
   deterministic resolver.

## The question

Does explicit, provenance-bound argumentation beat direct inference and plain retrieval?
The cases, the model family and the knowledge boundary stay fixed across all four
conditions, so the only thing that changes is what happens to the retrieved evidence.

What gets measured: family-level diagnostic accuracy, coverage, abstention rate, and how
many accepted answers carry a checked evidence binding. The argumentation traces are kept
separately as a record of which evidence was used and where a diagnosis changed.

## What it found

On 423 held-out cases, the knowledge graph is what improved accuracy and the argumentation
layer is what made the answers checkable:

| Condition | Accuracy | Abstained | Answers with checked evidence |
|---|---|---|---|
| Direct | 0.657 | 0 | none |
| Basic retrieval | 0.664 | 0 | none |
| Graph retrieval | 0.726 | 0 | none |
| Argumentation | 0.731 | 55 | 97.8% |

Both graph-grounded conditions beat direct prompting after correcting for multiple
comparisons. Argumentation and graph retrieval are statistically indistinguishable from
each other, so the layer buys no accuracy — but it is the only condition that can decline
to answer, and the only one whose answers come with evidence the pipeline has already
checked. Accuracy is scored at family level by a separate judge model, on a retrospective
benchmark, and no clinician has reviewed any output.

## How a case flows through

```text
Clinical case
    ├── Direct inference
    ├── Basic retrieval ──> Basic retrieval answer
    └── Graph retrieval ─┬─> Graph retrieval answer
                         └─> Argumentation
                               1. Build eight family evidence profiles
                               2. Activate up to four families
                               3. Select a family, then an evidenced subtype
                               4. Independently test the proposal
                               5. Validate any attack
                               6. Resolve, or keep a protected incumbent
                               7. Grade frozen outputs with an independent judge
```

## What the argumentation condition may and may not do

The model can only choose from candidates the server defined, and can only cite
patient/knowledge evidence pairs the server owns. The verifier can challenge a proposal on
seven grounds: citation failure, wrong subject or encounter, diagnostic insufficiency,
explicit contradiction, a better-supported alternative, unsupported specificity, or
material unexplained evidence.

A rival diagnosis does not win simply by having support behind it. Switching requires
evidence that both falsifies something the current diagnosis needs to be true and
independently establishes the alternative. Failing that, the deterministic resolver keeps
the diagnosis, drops to its parent family, or abstains and returns the leading
differential.

Where the direct comparison is inconclusive, the resolver may fall back on the existing
graph-retrieval answer, but only as a backstop. That answer has to map uniquely to an
active candidate through cited, candidate-owned evidence, it is reduced to the family
label, and it has to survive the same independent attack. A supported direct argument
always wins over it.

Once every prediction is frozen, a separate pinned judge model decides whether the
reference and predicted diagnoses share the same core disease identity. It has no
diagnostic or resolver authority and cannot change a prediction.

## Evidence rules

- Patient findings keep their section, polarity, temporality, quantities and units.
- Knowledge passages are broken down into atomic warrants before anything is bound to them.
- UMLS settles terminology identity and synonyms. It carries no diagnostic authority.
- Risk factors, manifestations, guideline authority and subtype features cannot establish a
  parent diagnosis on their own.
- A derived decisive-anchor view marks only existing, sourced diagnostic-test or
  numeric-threshold claims, preserving the original graph node, wording, diagnostic path
  and source IDs. Symptoms, signs, risk factors and generic guidance are excluded, and
  anchor coverage is audited before the view is allowed to affect resolution.
- Missing support is not treated as contradiction.
- Every cited identifier has to belong to the immutable case or retrieval bundle.
- Prompts use bounded JSON schemas and fixed completion limits, with reasoning and evidence
  fields ordered ahead of the fields they justify, so a verdict is generated after its own
  citations rather than before them.
- Gold labels are hidden from retrieval, generation, verification and resolution. They are
  used only when scoring.

## Repository layout

| Path | Purpose |
|---|---|
| `clinical_cds/` | Clinical cases, retrieval integration, argumentation, evaluation and reporting |
| `graphrag_runtime/` | GraphRAG corpus, retrieval, provenance and prompt-boundary controls |
| `job/` | Reproducible Hugging Face package preparation and remote runtime |
| `tests/` | Unit, boundary, provenance, retrieval and evaluation tests |
| `examples/` | Example input records |
| `RUNNING.md` | How to run the comparison, locally or as a remote job |

Generated outputs, local caches, licensed terminology data, downloaded model artifacts and
staged remote packages are all kept out of version control.

## Running it locally

Create a virtual environment and install the pinned requirements:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Then run the test suite:

```bash
venv/bin/pytest -q
```

The tests cover typed clinical bindings, deterministic argument resolution, retrieval
provenance, GraphRAG prompt construction, structured-output limits, privacy-safe
presentation, evaluation and remote package preparation.

## Data and models

The comparison runs on de-identified MIMIC-derived diagnostic cases, a local UMLS
installation, and provenance-backed diagnostic knowledge graphs. None of these are in the
repository — [RUNNING.md](RUNNING.md) lists where they are expected and how their
preparation is checked.

Remote runs use the pinned model and container configuration in `job/`. Nothing in the
runtime submits, retries or resubmits a paid job on its own.

## Running a comparison

[RUNNING.md](RUNNING.md) has the full setup, preparation, submission and result-collection
sequence.
