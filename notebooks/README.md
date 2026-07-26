# Analysis Notebooks

Launch Jupyter from the repository root so data and output paths resolve
consistently:

```bash
./venv/bin/jupyter lab
```

Run the notebooks in numerical order.

1. `01_direct_dataset_audit.ipynb` validates DiReCT corpus structure,
   annotation volume, graph inventory, and quality flags.
2. `02_retrieval_and_umls_analysis.ipynb` measures gold-path retrieval coverage,
   optional UMLS sensitivity, and MedQA coverage by the DiReCT graph inventory.
3. `03_ablation_results.ipynb` converts completed model runs into main results,
   paired baseline comparisons, incremental retrieval, graph, structured-agent,
   and symbolic-resolution effects, explanation-quality tables, and figures.
4. `04_robustness_and_error_analysis.ipynb` reports outcome profiles, paired
   corrections and regressions, symbolic decision changes, a text-free error
   catalogue, a selected argument-trace figure, and section-removal
   robustness.

Notebooks `01` and `02` do not call a language model. Notebook `03` requires
completed `run-experiment` artifacts. Notebook `04` requires at least one
completed experiment and adds section-removal results when a completed
`run-perturbations` artifact is available.

The standard run names in the root README are discovered automatically:

```text
output/experiments/direct_test
output/experiments/direct_test_strict
output/experiments/direct_test_umls
output/experiments/medqa_test
output/experiments/direct_section_removal
```

Use `EXPERIMENT_DIR` for one nonstandard experiment directory or
`EXPERIMENT_DIRS` for multiple entries separated by the operating system path
separator. Entries may use `label=path`. Use `PERTURBATION_DIR` or
`PERTURBATION_DIRS` in the same way for notebook `04`.
Set `TRACE_CASE_ID` to choose the case shown in the argument-trace figure. If
it is unset, notebook `04` uses the first complete trace it finds.

Set `UMLS_DB=output/cache/umls_local.sqlite3` before starting notebook `02` to
select a nondefault UMLS index. If no index exists, the notebook completes the
lexical analysis and explicitly skips UMLS results.

Notebook artifacts are written under `output/notebook_artifacts/`. Aggregate
tables and figures exclude clinical note text and model reasoning.
