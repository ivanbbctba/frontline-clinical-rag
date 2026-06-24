# Evaluation Package

This package implements ADR-009: the lean deterministic evaluation harness for the Clinical RAG pipeline.

The goal is to make clinical RAG behavior reviewable without adding non-deterministic judges, notebook-heavy reporting, or a second implementation of production routing and safety logic. The harness runs the existing graph, captures the final `ClinicalRAGState`, and computes deterministic PASS/FAIL metrics from that state.

## What Lives Here

- `fixtures.py` is the single source of truth for the four canonical Merck evaluation questions.
- `metrics.py` contains pure Layer A metric computation for M1-M12.
- `harness.py` orchestrates evaluation runs, prints console tables, and exports optional CSV/JSON artifacts.
- `__init__.py` exposes the public package API.

Evaluation configuration intentionally does not live in this package. Use `src.frontline_clinical_rag.core.config.EvaluationConfig` through the centralized project configuration, the same way other packages use `core.config`.

## Determinism Model

ADR-009 separates determinism into two practical layers:

- Layer A: deterministic metric computation. Given the same final `ClinicalRAGState` and `EvaluationConfig`, metric results are identical. Metrics do not call an LLM, use randomness, or depend on wall-clock time, except for the informational latency value.
- Layer B: execution of retrieval and generation. Mock-based harness runs are CI-friendly; live LLM runs use `temperature=0` but are still provider-dependent and should not be treated as strict reproducibility guarantees.

## Run Modes

- `metric_unit`: intended for tests that inject state fixtures directly into the metric layer.
- `harness_integration`: runs canonical fixtures through the graph with deterministic mock components.
- `live_eval`: runs the real retrieval and generation path at `temperature=0` for local portfolio evidence and regression review.

## CLI Usage

Run the default full graph demo. ADR-009 metrics print after every question:

```bash
python run.py
```

Run the standalone evaluation table with deterministic mock components:

```bash
python run.py --mode evaluation --eval-mode harness_integration
```

Compare retrieval strategies:

```bash
python run.py --mode evaluation --eval-mode harness_integration --compare-strategies
```

Run live evaluation and export artifacts:

```bash
python run.py --mode evaluation --eval-mode live_eval --strategy hierarchical --csv reports/eval.csv --json reports/eval.json
```

Run the package module directly:

```bash
python -m src.frontline_clinical_rag.evaluation.harness --mode harness_integration
```

## Python Usage

```python
from src.frontline_clinical_rag.core.config import EvaluationConfig, get_config
from src.frontline_clinical_rag.evaluation.harness import print_table, run_evaluation

app_config = get_config()
evaluation_config = EvaluationConfig.from_app_config(
    app_config,
    run_mode="harness_integration",
    strategies=["hierarchical"],
)

rows = run_evaluation(evaluation_config, app_config=app_config)
print_table(rows)
```

## Output Contract

The harness reports one row per question and retrieval strategy. The key reviewer signal is whether all gating metrics pass for each final graph state.

- M1-M11 are gating metrics.
- M12 latency is informational only.
- M7 uncertainty/escalation handling is marked as LLM-sensitive for CI review because live generation can affect whether escalation fields are populated.

## Guardrails

- Do not duplicate the four canonical question strings outside `fixtures.py`.
- Do not add LLM judges, RAGAS, or heavy notebook reporting to this Phase 1 package.
- Do not re-implement production clinical routing, safety assessment, or answer formatting in evaluation code.
- Do not create package-local evaluation config; keep evaluation settings in `core.config`.