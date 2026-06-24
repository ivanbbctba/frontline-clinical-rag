# Pipeline Package

This package owns the production Clinical RAG orchestration path.

The goal is to keep question validation, retrieval, answer generation, clinical safety assessment, routing, and final response formatting in one graph-driven workflow. Other packages provide specialized capabilities, but `pipeline` is where those capabilities are composed into the end-to-end application behavior that users and evaluation runs exercise.

## What Lives Here

- `graph.py` defines the Clinical RAG LangGraph, graph state, routing decisions, clinical assessment shape, and node-level transitions.
- `factory.py` builds production retrievers and applies the safety layer around generated clinical responses.
- `__init__.py` exposes lazy package imports so downstream code can use pipeline components without eagerly importing optional graph dependencies.

## Production Flow

The main graph path is intentionally explicit and reviewable:

```text
validate_input → retrieve → generate → assess_and_route → format_high_confidence
                                                    ↘ handle_low_confidence_escalation
```

The graph keeps the final `ClinicalRAGState` as the shared contract between runtime execution, logging, tests, and ADR-009 deterministic evaluation. That state includes the original question, normalized question, retrieved context, generated `ClinicalResponse`, clinical assessment, routing decision, formatted answer, trace information, and any validation or escalation errors.

## Key Public Entry Points

- `build_clinical_rag_graph()` constructs the graph application with injectable retriever, LLM, and logger dependencies.
- `run_clinical_rag_graph()` runs one clinical question through the graph and returns the final `ClinicalRAGState`.
- `create_retriever()` selects the configured retrieval implementation from centralized `core.config.AppConfig`.
- `apply_safety_layer()` applies safety validation and critique to generated responses without duplicating graph routing logic.
- `save_graph_visualization()` exports the graph structure for documentation and portfolio review when graph rendering dependencies are available.

## Configuration

Pipeline code should use centralized configuration from `src.frontline_clinical_rag.core.config`.

Retrieval strategy, top-k settings, force-rebuild behavior, safety thresholds, disclaimer text, and model provider settings belong in `AppConfig`; they should not be duplicated inside this package. Tests may inject fake retrievers or mock LLMs directly into graph entry points when they need deterministic behavior.

## Evaluation Relationship

ADR-009 evaluation stays external to this package.

The evaluation harness may run `run_clinical_rag_graph()` and inspect the returned `ClinicalRAGState`, but it must not re-implement validation, retrieval, generation, safety assessment, routing, or formatting. This keeps the production graph as the source of truth and lets deterministic metrics review the actual behavior of the system.

## Testing Guidance

- Use graph-level tests when verifying node ordering, routing decisions, state fields, and integration with mocked dependencies.
- Use factory tests when verifying retriever selection, configuration wiring, or safety-layer behavior.
- Prefer injected deterministic retrievers and mock LLMs for CI-safe tests.
- Keep live provider checks outside routine unit tests unless they are explicitly marked as local or integration-only.

## Guardrails

- Do not duplicate clinical routing logic outside `graph.py`.
- Do not bypass `apply_safety_layer()` for generated clinical responses.
- Do not add package-local configuration; use `core.config`.
- Do not hard-code evaluation fixture questions in pipeline tests; import canonical fixtures from `evaluation.fixtures` when needed.
- Do not make graph metrics depend on wall-clock time, randomness, or live LLM calls; deterministic evaluation belongs in `evaluation.metrics`.