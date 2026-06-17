# Safety & Validation Layer

ADR-006 introduced a focused safety package for the clinical RAG system, and ADR-008 now uses it inside an explicit deterministic LangGraph.

The layer is designed for trained medical professionals: it does not refuse high-risk clinical topics, but it does protect against prompt injection and validates that generated answers remain cited, cautious, and clearly marked as decision-support.

## Components

- `guardrails.py` provides deterministic prompt injection detection using the shared `INJECTION_KEYWORDS`.
- `schemas.py` defines the Pydantic v2 `ClinicalResponse` model. Key fields (`uncertainty_note`, `key_findings_to_verify`, `recommended_next_steps`) are now required to support explicit routing decisions in ADR-008. The automatic escalation validator was removed — escalation logic now lives in the generation layer and the graph.
- `critic.py` provides a deterministic Phase 1 `SafetyCritic` that can be used after generation.
- `prompts.py` is the safety package's single source of truth for guardrail keywords and critic prompt text.

## Runtime Role in the Graph

ADR-008 uses the safety package in two places:

1. **Before retrieval/generation**: `pipeline/graph.py` starts with a deterministic `validate_input` node that calls `detect_prompt_injection(...)`. If prompt injection is detected, the graph exits early before any retrieval or LLM work.
2. **After generation**: pipeline code passes the generated `ClinicalResponse` and retrieved context to `pipeline.factory.apply_safety_layer()`. The resulting `safe_response` is then consumed by the `assess_and_route` node, which builds the lightweight `assessment` object and makes the final deterministic routing decision.

## Design Notes (Post ADR-008)

- `ClinicalResponse` no longer auto-escalates `requires_human_review`. This responsibility was moved out of the model to keep routing decisions explicit and testable in the graph.
- `from_raw_sources()` was restored as a convenience helper for tests and fallbacks while preserving the stricter field requirements.
- The safety package now supports both entry-time input screening and post-generation validation. Final routing (high vs low confidence path) is handled in the ADR-008 graph.

## Phase 2 TODOs

- Expand beyond keyword-based prompt injection detection with richer policy/config support.
- Add citation-faithfulness and disclaimer-quality evaluation metrics.
- Introduce configurable safety profiles.