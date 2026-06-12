# Safety & Validation Layer

Phase 1 of ADR-006 adds a focused safety package for the clinical RAG system.

The layer is designed for trained medical professionals: it does not refuse high-risk clinical topics, but it does protect against prompt injection and validates that generated answers remain cited, cautious, and clearly marked as decision-support.

## Components

- `guardrails.py` provides deterministic prompt injection detection using the shared `INJECTION_KEYWORDS`.
- `schemas.py` defines the Pydantic v2 `ClinicalResponse` model. Key fields (`uncertainty_note`, `key_findings_to_verify`, `recommended_next_steps`) are now required to support explicit routing decisions in ADR-008. The automatic escalation validator was removed — escalation logic now lives in the generation layer and the graph.
- `critic.py` provides a deterministic Phase 1 `SafetyCritic` that can be used after generation.
- `prompts.py` is the safety package's single source of truth for guardrail keywords and critic prompt text.

Pipeline code should apply this layer after answer generation by passing a `ClinicalResponse` and retrieved context to `pipeline.factory.apply_safety_layer()`.

The resulting `safe_response` is then consumed by the `assess_and_route` node in `pipeline/graph.py` (ADR-008), which makes the final deterministic routing decision.

## Design Notes (Post ADR-008)

- `ClinicalResponse` no longer auto-escalates `requires_human_review`. This responsibility was moved out of the model to keep routing decisions explicit and testable in the graph.
- `from_raw_sources()` was restored as a convenience helper for tests and fallbacks while preserving the stricter field requirements.
- The Safety layer remains focused on post-generation validation and improvement. Final routing (high vs low confidence path) is handled in the ADR-008 graph.

## Phase 2 TODOs

- Wire `SafetyCritic` into a full LangGraph generation + assessment pipeline.
- Add citation-faithfulness and disclaimer-quality evaluation metrics.
- Introduce configurable safety profiles.