# Safety & Validation Layer

Phase 1 of ADR-006 adds a focused safety package for the clinical RAG system.

The layer is designed for trained medical professionals: it does not refuse high-risk clinical topics, but it does protect against prompt injection and validates that generated answers remain cited, cautious, and clearly marked as decision-support.

## Components

- `guardrails.py` provides deterministic prompt injection detection using the shared `INJECTION_KEYWORDS` prompt constants.
- `schemas.py` defines the Pydantic v2 `ClinicalResponse` and citation models used to enforce disclaimer, source, warning, confidence, and human-review fields.
- `critic.py` provides a deterministic Phase 1 `SafetyCritic` interface that can later become a LangGraph node using the shared `SAFETY_CRITIC_SYSTEM_PROMPT`.
- `prompts.py` is the safety package's single source of truth for guardrail keywords and critic prompt text.

## Phase 2 TODOs

- Wire `SafetyCritic.build_llm_messages()` into a full LangGraph generation pipeline.
- Add citation-faithfulness and disclaimer-quality evaluation metrics.
- Introduce configurable safety profiles and human-in-the-loop review workflows.