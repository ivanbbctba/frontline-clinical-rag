# Generation

This module is responsible for the **Generation Layer** of the clinical RAG pipeline. It transforms retrieved medical context into structured, grounded, and clinically useful responses.

## Responsibilities

- Format retrieved documents and metadata into clean context for the LLM.
- Define and manage production-grade system and user prompts.
- Call the LLM and parse its output into a validated `ClinicalResponse`.
- Handle uncertainty signals from the LLM while keeping escalation logic explicit.

## File Structure

| File            | Responsibility |
|-----------------|----------------|
| `prompts.py`    | Contains the main `CLINICAL_SYSTEM_PROMPT` and `CLINICAL_USER_TEMPLATE`. Defines expected behavior for high-confidence, low-confidence, and out-of-knowledge scenarios. |
| `context.py`    | Pure formatting utilities. Converts retrieved documents + metadata into a readable string for the LLM. |
| `chain.py`      | Core generation logic (`generate_clinical_answer`). Handles LLM invocation, JSON parsing, fallback behavior, and LangSmith traceability. Escalation of `requires_human_review` based on LLM signals (`uncertainty_note` / `key_findings_to_verify`) is applied here. |
| `__init__.py`   | Module exports. |

## Design Principles

- **Separation of concerns**: Prompts are kept pure (strings only). Formatting logic lives in `context.py`. LLM interaction, parsing, and light escalation logic live in `chain.py`.
- **Explicit escalation**: When the LLM returns an `uncertainty_note` or `key_findings_to_verify`, `requires_human_review` is set to `True` inside the generation function. This keeps the `ClinicalResponse` model clean while still surfacing uncertainty.
- **Strong grounding**: The model is strictly instructed to only use the provided context and to cite sources.
- **Deterministic routing support**: The output `ClinicalResponse` is designed to feed directly into the ADR-008 `assess_and_route` node, which makes final routing decisions (`HIGH_CONFIDENCE` vs `LOW_CONFIDENCE_ESCALATION`).

## Integration

This module is used by:
- `pipeline/graph.py` (inside the `generate` node)
- Direct calls in demos and tests

The output is always a validated `ClinicalResponse` from `safety.schemas`.