# Generation

This module is responsible for the **Generation Layer** of the clinical RAG pipeline. It transforms retrieved medical context into structured, grounded, and clinically useful responses.

## Responsibilities

- Format retrieved documents and metadata into clean context for the LLM.
- Define and manage production-grade system and user prompts.
- Call the LLM and parse its output into a validated `ClinicalResponse`.
- Handle uncertainty gracefully by supporting structured fields instead of binary refusal.

## File Structure

| File            | Responsibility |
|-----------------|----------------|
| `prompts.py`    | Contains the main `CLINICAL_SYSTEM_PROMPT` and `CLINICAL_USER_TEMPLATE`. Defines how the model should behave in high-confidence, low-confidence, and out-of-knowledge scenarios. |
| `context.py`    | Pure formatting utilities. Converts retrieved documents + metadata into a readable string for the LLM. |
| `chain.py`      | Core generation logic (`generate_clinical_answer`). Handles LLM invocation, JSON parsing, fallback behavior, and LangSmith traceability. |
| `__init__.py`   | Module exports. |

## Design Principles

- **Separation of concerns**: Prompts are kept pure (strings only). Formatting logic lives in `context.py`. LLM interaction and parsing live in `chain.py`.
- **Clinical usefulness over binary refusal**: When the model has partial information, it returns a hedged but useful answer along with `uncertainty_note`, `key_findings_to_verify`, and `recommended_next_steps`.
- **Strong grounding**: The model is strictly instructed to only use the provided context and to cite sources.
- **Safety by default**: Low confidence or uncertainty automatically escalates `requires_human_review`.

## Integration

This module is used by:
- `pipeline/graph.py` (inside the `generate` node)
- Direct calls in demos and tests

The output is always a validated `ClinicalResponse` from `safety.schemas`.