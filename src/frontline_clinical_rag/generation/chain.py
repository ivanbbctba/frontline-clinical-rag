"""Traceable grounded generation step for the clinical RAG pipeline."""

from __future__ import annotations

import json
from typing import Any, Protocol

from langsmith import traceable
from pydantic import ValidationError

from src.frontline_clinical_rag.generation.context import \
    format_context_with_metadata
from src.frontline_clinical_rag.generation.prompts import (
    CLINICAL_SYSTEM_PROMPT, CLINICAL_USER_TEMPLATE, OUT_OF_KNOWLEDGE_BASE)
from src.frontline_clinical_rag.safety.schemas import ClinicalResponse


class ClinicalLLM(Protocol):
    """Minimal model protocol used by the generation step."""

    def invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        """Invoke a chat model with LangChain-style messages."""


@traceable(name="generate_clinical_answer")
def generate_clinical_answer(
    question: str,
    documents: list[dict[str, Any]],
    *,
    llm: ClinicalLLM,
    run_name: str = "generate_clinical_answer",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ClinicalResponse:
    """Generate and validate a grounded clinical answer.

    The function is LangSmith-ready without hardcoding tracing configuration:
    callers can pass ``run_name``, ``tags``, and ``metadata`` for future tracer
    enrichment, while environment variables can enable or disable tracing.
    """

    context = format_context_with_metadata(documents)
    messages = [
        {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": CLINICAL_USER_TEMPLATE.format(
                context=context,
                question=question,
            ),
        },
    ]
    trace_metadata = {
        "question": question,
        "document_count": len(documents),
        "warning_level_detected": _warning_level_summary(documents),
        **(metadata or {}),
    }

    try:
        raw_output = llm.invoke(
            messages,
            run_name=run_name,
            tags=tags or [],
            metadata=trace_metadata,
        )
        return _parse_clinical_response(_content_from_llm_output(raw_output))
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        return _fallback_response(question, documents, exc)


def _parse_clinical_response(raw_output: str) -> ClinicalResponse:
    payload = json.loads(_extract_json_object(raw_output))
    return ClinicalResponse.model_validate(payload)


def _extract_json_object(raw_output: str) -> str:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM output did not contain a JSON object")
    return cleaned[start : end + 1]


def _content_from_llm_output(raw_output: Any) -> str:
    content = getattr(raw_output, "content", raw_output)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def _fallback_response(
    question: str,
    documents: list[dict[str, Any]],
    error: Exception,
) -> ClinicalResponse:
    return ClinicalResponse.from_raw_sources(
        answer=OUT_OF_KNOWLEDGE_BASE,
        sources=_sources_from_documents(documents),
        warning_level_summary=(
            "Generation output could not be parsed into ClinicalResponse: "
            f"{type(error).__name__}."
        ),
        confidence=0.0,
        requires_human_review=True,
    )


def _sources_from_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for document in documents:
        metadata = document.get("metadata", document)
        if not isinstance(metadata, dict):
            metadata = {}
        content = str(
            document.get(
                "page_content", document.get("content", document.get("text", ""))
            )
        )
        section = metadata.get("section") or metadata.get("section_title")
        hierarchy = metadata.get("section_hierarchy")
        if not section and isinstance(hierarchy, list):
            section = " > ".join(str(item) for item in hierarchy)
        sources.append(
            {
                "page": metadata.get("page_number", metadata.get("page", "unknown")),
                "section": section or metadata.get("source_title") or "unknown",
                "excerpt": " ".join(content.strip().split())[:300]
                or "No excerpt available.",
            }
        )
    return sources


def _warning_level_summary(documents: list[dict[str, Any]]) -> str:
    warning_levels: set[str] = set()
    for document in documents:
        metadata = document.get("metadata", document)
        if isinstance(metadata, dict) and metadata.get("warning_level"):
            warning_levels.add(str(metadata["warning_level"]))
    return ", ".join(sorted(warning_levels)) if warning_levels else "none"
