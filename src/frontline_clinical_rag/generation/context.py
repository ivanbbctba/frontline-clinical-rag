"""Pure context formatting utilities for grounded clinical generation.

This module is intentionally free of model calls, retriever construction, and
side effects. It turns retrieved chunks plus their clinical metadata into a
stable text contract that can be passed to the prompt layer and inspected in
LangSmith traces or future architecture diagrams.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_METADATA_KEYS: tuple[str, ...] = (
    "source_title",
    "title",
    "section_hierarchy",
    "section",
    "section_title",
    "warning_level",
    "chunk_type",
    "page_number",
    "page",
    "source",
)


def format_context_with_metadata(documents: list[dict[str, Any]]) -> str:
    """Format retrieved medical chunks and metadata for clinical generation.

    Args:
        documents: Retrieved chunks represented as dictionaries. Each item may
            expose content as ``page_content``, ``content``, or ``text`` and may
            either carry metadata under a ``metadata`` key or as top-level keys.

    Returns:
        A deterministic, human-readable context block. Each chunk includes its
        content plus citation-relevant metadata such as section hierarchy,
        warning level, page, and source.

    The function is pure by design: it does not mutate inputs, read files, log,
    call models, or depend on retriever classes.
    """

    if not documents:
        return "No retrieved medical context was provided."

    formatted_chunks: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = _metadata_for(document)
        content = _content_for(document)
        metadata_lines = _format_metadata(metadata)
        formatted_chunks.append(
            "\n".join(
                [
                    f"[Chunk {index}]",
                    *metadata_lines,
                    "Content:",
                    content or "No chunk content available.",
                ]
            )
        )

    return "\n\n".join(formatted_chunks)


def _metadata_for(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    merged = dict(metadata)
    for key, value in document.items():
        if key not in {"metadata", "page_content", "content", "text"}:
            merged.setdefault(key, value)
    return merged


def _content_for(document: dict[str, Any]) -> str:
    for key in ("page_content", "content", "text"):
        value = document.get(key)
        if value is not None:
            return _normalize_text(str(value))
    return ""


def _format_metadata(metadata: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    normalized = _normalized_metadata(metadata)

    for key in _METADATA_KEYS:
        value = normalized.get(key)
        if value not in (None, "", []):
            label = key.replace("_", " ").title()
            lines.append(f"{label}: {_format_value(value)}")

    extra_keys = sorted(set(normalized) - set(_METADATA_KEYS))
    for key in extra_keys:
        value = normalized[key]
        if value not in (None, "", []):
            label = key.replace("_", " ").title()
            lines.append(f"{label}: {_format_value(value)}")

    return lines or ["Metadata: none provided"]


def _normalized_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    source = normalized.get("source")
    if source and not normalized.get("source_title"):
        normalized["source_title"] = Path(str(source)).stem
    return normalized


def _format_value(value: Any) -> str:
    if isinstance(value, list | tuple):
        return " > ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={item}" for key, item in sorted(value.items()))
    return str(value)


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())
