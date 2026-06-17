"""Safety critic for post-generation clinical response validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from src.frontline_clinical_rag.safety.prompts import \
    SAFETY_CRITIC_SYSTEM_PROMPT
from src.frontline_clinical_rag.safety.schemas import ClinicalResponse

CitationQuality = Literal["good", "needs_improvement", "poor"]

_HIGH_WARNING_LEVELS = {"high", "black_box", "boxed_warning", "warning"}
_OVERCONFIDENT_PATTERNS = (
    re.compile(
        r"\b(always|never|guaranteed|definitely|certainly|cure|safe for all)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(no risk|without risk|cannot cause)\b", re.IGNORECASE),
)


class SafetyCriticFeedback(BaseModel):
    """Structured feedback produced by the Safety Critic."""

    citation_quality: CitationQuality
    disclaimer_sufficient: bool
    warning_level_handled: bool
    overconfidence_detected: bool
    suggested_improvements: str
    revised_disclaimer: str | None = None


@dataclass(slots=True)
class SafetyCritic:
    """Review clinical RAG answers for citations, disclaimers, and phrasing.

    The class exposes deterministic review behavior for Phase 1 and carries the
    full ``SAFETY_CRITIC_SYSTEM_PROMPT`` so the same interface can later delegate
    to an LLM or become a LangGraph node. The critic never refuses clinical
    questions; it returns constructive feedback and can strengthen a structured
    ``ClinicalResponse`` while preserving clinical utility for professionals.
    """

    system_prompt: str = SAFETY_CRITIC_SYSTEM_PROMPT
    high_warning_levels: set[str] = field(
        default_factory=lambda: _HIGH_WARNING_LEVELS.copy()
    )

    def review(self, answer: str, retrieved_context: list[Any]) -> SafetyCriticFeedback:
        """Return structured safety feedback for a generated answer and context."""

        sources = self._extract_sources(retrieved_context)
        high_warning_present = self._has_high_warning(retrieved_context)
        has_disclaimer = (
            "clinical judgment" in answer.casefold()
            or "decision-support" in answer.casefold()
        )
        overconfident = self._has_overconfident_language(answer)
        citation_quality = self._citation_quality(answer, sources)
        warning_handled = not high_warning_present or self._mentions_warning(answer)

        improvements: list[str] = []
        if citation_quality != "good":
            improvements.append(
                "Add explicit Merck Manual citations with page, section, and short excerpts."
            )
        if not has_disclaimer:
            improvements.append(
                "Add the standardized clinical decision-support disclaimer prominently."
            )
        if high_warning_present and not warning_handled:
            improvements.append(
                "Call out high-warning source metadata and recommend clinician review before action."
            )
        if overconfident:
            improvements.append(
                "Replace absolute phrasing with conservative language such as 'consider' and 'correlate clinically'."
            )
        if not improvements:
            improvements.append(
                "Response is appropriately grounded, cautious, and useful for a trained clinician."
            )

        return SafetyCriticFeedback(
            citation_quality=citation_quality,
            disclaimer_sufficient=has_disclaimer,
            warning_level_handled=warning_handled,
            overconfidence_detected=overconfident,
            suggested_improvements=" ".join(improvements),
            revised_disclaimer=(
                ClinicalResponse.default_disclaimer()
                if not has_disclaimer or high_warning_present
                else None
            ),
        )

    def improve_response(
        self, response: ClinicalResponse, retrieved_context: list[Any]
    ) -> ClinicalResponse:
        """Return a safer ``ClinicalResponse`` using deterministic critic feedback."""

        feedback = self.review(response.answer, retrieved_context)
        high_warning_present = self._has_high_warning(retrieved_context)

        return response.model_copy(
            update={
                "disclaimer": feedback.revised_disclaimer or response.disclaimer,
                "requires_human_review": response.requires_human_review
                or high_warning_present
                or feedback.overconfidence_detected,
                "warning_level_summary": self._warning_level_summary(retrieved_context),
            }
        )

    def build_llm_messages(
        self, answer: str, retrieved_context: list[Any]
    ) -> list[dict[str, str]]:
        """Build future LLM critic messages without coupling Phase 1 to a model client.

        TODO(Phase 2): Wire these messages into the LangGraph safety critic node
        and parse the JSON response into ``SafetyCriticFeedback``.
        """

        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Generated answer:\n{answer}\n\nRetrieved context:\n{self._format_context(retrieved_context)}",
            },
        ]

    def _citation_quality(
        self, answer: str, sources: list[dict[str, Any]]
    ) -> CitationQuality:
        if not sources:
            return "poor"
        if any(
            str(source.get("page", "")) in answer
            or str(source.get("section", "")) in answer
            for source in sources
        ):
            return "good"
        return "needs_improvement"

    def _extract_sources(self, retrieved_context: list[Any]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for item in retrieved_context:
            metadata = self._metadata_for(item)
            sources.append(
                {
                    "page": metadata.get(
                        "page", metadata.get("page_number", "unknown")
                    ),
                    "section": metadata.get(
                        "section", metadata.get("section_title", "unknown")
                    ),
                }
            )
        return sources

    def _has_high_warning(self, retrieved_context: list[Any]) -> bool:
        for item in retrieved_context:
            metadata = self._metadata_for(item)
            warning_level = str(metadata.get("warning_level", "")).casefold()
            chunk_type = str(metadata.get("chunk_type", "")).casefold()
            if warning_level in self.high_warning_levels or chunk_type == "warning":
                return True
        return False

    def _warning_level_summary(self, retrieved_context: list[Any]) -> str:
        warnings: list[str] = []
        for item in retrieved_context:
            metadata = self._metadata_for(item)
            warning_level = metadata.get("warning_level")
            if warning_level:
                warnings.append(str(warning_level))
        return (
            "High-warning source metadata present: " + ", ".join(sorted(set(warnings)))
            if warnings
            else "No high-warning source metadata reported."
        )

    def _has_overconfident_language(self, answer: str) -> bool:
        return any(pattern.search(answer) for pattern in _OVERCONFIDENT_PATTERNS)

    def _mentions_warning(self, answer: str) -> bool:
        lowered = answer.casefold()
        return any(
            term in lowered
            for term in ("warning", "contraindication", "risk", "caution", "review")
        )

    def _format_context(self, retrieved_context: list[Any]) -> str:
        lines: list[str] = []
        for index, item in enumerate(retrieved_context, start=1):
            metadata = self._metadata_for(item)
            content = self._content_for(item)
            lines.append(f"[{index}] metadata={metadata}; excerpt={content[:700]}")
        return "\n".join(lines)

    def _metadata_for(self, item: Any) -> dict[str, Any]:
        if hasattr(item, "metadata"):
            return dict(item.metadata)
        if isinstance(item, dict):
            metadata = item.get("metadata", item)
            return dict(metadata) if isinstance(metadata, dict) else {}
        return {}

    def _content_for(self, item: Any) -> str:
        if hasattr(item, "page_content"):
            return str(item.page_content)
        if isinstance(item, dict):
            return str(item.get("page_content", item.get("content", "")))
        return str(item)
