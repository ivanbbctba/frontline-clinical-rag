"""Structured clinical response schemas for the safety layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ClinicalSource(BaseModel):
    """A cited Merck Manual source supporting a clinical answer."""

    page: int | str
    section: str
    excerpt: str

    @field_validator("section", "excerpt")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        """Ensure citations contain usable section labels and excerpts."""

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source section and excerpt must be non-empty")
        return cleaned


class ClinicalResponse(BaseModel):
    """Validated final response shape for clinical decision-support output.

    The schema keeps disclaimer, citation, warning, confidence, and review flags
    explicit so downstream formatters and future LangGraph nodes can audit every
    generated answer before it is shown to a medical professional.
    """

    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    disclaimer: str = Field(
        default_factory=lambda: ClinicalResponse.default_disclaimer()
    )
    warning_level_summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool

    @classmethod
    def default_disclaimer(cls) -> str:
        """Return the standardized clinical decision-support disclaimer."""

        return (
            "Clinical decision-support only for trained medical professionals. "
            "Verify all recommendations against the cited Merck Manual content, "
            "local protocols, patient-specific factors, and independent clinical judgment. "
            "This output is not a diagnosis, treatment order, or substitute for clinician review."
        )

    @field_validator("answer", "disclaimer", "warning_level_summary")
    @classmethod
    def require_non_empty_fields(cls, value: str) -> str:
        """Reject empty clinical text fields."""

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must be non-empty")
        return cleaned

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure each citation includes page, section, and excerpt fields."""

        return [ClinicalSource.model_validate(source).model_dump() for source in value]

    @model_validator(mode="after")
    def require_sources_for_substantial_answer(self) -> ClinicalResponse:
        """Require citations when an answer contains substantive clinical content."""

        if len(self.answer.split()) >= 20 and not self.sources:
            raise ValueError("substantial clinical answers require at least one source")
        return self

    @classmethod
    def from_raw_sources(
        cls,
        *,
        answer: str,
        sources: list[dict[str, Any]],
        disclaimer: str | None = None,
        warning_level_summary: str = "No high-warning source metadata reported.",
        confidence: float = 0.0,
        requires_human_review: bool = False,
    ) -> ClinicalResponse:
        """Build a validated response from raw citation dictionaries."""

        return cls(
            answer=answer,
            sources=sources,
            disclaimer=disclaimer or cls.default_disclaimer(),
            warning_level_summary=warning_level_summary,
            confidence=confidence,
            requires_human_review=requires_human_review,
        )
