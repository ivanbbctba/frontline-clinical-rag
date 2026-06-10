"""Deterministic input guardrails for clinical RAG queries."""

from __future__ import annotations

import logging
import re

from src.frontline_clinical_rag.safety.prompts import INJECTION_KEYWORDS

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[\s\-_:;,.!?()\[\]{}]+")


def _normalize_text(text: str) -> str:
    """Normalize user input for low-cost prompt injection matching."""

    return _TOKEN_PATTERN.sub(" ", text.casefold()).strip()


def detect_prompt_injection(text: str | None) -> bool:
    """Return ``True`` when input contains prompt injection or jailbreak markers.

    The detector intentionally focuses on explicit instruction-hijacking phrases
    from ``INJECTION_KEYWORDS``. It does not classify clinical risk, acuity, or
    sensitive medical topics as malicious because this system is intended to help
    trained medical professionals with high-risk clinical questions.
    """

    if not text:
        return False

    normalized_text = _normalize_text(text)
    if not normalized_text:
        return False

    for keyword in INJECTION_KEYWORDS:
        normalized_keyword = _normalize_text(keyword)
        if normalized_keyword and normalized_keyword in normalized_text:
            logger.warning(
                "Prompt injection signal detected.",
                extra={"matched_keyword": keyword},
            )
            return True

    return False
