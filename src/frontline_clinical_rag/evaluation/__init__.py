"""ADR-009 deterministic evaluation harness for clinical RAG.

The package keeps Phase 1 evaluation deliberately lean: canonical fixtures live
in one place, metric computation is pure Layer A logic, and the external harness
invokes the production LangGraph rather than duplicating clinical routing or
safety behavior.
"""

from __future__ import annotations

from src.frontline_clinical_rag.core.config import EvaluationConfig
from src.frontline_clinical_rag.evaluation.fixtures import (
    CANONICAL_MERCK_FIXTURES,
    CANONICAL_MERCK_QUESTIONS,
    ClinicalQuestionFixture,
)
from src.frontline_clinical_rag.evaluation.metrics import (
    EvaluationMetricResult,
    EvaluationResult,
    compute_deterministic_metrics,
)

__all__ = [
    "CANONICAL_MERCK_FIXTURES",
    "CANONICAL_MERCK_QUESTIONS",
    "ClinicalQuestionFixture",
    "EvaluationConfig",
    "EvaluationMetricResult",
    "EvaluationResult",
    "compute_deterministic_metrics",
]
