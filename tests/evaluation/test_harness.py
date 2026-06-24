from __future__ import annotations

from src.frontline_clinical_rag.core.config import EvaluationConfig
from src.frontline_clinical_rag.evaluation.fixtures import CANONICAL_MERCK_QUESTIONS
from src.frontline_clinical_rag.evaluation.harness import run_evaluation


def test_harness_integration_runs_all_canonical_questions_with_mock_llm():
    config = EvaluationConfig(
        run_mode="harness_integration",
        strategies=["hierarchical"],
        top_k=5,
    )

    rows = run_evaluation(config)

    assert len(rows) == len(CANONICAL_MERCK_QUESTIONS)
    assert all(row["passed"] is True for row in rows)
    assert all(row["M10"] == "PASS" for row in rows)
    assert all(row["M11"] == "PASS" for row in rows)