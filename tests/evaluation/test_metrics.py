from __future__ import annotations

from src.frontline_clinical_rag.core.config import EvaluationConfig
from src.frontline_clinical_rag.evaluation.fixtures import CANONICAL_MERCK_QUESTIONS
from src.frontline_clinical_rag.evaluation.metrics import compute_deterministic_metrics
from src.frontline_clinical_rag.pipeline.graph import ClinicalRAGState, RoutingDecision
from src.frontline_clinical_rag.safety.schemas import ClinicalResponse


def _state() -> ClinicalRAGState:
    response = ClinicalResponse.from_raw_sources(
        answer=(
            "The retrieved Merck context supports this clinical answer with enough "
            "detail to require citations and deterministic safety review."
        ),
        sources=[{"page": 5, "section": "Clinical Section", "excerpt": "Grounded context."}],
        warning_level_summary="High-warning source metadata present: boxed_warning",
        confidence=0.75,
        requires_human_review=False,
    )
    return {
        "question": CANONICAL_MERCK_QUESTIONS[0],
        "documents": [
            {
                "page_content": "Grounded context.",
                "metadata": {"warning_level": "boxed_warning", "page": 5},
            }
        ],
        "generated_response": response,
        "safe_response": response,
        "assessment": {
            "confidence": 0.75,
            "requires_human_review": False,
            "has_uncertainty_signal": False,
            "source_count": 1,
            "warning_level_summary": response.warning_level_summary,
        },
        "routing_decision": RoutingDecision.HIGH_CONFIDENCE,
        "routing_history": [RoutingDecision.HIGH_CONFIDENCE.value],
        "output": response,
        "generate_answer": True,
        "node_log": [
            "validate_input",
            "retrieve",
            "generate",
            "assess_and_route",
            "format_high_confidence",
        ],
    }


def test_layer_a_metrics_are_deterministic_for_identical_state_and_config():
    config = EvaluationConfig(top_k=5, high_warning_levels={"boxed_warning"})
    state = _state()

    first = compute_deterministic_metrics(state, config, latency_ms=12.0)
    second = compute_deterministic_metrics(state, config, latency_ms=12.0)

    assert first == second
    assert first.passed is True
    assert [metric.id for metric in first.metrics] == [f"M{index}" for index in range(1, 13)]
    assert first.metrics[-1].gating is False


def test_m7_fails_when_escalation_lacks_uncertainty_fields():
    config = EvaluationConfig(top_k=5)
    state = _state()
    response = ClinicalResponse.from_raw_sources(
        answer=(
            "The retrieved context only partially supports this answer and therefore "
            "must be escalated for careful human clinical review."
        ),
        sources=[{"page": 5, "section": "Clinical Section", "excerpt": "Grounded context."}],
        confidence=0.25,
        requires_human_review=True,
    )
    state.update(
        {
            "generated_response": response,
            "safe_response": response,
            "output": response,
            "assessment": {
                "confidence": 0.25,
                "requires_human_review": True,
                "has_uncertainty_signal": False,
                "source_count": 1,
                "warning_level_summary": response.warning_level_summary,
            },
            "routing_decision": RoutingDecision.LOW_CONFIDENCE_ESCALATION,
            "node_log": [
                "validate_input",
                "retrieve",
                "generate",
                "assess_and_route",
                "handle_low_confidence_escalation",
            ],
        }
    )

    result = compute_deterministic_metrics(state, config, latency_ms=1.0)
    m7 = next(metric for metric in result.metrics if metric.id == "M7")

    assert m7.passed is False
    assert result.passed is False