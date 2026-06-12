from langchain_core.documents import Document

from src.frontline_clinical_rag.safety.critic import SafetyCritic
from src.frontline_clinical_rag.safety.prompts import SAFETY_CRITIC_SYSTEM_PROMPT
from src.frontline_clinical_rag.safety.schemas import ClinicalResponse


def test_safety_critic_flags_missing_citations_disclaimer_warning_and_overconfidence():
    critic = SafetyCritic()
    context = [
        Document(
            page_content="Black box warning: monitor closely.",
            metadata={"page": 10, "section": "Anticoagulants", "warning_level": "black_box"},
        )
    ]

    feedback = critic.review("This is definitely safe for all patients and cannot cause harm.", context)

    assert feedback.citation_quality == "needs_improvement"
    assert feedback.disclaimer_sufficient is False
    assert feedback.warning_level_handled is False
    assert feedback.overconfidence_detected is True
    assert feedback.revised_disclaimer is not None


def test_safety_critic_accepts_grounded_cautious_answer():
    critic = SafetyCritic()
    context = [
        Document(
            page_content="Assess volume status and correlate clinically.",
            metadata={"page": 22, "section": "Heart Failure"},
        )
    ]
    answer = (
        "Clinical decision-support: consider assessment from Heart Failure, page 22, "
        "and correlate with independent clinical judgment."
    )

    feedback = critic.review(answer, context)

    assert feedback.citation_quality == "good"
    assert feedback.disclaimer_sufficient is True
    assert feedback.warning_level_handled is True
    assert feedback.overconfidence_detected is False
    assert feedback.revised_disclaimer is None


def test_safety_critic_improves_response_for_high_warning_context():
    critic = SafetyCritic()
    response = ClinicalResponse(
        answer="Consider anticoagulation precautions using cited context.",
        sources=[{"page": 10, "section": "Anticoagulants", "excerpt": "Monitor closely."}],
        disclaimer="Brief disclaimer.",
        warning_level_summary="No high-warning source metadata reported.",
        confidence=0.6,
        requires_human_review=False,
        uncertainty_note=None,
        key_findings_to_verify=[],
        recommended_next_steps=[],
    )
    context = [{"metadata": {"page_number": 10, "section_title": "Anticoagulants", "chunk_type": "warning"}}]

    improved = critic.improve_response(response, context)

    assert improved.requires_human_review is True
    assert improved.disclaimer == ClinicalResponse.default_disclaimer()
    assert improved.warning_level_summary == "No high-warning source metadata reported."


def test_safety_critic_builds_langgraph_ready_messages():
    critic = SafetyCritic()
    context = [{"content": "Excerpt", "metadata": {"page": 5, "section": "Sepsis"}}]

    messages = critic.build_llm_messages("Generated answer", context)

    assert messages[0] == {"role": "system", "content": SAFETY_CRITIC_SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert "Generated answer" in messages[1]["content"]
    assert "Sepsis" in messages[1]["content"]
