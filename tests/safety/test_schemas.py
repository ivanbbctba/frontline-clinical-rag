import pytest
from pydantic import ValidationError

from src.frontline_clinical_rag.safety.schemas import ClinicalResponse, ClinicalSource


def test_clinical_source_requires_non_empty_citation_fields():
    source = ClinicalSource(page=12, section=" Sepsis ", excerpt=" Initial treatment excerpt. ")

    assert source.page == 12
    assert source.section == "Sepsis"
    assert source.excerpt == "Initial treatment excerpt."

    with pytest.raises(ValidationError):
        ClinicalSource(page=12, section=" ", excerpt="Initial treatment excerpt.")


def test_clinical_response_adds_default_disclaimer_and_normalizes_sources():
    response = ClinicalResponse.from_raw_sources(
        answer="Consider prompt antimicrobial therapy per cited source.",
        sources=[{"page": "45", "section": " Sepsis ", "excerpt": " Treat promptly. "}],
        warning_level_summary="No high-warning source metadata reported.",
        confidence=0.72,
    )

    assert "Clinical decision-support only" in response.disclaimer
    assert response.sources == [{"page": "45", "section": "Sepsis", "excerpt": "Treat promptly."}]
    assert response.requires_human_review is False


def test_clinical_response_requires_sources_for_substantial_answers():
    substantial_answer = " ".join(["clinical"] * 20)

    with pytest.raises(ValidationError, match="substantial clinical answers require at least one source"):
        ClinicalResponse(
            answer=substantial_answer,
            sources=[],
            warning_level_summary="No warnings.",
            confidence=0.5,
            requires_human_review=False,
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_clinical_response_rejects_confidence_outside_unit_interval(confidence):
    with pytest.raises(ValidationError):
        ClinicalResponse.from_raw_sources(
            answer="Short answer.",
            sources=[],
            warning_level_summary="No warnings.",
            confidence=confidence,
        )
