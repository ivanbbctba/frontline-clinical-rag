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


def test_clinical_response_from_raw_sources_adds_disclaimer_and_normalizes_sources():
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
            disclaimer=ClinicalResponse.default_disclaimer(),
            warning_level_summary="No warnings.",
            confidence=0.5,
            requires_human_review=False,
            uncertainty_note=None,
            key_findings_to_verify=[],
            recommended_next_steps=[],
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


def test_clinical_response_uncertainty_fields_round_trip():
    response = ClinicalResponse.from_raw_sources(
        answer="Likely alopecia areata; see source.",
        sources=[{"page": 521, "section": "Hair Disorders", "excerpt": "Patchy hair loss."}],
        warning_level_summary="No high-warning source metadata reported.",
        confidence=0.55,
        requires_human_review=True,
        uncertainty_note="Context describes pathophysiology but not a complete treatment regimen.",
        key_findings_to_verify=[" Dosage of topical corticosteroids ", "", "Suitability for pediatric patients"],
        recommended_next_steps=["Refer to dermatology for confirmation."],
    )

    assert response.uncertainty_note.startswith("Context describes")
    assert response.key_findings_to_verify == [
        "Dosage of topical corticosteroids",
        "Suitability for pediatric patients",
    ]
    assert response.recommended_next_steps == ["Refer to dermatology for confirmation."]
    assert response.requires_human_review is True


def test_clinical_response_keeps_review_flag_explicit_when_confidence_low():
    response = ClinicalResponse.from_raw_sources(
        answer="Short hedged answer.",
        sources=[],
        warning_level_summary="No warnings.",
        confidence=0.2,
    )

    assert response.requires_human_review is False


def test_clinical_response_keeps_review_false_when_confident_and_no_uncertainty():
    response = ClinicalResponse.from_raw_sources(
        answer="Confident grounded answer.",
        sources=[{"page": 1, "section": "Section", "excerpt": "Excerpt."}],
        warning_level_summary="No warnings.",
        confidence=0.85,
    )

    assert response.requires_human_review is False
    assert response.uncertainty_note is None
    assert response.key_findings_to_verify == []
    assert response.recommended_next_steps == []


def test_clinical_response_normalizes_blank_uncertainty_note_to_none():
    response = ClinicalResponse.from_raw_sources(
        answer="Confident grounded answer.",
        sources=[{"page": 1, "section": "Section", "excerpt": "Excerpt."}],
        warning_level_summary="No warnings.",
        confidence=0.85,
        uncertainty_note="   ",
    )

    assert response.uncertainty_note is None
    assert response.requires_human_review is False
