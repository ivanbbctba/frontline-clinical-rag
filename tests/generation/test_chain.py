import json

from src.frontline_clinical_rag.generation.chain import generate_clinical_answer
from src.frontline_clinical_rag.generation.prompts import CLINICAL_SYSTEM_PROMPT
from src.frontline_clinical_rag.safety.schemas import ClinicalResponse


class MockLLM:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.output


def test_generate_clinical_answer_invokes_llm_and_validates_schema():
    output = json.dumps(
        {
            "answer": "Consider source-guided care. Source: Appendicitis, p. 7.",
            "sources": [
                {"page": 7, "section": "Appendicitis", "excerpt": "Surgery is definitive."}
            ],
            "disclaimer": ClinicalResponse.default_disclaimer(),
            "warning_level_summary": "None identified",
            "confidence": 0.8,
            "requires_human_review": False,
        }
    )
    llm = MockLLM(output)

    # Proper document structure
    documents = [
        {
            "page_content": "Surgery is definitive.",
            "metadata": {"section": "Appendicitis", "page": 7},
        }
    ]

    response = generate_clinical_answer(
        "Appendicitis symptoms and treatment.",
        documents,
        llm=llm,
        run_name="test-run",
        tags=["unit"],
        metadata={"strategy": "mocked"},
    )

    assert isinstance(response, ClinicalResponse)
    assert response.confidence == 0.8
    assert llm.calls[0]["messages"][0]["content"] == CLINICAL_SYSTEM_PROMPT


def test_generate_clinical_answer_returns_safe_fallback_on_parse_error():
    llm = MockLLM("not json")

    response = generate_clinical_answer(
        "TBI treatment.",
        [
            {
                "page_content": "Monitor intracranial pressure.",
                "metadata": {"section": "TBI", "page": 12},
            }
        ],
        llm=llm,
    )

    assert isinstance(response, ClinicalResponse)
    assert response.confidence == 0.0
    assert response.requires_human_review is True
    assert "could not be parsed" in response.warning_level_summary
    assert response.uncertainty_note is not None
    assert response.recommended_next_steps


def test_generate_clinical_answer_preserves_uncertainty_fields_from_llm():
    output = json.dumps(
        {
            "answer": (
                "Based on the available context, the most likely interpretation is "
                "alopecia areata. Source: Hair Disorders, p. 521."
            ),
            "sources": [
                {
                    "page": 521,
                    "section": "Hair Disorders",
                    "excerpt": "Patchy nonscarring hair loss is characteristic.",
                }
            ],
            "disclaimer": ClinicalResponse.default_disclaimer(),
            "warning_level_summary": "None identified",
            "confidence": 0.55,
            "requires_human_review": False,
            "uncertainty_note": (
                "Context describes pathophysiology but lacks a complete first-line "
                "treatment regimen."
            ),
            "key_findings_to_verify": [
                "Patient age suitability for intralesional corticosteroids",
                "Extent of scalp involvement",
            ],
            "recommended_next_steps": [
                "Dermatology referral for confirmation and management.",
                "Review current local alopecia guidelines.",
            ],
        }
    )
    llm = MockLLM(output)

    response = generate_clinical_answer(
        "Sudden patchy hair loss — likely cause and treatment?",
        [
            {
                "page_content": "Patchy nonscarring hair loss is characteristic.",
                "metadata": {"section": "Hair Disorders", "page": 521},
            }
        ],
        llm=llm,
    )

    assert response.uncertainty_note.startswith("Context describes")
    assert "Patient age suitability" in response.key_findings_to_verify[0]
    assert response.recommended_next_steps[0].startswith("Dermatology referral")
    # An LLM-declared uncertainty note must force human review on.
    assert response.requires_human_review is True