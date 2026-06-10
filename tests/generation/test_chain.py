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

    response = generate_clinical_answer(
        "Appendicitis symptoms and treatment.",
        [
            {
                "page_content": "Surgery is definitive.",
                "metadata": {"section": "Appendicitis", "page": 7},
            }
        ],
        llm=llm,
        run_name="test-run",
        tags=["unit"],
        metadata={"strategy": "mocked"},
    )

    assert isinstance(response, ClinicalResponse)
    assert response.confidence == 0.8
    assert llm.calls[0]["messages"][0]["content"] == CLINICAL_SYSTEM_PROMPT
    assert llm.calls[0]["kwargs"]["run_name"] == "test-run"
    assert llm.calls[0]["kwargs"]["metadata"]["strategy"] == "mocked"


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