"""Canonical ADR-009 clinical evaluation fixtures.

ADR-009 makes this module the single source of truth for the four Merck Manual
benchmark questions. Keeping the natural-language strings here prevents drift
between demos, tests, CLI entry points, and future reports.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ClinicalQuestionFixture(BaseModel):
    """Stable benchmark question metadata for ADR-009 reports.

    The fixture id and topic provide deterministic CSV/JSON columns while the
    question field preserves the exact natural-language prompt used by the RAG
    graph and tests.
    """

    id: str
    question: str
    topic: Literal["sepsis", "appendicitis", "alopecia_areata", "traumatic_brain_injury"]

    model_config = ConfigDict(frozen=True)


CANONICAL_MERCK_FIXTURES: tuple[ClinicalQuestionFixture, ...] = (
    ClinicalQuestionFixture(
        id="merck_sepsis_protocol",
        topic="sepsis",
        question="What is the protocol for managing sepsis in a critical care unit?",
    ),
    ClinicalQuestionFixture(
        id="merck_appendicitis_symptoms_treatment",
        topic="appendicitis",
        question=(
            "What are the common symptoms for appendicitis, and can it be cured via "
            "medicine? If not, what surgical procedure should be followed to treat it?"
        ),
    ),
    ClinicalQuestionFixture(
        id="merck_alopecia_areata_treatment_causes",
        topic="alopecia_areata",
        question=(
            "What are the effective treatments or solutions for addressing sudden "
            "patchy hair loss, commonly seen as localized bald spots on the scalp, "
            "and what could be the possible causes behind it?"
        ),
    ),
    ClinicalQuestionFixture(
        id="merck_traumatic_brain_injury_treatment",
        topic="traumatic_brain_injury",
        question=(
            "What treatments are recommended for a person who has sustained a physical "
            "injury to brain tissue, resulting in temporary or permanent impairment "
            "of brain function?"
        ),
    ),
)

CANONICAL_MERCK_QUESTIONS: tuple[str, ...] = tuple(
    fixture.question for fixture in CANONICAL_MERCK_FIXTURES
)