import pytest
from src.frontline_clinical_rag.ingestion.loader import _detect_chunk_type


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Black Box Warning: This drug can cause serious side effects.", "warning"),
        ("BOXED WARNING", "warning"),
        ("Warning: Monitor liver function", "warning"),
        ("Table 3. Laboratory Findings", "table"),
        ("Figure 2. Pathophysiology", "figure"),
        ("Fig. 1. Clinical presentation", "figure"),
        ("This is a normal paragraph about hypertension.", "text"),
        ("   CAUTION: Use with care in elderly patients", "warning"),
    ],
)
def test_detect_chunk_type(text, expected):
    assert _detect_chunk_type(text) == expected