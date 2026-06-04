import pytest
from src.frontline_clinical_rag.ingestion.loader import (
    _is_layout_heading,
    _is_probable_heading,
)


class TestIsLayoutHeading:
    def test_detects_large_bold_text_as_heading(self):
        assert _is_layout_heading("Heart Failure", size=14.0, body_size=10.0, is_bold=True) is True

    def test_rejects_small_text(self):
        assert _is_layout_heading("Heart Failure", size=9.0, body_size=10.0, is_bold=False) is False

    def test_rejects_text_with_too_many_words(self):
        # Create a string with more than 16 words
        long_text = " ".join(["word"] * 20)
        assert (
            _is_layout_heading(long_text, size=14.0, body_size=10.0, is_bold=True)
            is False
        )

    def test_rejects_sentences_ending_with_period(self):
        assert _is_layout_heading("This is not a heading.", size=14.0, body_size=10.0, is_bold=True) is False


class TestIsProbableHeading:
    def test_detects_title_case_short_text(self):
        assert _is_probable_heading("Acute Coronary Syndrome") is True

    def test_rejects_long_text(self):
        assert _is_probable_heading("This is a very long heading that exceeds reasonable length") is False

    def test_rejects_text_ending_with_punctuation(self):
        assert _is_probable_heading("Heart Failure.") is False