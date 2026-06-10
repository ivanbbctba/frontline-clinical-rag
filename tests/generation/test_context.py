from src.frontline_clinical_rag.generation.context import \
    format_context_with_metadata


def test_format_context_with_metadata_includes_clinical_metadata():
    documents = [
        {
            "page_content": "Sepsis requires rapid recognition and source control.\nGive antibiotics.",
            "metadata": {
                "source": "data/raw/merck.pdf",
                "section_hierarchy": ["Critical Care", "Sepsis"],
                "warning_level": "boxed_warning",
                "page_number": 42,
                "chunk_type": "warning",
            },
        }
    ]

    formatted = format_context_with_metadata(documents)

    assert "[Chunk 1]" in formatted
    assert "Section Hierarchy: Critical Care > Sepsis" in formatted
    assert "Warning Level: boxed_warning" in formatted
    assert "Page Number: 42" in formatted
    assert "Source Title: merck" in formatted
    assert "Sepsis requires rapid recognition and source control. Give antibiotics." in formatted


def test_format_context_with_metadata_is_pure_for_empty_input():
    assert (
        format_context_with_metadata([])
        == "No retrieved medical context was provided."
    )