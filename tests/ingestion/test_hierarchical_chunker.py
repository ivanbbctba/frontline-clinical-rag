import pytest
from langchain_core.documents import Document

from src.frontline_clinical_rag.ingestion.loader import (
    HierarchicalMedicalChunker,
    RecursiveMedicalChunker,
)


class TestHierarchicalMedicalChunker:
    def test_produces_chunks_with_rich_metadata(self):
        chunker = HierarchicalMedicalChunker()

        docs = [
            Document(
                page_content="Heart Failure\n\nHeart failure is a clinical syndrome...",
                metadata={"source": "test.pdf", "page": 42},
            )
        ]

        chunks = chunker.split_documents(docs)

        assert len(chunks) > 0
        chunk = chunks[0]

        # Check that rich metadata is present
        assert "chunk_id" in chunk.metadata
        assert "chunk_type" in chunk.metadata
        assert "section_hierarchy" in chunk.metadata
        assert chunk.metadata["strategy"] == "hierarchical"

    def test_falls_back_when_no_layout_data(self):
        """Test that the chunker still works when layout analysis fails"""
        chunker = HierarchicalMedicalChunker()

        docs = [
            Document(
                page_content="This is a simple document without layout information.\n\n"
                "It should still be chunked properly using the fallback method.",
                metadata={"source": "simple.pdf", "page": 1},
            )
        ]

        chunks = chunker.split_documents(docs)
        assert len(chunks) > 0
        assert chunks[0].metadata.get("strategy") == "hierarchical"

    def test_detects_warning_chunks(self):
        chunker = HierarchicalMedicalChunker()

        docs = [
            Document(
                page_content="Black Box Warning: This medication can cause severe liver damage.",
                metadata={"source": "test.pdf", "page": 15},
            )
        ]

        chunks = chunker.split_documents(docs)
        assert any(chunk.metadata.get("chunk_type") == "warning" for chunk in chunks)


class TestChunkerComparison:
    """Basic smoke test to ensure both strategies can run"""

    def test_both_chunkers_produce_output(self):
        docs = [
            Document(
                page_content="Sample medical text about diabetes management.",
                metadata={},
            )
        ]

        hierarchical = HierarchicalMedicalChunker().split_documents(docs)
        recursive = RecursiveMedicalChunker().split_documents(docs)

        assert len(hierarchical) > 0
        assert len(recursive) > 0
