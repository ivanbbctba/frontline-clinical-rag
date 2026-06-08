"""Unit tests for MetadataAwareHybridRetriever (ADR-004)."""

from typing import Any, List

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.frontline_clinical_rag.retrieval.hybrid_retriever import (
    MetadataAwareHybridRetriever,
)


class FakeRetriever(BaseRetriever):
    """Lightweight test double that properly inherits from BaseRetriever."""

    def __init__(self, documents: List[Document]):
        super().__init__()
        self._documents = documents

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        return self._documents


def make_doc(
    content: str,
    chunk_id: str | int = None,
    chunk_type: str = "section",
    warning_level: str = None,
    page: int = 1,
) -> Document:
    """Helper to create realistic Document objects with clinical metadata."""
    metadata = {
        "chunk_id": chunk_id,
        "chunk_type": chunk_type,
        "warning_level": warning_level,
        "page_number": page,
        "section_hierarchy": ["Merck Manual", "Cardiology"],
    }
    return Document(page_content=content, metadata=metadata)


def test_basic_hybrid_retrieval_ranking():
    """RRF should rank documents that appear in both retrievers higher."""
    dense_docs = [
        make_doc("Heart failure symptoms", chunk_id=1, chunk_type="section"),
        make_doc("Warning about digoxin", chunk_id=2, chunk_type="warning"),
    ]
    sparse_docs = [
        make_doc("Warning about digoxin", chunk_id=2, chunk_type="warning"),
        make_doc("Beta blockers overview", chunk_id=3, chunk_type="section"),
    ]

    dense_retriever = FakeRetriever(dense_docs)
    sparse_retriever = FakeRetriever(sparse_docs)

    retriever = MetadataAwareHybridRetriever(
        vector_retriever=dense_retriever,
        bm25_retriever=sparse_retriever,
        k_final=3,
        boost_factors={"warning": 1.0},  # neutral for this test
    )

    results = retriever.invoke("heart failure treatment")

    assert len(results) == 3
    assert results[0].metadata["chunk_id"] == 2
    assert "retrieval_score" in results[0].metadata


def test_warning_boost_increases_ranking_for_near_tie():
    """Warning metadata should be a bounded prior, not override stronger evidence."""
    normal_doc = make_doc("Heart failure definition", chunk_id=10, chunk_type="section")
    warning_doc = make_doc("Black box warning", chunk_id=11, chunk_type="warning")

    dense_retriever = FakeRetriever([normal_doc, warning_doc])
    sparse_retriever = FakeRetriever([warning_doc, normal_doc])

    retriever = MetadataAwareHybridRetriever(
        vector_retriever=dense_retriever,
        bm25_retriever=sparse_retriever,
        k_final=2,
        boost_factors={"warning": 2.0},
    )

    results = retriever.invoke("digoxin")

    assert results[0].metadata["chunk_id"] == 11
    assert results[0].metadata["retrieval_score"] > results[1].metadata["retrieval_score"]


def test_warning_boost_does_not_override_stronger_hybrid_evidence():
    """Warning metadata should not beat a document supported by both retrievers."""
    normal_doc = make_doc("Heart failure definition", chunk_id=10, chunk_type="section")
    warning_doc = make_doc("Black box warning", chunk_id=11, chunk_type="warning")

    dense_retriever = FakeRetriever([normal_doc, warning_doc])
    sparse_retriever = FakeRetriever([normal_doc])

    retriever = MetadataAwareHybridRetriever(
        vector_retriever=dense_retriever,
        bm25_retriever=sparse_retriever,
        k_final=2,
        boost_factors={"warning": 1.7},
    )

    results = retriever.invoke("digoxin")

    assert results[0].metadata["chunk_id"] == 10
    assert results[0].metadata["retrieval_score"] > results[1].metadata["retrieval_score"]


def test_safety_downweight_when_no_safety_terms_in_query():
    """Black-box warnings should be down-weighted unless query mentions safety terms."""
    warning_doc = make_doc(
        "Serious cardiac risk",
        chunk_id=20,
        chunk_type="warning",
        warning_level="black_box",
    )
    normal_doc = make_doc("Standard treatment", chunk_id=21, chunk_type="section")

    dense_retriever = FakeRetriever([warning_doc, normal_doc])
    sparse_retriever = FakeRetriever([normal_doc])

    retriever = MetadataAwareHybridRetriever(
        vector_retriever=dense_retriever,
        bm25_retriever=sparse_retriever,
        k_final=2,
        enable_metadata_filter=True,
    )

    results = retriever.invoke("heart failure")

    warning_result = next(r for r in results if r.metadata["chunk_id"] == 20)
    assert warning_result.metadata["retrieval_score"] < 1.0


def test_safety_not_downweighted_when_query_has_safety_term():
    """Safety down-weighting should be skipped when query contains safety terms."""
    warning_doc = make_doc(
        "Black box warning",
        chunk_id=30,
        chunk_type="warning",
        warning_level="black_box",
    )

    dense_retriever = FakeRetriever([warning_doc])
    sparse_retriever = FakeRetriever([warning_doc])

    retriever = MetadataAwareHybridRetriever(
        vector_retriever=dense_retriever,
        bm25_retriever=sparse_retriever,
        k_final=1,
        enable_metadata_filter=True,
    )

    results = retriever.invoke("black box warning digoxin")

    assert results[0].metadata["chunk_id"] == 30
    assert results[0].metadata["retrieval_score"] > 0.8


def test_constructor_override_works():
    """Constructor parameters should override settings defaults."""
    dense_retriever = FakeRetriever([])
    sparse_retriever = FakeRetriever([])

    retriever = MetadataAwareHybridRetriever(
        vector_retriever=dense_retriever,
        bm25_retriever=sparse_retriever,
        k_final=10,
        boost_factors={"warning": 5.0},
        safety_downweight_factor=0.4,
    )

    assert retriever.k_final == 10
    assert retriever.boost_factors["warning"] == 5.0
    assert retriever.safety_downweight_factor == 0.4


def test_normalization_of_chunk_type_and_warning_level():
    """Various string formats should be normalized correctly for boosting/filtering."""
    doc = make_doc(
        "Risk information",
        chunk_id=40,
        chunk_type="Black-Box Warning",
        warning_level="Boxed Warning",
    )

    dense_retriever = FakeRetriever([doc])
    sparse_retriever = FakeRetriever([])

    retriever = MetadataAwareHybridRetriever(
        vector_retriever=dense_retriever,
        bm25_retriever=sparse_retriever,
        k_final=1,
    )

    results = retriever.invoke("test")
    assert results[0].metadata["chunk_id"] == 40