"""Recursive retriever assembly owned by the retrieval layer."""

from __future__ import annotations

from src.frontline_clinical_rag.core.config import AppConfig
from src.frontline_clinical_rag.ingestion.loader import RecursiveMedicalChunker
from src.frontline_clinical_rag.retrieval.hybrid_retriever import (
    MetadataAwareHybridRetriever, _build_hybrid_retriever)


def create_recursive_retriever(
    config: AppConfig,
    *,
    force_rebuild_index: bool = False,
) -> MetadataAwareHybridRetriever:
    """Create the configured recursive hybrid retriever."""

    return _build_hybrid_retriever(
        config,
        strategy="recursive",
        chunker=RecursiveMedicalChunker(),
        force_rebuild_index=force_rebuild_index,
    )
