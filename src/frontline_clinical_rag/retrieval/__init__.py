"""Retrieval package for frontline-clinical-rag (ADR-004).

Exports the production hybrid retriever and factory helpers.
"""

from .hybrid_retriever import MetadataAwareHybridRetriever

__all__ = ["MetadataAwareHybridRetriever"]
