"""Retrieval package for frontline-clinical-rag (ADR-004/ADR-005).

Exports production retrievers and retrieval-layer assembly helpers.
"""

from .hierarchical_retriever import create_hierarchical_retriever
from .hybrid_retriever import MetadataAwareHybridRetriever
from .recursive_retriever import create_recursive_retriever

__all__ = [
    "MetadataAwareHybridRetriever",
    "create_hierarchical_retriever",
    "create_recursive_retriever",
]
