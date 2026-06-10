"""Thin RAG pipeline assembly factory (ADR-005).

The pipeline layer is the single entry point for deciding which retriever to
build, but it does not own retrieval implementation details. All retriever
construction is delegated to the retrieval package.
"""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever

from src.frontline_clinical_rag.core.config import AppConfig, get_config
from src.frontline_clinical_rag.retrieval.hierarchical_retriever import \
    create_hierarchical_retriever
from src.frontline_clinical_rag.retrieval.recursive_retriever import \
    create_recursive_retriever


def create_retriever(config: AppConfig | None = None) -> BaseRetriever:
    """Create the configured retriever for the RAG pipeline.

    ADR-005 responsibility: this factory reads application configuration,
    selects the retrieval strategy, delegates construction to `retrieval/`, and
    returns a LangChain `BaseRetriever` to entrypoints and demos.
    """

    resolved_config = config or get_config()

    if resolved_config.retrieval.strategy == "recursive":
        return create_recursive_retriever(
            resolved_config,
            force_rebuild_index=resolved_config.retrieval.force_rebuild_index,
        )

    return create_hierarchical_retriever(
        resolved_config,
        force_rebuild_index=resolved_config.retrieval.force_rebuild_index,
    )


def create_hybrid_retriever(
    config: AppConfig | None = None,
    *,
    strategy: str | None = None,
    force_rebuild_index: bool = False,
) -> BaseRetriever:
    """Backward-compatible thin wrapper around `create_retriever`.

    New code should configure `retrieval.strategy` and call `create_retriever`.
    """

    resolved_config = config or get_config()
    if strategy is not None:
        resolved_config.retrieval.strategy = strategy  # type: ignore[assignment]
    if force_rebuild_index:
        resolved_config.retrieval.force_rebuild_index = True
    return create_retriever(resolved_config)
