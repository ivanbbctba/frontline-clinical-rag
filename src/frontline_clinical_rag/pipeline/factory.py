"""Thin RAG pipeline assembly factory (ADR-005).

The pipeline layer is the single entry point for deciding which retriever to
build, but it does not own retrieval implementation details. All retriever
construction is delegated to the retrieval package.
"""

from __future__ import annotations

from typing import Any

from langchain_core.retrievers import BaseRetriever

from src.frontline_clinical_rag.core.config import AppConfig, get_config
from src.frontline_clinical_rag.safety import ClinicalResponse, SafetyCritic


def create_hierarchical_retriever(
    config: AppConfig,
    *,
    force_rebuild_index: bool = False,
) -> BaseRetriever:
    """Lazily delegate hierarchical retriever construction."""

    from src.frontline_clinical_rag.retrieval.hierarchical_retriever import (
        create_hierarchical_retriever as _create_hierarchical_retriever,
    )

    return _create_hierarchical_retriever(
        config,
        force_rebuild_index=force_rebuild_index,
    )


def create_recursive_retriever(
    config: AppConfig,
    *,
    force_rebuild_index: bool = False,
) -> BaseRetriever:
    """Lazily delegate recursive retriever construction."""

    from src.frontline_clinical_rag.retrieval.recursive_retriever import (
        create_recursive_retriever as _create_recursive_retriever,
    )

    return _create_recursive_retriever(
        config,
        force_rebuild_index=force_rebuild_index,
    )


def create_retriever(config: AppConfig | None = None) -> BaseRetriever:
    """Create the configured retriever for the RAG pipeline.

    ADR-005 responsibility: this factory reads application configuration,
    selects the retrieval strategy, delegates construction to `retrieval/`, and
    returns a LangChain `BaseRetriever` to entrypoints and demos.

    ADR-006 safety is intentionally not applied here. Safety validation is a
    post-generation concern: generation pipelines should retrieve context,
    generate a structured ``ClinicalResponse``, then call ``apply_safety_layer``
    before returning the final answer.
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


# ADR-006 Safety Integration


def create_safety_critic() -> SafetyCritic:
    """Create the ADR-006 Phase 1 safety critic.

    The critic is separate from retriever construction so existing ADR-005
    retrieval-only flows remain unchanged. Generation pipelines and future
    LangGraph nodes can reuse this helper to obtain the deterministic Phase 1
    critic that validates grounding, disclaimer strength, warning metadata, and
    overconfident medical phrasing after an answer has been generated.
    """

    return SafetyCritic()


def apply_safety_layer(
    response: ClinicalResponse,
    retrieved_context: list[Any],
    *,
    critic: SafetyCritic | None = None,
) -> ClinicalResponse:
    """Apply ADR-006 post-generation validation to a clinical response.

    This helper is the intentional Phase 1 integration seam for the Safety &
    Validation Layer. It runs after retrieval and answer generation, never while
    constructing retrievers, and returns a validated ``ClinicalResponse`` with
    strengthened disclaimers, warning summaries, and human-review flags when the
    retrieved context warrants additional caution.

    Pass ``critic`` to reuse a configured critic instance across calls.

    Example:
        ```python
        retriever = create_retriever()
        retrieved_context = retriever.invoke(question)
        response = generate_clinical_answer(question, retrieved_context, llm=llm)
        safe_response = apply_safety_layer(response, retrieved_context)
        ```
    """

    safety_critic = critic or create_safety_critic()
    return safety_critic.improve_response(response, retrieved_context)
