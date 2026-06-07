"""Metadata-aware hybrid retriever for clinical RAG (ADR-004).

Implements true hybrid retrieval using Reciprocal Rank Fusion (RRF)
combined with configurable metadata boosting (especially for warnings
and other high-severity chunk types).

Design principles:
- Composition over inheritance for the underlying retrievers.
- Relevance (RRF) is the primary signal; metadata boost is a secondary
  safety/relevance adjustment, never the only ranking factor.
- All original chunk metadata is preserved for citations and downstream
  safety layers.
- Validation and observability are first-class.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, model_validator

from src.frontline_clinical_rag.core.config import settings

logger = logging.getLogger(__name__)


def _reciprocal_rank_fusion(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    k: int = 60,
) -> List[Tuple[float, Document]]:
    """Reciprocal Rank Fusion using the common 1 / (k + rank + 1) form."""
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    def _get_stable_key(doc: Document) -> str:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id is not None:
            return str(chunk_id)

        content = doc.page_content or ""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        source = doc.metadata.get("source") or doc.metadata.get("source_title", "")
        page = doc.metadata.get("page_number") or doc.metadata.get("page", "")
        return f"{source}:{page}:{content_hash}"

    for rank, doc in enumerate(dense_docs):
        key = _get_stable_key(doc)
        doc_map.setdefault(key, doc)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc in enumerate(sparse_docs):
        key = _get_stable_key(doc)
        if key not in doc_map:
            doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))
    return [(score, doc_map[key]) for key, score in sorted_items]


def _normalize_chunk_type(value: Any) -> str:
    if value is None:
        return "section"
    try:
        normalized = str(value).lower().strip().replace("-", "_")
    except Exception:
        return "section"
    if normalized in {
        "warning",
        "warnings",
        "black_box",
        "boxed_warning",
        "safety_warning",
    }:
        return "warning"
    if normalized in {"table", "table_row", "clinical_table"}:
        return "table"
    return normalized or "section"


def _normalize_warning_level(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value).lower().strip().replace(" ", "_").replace("-", "_")
    except Exception:
        return ""


class MetadataAwareHybridRetriever(BaseRetriever):
    """Hybrid retriever using RRF + metadata boosting for clinical documents.

    Defaults are sourced from core.config.settings (FRONTLINE_* env vars).
    All parameters remain overridable in the constructor for full testability
    and experimentation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_retriever: BaseRetriever = Field(...)
    bm25_retriever: BaseRetriever = Field(...)
    k_final: int = Field(default_factory=lambda: settings.retriever_k_final)
    k_dense: int = Field(default_factory=lambda: settings.retriever_k_dense)
    k_sparse: int = Field(default_factory=lambda: settings.retriever_k_sparse)
    rrf_k: int = Field(default_factory=lambda: settings.retriever_rrf_k)
    boost_factors: Dict[str, float] = Field(
        default_factory=lambda: settings.retriever_boost_factors.copy()
    )
    safety_warning_levels: List[str] = Field(
        default_factory=lambda: settings.retriever_safety_warning_levels.copy()
    )
    safety_query_terms: List[str] = Field(
        default_factory=lambda: settings.retriever_safety_query_terms.copy()
    )
    enable_metadata_filter: bool = Field(True)

    @model_validator(mode="after")
    def _validate_and_normalize(self) -> MetadataAwareHybridRetriever:
        normalized_boosts: Dict[str, float] = {}
        for raw_key, value in self.boost_factors.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"boost_factors values must be numeric, got {type(value)}"
                )
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"boost_factors must be positive finite numbers, got {value}"
                )
            norm_key = _normalize_chunk_type(raw_key)
            normalized_boosts[norm_key] = float(value)
        self.boost_factors = normalized_boosts
        self.safety_warning_levels = [
            normalized
            for item in self.safety_warning_levels
            if (normalized := _normalize_warning_level(item))
        ]
        self.safety_query_terms = [
            normalized
            for item in self.safety_query_terms
            if (normalized := str(item).lower().strip())
        ]
        return self

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[Any] = None, **kwargs: Any
    ) -> List[Document]:
        """Execute hybrid retrieval."""
        query_text = str(query or "")
        query_lower = query_text.lower()

        original_bm25_k = getattr(self.bm25_retriever, "k", None)
        config = (
            {"callbacks": getattr(run_manager, "handlers", None)}
            if run_manager
            else None
        )

        # Dense retrieval with fallback
        try:
            dense_docs: List[Document] = self.vector_retriever.invoke(
                query_text, config=config, **kwargs
            )
            dense_docs = list(dense_docs or [])[: self.k_dense]
        except Exception as exc:
            logger.warning(
                "Dense retriever failed: %s. Continuing with sparse only.", exc
            )
            dense_docs = []

        # Sparse retrieval with exception handling + safe k mutation
        sparse_docs: List[Document] = []
        try:
            if original_bm25_k is not None:
                self.bm25_retriever.k = self.k_sparse
            sparse_docs = self.bm25_retriever.invoke(
                query_text, config=config, **kwargs
            )
            sparse_docs = list(sparse_docs or [])
        except Exception as exc:
            logger.warning(
                "Sparse retriever failed: %s. Continuing with dense only.", exc
            )
            sparse_docs = []
        finally:
            if original_bm25_k is not None:
                self.bm25_retriever.k = original_bm25_k

        # RRF + boost
        rrf_results = _reciprocal_rank_fusion(dense_docs, sparse_docs, k=self.rrf_k)

        scored: List[Tuple[float, Document]] = []
        for rrf_score, doc in rrf_results:
            chunk_type = _normalize_chunk_type(doc.metadata.get("chunk_type"))
            boost = self.boost_factors.get(chunk_type, 1.0)
            final_score = rrf_score * boost

            if self.enable_metadata_filter:
                wl = _normalize_warning_level(doc.metadata.get("warning_level"))
                if wl in self.safety_warning_levels and not any(
                    term in query_lower for term in self.safety_query_terms
                ):
                    final_score *= 0.55

            new_metadata = {**doc.metadata, "retrieval_score": round(final_score, 6)}
            new_doc = Document(page_content=doc.page_content, metadata=new_metadata)

            scored.append((final_score, new_doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[: self.k_final]]
