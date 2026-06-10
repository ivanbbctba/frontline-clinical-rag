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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import ConfigDict, Field, model_validator

from src.frontline_clinical_rag.core.config import AppConfig, get_config
from src.frontline_clinical_rag.ingestion.loader import MedicalDocumentLoader

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


def _normalize_query_for_safety_terms(value: str) -> str:
    return str(value or "").lower().strip().replace(" ", "_").replace("-", "_")


def _build_hybrid_retriever(
    config: AppConfig,
    *,
    strategy: str,
    chunker: Any,
    force_rebuild_index: bool = False,
) -> MetadataAwareHybridRetriever:
    """Assemble a strategy-specific hybrid retriever (ADR-005 retrieval layer)."""

    if not config.retrieval.use_hybrid:
        raise ValueError("ADR-005 factory requires retrieval.use_hybrid=True")

    vectorstore = _get_or_create_vectorstore(
        config,
        strategy=strategy,
        chunker=chunker,
        force_rebuild=force_rebuild_index,
    )
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.retrieval.dense_top_k},
    )
    docs = _extract_documents_from_vectorstore(vectorstore)
    if not docs:
        raise ValueError("Cannot create BM25 retriever from an empty vector store")

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = config.retrieval.sparse_top_k

    return MetadataAwareHybridRetriever(
        vector_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        k_final=config.retrieval.top_k,
        k_dense=config.retrieval.dense_top_k,
        k_sparse=config.retrieval.sparse_top_k,
        rrf_k=config.retrieval.rrf_k,
        boost_factors=config.retrieval.metadata_boosting,
        safety_warning_levels=config.retrieval.safety_warning_levels,
        safety_query_terms=config.retrieval.safety_query_terms,
        safety_downweight_factor=config.retrieval.safety_downweight_factor,
    )


def _get_or_create_vectorstore(
    config: AppConfig,
    *,
    strategy: str,
    chunker: Any,
    force_rebuild: bool = False,
) -> FAISS:
    if config.vector_store.backend != "faiss":
        raise ValueError("ADR-005 is configured for FAISS vector stores")
    if config.embedding.provider != "local":
        raise ValueError("ADR-005 is configured for local embeddings")

    embeddings = _create_embeddings(config)
    persist_dir = Path(config.vector_store.persist_directory)
    index_path = _resolve_index_path(persist_dir, strategy)

    if not force_rebuild and (index_path / "index.faiss").exists():
        return FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )

    pdfs = list(Path(config.raw_data_path).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {config.raw_data_path}")

    loader = MedicalDocumentLoader(config)
    loader.embeddings = embeddings
    loader.embedding_model_name = config.embedding.model_name
    loader.raw_data_path = config.raw_data_path
    loader.vector_store_path = (
        persist_dir if persist_dir.name != strategy else persist_dir.parent
    )

    return loader.create_vector_store(chunker, strategy)


def _create_embeddings(config: AppConfig) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.embedding.model_name,
        model_kwargs={"device": config.embedding.device},
    )


def _extract_documents_from_vectorstore(vectorstore: Any) -> list[Any]:
    docstore = getattr(vectorstore, "docstore", None)
    raw_docs = getattr(docstore, "_dict", {}) if docstore is not None else {}
    return list(raw_docs.values())


def _resolve_index_path(persist_dir: Path, strategy: str) -> Path:
    if (persist_dir / strategy / "index.faiss").exists():
        return persist_dir / strategy
    if persist_dir.name == strategy and (persist_dir / "index.faiss").exists():
        return persist_dir
    if (persist_dir / "index.faiss").exists():
        return persist_dir
    return persist_dir / strategy


class MetadataAwareHybridRetriever(BaseRetriever):
    """Hybrid retriever using RRF + metadata boosting for clinical documents.

    Defaults are sourced from core.config.get_config().retrieval.
    All parameters remain overridable in the constructor for full testability
    and experimentation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_retriever: BaseRetriever = Field(...)
    bm25_retriever: BaseRetriever = Field(...)
    k_final: int = Field(default_factory=lambda: get_config().retrieval.top_k)
    k_dense: int = Field(default_factory=lambda: get_config().retrieval.dense_top_k)
    k_sparse: int = Field(default_factory=lambda: get_config().retrieval.sparse_top_k)
    rrf_k: int = Field(default_factory=lambda: get_config().retrieval.rrf_k)
    boost_factors: Dict[str, float] = Field(
        default_factory=lambda: get_config().retrieval.metadata_boosting.copy()
    )
    safety_warning_levels: List[str] = Field(
        default_factory=lambda: get_config().retrieval.safety_warning_levels.copy()
    )
    safety_query_terms: List[str] = Field(
        default_factory=lambda: get_config().retrieval.safety_query_terms.copy()
    )
    safety_downweight_factor: float = Field(
        default_factory=lambda: get_config().retrieval.safety_downweight_factor
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
            if (normalized := _normalize_query_for_safety_terms(str(item)))
        ]
        if (
            isinstance(self.safety_downweight_factor, bool)
            or not isinstance(self.safety_downweight_factor, (int, float))
            or not math.isfinite(self.safety_downweight_factor)
            or self.safety_downweight_factor <= 0
        ):
            raise ValueError(
                "safety_downweight_factor must be a positive finite number"
            )
        self.safety_downweight_factor = float(self.safety_downweight_factor)
        return self

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[Any] = None, **kwargs: Any
    ) -> List[Document]:
        """Execute hybrid retrieval."""
        query_text = str(query or "")

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
        max_rrf_score = max((score for score, _ in rrf_results), default=1.0)
        query_normalized = _normalize_query_for_safety_terms(query_text)

        scored: List[Tuple[float, Document]] = []
        for rrf_score, doc in rrf_results:
            chunk_type = _normalize_chunk_type(doc.metadata.get("chunk_type"))
            boost = self.boost_factors.get(chunk_type, 1.0)
            normalized_score = rrf_score / max_rrf_score if max_rrf_score > 0 else 0.0
            final_score = normalized_score * boost

            if self.enable_metadata_filter:
                wl = _normalize_warning_level(doc.metadata.get("warning_level"))
                if wl in self.safety_warning_levels and not any(
                    term in query_normalized for term in self.safety_query_terms
                ):
                    final_score *= self.safety_downweight_factor

            new_metadata = {**doc.metadata, "retrieval_score": round(final_score, 6)}
            new_doc = Document(page_content=doc.page_content, metadata=new_metadata)

            scored.append((final_score, new_doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[: self.k_final]]
