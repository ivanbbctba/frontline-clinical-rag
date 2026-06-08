"""Lightweight RAG pipeline assembly factory (ADR-005).

This module is deliberately thin: it wires configured components together but
does not implement retrieval scoring, prompts, generation, or safety rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.frontline_clinical_rag.core.config import AppConfig, get_config
from src.frontline_clinical_rag.ingestion.loader import (
    HierarchicalMedicalChunker, MedicalDocumentLoader)
from src.frontline_clinical_rag.retrieval.hybrid_retriever import \
    MetadataAwareHybridRetriever


def create_hybrid_retriever(
    config: AppConfig | None = None, *, force_rebuild_index: bool = False
) -> MetadataAwareHybridRetriever:
    """Create a configured metadata-aware hybrid retriever.

    Args:
        config: Optional configuration override for tests and experiments.
        force_rebuild_index: Rebuild the FAISS index from configured PDFs.
    """

    resolved_config = config or get_config()
    return _get_hybrid_retriever(resolved_config, force_rebuild=force_rebuild_index)


def _get_hybrid_retriever(
    config: AppConfig, *, force_rebuild: bool = False
) -> MetadataAwareHybridRetriever:
    if not config.retrieval.use_hybrid:
        raise ValueError("ADR-005 factory requires retrieval.use_hybrid=True")

    vectorstore = _get_or_create_vectorstore(config, force_rebuild=force_rebuild)
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
    config: AppConfig, *, force_rebuild: bool = False
) -> FAISS:
    if config.vector_store.backend != "faiss":
        raise ValueError("ADR-005 is configured for FAISS vector stores")
    if config.embedding.provider != "local":
        raise ValueError("ADR-005 is configured for local embeddings")

    embeddings = _create_embeddings(config)
    persist_dir = Path(config.vector_store.persist_directory)
    # Strategy-aware loading: check both the base dir and the strategy subfolder
    index_path = persist_dir
    if not (index_path / "index.faiss").exists():
        # Fallback to hierarchical if it's a known strategy subfolder
        if (persist_dir / "hierarchical" / "index.faiss").exists():
            index_path = persist_dir / "hierarchical"

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
    
    # Ensure loader uses the correct parent directory if we are using a strategy subfolder
    if persist_dir.name in ["hierarchical", "recursive"]:
        loader.vector_store_path = persist_dir.parent
        strategy_name = persist_dir.name
    else:
        loader.vector_store_path = persist_dir
        strategy_name = "hierarchical"
        
    chunker = HierarchicalMedicalChunker()
    return loader.create_vector_store(chunker, strategy_name)


def _create_embeddings(config: AppConfig) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.embedding.model_name,
        model_kwargs={"device": config.embedding.device},
    )


def _extract_documents_from_vectorstore(vectorstore: Any) -> list[Any]:
    docstore = getattr(vectorstore, "docstore", None)
    raw_docs = getattr(docstore, "_dict", {}) if docstore is not None else {}
    return list(raw_docs.values())
