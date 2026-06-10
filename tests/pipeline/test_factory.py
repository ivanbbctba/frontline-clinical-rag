from pathlib import Path
from unittest.mock import Mock

import pytest

from src.frontline_clinical_rag.core.config import (
    AppConfig,
    EmbeddingConfig,
    LLMConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from src.frontline_clinical_rag.pipeline import factory
from src.frontline_clinical_rag.retrieval import hierarchical_retriever
from src.frontline_clinical_rag.retrieval import hybrid_retriever


@pytest.fixture
def factory_config(tmp_path):
    raw_data_path = tmp_path / "raw"
    raw_data_path.mkdir()
    (raw_data_path / "merck.pdf").write_text("pdf placeholder")

    return AppConfig(
        llm=LLMConfig(api_key="test-key"),
        embedding=EmbeddingConfig(),
        vector_store=VectorStoreConfig(
            backend="faiss",
            persist_directory=str(tmp_path / "faiss_index"),
            collection_name="test_collection",
        ),
        retrieval=RetrievalConfig(top_k=3, dense_top_k=4, sparse_top_k=5),
        raw_data_path=raw_data_path,
    )


def test_loads_existing_faiss_and_composes_hybrid_retriever(monkeypatch, factory_config):
    persist_dir = Path(factory_config.vector_store.persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)
    (persist_dir / "index.faiss").write_text("existing index")

    embeddings = Mock(name="embeddings")
    dense_retriever = Mock(name="dense_retriever")
    assembled_retriever = Mock(name="hybrid_retriever")
    bm25_retriever = Mock(name="bm25_retriever")
    docs = [Mock(name="doc")]
    vectorstore = Mock(name="vectorstore")
    vectorstore.as_retriever.return_value = dense_retriever
    vectorstore.docstore._dict = {"doc-1": docs[0]}
    faiss = Mock()
    faiss.load_local.return_value = vectorstore
    bm25 = Mock()
    bm25.from_documents.return_value = bm25_retriever

    monkeypatch.setattr(
        hybrid_retriever, "HuggingFaceEmbeddings", Mock(return_value=embeddings)
    )
    monkeypatch.setattr(hybrid_retriever, "FAISS", faiss)
    monkeypatch.setattr(hybrid_retriever, "BM25Retriever", bm25)
    monkeypatch.setattr(
        hybrid_retriever,
        "MetadataAwareHybridRetriever",
        Mock(return_value=assembled_retriever),
    )

    result = hierarchical_retriever.create_hierarchical_retriever(factory_config)

    assert result is assembled_retriever
    faiss.load_local.assert_called_once_with(
        str(persist_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    vectorstore.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": factory_config.retrieval.dense_top_k},
    )
    bm25.from_documents.assert_called_once_with(docs)
    assert bm25_retriever.k == factory_config.retrieval.sparse_top_k
    hybrid_retriever.MetadataAwareHybridRetriever.assert_called_once_with(
        vector_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        k_final=factory_config.retrieval.top_k,
        k_dense=factory_config.retrieval.dense_top_k,
        k_sparse=factory_config.retrieval.sparse_top_k,
        rrf_k=factory_config.retrieval.rrf_k,
        boost_factors=factory_config.retrieval.metadata_boosting,
        safety_warning_levels=factory_config.retrieval.safety_warning_levels,
        safety_query_terms=factory_config.retrieval.safety_query_terms,
        safety_downweight_factor=factory_config.retrieval.safety_downweight_factor,
    )


def test_builds_faiss_with_hierarchical_chunker_when_forced(monkeypatch, factory_config):
    embeddings = Mock(name="embeddings")
    vectorstore = Mock(name="vectorstore")
    loader = Mock()
    loader.create_vector_store.return_value = vectorstore
    chunker = Mock(name="hierarchical_chunker")

    monkeypatch.setattr(
        hybrid_retriever, "HuggingFaceEmbeddings", Mock(return_value=embeddings)
    )
    medical_loader = Mock(return_value=loader)
    monkeypatch.setattr(hybrid_retriever, "MedicalDocumentLoader", medical_loader)
    monkeypatch.setattr(
        hierarchical_retriever, "HierarchicalMedicalChunker", Mock(return_value=chunker)
    )

    result = hybrid_retriever._get_or_create_vectorstore(
        factory_config,
        strategy="hierarchical",
        chunker=chunker,
        force_rebuild=True,
    )

    assert result is vectorstore
    assert loader.embeddings is embeddings
    assert loader.embedding_model_name == "BAAI/bge-m3"
    assert loader.raw_data_path == factory_config.raw_data_path
    assert loader.vector_store_path == Path(
        factory_config.vector_store.persist_directory
    )
    medical_loader.assert_called_once_with(factory_config)
    loader.create_vector_store.assert_called_once_with(chunker, "hierarchical")


def test_missing_pdf_directory_raises_before_loading(monkeypatch, factory_config):
    for pdf in factory_config.raw_data_path.glob("*.pdf"):
        pdf.unlink()
    loader = Mock()

    monkeypatch.setattr(
        hybrid_retriever, "HuggingFaceEmbeddings", Mock(return_value=Mock())
    )
    monkeypatch.setattr(hybrid_retriever, "MedicalDocumentLoader", loader)

    with pytest.raises(FileNotFoundError, match="No PDF files found"):
        hybrid_retriever._get_or_create_vectorstore(
            factory_config,
            strategy="hierarchical",
            chunker=Mock(),
            force_rebuild=True,
        )

    loader.assert_not_called()


def test_create_retriever_uses_config_fallback(monkeypatch, factory_config):
    retriever = Mock(name="retriever")
    get_config = Mock(return_value=factory_config)
    create_hierarchical_retriever = Mock(return_value=retriever)

    monkeypatch.setattr(factory, "get_config", get_config)
    monkeypatch.setattr(
        factory,
        "create_hierarchical_retriever",
        create_hierarchical_retriever,
    )

    result = factory.create_retriever()

    assert result is retriever
    get_config.assert_called_once_with()
    create_hierarchical_retriever.assert_called_once_with(
        factory_config,
        force_rebuild_index=False,
    )


def test_create_retriever_delegates_recursive_strategy(monkeypatch, factory_config):
    retriever = Mock(name="retriever")
    factory_config.retrieval.strategy = "recursive"
    factory_config.retrieval.force_rebuild_index = True
    create_recursive_retriever = Mock(return_value=retriever)

    monkeypatch.setattr(
        factory,
        "create_recursive_retriever",
        create_recursive_retriever,
    )

    result = factory.create_retriever(factory_config)

    assert result is retriever
    create_recursive_retriever.assert_called_once_with(
        factory_config,
        force_rebuild_index=True,
    )


def test_default_config_matches_faiss_local_hybrid_decisions():
    config = AppConfig()

    assert config.embedding.provider == "local"
    assert config.embedding.model_name == "BAAI/bge-m3"
    assert config.embedding.dimensions == 1024
    assert config.vector_store.backend == "faiss"
    assert config.retrieval.strategy == "hierarchical"
    assert config.retrieval.use_hybrid is True