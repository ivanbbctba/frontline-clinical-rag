# Retrieval

This module implements the **Retrieval Layer** of the clinical RAG pipeline. It is responsible for fetching relevant medical context from the vector store using a hybrid retrieval approach, while supporting two different chunking/retrieval strategies.

## Purpose

The retrieval layer combines dense vector search with sparse (BM25) retrieval using **Reciprocal Rank Fusion (RRF)** and configurable **metadata boosting**. This design improves relevance, especially for clinical documents that contain important metadata such as `warning_level`, section hierarchy, and page numbers.

## Key Concepts

- **Hybrid Retrieval**: Combines semantic (dense) embeddings with keyword-based (sparse) search for better recall.
- **Reciprocal Rank Fusion (RRF)**: Merges results from dense and sparse retrievers into a single ranked list.
- **Metadata Boosting**: Increases the score of chunks that match specific metadata fields (e.g. boosting chunks with `warning_level = black_box`).
- **Strategy-based Retrieval**: Supports two distinct chunking and retrieval strategies.

## File Overview

| File                        | Responsibility |
|----------------------------|----------------|
| `hybrid_retriever.py`      | Core hybrid retrieval logic. Combines dense + sparse search with RRF and metadata boosting. |
| `hierarchical_retriever.py`| Implements retrieval using the hierarchical chunking strategy (layout-aware, preserves document structure). |
| `recursive_retriever.py`   | Implements retrieval using the recursive chunking strategy (standard recursive splitting). |
| `__init__.py`              | Module exports and factory access. |

## Available Strategies

The module supports two retrieval strategies, both accessible via the factory in `pipeline/factory.py`:

| Strategy      | Description | Best For | Trade-offs |
|---------------|-------------|----------|----------|
| **Hierarchical** | Uses layout-aware chunking that preserves document hierarchy (sections, subsections). Rich metadata is attached during ingestion. | Clinical documents with clear structure (e.g. Merck Manual). Better context preservation. | Slightly more complex indexing. |
| **Recursive**    | Uses standard recursive character splitting. Simpler and more general-purpose. | Quick experiments or when document structure is less important. | Loses some hierarchical context. |

Both strategies can be selected via configuration (`RetrievalConfig.strategy`).

## Design Principles

- **Config-driven**: Retrieval behavior is controlled through `AppConfig` (embedding model, vector store, hybrid weights, boosting rules, strategy, etc.).
- **Metadata-aware**: Special attention is given to clinical metadata (`warning_level`, section hierarchy, page numbers) to improve relevance and safety.
- **Separation of concerns**: Strategy-specific logic lives in dedicated files, while common hybrid logic lives in `hybrid_retriever.py`.
- **Testability**: Retrievers can be instantiated with configuration overrides for unit and integration testing.

## Integration

This module is primarily used by:
- `pipeline/factory.py` → via `create_retriever(strategy=...)`
- `pipeline/graph.py` → inside the `retrieve` node of the LangGraph

The output is a list of documents that are then passed to the Generation Layer (`generation/chain.py`).