# src/frontline_clinical_rag/ingestion/loader.py
"""
Medical Document Loader for frontline-clinical-rag

Responsible for:
- Loading PDFs from data/raw/
- Applying two DIFFERENT chunking strategies (for fair comparison)
- Creating rich medical metadata on every chunk
- Persisting FAISS vector stores

Follows ADR-002: HierarchicalMedicalChunker is the production choice.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from frontline_clinical_rag.core.config import settings


class BaseMedicalChunker:
    """Abstract base for chunking strategies (ADR-002)."""

    def split_documents(self, docs: List[Document]) -> List[Document]:
        raise NotImplementedError


class RecursiveMedicalChunker(BaseMedicalChunker):
    """Simple recursive splitter — the baseline that 'everyone uses'."""

    def split_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(docs)


class HierarchicalMedicalChunker(BaseMedicalChunker):
    """
    Production-grade hierarchical chunker for medical documents (Merck-style).

    Currently adds rich medical metadata based on real Merck structure.
    TODO: Replace with true hierarchical parser (Unstructured.io or PyMuPDF layout analysis)
          to detect: Book/Part → Section → Chapter → Subsection → Table → Black-box warning.
    """

    def split_documents(self, docs: List[Document]) -> List[Document]:
        # Baseline splitter (same as recursive for now)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents(docs)

        # Enrich every chunk with realistic medical metadata (based on Merck structure)
        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata
            text = chunk.page_content.lower()

            # Detect common Merck patterns
            section = metadata.get("source", "unknown")
            page = metadata.get("page", 0)

            # Simple heuristic for chapter/subsection (can be improved later)
            chunk_type = "text"
            if "table" in text or "fig." in text or "figure" in text:
                chunk_type = "table"
            elif any(w in text for w in ["warning", "caution", "black box", "adverse"]):
                chunk_type = "warning"

            chunk.metadata.update({
                "source": metadata.get("source", "unknown.pdf"),
                "page": page,
                "chunk_id": i,
                "chunk_type": chunk_type,
                "warning_level": "high" if chunk_type == "warning" else None,
                "section": section,  # e.g., "1 - Nutritional Disorders"
                "chapter_title": "unknown",  # TODO: extract real chapter title
                "subsection": "unknown",  # TODO: extract "Introduction", "Etiology", etc.
                "strategy": "hierarchical",
                # Future: part_number, chapter_number, has_table, has_figure, etc.
            })

        return chunks


class MedicalDocumentLoader:
    """Main orchestrator for loading, chunking and indexing medical PDFs."""

    def __init__(self):
        self.embeddings = SentenceTransformer(settings.embedding_model)
        self.vector_store_path = settings.faiss_index_path

    def load_pdfs(self) -> List[Document]:
        """Load all PDFs from data/raw/."""
        loader = PyPDFDirectoryLoader(str(Path("data/raw")))
        docs = loader.load()
        print(f"📄 Loaded {len(docs)} PDF documents from data/raw/")
        return docs

    def create_vector_store(
            self,
            chunker: BaseMedicalChunker,
            strategy_name: str = "hierarchical"
    ) -> FAISS:
        """Full pipeline: load → chunk → embed → persist FAISS index."""
        print(f"\n=== Starting {strategy_name.upper()} strategy ===")

        docs = self.load_pdfs()
        chunks = chunker.split_documents(docs)

        print(f"   → Created {len(chunks)} chunks")
        print(f"   → Embedding with {settings.embedding_model}...")

        vector_store = FAISS.from_documents(chunks, self.embeddings)

        vector_store.save_local(str(self.vector_store_path))
        print(f"✅ {strategy_name.capitalize()} vector store saved to {self.vector_store_path}")

        return vector_store


# Simple entry point for manual testing / comparison
if __name__ == "__main__":
    loader = MedicalDocumentLoader()

    print("🚀 Testing both chunking strategies for comparison...")

    # Production choice (rich metadata)
    hierarchical_store = loader.create_vector_store(
        HierarchicalMedicalChunker(), "hierarchical"
    )

    # Simple baseline for fair comparison
    recursive_store = loader.create_vector_store(
        RecursiveMedicalChunker(), "recursive"
    )

    print("\n🎉 Both vector stores created successfully!")
    print("You can now compare retrieval quality between the two strategies.")
