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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from tqdm import tqdm

from src.frontline_clinical_rag.core.config import settings


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

        # Enrich every chunk with medical-specific metadata
        for i, chunk in enumerate(chunks):
            text = chunk.page_content.lower()
            chunk_type = "text"
            if "table" in text or "fig." in text or "figure" in text:
                chunk_type = "table"
            elif any(w in text for w in ["warning", "caution", "black box", "adverse"]):
                chunk_type = "warning"

            chunk.metadata.update({
                "source": chunk.metadata.get("source", "unknown.pdf"),
                "page": chunk.metadata.get("page", 0),
                "chunk_id": i,
                "chunk_type": chunk_type,
                "warning_level": "high" if chunk_type == "warning" else None,
                "section": "unknown",                     # TODO: extract real section title
                "chapter_title": "unknown",               # TODO: extract real chapter title
                "subsection": "unknown",                  # TODO: extract "Introduction", "Etiology", etc.
                "strategy": "hierarchical",
            })

        return chunks


class MedicalDocumentLoader:
    """Main orchestrator for loading, chunking and indexing medical PDFs."""

    def __init__(self):
        # Options:
        #   device="cpu"          → safest and most stable (what we use now)
        #   device="cuda"         → try this if you have ROCm / DirectML working
        #   device="mps"          → not supported on AMD Windows
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device}
        )
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

        # Minimal change: add progress bar during embedding
        vector_store = FAISS.from_documents(
            tqdm(chunks, desc="Embedding chunks", unit="chunk"),
            self.embeddings
        )

        # Persist index
        vector_store.save_local(str(self.vector_store_path))
        print(f"✅ {strategy_name.capitalize()} vector store saved to {self.vector_store_path}")

        return vector_store


# Simple entry point for manual testing / comparison
if __name__ == "__main__":
    loader = MedicalDocumentLoader()

    print("🚀 Testing both chunking strategies for comparison...")

    # Production choice
    hierarchical_store = loader.create_vector_store(
        HierarchicalMedicalChunker(), "hierarchical"
    )

    # Baseline for fair comparison
    recursive_store = loader.create_vector_store(
        RecursiveMedicalChunker(), "recursive"
    )

    print("\n🎉 Both vector stores created successfully!")
    print("You can now compare retrieval quality between the two strategies.")