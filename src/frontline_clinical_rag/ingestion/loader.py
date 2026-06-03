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

import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tqdm import tqdm

from src.frontline_clinical_rag.core.config import settings


def _slugify(value: str) -> str:
    """Create a stable, readable slug for metadata identifiers."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def _detect_chunk_type(text: str) -> str:
    """Detect high-signal medical chunk types without broad keyword matches."""
    lower = text.strip().lower()

    if lower.startswith(("black box warning", "boxed warning", "warning", "caution")):
        return "warning"
    if lower.startswith("table "):
        return "table"
    if lower.startswith(("figure ", "fig. ")):
        return "figure"
    return "text"


def _make_chunk_id(source: str, page_number: int, chunk_index: int) -> str:
    """Build a deterministic chunk id from stable source metadata."""
    source_title = Path(source).stem if source else "unknown"
    return f"{_slugify(source_title)}:p{page_number}:c{chunk_index:04d}"


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
            source = chunk.metadata.get("source", "unknown.pdf")
            page_number = chunk.metadata.get("page", 0)
            source_title = Path(source).stem if source else "unknown"
            section_hierarchy = [source_title]
            chunk_type = _detect_chunk_type(chunk.page_content)
            chunk_id = _make_chunk_id(source, page_number, i)

            chunk.metadata.update({
                "source": source,
                "source_title": source_title,
                "page": page_number,
                "page_number": page_number,
                "chunk_id": chunk_id,
                "parent_chunk_id": _slugify(source_title),
                "chunk_type": chunk_type,
                "warning_level": "high" if chunk_type == "warning" else None,
                "section_hierarchy": section_hierarchy,
                "section": source_title,
                "chapter_title": None,                    # TODO: extract real chapter title
                "subsection": None,                       # TODO: extract "Introduction", "Etiology", etc.
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

        vector_store = FAISS.from_documents(
            tqdm(chunks, desc="Embedding chunks", unit="chunk"),
            self.embeddings
        )

        # Persist index
        strategy_path = Path(self.vector_store_path) / strategy_name
        vector_store.save_local(str(strategy_path))
        print(f"✅ {strategy_name.capitalize()} vector store saved to {strategy_path}")

        return vector_store

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