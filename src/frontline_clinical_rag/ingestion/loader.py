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

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import fitz
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.frontline_clinical_rag.core.config import get_config

logger = logging.getLogger(__name__)


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


def _clean_heading(text: str) -> str:
    """Normalize a possible heading while preserving readable casing."""
    heading = re.sub(r"\s+", " ", text).strip(" :-–—\t")
    return heading


def _is_probable_heading(text: str) -> bool:
    """Fallback heading detector for documents without PyMuPDF layout data."""
    heading = _clean_heading(text)
    if not heading or len(heading) > get_config().max_heading_length:
        return False

    words = heading.split()
    if len(words) > 14:
        return False
    if heading.endswith((".", ",", ";")):
        return False

    if heading.isupper() and len(words) <= 10:
        return True
    if heading.istitle() and len(words) <= 12:
        return True
    return False


def _section_metadata(source: str, page_number: int, hierarchy: List[str]) -> dict:
    """Build hierarchy-aware metadata shared by all chunks in a section."""
    source_title = Path(source).stem if source else "unknown"
    section_hierarchy = hierarchy or [source_title]
    chapter_title = section_hierarchy[1] if len(section_hierarchy) >= 2 else None
    subsection = section_hierarchy[2] if len(section_hierarchy) >= 3 else None

    return {
        "source": source,
        "source_title": source_title,
        "page": page_number,
        "page_number": page_number,
        "parent_chunk_id": ":".join(_slugify(part) for part in section_hierarchy),
        "section_hierarchy": section_hierarchy,
        "section": (
            section_hierarchy[1] if len(section_hierarchy) >= 2 else source_title
        ),
        "chapter_title": chapter_title,
        "subsection": subsection,
        "strategy": "hierarchical",
    }


def _line_text(block: dict) -> str:
    """Extract text from a PyMuPDF line/block dictionary."""
    spans = block.get("spans", [])
    return "".join(span.get("text", "") for span in spans)


def _line_size(block: dict) -> float:
    """Return the largest span font size for a PyMuPDF line/block dictionary."""
    spans = block.get("spans", [])
    return max((span.get("size", 0.0) for span in spans), default=0.0)


def _line_is_bold(block: dict) -> bool:
    """Infer boldness from PyMuPDF font names or font flags."""
    for span in block.get("spans", []):
        font = span.get("font", "").lower()
        flags = span.get("flags", 0)
        if "bold" in font or flags & 16:
            return True
    return False


def _is_layout_heading(text: str, size: float, body_size: float, is_bold: bool) -> bool:
    """Detect headings from document layout, not from hardcoded medical titles."""
    heading = _clean_heading(text)
    if not heading or len(heading) > get_config().max_heading_length:
        return False
    if heading.isdigit() or heading.count(".") >= 4:
        return False
    if ":" in heading and size < body_size + 1.5:
        return False

    words = heading.split()
    if len(words) > 16 or heading.endswith((".", ",", ";")):
        return False
    if size >= body_size + 1.5:
        return True
    if is_bold and size >= body_size and len(words) <= 12:
        return True
    if heading.isupper() and size >= body_size and len(words) <= 10:
        return True
    return False


def _heading_level(size: float, heading_sizes: List[float]) -> int:
    """Map corpus-discovered heading font sizes to hierarchy levels."""
    larger_sizes = sorted({round(value, 1) for value in heading_sizes}, reverse=True)
    for index, known_size in enumerate(larger_sizes[:3], start=1):
        if round(size, 1) >= known_size - 0.2:
            return index
    return min(len(larger_sizes), 3) or 1


def _body_font_size(font_sizes: List[float]) -> float:
    """Infer body text size from document-local font statistics."""
    rounded_sizes = [round(size, 1) for size in font_sizes if size > 0]
    if not rounded_sizes:
        return 0.0

    counts = Counter(rounded_sizes)
    highest_count = max(counts.values())
    common_sizes = [size for size, count in counts.items() if count == highest_count]
    return min(common_sizes)


class BaseMedicalChunker:
    """Abstract base for chunking strategies (ADR-002)."""

    def split_documents(self, docs: List[Document]) -> List[Document]:
        raise NotImplementedError


class RecursiveMedicalChunker(BaseMedicalChunker):
    """Simple recursive splitter — the baseline that 'everyone uses'."""

    def split_documents(self, docs: List[Document]) -> List[Document]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=get_config().chunk_size,
            chunk_overlap=get_config().chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(docs)


class HierarchicalMedicalChunker(BaseMedicalChunker):
    """
    Production-grade hierarchical chunker for medical documents.

    Uses PyMuPDF layout analysis to infer hierarchy from document-specific
    font sizes/styles instead of hardcoded clinical heading titles.
    """

    def split_documents(self, docs: List[Document]) -> List[Document]:

        section_docs = self._add_section_metadata(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=get_config().chunk_size,
            chunk_overlap=get_config().chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents(section_docs)

        # Enrich every chunk with medical-specific metadata
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown.pdf")
            page_number = chunk.metadata.get("page", 0)
            chunk_type = _detect_chunk_type(chunk.page_content)
            chunk_id = _make_chunk_id(source, page_number, i)

            chunk.metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_type": chunk_type,
                    "warning_level": "high" if chunk_type == "warning" else None,
                }
            )

        return chunks

    def _add_section_metadata(self, docs: List[Document]) -> List[Document]:
        """Split pages into section-aware documents using PyMuPDF when possible."""
        layout_docs = self._add_layout_section_metadata(docs)
        if layout_docs:
            return layout_docs
        return self._add_fallback_section_metadata(docs)

    def _add_layout_section_metadata(self, docs: List[Document]) -> List[Document]:
        """Infer hierarchy from PDF layout styles discovered by PyMuPDF."""
        section_docs: List[Document] = []
        sources = self._ordered_sources(docs)

        for source in sources:
            pdf_path = Path(source)
            if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
                logger.warning("Skipping non-PDF or missing file: %s", source)
                continue

            try:
                pdf = fitz.open(pdf_path)
            except Exception as exc:
                logger.warning("Failed to open PDF with PyMuPDF (%s): %s", source, exc)
                continue

            try:
                source_title = pdf_path.stem
                lines_by_page = self._extract_layout_lines(pdf)
                toc_by_page = self._toc_hierarchy_by_page(pdf)
                font_sizes = [line["size"] for page in lines_by_page for line in page]
                if not font_sizes:
                    logger.warning(
                        "No font size information extracted from %s — falling back to text-based hierarchy",
                        source,
                    )
                    continue

                body_size = _body_font_size(font_sizes)
                heading_sizes = [
                    line["size"]
                    for page in lines_by_page
                    for line in page
                    if _is_layout_heading(
                        line["text"], line["size"], body_size, line["bold"]
                    )
                ]
                if not heading_sizes:
                    logger.warning(
                        "No layout headings detected in %s (font/bold criteria) — will use fallback section splitter",
                        source,
                    )
                    continue

                hierarchy_by_level: Dict[int, str] = {}
                buffer: List[str] = []
                buffer_page = 0

                def current_hierarchy() -> List[str]:
                    hierarchy = [source_title]
                    for level in sorted(hierarchy_by_level):
                        hierarchy.append(hierarchy_by_level[level])
                    return hierarchy

                def flush_buffer() -> None:
                    nonlocal buffer_page
                    text = "\n".join(line for line in buffer if line.strip()).strip()
                    if not text:
                        buffer.clear()
                        return

                    hierarchy = current_hierarchy()
                    metadata = _section_metadata(source, buffer_page, hierarchy)
                    section_docs.append(Document(page_content=text, metadata=metadata))
                    buffer.clear()

                for page_number, page_lines in enumerate(lines_by_page):
                    if page_number in toc_by_page:
                        flush_buffer()
                        hierarchy_by_level = toc_by_page[page_number].copy()

                    for line in page_lines:
                        text = line["text"]
                        if _is_layout_heading(
                            text, line["size"], body_size, line["bold"]
                        ):
                            flush_buffer()
                            level = _heading_level(line["size"], heading_sizes)
                            if page_number in toc_by_page:
                                level = max(level, len(toc_by_page[page_number]) + 1)
                            hierarchy_by_level = {
                                known_level: title
                                for known_level, title in hierarchy_by_level.items()
                                if known_level < level
                            }
                            hierarchy_by_level[level] = _clean_heading(text)
                            buffer_page = page_number
                            buffer.append(text)
                            continue

                        if not buffer:
                            buffer_page = page_number
                        buffer.append(text)

                flush_buffer()
            finally:
                pdf.close()

        return section_docs

    def _extract_layout_lines(self, pdf: fitz.Document) -> List[List[dict]]:
        """Extract page lines with text, font size and bold metadata."""
        pages: List[List[dict]] = []
        for page in pdf:
            page_lines: List[dict] = []
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                for line in block.get("lines", []):
                    text = _clean_heading(_line_text(line))
                    if not text:
                        continue
                    page_lines.append(
                        {
                            "text": text,
                            "size": _line_size(line),
                            "bold": _line_is_bold(line),
                        }
                    )
            pages.append(page_lines)
        return pages

    def _toc_hierarchy_by_page(self, pdf: fitz.Document) -> Dict[int, Dict[int, str]]:
        """Build page-level hierarchy from PDF outline/bookmark metadata."""
        toc = pdf.get_toc(simple=True)
        toc_by_page: Dict[int, Dict[int, str]] = {}
        active: Dict[int, str] = {}

        for level, title, page_number in toc:
            if page_number < 1:
                continue

            active = {
                known_level: value
                for known_level, value in active.items()
                if known_level < level
            }
            active[level] = _clean_heading(title)
            toc_by_page[page_number - 1] = active.copy()

        return toc_by_page

    def _ordered_sources(self, docs: List[Document]) -> List[str]:
        """Keep PDF source order stable while de-duplicating pages from PyPDF loader."""
        sources: List[str] = []
        for doc in docs:
            source = doc.metadata.get("source")
            if source and source not in sources:
                sources.append(source)
        return sources

    def _add_fallback_section_metadata(self, docs: List[Document]) -> List[Document]:
        """Fallback splitter for already-loaded text without PDF layout metadata."""
        section_docs: List[Document] = []
        current_source = None
        current_chapter = None
        current_subsection = None

        for doc in docs:
            source = doc.metadata.get("source", "unknown.pdf")
            page_number = doc.metadata.get("page", 0)
            source_title = Path(source).stem if source else "unknown"

            if source != current_source:
                current_source = source
                current_chapter = None
                current_subsection = None

            lines = [line.rstrip() for line in doc.page_content.splitlines()]
            buffer: List[str] = []

            def flush_buffer() -> None:
                text = "\n".join(line for line in buffer if line.strip()).strip()
                if not text:
                    buffer.clear()
                    return

                hierarchy = [source_title]
                if current_chapter:
                    hierarchy.append(current_chapter)
                if current_subsection:
                    hierarchy.append(current_subsection)

                metadata = {
                    **doc.metadata,
                    **_section_metadata(source, page_number, hierarchy),
                }
                section_docs.append(Document(page_content=text, metadata=metadata))
                buffer.clear()

            for line in lines:
                heading = _clean_heading(line)
                if _is_probable_heading(heading):
                    flush_buffer()
                    if current_chapter and len(heading.split()) <= 4:
                        current_subsection = heading
                    else:
                        current_chapter = heading
                        current_subsection = None
                    buffer.append(line)
                    continue

                buffer.append(line)

            flush_buffer()

        return section_docs


class MedicalDocumentLoader:
    """Main orchestrator for loading, chunking and indexing medical PDFs."""

    def __init__(self):
        config = get_config()

        # Options:
        #   device="cpu"          → safest and most stable (what we use now)
        #   device="cuda"         → try this if you have ROCm / DirectML working
        #   device="mps"          → not supported on AMD Windows
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding.model_name,
            model_kwargs={"device": config.embedding.device},
        )
        self.embedding_model_name = config.embedding.model_name
        self.raw_data_path = config.raw_data_path
        self.vector_store_path = Path(config.vector_store.persist_directory)

    def load_pdfs(self) -> List[Document]:
        """Load all PDFs from the configured raw data directory."""

        raw_path = Path(self.raw_data_path)
        loader = PyPDFDirectoryLoader(str(raw_path))
        docs = loader.load()
        print(f"📄 Loaded {len(docs)} PDF documents from {raw_path}")
        return docs

    def create_vector_store(
        self, chunker: BaseMedicalChunker, strategy_name: str = "hierarchical"
    ) -> FAISS:
        """Full pipeline: load → chunk → embed → persist FAISS index."""

        print(f"\n=== Starting {strategy_name.upper()} strategy ===")

        docs = self.load_pdfs()
        chunks = chunker.split_documents(docs)

        print(f"   → Created {len(chunks)} chunks")
        print(f"   → Embedding with {self.embedding_model_name}...")

        vector_store = FAISS.from_documents(
            tqdm(chunks, desc="Embedding chunks", unit="chunk"), self.embeddings
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
    recursive_store = loader.create_vector_store(RecursiveMedicalChunker(), "recursive")

    print("\n🎉 Both vector stores created successfully!")
    print("You can now compare retrieval quality between the two strategies.")
