# Ingestion Package

Production document ingestion for `frontline-clinical-rag`.

This package turns large clinical PDFs into citation-ready, metadata-rich chunks and persists them as FAISS indexes for downstream hybrid retrieval. It is intentionally small, explicit, and ADR-governed: ingestion owns document loading, chunking, embedding, and index persistence; retrieval owns ranking and query-time behavior.

## What ingestion does

The ingestion flow is:

```text
PDF files in data/raw/
    ↓
PyPDFDirectoryLoader page loading
    ↓
Chunking strategy
    ├── HierarchicalMedicalChunker  ← production strategy
    └── RecursiveMedicalChunker     ← baseline strategy
    ↓
Local BAAI/bge-m3 embeddings
    ↓
Persisted FAISS index under data/vector_store/faiss_index/<strategy>/
```

The production path is `HierarchicalMedicalChunker`, documented by `docs/adr/ADR-002.md`. It preserves medical document structure before recursive text splitting, so chunks carry the clinical context required for safer retrieval, citations, and evaluation.

## Package files

| File | Responsibility |
|------|----------------|
| `loader.py` | Loader, chunking strategies, metadata enrichment, and FAISS persistence. |
| `__init__.py` | Package boundary for ingestion imports. |

Important public classes:

| Class | Purpose |
|-------|---------|
| `MedicalDocumentLoader` | Orchestrates `load → chunk → embed → save`. |
| `HierarchicalMedicalChunker` | Production chunker that uses PyMuPDF layout/TOC signals and fallback heading detection. |
| `RecursiveMedicalChunker` | Baseline splitter for fair comparison against the production strategy. |
| `BaseMedicalChunker` | Minimal interface shared by chunking strategies. |

## ADR alignment

| ADR | Ingestion impact |
|-----|------------------|
| `ADR-002` | Chooses hierarchical chunking as the production strategy and defines required metadata such as `section_hierarchy`, `page_number`, `chunk_id`, and `parent_chunk_id`. |
| `ADR-003` | Adds ingestion observability and tunability: warnings for layout fallback cases plus externalized settings for raw data path and heading thresholds. |
| `ADR-004` | Uses ingestion metadata as a retrieval signal for hybrid ranking, warning boosts, and source-cited responses. |
| `ADR-005` | Keeps assembly in the pipeline/retrieval factory while ingestion remains focused on document preparation and index creation. |
| `ADR-009` | Depends on stable chunks and metadata for deterministic retrieval/evaluation fixtures. |

## Configuration

Copy the environment template first:

```bash
cp env.example .env
```

On Windows PowerShell, use:

```powershell
Copy-Item env.example .env
```

Ingestion-relevant settings are defined in `src/frontline_clinical_rag/core/config.py` and mirrored in `env.example`.

| Setting | Default | Why it matters |
|---------|---------|----------------|
| `FRONTLINE_RAW_DATA_PATH` | `data/raw` | Directory scanned for `*.pdf` files. |
| `FRONTLINE_MERCK_PDF_PATH` | `data/raw/Merk medical_diagnosis_manual-1-1999.pdf` | Canonical Merck PDF path used by the application config. |
| `FRONTLINE_CHUNK_SIZE` | `800` | Target chunk size used by both chunking strategies. |
| `FRONTLINE_CHUNK_OVERLAP` | `200` | Preserves context across section boundaries. |
| `FRONTLINE_MAX_HEADING_LENGTH` | `140` | Caps fallback heading detection to avoid false positives. |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Local embedding model used for FAISS vectors. |
| `EMBEDDING_DEVICE` | `cpu` | Use `cpu` for portability, `cuda` for supported NVIDIA GPU setups, or `mps` on Apple Silicon. |
| `VECTOR_STORE_PATH` | `data/vector_store/faiss_index` | Base directory for persisted indexes. |
| `RETRIEVER_STRATEGY` | `hierarchical` | Selects `hierarchical` or `recursive` retriever assembly. |
| `RETRIEVER_FORCE_REBUILD_INDEX` | `false` | Rebuilds the selected index instead of loading an existing one. |

Keep clinical PDFs and generated vector stores out of git. The repository is designed for local, reproducible ingestion rather than committing large/private artifacts.

## Commands

Run all commands from the project root.

### Install dependencies

```bash
pipenv install
pipenv shell
```

### Verify ingestion configuration

```bash
python -c "from src.frontline_clinical_rag.core.config import get_config; c=get_config(); print(c.raw_data_path); print(c.vector_store.persist_directory); print(c.retrieval.strategy)"
```

PowerShell-friendly multi-line version:

```powershell
python -c "from src.frontline_clinical_rag.core.config import get_config; c=get_config(); print(c.raw_data_path); print(c.vector_store.persist_directory); print(c.retrieval.strategy)"
```

### Build both indexes directly

Use this when you want to compare the production hierarchical strategy against the recursive baseline:

```bash
python -m src.frontline_clinical_rag.ingestion.loader
```

Expected outputs:

```text
data/vector_store/faiss_index/hierarchical/index.faiss
data/vector_store/faiss_index/hierarchical/index.pkl
data/vector_store/faiss_index/recursive/index.faiss
data/vector_store/faiss_index/recursive/index.pkl
```

### Build only the configured retriever index

The normal application path builds or loads the configured strategy through the ADR-005 factory:

```bash
python -c "from src.frontline_clinical_rag.pipeline.factory import create_retriever; create_retriever(); print('retriever ready')"
```

To force a rebuild for the configured strategy, set `RETRIEVER_FORCE_REBUILD_INDEX=true` in `.env`, then run:

```bash
python -c "from src.frontline_clinical_rag.pipeline.factory import create_retriever; create_retriever(); print('rebuilt retriever ready')"
```

PowerShell one-off without editing `.env`:

```powershell
$env:RETRIEVER_FORCE_REBUILD_INDEX="true"; python -c "from src.frontline_clinical_rag.pipeline.factory import create_retriever; create_retriever(); print('rebuilt retriever ready')"
```

### Switch strategies

Set one of these in `.env`:

```dotenv
RETRIEVER_STRATEGY=hierarchical
```

or:

```dotenv
RETRIEVER_STRATEGY=recursive
```

Then run:

```bash
python -c "from src.frontline_clinical_rag.pipeline.factory import create_retriever; r=create_retriever(); print(type(r).__name__)"
```

### Run the full graph demo after ingestion

```bash
python run.py
```

### Run deterministic evaluation after ingestion

```bash
python run.py --mode evaluation --eval-mode harness_integration
python run.py --mode evaluation --eval-mode harness_integration --compare-strategies
python run.py --mode evaluation --eval-mode live_eval --strategy hierarchical
```

## How to use the package in code

### Production hierarchical index

```python
from src.frontline_clinical_rag.ingestion.loader import (
    HierarchicalMedicalChunker,
    MedicalDocumentLoader,
)

loader = MedicalDocumentLoader()
vector_store = loader.create_vector_store(
    HierarchicalMedicalChunker(),
    strategy_name="hierarchical",
)
```

### Recursive baseline index

```python
from src.frontline_clinical_rag.ingestion.loader import (
    MedicalDocumentLoader,
    RecursiveMedicalChunker,
)

loader = MedicalDocumentLoader()
vector_store = loader.create_vector_store(
    RecursiveMedicalChunker(),
    strategy_name="recursive",
)
```

### Preferred application usage

Most callers should not instantiate ingestion directly. Use the pipeline factory so config, strategy selection, index loading, and retriever construction stay centralized:

```python
from src.frontline_clinical_rag.pipeline.factory import create_retriever

retriever = create_retriever()
docs = retriever.invoke("What are red flags for appendicitis?")
```

## Metadata contract

Hierarchical ingestion enriches chunks with metadata used by retrieval, generation, citations, safety, and evaluation.

| Metadata field | Meaning |
|----------------|---------|
| `source` | Original PDF path. |
| `source_title` | Filename stem for display and citation. |
| `page` / `page_number` | Source page number from the loader. |
| `section_hierarchy` | Ordered section path inferred from TOC/layout/fallback headings. |
| `section` | Primary section value used by retrieval and citation formatting. |
| `chapter_title` | Chapter-level hierarchy entry when available. |
| `subsection` | Subsection-level hierarchy entry when available. |
| `parent_chunk_id` | Stable hierarchy identifier shared by chunks in the same section. |
| `chunk_id` | Deterministic chunk identifier built from source, page, and chunk index. |
| `chunk_type` | `warning`, `table`, `figure`, or `text`. |
| `warning_level` | `high` when the chunk is classified as a warning, otherwise `None`. |
| `strategy` | `hierarchical` for hierarchical chunks. |

The retriever preserves this metadata so downstream layers can cite sources as:

```text
Source: <source_title> - Section: <section_hierarchy> - Page <page_number>
```

## Hierarchical chunking behavior

`HierarchicalMedicalChunker` follows a layered strategy:

1. Load page text through `PyPDFDirectoryLoader`.
2. Re-open the original PDF with PyMuPDF (`fitz`) to inspect layout.
3. Prefer embedded table-of-contents hierarchy when available.
4. Infer headings from document-specific font sizes and boldness rather than hardcoded medical titles.
5. Fall back to conservative text heading detection when layout extraction fails.
6. Split section-aware documents with `RecursiveCharacterTextSplitter` using configured chunk size and overlap.
7. Add stable chunk IDs, chunk type, warning level, and hierarchy metadata.

This design avoids the main failure mode of naive RAG ingestion: chunking a large clinical manual into text fragments that no longer know which section, warning, table, or page they came from.

## Operational guidance

Use `hierarchical` for real application behavior. Use `recursive` only for benchmark comparisons, demos, and regression checks.

Rebuild the index when any of these change:

- Source PDFs in `FRONTLINE_RAW_DATA_PATH`
- `FRONTLINE_CHUNK_SIZE`
- `FRONTLINE_CHUNK_OVERLAP`
- `FRONTLINE_MAX_HEADING_LENGTH`
- `EMBEDDING_MODEL_NAME`
- `EMBEDDING_DEVICE` when the embedding backend output can differ
- Chunking code in `loader.py`

You usually do not need to rebuild when changing only retrieval-time settings such as `RETRIEVER_K_FINAL`, `RETRIEVER_RRF_K`, or metadata boost factors.

## Troubleshooting

### `No PDF files found in data/raw`

Place the Merck Manual or other clinical PDF files under `data/raw`, or set `FRONTLINE_RAW_DATA_PATH` to the directory containing your PDFs.

### Existing index is reused when you expected a rebuild

Set `RETRIEVER_FORCE_REBUILD_INDEX=true` and rerun the factory command. The retriever loads an existing `index.faiss` unless forced to rebuild.

### PyMuPDF cannot open the PDF

The chunker logs a warning and falls back when possible. Confirm the file exists, has a `.pdf` suffix, and is not encrypted or corrupted.

### No layout headings detected

This is an expected fallback path for some PDFs. The chunker then uses conservative text-based heading detection controlled by `FRONTLINE_MAX_HEADING_LENGTH`.

### Embedding is slow on CPU

`BAAI/bge-m3` is high quality but can be slow for large manuals. Keep `EMBEDDING_DEVICE=cpu` for portability, or use `cuda` when a supported GPU stack is available.

### FAISS deserialization warning

The retriever uses LangChain FAISS local loading for indexes generated by this project. Treat persisted indexes as trusted local artifacts and do not load indexes from untrusted sources.

## Review checklist

Before relying on a newly ingested corpus, verify:

- `python -m src.frontline_clinical_rag.ingestion.loader` completes without unexpected warnings.
- Both `hierarchical` and `recursive` indexes exist if you need comparison evidence.
- Sample retrieved documents include `source_title`, `section_hierarchy`, `page_number`, and `chunk_id`.
- Warning-related chunks have `chunk_type="warning"` and `warning_level="high"` where applicable.
- `python run.py --mode evaluation --eval-mode harness_integration` passes for deterministic wiring.
- Any live evaluation result is treated as non-deterministic evidence, not a CI gate.

## Medical safety note

This package prepares source material for an educational clinical RAG system. It does not validate clinical truth by itself and must not be used as a substitute for primary sources, qualified clinical judgment, or professional medical advice.