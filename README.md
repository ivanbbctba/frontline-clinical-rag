# frontline-clinical-rag

**Production-grade RAG system for clinical decision support**  
*Layout-aware hierarchical chunking • Metadata-rich hybrid retrieval • ADR-governed architecture • Clinical safety by design*

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)](https://www.python.org/)  
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-1C3C3C?logo=langchain)](https://python.langchain.com/)  
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-FF6B6B)](https://github.com/facebookresearch/faiss)  
[![bge-m3](https://img.shields.io/badge/bge--m3-Embeddings-FFD93D)](https://huggingface.co/BAAI/bge-m3)  
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063)](https://docs.pydantic.dev/)  
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-Layout%20Aware-00A86B)](https://pymupdf.readthedocs.io/)  
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)  
[![xAI / Grok](https://img.shields.io/badge/xAI-Grok%20%7C%20Local%20LLMs-000000)](https://x.ai/)

---

## The Real Problem

Frontline healthcare workers and clinicians face **information overload** when consulting thousands of pages of trusted references like *The Merck Manual of Diagnosis & Therapy* (≈2,000 pages across 23 sections). In time-critical scenarios — sepsis protocols, appendicitis differentials, traumatic brain injury management, alopecia areata workups — they need **fast, precise, source-cited answers** grounded in authoritative text, not generic LLM hallucinations.

Traditional RAG approaches (flat RecursiveCharacterTextSplitter + basic vector search) lose critical structure in long, hierarchically organized medical documents. They also ignore clinical safety signals (black-box warnings, contraindications) that should influence retrieval ranking.

## Senior Solution: Production RAG with Clinical Intelligence

We built a **maintainable, testable, safety-conscious RAG pipeline** that treats medical documents as first-class structured artifacts:

- **HierarchicalMedicalChunker** (PyMuPDF + TOC analysis): Extracts true document hierarchy (`section_hierarchy` metadata) and detects clinical warning levels (`black_box`, `boxed_warning`, `has_warning`). Chunks carry rich, queryable metadata instead of plain text.
- **Metadata-Aware Hybrid Retrieval**: Dense embeddings (bge-m3 or OpenAI) + sparse BM25 + Reciprocal Rank Fusion (RRF) + configurable field boosting (warnings boosted up to 1.7×, hierarchy 1.2×). Two retrieval strategies side-by-side for comparison.
- **Config-Driven Factory Assembly** (ADR-005): Thin, explicit `pipeline/factory.py` wires everything from a single Pydantic `AppConfig`. No hidden singletons, full testability with overrides, clear dependency boundaries.
- **Clinical Safety Posture**: Metadata enables future guardrails, citation enforcement, and refusal behavior. Every answer path is designed to be auditable.
- **ADR Governance**: Five Architecture Decision Records document context, alternatives considered, trade-offs, and consequences. This is how senior engineers ship systems that teams can maintain and audit.

This project directly exercises the core competencies hiring managers screen for in AI Engineer / ML Engineer roles (see *Regra das 10 Vagas*): **Python**, **RAG**, **LangChain**, **Vector Databases**, **Embeddings**, **hybrid retrieval**, **evaluation harness**, **modular architecture**, **Pydantic config**, and **production patterns** in a high-stakes domain.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        run.py / Notebooks                       │
│              (Clinical questions • Strategy comparison)         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     pipeline/factory.py                         │
│   create_retriever(config)  →  HybridRetriever (RRF + boost)   │
│   (thin composition, config-driven, testable overrides)        │
└──────────────┬───────────────────────────────┬─────────────────┘
               │                               │
   ┌───────────▼───────────┐       ┌───────────▼───────────┐
   │   retrieval/          │       │   ingestion/          │
   │   hybrid_retriever.py │       │   loader.py           │
   │   (RRF, metadata boost)│       │   (HierarchicalMedicalChunker)│
   └───────────┬───────────┘       │   PyMuPDF + TOC       │
               │                   └───────────┬───────────┘
               │                               │
   ┌───────────▼───────────┐       ┌───────────▼───────────┐
   │   Vector Store        │       │   core/config.py      │
   │   (FAISS / Chroma)    │       │   (Pydantic v2, single source)│
   └───────────────────────┘       └───────────────────────┘
```

**Key Design Decisions & Trade-offs** (documented in ADRs):
- **Why hierarchical chunking?** Medical manuals are sequential and nested. Flat chunking destroys section context; hierarchical preserves it and adds safety metadata.
- **Why thin factory?** Keeps assembly logic in one place without over-abstracting. Retrieval logic stays pure and independently testable (ADR-005).
- **Why metadata boosting?** Clinical warnings and hierarchy are stronger signals than pure semantic similarity in medical QA.
- **Why Pydantic everywhere?** Type safety, validation, easy test overrides, and future guardrail flags in one maintainable source of truth.

## Tech Stack with Senior Rationale

| Layer              | Technology                          | Why (Trade-off / Production Thinking)                          |
|--------------------|-------------------------------------|----------------------------------------------------------------|
| Language & Types   | Python 3.11 + Pydantic v2           | Type-safe config, runtime validation, excellent DX in PyCharm  |
| Orchestration      | LangChain 0.3 (LCEL ready)          | Mature RAG primitives + future chain composability             |
| Embeddings         | BAAI/bge-m3 (local) or OpenAI       | Strong medical-domain performance; local-first privacy option  |
| Vector Database    | FAISS (default) / Chroma / Weaviate | Fast local retrieval; easy swap via config                     |
| Chunking           | Custom HierarchicalMedicalChunker + PyMuPDF | Layout-aware TOC hierarchy + clinical warning detection     |
| Retrieval          | Hybrid (dense + sparse) + RRF + metadata boost | Best of semantic + lexical; safety signals influence ranking |
| LLM                | Grok (xAI) / local Ollama (llama3.1) / OpenAI fallback | Flexible, production-ready, cost/privacy options            |
| Config             | Single Pydantic AppConfig           | One source of truth; overrides for tests & experiments         |
| Testing & Eval     | Strategy comparison harness + RAGAS (roadmap) | Reproducible A/B of retrieval strategies; future grounded metrics |
| Packaging          | pipenv + src layout                 | Reproducible environments, clean imports                       |

## Project Structure

```text
frontline-clinical-rag/
├── src/
│   └── frontline_clinical_rag/
│       ├── core/           # Pydantic AppConfig (single source of truth)
│       ├── ingestion/      # HierarchicalMedicalChunker (PyMuPDF + TOC)
│       ├── retrieval/      # Hybrid retriever, RRF, metadata boosting
│       ├── pipeline/       # Factory assembly (create_retriever)
│       ├── generation/     # (Future) LCEL chains + prompts
│       ├── safety/         # Guardrails, disclaimers, refusal logic
│       └── evaluation/     # RAGAS + clinical-specific metrics
├── data/
│   ├── raw/              # Merck Manual PDF (gitignored)
│   └── vector_store/     # Persisted FAISS indices
├── docs/
│   └── adr/              # 5 Architecture Decision Records
├── tests/                # Unit tests with config overrides
├── run.py                # End-to-end comparison (hierarchical vs recursive)
├── Pipfile / Pipfile.lock
├── env.example
└── README.md
```

## Quickstart

```bash
# 1. Environment
pipenv install
pipenv shell
cp env.example .env   # Fill OPENAI_API_KEY or LLM_LOCAL_*, FRONTLINE_MERCK_PDF_PATH

# 2. (Optional) Ingest / rebuild index
python -c "from src.frontline_clinical_rag.core.config import get_config; print(get_config().force_rebuild_index)"
# Set FRONTLINE_RETRIEVER_FORCE_REBUILD_INDEX=true to rebuild

# 3. Run clinical question comparison (hierarchical vs recursive)
python run.py
```

`run.py` executes the four canonical questions from the Merck Manual use-case and prints retrieved chunks with full `section_hierarchy` and page provenance for both strategies. This is your living benchmark.

## Evaluation & Comparison Harness

The project ships with a built-in strategy comparator:

```python
for strategy in ("hierarchical", "recursive"):
    retriever = create_retriever(strategy_config)
    docs = retriever.invoke(question)
    # prints Source: title - Section: hierarchy - Page X
```

**Why this matters**: Hierarchical + metadata boosting consistently surfaces more contextually relevant and safety-aware passages for long-form medical documents. The harness makes this visible and reproducible — exactly the kind of engineering rigor expected in production RAG systems.

## Architecture Decision Records (ADRs)

Transparent senior decision-making is a first-class deliverable:

| ADR     | Focus                                      | Key Outcome                              |
|---------|--------------------------------------------|------------------------------------------|
| ADR-001 | Initial core setup & package boundaries    | Clean src/ layout, config foundation     |
| ADR-002 | Hierarchical chunking & clinical metadata  | section_hierarchy + warning_level flags  |
| ADR-003 | (Intermediate)                             | -                                        |
| ADR-004 | Metadata-aware hybrid retrieval + RRF      | Boosting rules, hybrid composition       |
| ADR-005 | Lightweight pipeline factory               | Thin assembly, testability, safety posture |

All ADRs live in `docs/adr/`. Reading them shows how we evaluate alternatives, document trade-offs, and protect long-term maintainability — a hallmark of Staff-level engineering.

## Production Readiness & Roadmap

**Current strengths**:
- Fully config-driven and reproducible
- Clear dependency boundaries (no circular imports)
- Rich clinical metadata for future safety layers
- Comparison harness for continuous retrieval quality

**Next milestones** (tracked via ADRs/issues):
- Generation layer with Grok / local LLMs + source citation enforcement
- Full evaluation harness (RAGAS + clinical metrics: citation recall, warning grounding, refusal rate, consistency)
- Output guardrails & refusal for out-of-scope / contradictory queries
- Docker packaging + CI/CD pipeline
- Observability (LangSmith alternative or custom tracing)
- Multi-profile pipelines (dev vs. safety-eval vs. production)

This roadmap deliberately prioritizes **safety and evaluation before full generation** — non-negotiable for any clinical decision-support system.

## Medical Disclaimer

**This is an educational and portfolio project only.**  
It is **not intended for clinical use**, diagnosis, or treatment decisions. All information retrieved must be verified against primary sources and qualified medical judgment. The authors and contributors accept no liability for any clinical decisions made based on this system.

Designed with explicit safety metadata and guardrail hooks precisely because we understand the stakes.

## Why This Repository Signals Senior AI Engineering Capability

- **End-to-end production RAG**, not a toy notebook
- **Domain adaptation** for safety-critical long documents (layout intelligence + metadata)
- **Architectural discipline** via ADRs and clean boundaries
- **Testability & reproducibility** built in from day one
- **Direct keyword alignment** with top AI Engineer job descriptions: Python, LLMs, RAG, LangChain, Vector Databases, Embeddings, Hybrid Retrieval, Evaluation, Modular Python Architecture, Pydantic, Clinical AI

If you are hiring for AI Engineer, ML Engineer, or Agentic/RAG-focused roles and value engineers who can ship maintainable systems in regulated or high-stakes domains — this project demonstrates exactly that.

---

*Built with ❤️ for clinical excellence and engineering craft. Power ahead.*
