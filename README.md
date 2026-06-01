# frontline-clinical-rag

**Production-grade, privacy-first RAG assistant** for frontline healthcare workers needing fast, source-cited clinical information from structured medical references.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-orange)](https://python.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-green)](https://github.com/facebookresearch/faiss)
[![bge-m3](https://img.shields.io/badge/bge-m3-yellow)](https://huggingface.co/BAAI/bge-m3)
[![Grok API](https://img.shields.io/badge/Grok-purple)](https://x.ai/)
[![Docker](https://img.shields.io/badge/Docker-blue)](https://www.docker.com/)

## Architecture Overview
Clean `src/` architecture with clear package boundaries:
- `ingestion/` – loading + hierarchical chunking
- `retrieval/` – FAISS vector store
- `generation/` – Grok API + prompts
- `safety/` – guardrails + disclaimer
- `evaluation/` – RAGAS metrics
- `core/` – config + utilities

## Tech Stack (with rationale)

| Layer       | Technology                  | Why we chose it                     |
|-------------|-----------------------------|-------------------------------------|
| LLM         | Grok via xAI REST API       | Production REST integration         |
| Embeddings  | BAAI/bge-m3 (local)         | Best medical-domain performance     |
| Vector Store| FAISS (local)               | Fast, lightweight, local-first      |
| Framework   | LangChain                   | Modular RAG development             |
| Environment | pipenv + Python 3.11+       | Reproducible                        |
| Config      | .env + env.example          | Secure secrets                      |
| Deployment  | Docker (planned)            | Production-ready                    |

## Project Structure

```text
frontline-clinical-rag/
├── src/
│   └── frontline_clinical_rag/
│       ├── __init__.py
│       ├── core/
│       ├── ingestion/
│       ├── retrieval/
│       ├── generation/
│       ├── safety/
│       └── evaluation/
├── docs/
│   └── adr/
│       └── ADR-001.md
├── tests/
├── env.example
├── Pipfile
├── pyproject.toml
├── Dockerfile
└── README.md
```

## Medical Disclaimer
**This project is for educational and portfolio purposes only.** Not for clinical use. Always verify with qualified healthcare professionals.

Built to demonstrate production-grade RAG (LangChain, Vector DB, embeddings, safety/guardrails) in high-stakes medical domain.