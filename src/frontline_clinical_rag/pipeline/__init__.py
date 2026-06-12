"""Pipeline assembly package for frontline-clinical-rag.

Factory helpers are loaded lazily so lightweight orchestration tests can import
``pipeline.graph`` without importing retrieval backends such as PyMuPDF/FAISS.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_retriever"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from src.frontline_clinical_rag.pipeline import factory

        return getattr(factory, name)
    raise AttributeError(name)
