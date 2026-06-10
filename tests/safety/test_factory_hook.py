import importlib
import sys
import types

from src.frontline_clinical_rag.safety import SafetyCritic


def test_factory_safety_hook_helpers_import_without_building_retrievers(monkeypatch):
    """Exercise ADR-006 factory helpers without importing heavy retrieval backends."""

    hierarchical = types.ModuleType("src.frontline_clinical_rag.retrieval.hierarchical_retriever")
    recursive = types.ModuleType("src.frontline_clinical_rag.retrieval.recursive_retriever")
    hierarchical.create_hierarchical_retriever = lambda *args, **kwargs: object()
    recursive.create_recursive_retriever = lambda *args, **kwargs: object()
    monkeypatch.setitem(
        sys.modules,
        "src.frontline_clinical_rag.retrieval.hierarchical_retriever",
        hierarchical,
    )
    monkeypatch.setitem(
        sys.modules,
        "src.frontline_clinical_rag.retrieval.recursive_retriever",
        recursive,
    )

    factory = importlib.import_module("src.frontline_clinical_rag.pipeline.factory")
    factory = importlib.reload(factory)
    retriever = object()

    assert isinstance(factory.create_safety_critic(), SafetyCritic)
    assert factory.apply_safety_layer(retriever) is retriever
