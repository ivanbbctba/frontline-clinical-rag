import importlib
import sys
import types

from src.frontline_clinical_rag.safety import SafetyCritic
from src.frontline_clinical_rag.safety.schemas import ClinicalResponse


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

    assert isinstance(factory.create_safety_critic(), SafetyCritic)


def test_apply_safety_layer_uses_default_critic_to_improve_response(monkeypatch):
    factory = _import_factory_with_mocked_retrieval(monkeypatch)
    response = ClinicalResponse.from_raw_sources(
        answer="This is definitely safe for all patients per Anticoagulants page 10.",
        sources=[{"page": 10, "section": "Anticoagulants", "excerpt": "Monitor closely."}],
        disclaimer="Brief disclaimer.",
        warning_level_summary="No high-warning source metadata reported.",
        confidence=0.6,
    )
    context = [
        {
            "metadata": {
                "page": 10,
                "section": "Anticoagulants",
                "warning_level": "black_box",
            }
        }
    ]

    improved = factory.apply_safety_layer(response, context)

    assert improved.requires_human_review is True
    assert improved.disclaimer == ClinicalResponse.default_disclaimer()
    assert improved.warning_level_summary == "High-warning source metadata present: black_box"


def test_apply_safety_layer_accepts_custom_critic(monkeypatch):
    factory = _import_factory_with_mocked_retrieval(monkeypatch)
    response = ClinicalResponse.from_raw_sources(
        answer="Consider source-guided care.",
        sources=[{"page": 1, "section": "Care", "excerpt": "Use clinical judgment."}],
        warning_level_summary="No warnings.",
        confidence=0.5,
    )

    class TrackingCritic(SafetyCritic):
        def __init__(self):
            super().__init__()
            self.called = False

        def improve_response(self, response, retrieved_context):
            self.called = True
            return response.model_copy(update={"requires_human_review": True})

    critic = TrackingCritic()

    improved = factory.apply_safety_layer(response, [], critic=critic)

    assert critic.called is True
    assert improved.requires_human_review is True


def _import_factory_with_mocked_retrieval(monkeypatch):
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
    return importlib.reload(factory)
