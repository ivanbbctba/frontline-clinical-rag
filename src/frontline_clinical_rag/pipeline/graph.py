"""Deterministic ADR-008 LangGraph orchestration for clinical RAG.

Data Flow:
    question -> retrieve -> generate -> assess_and_route -> conditional handler

The graph keeps orchestration explicit and observable: retrieval is delegated to
the factory-provided retriever, generation is delegated to ``generation.chain``,
and clinical validation/safety is applied inside the central ``assess_and_route``
node. Every generated answer must pass through that node before it can be
formatted for downstream callers.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, NotRequired, TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable

from src.frontline_clinical_rag.core.config import get_config
from src.frontline_clinical_rag.generation.chain import (
    ClinicalLLM,
    generate_clinical_answer,
)
from src.frontline_clinical_rag.safety.schemas import ClinicalResponse


class RoutingDecision(StrEnum):
    """Deterministic ADR-008 graph routing outcomes."""

    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    LOW_CONFIDENCE_ESCALATION = "LOW_CONFIDENCE_ESCALATION"


class ClinicalAssessment(TypedDict):
    """Lean deterministic assessment contract for graph routing."""

    confidence: float
    requires_human_review: bool
    has_uncertainty_signal: bool
    source_count: int
    warning_level_summary: str


class ClinicalRAGState(TypedDict):
    """Explicit graph state passed between ADR-008 nodes."""

    question: str
    documents: NotRequired[list[dict[str, Any]]]
    generated_response: NotRequired[ClinicalResponse]
    safe_response: NotRequired[ClinicalResponse]
    assessment: NotRequired[ClinicalAssessment]
    routing_decision: NotRequired[RoutingDecision]
    routing_history: NotRequired[list[str]]
    output: NotRequired[ClinicalResponse | list[dict[str, Any]]]
    generate_answer: NotRequired[bool]
    run_name: NotRequired[str]
    tags: NotRequired[list[str]]
    metadata: NotRequired[dict[str, Any]]
    node_log: NotRequired[list[str]]


GraphLogger = Callable[[str], None]


def build_clinical_rag_graph(
    *,
    retriever: Any | None = None,
    llm: ClinicalLLM | None = None,
    logger: GraphLogger | None = None,
):
    """Build the deterministic ADR-008 clinical RAG StateGraph."""

    if retriever is None:
        from src.frontline_clinical_rag.pipeline.factory import create_retriever

        resolved_retriever = create_retriever()
    else:
        resolved_retriever = retriever

    workflow = StateGraph(ClinicalRAGState)
    workflow.add_node("retrieve", _retrieve_node(resolved_retriever, logger))
    workflow.add_node("generate", _generate_node(llm, logger))
    workflow.add_node("assess_and_route", _assess_and_route_node(logger))
    workflow.add_node("format_high_confidence", _format_high_confidence_node(logger))
    workflow.add_node(
        "handle_low_confidence_escalation",
        _handle_low_confidence_escalation_node(logger),
    )
    workflow.add_node("format_output", _format_output_node(logger))

    workflow.set_entry_point("retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"generate": "generate", "format_output": "format_output"},
    )
    workflow.add_edge("generate", "assess_and_route")
    workflow.add_conditional_edges(
        "assess_and_route",
        _route_after_assessment,
        {
            RoutingDecision.HIGH_CONFIDENCE: "format_high_confidence",
            RoutingDecision.LOW_CONFIDENCE_ESCALATION: (
                "handle_low_confidence_escalation"
            ),
        },
    )
    workflow.add_edge("format_high_confidence", END)
    workflow.add_edge("handle_low_confidence_escalation", END)
    workflow.add_edge("format_output", END)

    return workflow.compile()


def run_clinical_rag_graph(
    question: str,
    *,
    retriever: Any | None = None,
    llm: ClinicalLLM | None = None,
    generate_answer: bool = True,
    run_name: str = "clinical_rag_graph",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    logger: GraphLogger | None = None,
) -> ClinicalRAGState:
    """Execute the deterministic ADR-008 graph for one clinical question."""

    graph = build_clinical_rag_graph(retriever=retriever, llm=llm, logger=logger)
    initial_state: ClinicalRAGState = {
        "question": question,
        "generate_answer": generate_answer,
        "run_name": run_name,
        "tags": tags or [],
        "metadata": metadata or {},
        "node_log": [],
        "routing_history": [],
    }
    return graph.invoke(initial_state)


@traceable(name="retrieve")
def _retrieve_node(retriever: Any, logger: GraphLogger | None):
    def retrieve(state: ClinicalRAGState) -> ClinicalRAGState:
        _log_transition(state, "retrieve", logger)
        raw_documents = retriever.invoke(state["question"])
        return {**state, "documents": [_document_to_dict(doc) for doc in raw_documents]}

    return retrieve


@traceable(name="generate")
def _generate_node(llm: ClinicalLLM | None, logger: GraphLogger | None):
    def generate(state: ClinicalRAGState) -> ClinicalRAGState:
        _log_transition(state, "generate", logger)
        if llm is None:
            raise ValueError("An LLM must be provided when generate_answer=True.")
        response = generate_clinical_answer(
            state["question"],
            state.get("documents", []),
            llm=llm,
            run_name=state.get("run_name", "generate_clinical_answer"),
            tags=state.get("tags", []),
            metadata={
                "strategy": "clinical_rag_phase_1",
                **state.get("metadata", {}),
            },
        )
        return {**state, "generated_response": response}

    return generate


@traceable(name="assess_and_route")
def _assess_and_route_node(logger: GraphLogger | None):
    def assess_and_route(state: ClinicalRAGState) -> ClinicalRAGState:
        _log_transition(state, "assess_and_route", logger)
        generated_response = state.get("generated_response")
        if generated_response is None:
            raise ValueError("assess_and_route requires a generated ClinicalResponse.")
        from src.frontline_clinical_rag.pipeline.factory import apply_safety_layer

        safe_response = apply_safety_layer(
            generated_response,
            state.get("documents", []),
        )
        assessment = _build_assessment(safe_response)
        routing_decision = _determine_routing_decision(
            assessment,
            low_confidence_threshold=get_config().safety.low_confidence_threshold,
        )
        routing_history = [
            *state.get("routing_history", []),
            routing_decision.value,
        ]
        return {
            **state,
            "safe_response": safe_response,
            "assessment": assessment,
            "routing_decision": routing_decision,
            "routing_history": routing_history,
        }

    return assess_and_route


@traceable(name="format_high_confidence")
def _format_high_confidence_node(logger: GraphLogger | None):
    def format_high_confidence(state: ClinicalRAGState) -> ClinicalRAGState:
        _log_transition(state, "format_high_confidence", logger)
        safe_response = state.get("safe_response")
        if safe_response is None:
            raise ValueError("Generated answers must pass through assess_and_route.")
        return {**state, "output": safe_response}

    return format_high_confidence


@traceable(name="handle_low_confidence_escalation")
def _handle_low_confidence_escalation_node(logger: GraphLogger | None):
    def handle_low_confidence_escalation(state: ClinicalRAGState) -> ClinicalRAGState:
        _log_transition(state, "handle_low_confidence_escalation", logger)
        safe_response = state.get("safe_response")
        if safe_response is None:
            raise ValueError("Generated answers must pass through assess_and_route.")
        return {**state, "output": safe_response}

    return handle_low_confidence_escalation


@traceable(name="format_output")
def _format_output_node(logger: GraphLogger | None):
    def format_output(state: ClinicalRAGState) -> ClinicalRAGState:
        _log_transition(state, "format_output", logger)
        if state.get("generate_answer", True):
            raise ValueError("Generated answers must pass through assess_and_route.")
        return {**state, "output": state.get("documents", [])}

    return format_output


def _build_assessment(response: ClinicalResponse) -> ClinicalAssessment:
    return {
        "confidence": response.confidence,
        "requires_human_review": response.requires_human_review,
        "has_uncertainty_signal": bool(
            response.uncertainty_note or response.key_findings_to_verify
        ),
        "source_count": len(response.sources),
        "warning_level_summary": response.warning_level_summary,
    }


@traceable(name="determine_routing_decision")
def _determine_routing_decision(
    assessment: ClinicalAssessment,
    *,
    low_confidence_threshold: float = 0.5,
) -> RoutingDecision:
    """Return the deterministic clinical routing decision for an assessment."""

    if (
        assessment["confidence"] < low_confidence_threshold
        or assessment["requires_human_review"]
        or assessment["has_uncertainty_signal"]
    ):
        return RoutingDecision.LOW_CONFIDENCE_ESCALATION
    return RoutingDecision.HIGH_CONFIDENCE


def _route_after_retrieve(state: ClinicalRAGState) -> str:
    return "generate" if state.get("generate_answer", True) else "format_output"


@traceable(name="route_after_assessment")
def _route_after_assessment(state: ClinicalRAGState) -> RoutingDecision:
    routing_decision = state.get("routing_decision")
    if routing_decision is None:
        raise ValueError("assess_and_route must set routing_decision.")
    return routing_decision


def save_graph_visualization(
    graph_app,
    filename: str = "clinical_rag_graph.png",
    output_dir: str = "docs",
) -> str:
    """Save a visualization of the LangGraph.

    Uses LangGraph's built-in ``draw_mermaid_png()`` when possible. Falls back
    to saving a Mermaid ``.mmd`` file if PNG rendering fails.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / filename

    try:
        png_bytes = graph_app.get_graph().draw_mermaid_png()
        target.write_bytes(png_bytes)
        return str(target)
    except Exception:
        mermaid_target = target.with_suffix(".mmd")
        mermaid_target.write_text(
            graph_app.get_graph().draw_mermaid(),
            encoding="utf-8",
        )
        return str(mermaid_target)


def _document_to_dict(document: Any) -> dict[str, Any]:
    if isinstance(document, dict):
        metadata = document.get("metadata", {})
        return {
            "page_content": document.get(
                "page_content", document.get("content", document.get("text", ""))
            ),
            "metadata": metadata if isinstance(metadata, dict) else {},
        }
    return {
        "page_content": str(getattr(document, "page_content", document)),
        "metadata": dict(getattr(document, "metadata", {}) or {}),
    }


def _log_transition(
    state: ClinicalRAGState,
    node_name: str,
    logger: GraphLogger | None,
) -> None:
    state.setdefault("node_log", []).append(node_name)
    if logger is not None:
        logger(f"[clinical-rag] node={node_name}")
