"""LangGraph orchestration for ADR-007 Phase 1 clinical RAG.

Data Flow:
    question -> retrieve -> generate -> apply_safety -> format_output

The graph keeps orchestration thin and observable: retrieval is delegated to the
factory-provided retriever, generation is delegated to ``generation.chain``, and
clinical validation/safety is delegated to ADR-006 ``apply_safety_layer``. Phase
1 also supports retrieval-only runs; however, any state containing a generated
answer is always routed through the explicit ``apply_safety`` node before final
output formatting.
"""

from __future__ import annotations

from typing import Any, Callable, NotRequired, TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable

from src.frontline_clinical_rag.generation.chain import (
    ClinicalLLM, generate_clinical_answer)
from src.frontline_clinical_rag.safety.schemas import ClinicalResponse


class ClinicalRAGState(TypedDict):
    """Explicit graph state passed between ADR-007 Phase 1 nodes."""

    question: str
    documents: NotRequired[list[dict[str, Any]]]
    generated_response: NotRequired[ClinicalResponse]
    safe_response: NotRequired[ClinicalResponse]
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
    """Build the minimal ADR-007 Phase 1 clinical RAG StateGraph."""

    if retriever is None:
        from src.frontline_clinical_rag.pipeline.factory import \
            create_retriever

        resolved_retriever = create_retriever()
    else:
        resolved_retriever = retriever

    workflow = StateGraph(ClinicalRAGState)
    workflow.add_node("retrieve", _retrieve_node(resolved_retriever, logger))
    workflow.add_node("generate", _generate_node(llm, logger))
    workflow.add_node("apply_safety", _apply_safety_node(logger))
    workflow.add_node("format_output", _format_output_node(logger))

    workflow.set_entry_point("retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"generate": "generate", "format_output": "format_output"},
    )
    workflow.add_edge("generate", "apply_safety")
    workflow.add_edge("apply_safety", "format_output")
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
    """Execute the Phase 1 graph for one clinical question."""

    graph = build_clinical_rag_graph(retriever=retriever, llm=llm, logger=logger)
    initial_state: ClinicalRAGState = {
        "question": question,
        "generate_answer": generate_answer,
        "run_name": run_name,
        "tags": tags or [],
        "metadata": metadata or {},
        "node_log": [],
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


@traceable(name="apply_safety")
def _apply_safety_node(logger: GraphLogger | None):
    def apply_safety(state: ClinicalRAGState) -> ClinicalRAGState:
        _log_transition(state, "apply_safety", logger)
        generated_response = state.get("generated_response")
        if generated_response is None:
            raise ValueError("apply_safety requires a generated ClinicalResponse.")
        from src.frontline_clinical_rag.pipeline.factory import \
            apply_safety_layer

        safe_response = apply_safety_layer(
            generated_response,
            state.get("documents", []),
        )
        return {**state, "safe_response": safe_response}

    return apply_safety


@traceable(name="format_output")
def _format_output_node(logger: GraphLogger | None):
    def format_output(state: ClinicalRAGState) -> ClinicalRAGState:
        _log_transition(state, "format_output", logger)
        if state.get("generate_answer", True):
            output = state.get("safe_response")
            if output is None:
                raise ValueError("Generated answers must pass through apply_safety.")
            return {**state, "output": output}
        return {**state, "output": state.get("documents", [])}

    return format_output


def _route_after_retrieve(state: ClinicalRAGState) -> str:
    return "generate" if state.get("generate_answer", True) else "format_output"


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
