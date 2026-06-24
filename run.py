import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.frontline_clinical_rag.core.config import EvaluationConfig, get_config
from src.frontline_clinical_rag.evaluation.fixtures import CANONICAL_MERCK_QUESTIONS
from src.frontline_clinical_rag.evaluation.harness import (
    export_rows,
    print_table,
    run_evaluation,
)
from src.frontline_clinical_rag.evaluation.metrics import compute_deterministic_metrics
from src.frontline_clinical_rag.pipeline.factory import create_retriever
from src.frontline_clinical_rag.pipeline.graph import run_clinical_rag_graph

load_dotenv()


CLINICAL_QUESTIONS = CANONICAL_MERCK_QUESTIONS

def _build_xai_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("LLM_XAI_MODEL_NAME"),
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_XAI_BASE_URL", "https://api.x.ai/v1"),
        temperature=float(os.environ.get("FRONTLINE_TEMPERATURE", "0.0")),
        max_tokens=int(os.environ.get("FRONTLINE_MAX_TOKENS", "1024")),
    )

def _format_source(metadata: dict) -> str:
    source = metadata.get("source") or "Unknown"
    title = metadata.get("source_title") or (Path(source).stem if source != "Unknown" else "Unknown source")
    hierarchy = metadata.get("section_hierarchy") or []
    section = " > ".join(hierarchy) if isinstance(hierarchy, list) else str(hierarchy)

    page = metadata.get("page_number")
    if page is None:
        page = metadata.get("page")
        if page is not None:
            try:
                page = int(page) + 1  # Assume 0-indexed if it's 'page' from LangChain
            except (ValueError, TypeError):
                pass

    return f"Source: {title} - Section: {section or 'unknown'} - Page {page or 'unknown'}"


def main() -> None:
    """Run demos or the ADR-009 deterministic evaluation harness."""

    parser = argparse.ArgumentParser(description="Clinical RAG demo and evaluation runner.")
    parser.add_argument(
        "--mode",
        choices=["demo", "retrieval", "evaluation"],
        default="demo",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["metric_unit", "harness_integration", "live_eval"],
        default="live_eval",
    )
    parser.add_argument("--strategy", choices=["hierarchical", "recursive"], action="append")
    parser.add_argument("--compare-strategies", action="store_true")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.mode == "retrieval":
        run_retrieval_demo()
        return
    if args.mode == "demo":
        run_generation_graph_demo(_build_xai_llm(), strategy=args.strategy[0] if args.strategy else None)
        return
    run_deterministic_evaluation(
        run_mode=args.eval_mode,
        strategies=(
            ["hierarchical", "recursive"]
            if args.compare_strategies
            else args.strategy or ["hierarchical"]
        ),
        csv_path=args.csv,
        json_path=args.json,
    )


def run_deterministic_evaluation(
    *,
    run_mode: str,
    strategies: list[str],
    csv_path: Path | None = None,
    json_path: Path | None = None,
) -> None:
    """Run ADR-009 evaluation from run.py using canonical fixtures only."""

    app_config = get_config()
    config = EvaluationConfig.from_app_config(app_config, run_mode=run_mode, strategies=strategies)
    rows = run_evaluation(config, app_config=app_config, llm_factory=_build_xai_llm)
    print_table(rows)
    export_rows(rows, csv_path=csv_path, json_path=json_path)


def run_retrieval_demo() -> None:
    config = get_config()
    retrievers = {}
    for strategy in ("hierarchical", "recursive"):
        strategy_config = config.model_copy(deep=True)
        strategy_config.retrieval.strategy = strategy
        print(f"Initializing {strategy.title()} Retriever...")
        retrievers[strategy] = create_retriever(strategy_config)

    for question in CLINICAL_QUESTIONS:
        print(f"\n" + "="*100)
        print(f"QUESTION: {question}")
        print("="*100)

        for strategy, retriever in retrievers.items():
            print(f"\n[ STRATEGY: {strategy.upper()} ]")
            docs = retriever.invoke(question)
            for index, doc in enumerate(docs, start=1):
                print(f"\n  ({index}) {_format_source(doc.metadata)}")
                content = doc.page_content.strip().replace("\n", " ")
                print(f"      {content}...")


def run_generation_graph_demo(llm: Any, *, strategy: str | None = None) -> None:
    """Run the full graph and append ADR-009 metrics after every question.

    This preserves the original end-to-end demo behavior while making the lean
    deterministic evaluation visible at the point reviewers inspect each graph
    run.
    """

    config = get_config()
    if strategy is not None:
        config = config.model_copy(deep=True)
        config.retrieval.strategy = strategy
    evaluation_config = EvaluationConfig.from_app_config(
        config,
        run_mode="live_eval",
        strategies=[config.retrieval.strategy],
    )
    retriever = create_retriever(config)
    for question in CLINICAL_QUESTIONS:
        print(f"\n{'=' * 100}")
        print(f"GRAPH QUESTION: {question}")
        print("=" * 100)
        state = run_clinical_rag_graph(
            question,
            retriever=retriever,
            llm=llm,
            generate_answer=True,
            tags=["adr-007", "phase-1", "demo"],
            metadata={"strategy": config.retrieval.strategy},
            logger=print,
        )
        response = state["output"]
        print(response.model_dump_json(indent=2))
        evaluation = compute_deterministic_metrics(state, evaluation_config, latency_ms=0.0)
        print("\nADR-009 DETERMINISTIC EVALUATION")
        for metric in evaluation.metrics:
            status = "PASS" if metric.passed else "FAIL"
            gate = "gating" if metric.gating else "info"
            print(f"  {metric.id} {metric.name}: {status} ({gate})")
        print(f"  QUESTION RESULT: {'PASS' if evaluation.passed else 'FAIL'}")


if __name__ == "__main__":
    main()
