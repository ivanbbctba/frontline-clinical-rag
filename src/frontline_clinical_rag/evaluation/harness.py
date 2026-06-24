"""External ADR-009 harness for deterministic clinical RAG evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from langchain_openai import ChatOpenAI

from src.frontline_clinical_rag.core.config import AppConfig, EvaluationConfig, get_config
from src.frontline_clinical_rag.evaluation.fixtures import (
    CANONICAL_MERCK_FIXTURES,
    ClinicalQuestionFixture,
)
from src.frontline_clinical_rag.evaluation.metrics import (
    EvaluationResult,
    compute_deterministic_metrics,
)
from src.frontline_clinical_rag.pipeline.factory import create_retriever
from src.frontline_clinical_rag.pipeline.graph import ClinicalRAGState, run_clinical_rag_graph


class MockClinicalLLM:
    """Fixed MockLLM for ADR-009 harness integration mode.

    CI uses this model to prove the harness executes the real graph path while
    avoiding Layer B provider variance. The JSON intentionally contains the
    uncertainty fields required by M7 when the graph escalates.
    """

    def invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Return a deterministic ClinicalResponse-compatible JSON payload."""

        return json.dumps(
            {
                "answer": (
                    "The retrieved Merck context supports source-guided clinical "
                    "decision support for this question. Confirm details against "
                    "the cited passage before applying patient-specific care."
                ),
                "sources": [
                    {
                        "page": 5,
                        "section": "Clinical Section",
                        "excerpt": "Grounded Merck context for deterministic evaluation.",
                    }
                ],
                "disclaimer": (
                    "Clinical decision-support only for trained medical professionals. "
                    "Verify all recommendations against the cited Merck Manual content, "
                    "local protocols, patient-specific factors, and independent clinical judgment. "
                    "This output is not a diagnosis, treatment order, or substitute for clinician review."
                ),
                "warning_level_summary": "No high-warning source metadata reported.",
                "confidence": 0.75,
                "requires_human_review": False,
                "uncertainty_note": None,
                "key_findings_to_verify": [],
                "recommended_next_steps": [],
            }
        )


class MockRetriever:
    """Small deterministic retriever for ADR-009 harness integration mode."""

    def invoke(self, question: str) -> list[dict[str, Any]]:
        """Return one Merck-shaped document so the real graph can run in CI."""

        return [
            {
                "page_content": f"Grounded Merck context for {question}",
                "metadata": {
                    "section": "Clinical Section",
                    "section_hierarchy": ["Merck", "Clinical Section"],
                    "page": 5,
                    "warning_level": "standard",
                    "source": "merck.pdf",
                },
            }
        ]


class QuestionEvaluationRow(dict[str, Any]):
    """Dictionary row used for lean ADR-009 console/CSV/JSON reporting."""


def run_evaluation(
    config: EvaluationConfig | None = None,
    *,
    app_config: AppConfig | None = None,
    llm_factory: Callable[[], Any] | None = None,
) -> list[QuestionEvaluationRow]:
    """Run ADR-009 benchmark fixtures and return deterministic report rows.

    The harness remains external: it creates retrievers, invokes
    ``run_clinical_rag_graph`` for each benign fixture, then computes pure Layer
    A metrics from the final state.
    """

    base_config = app_config or get_config()
    evaluation_config = config or base_config.evaluation
    rows: list[QuestionEvaluationRow] = []

    for strategy in evaluation_config.strategies:
        strategy_config = base_config.model_copy(deep=True)
        strategy_config.retrieval.strategy = strategy
        retriever = (
            MockRetriever()
            if evaluation_config.run_mode in {"metric_unit", "harness_integration"}
            else create_retriever(strategy_config)
        )
        llm = _resolve_llm(evaluation_config, llm_factory)

        for fixture in CANONICAL_MERCK_FIXTURES:
            state, latency_ms = _run_question(fixture, strategy, retriever, llm, evaluation_config)
            result = compute_deterministic_metrics(state, evaluation_config, latency_ms=latency_ms)
            rows.append(_row_from_result(fixture, strategy, latency_ms, result))

    return rows


def print_table(rows: list[QuestionEvaluationRow]) -> None:
    """Print the compact ADR-009 per-question PASS/FAIL table."""

    if not rows:
        print("No evaluation rows produced.")
        return
    headers = ["strategy", "question_id", "topic", "pass", "passed_gating", "latency_ms"]
    widths = {header: max(len(header), *(len(str(row.get(header, ""))) for row in rows)) for header in headers}
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def export_rows(rows: list[QuestionEvaluationRow], *, csv_path: Path | None = None, json_path: Path | None = None) -> None:
    """Write optional ADR-009 CSV/JSON artifacts using only stdlib formats."""

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m src.frontline_clinical_rag.evaluation.harness``."""

    parser = argparse.ArgumentParser(description="Run ADR-009 deterministic evaluation.")
    parser.add_argument("--mode", choices=["metric_unit", "harness_integration", "live_eval"], default="harness_integration")
    parser.add_argument("--strategy", choices=["hierarchical", "recursive"], action="append")
    parser.add_argument("--compare-strategies", action="store_true")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    strategies = ["hierarchical", "recursive"] if args.compare_strategies else args.strategy or ["hierarchical"]
    app_config = get_config()
    config = EvaluationConfig.from_app_config(app_config, run_mode=args.mode, strategies=strategies)
    rows = run_evaluation(config, app_config=app_config)
    print_table(rows)
    export_rows(rows, csv_path=args.csv, json_path=args.json)
    return 0 if all(row["passed"] for row in rows) else 1


def _run_question(
    fixture: ClinicalQuestionFixture,
    strategy: str,
    retriever: Any,
    llm: Any,
    config: EvaluationConfig,
) -> tuple[ClinicalRAGState, float]:
    started = time.perf_counter()
    state = run_clinical_rag_graph(
        fixture.question,
        retriever=retriever,
        llm=llm,
        generate_answer=True,
        tags=["adr-009", config.run_mode, strategy],
        metadata={"strategy": strategy, "fixture_id": fixture.id},
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return state, round(latency_ms, 3)


def _resolve_llm(config: EvaluationConfig, llm_factory: Callable[[], Any] | None) -> Any:
    """Resolve the LLM for ADR-009's three explicit run modes."""

    if config.run_mode == "metric_unit":
        return MockClinicalLLM()
    if config.run_mode == "harness_integration":
        return MockClinicalLLM()
    return llm_factory() if llm_factory else _build_live_llm()


def _build_live_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("LLM_XAI_MODEL_NAME"),
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_XAI_BASE_URL", "https://api.x.ai/v1"),
        temperature=0.0,
        max_tokens=int(os.environ.get("FRONTLINE_MAX_TOKENS", "1024")),
    )


def _row_from_result(
    fixture: ClinicalQuestionFixture,
    strategy: str,
    latency_ms: float,
    result: EvaluationResult,
) -> QuestionEvaluationRow:
    metric_status = {metric.id: "PASS" if metric.passed else "FAIL" for metric in result.metrics}
    row = QuestionEvaluationRow(
        strategy=strategy,
        question_id=fixture.id,
        topic=fixture.topic,
        passed=result.passed,
        passed_gating=f"{sum(metric.passed for metric in result.metrics if metric.gating)}/11",
        latency_ms=latency_ms,
    )
    row.update(metric_status)
    return row


if __name__ == "__main__":
    raise SystemExit(main())