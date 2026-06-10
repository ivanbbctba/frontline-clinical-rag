from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.frontline_clinical_rag.core.config import get_config
from src.frontline_clinical_rag.pipeline.factory import create_retriever
from src.frontline_clinical_rag.pipeline.graph import run_clinical_rag_graph

load_dotenv()


CLINICAL_QUESTIONS = [
    #"What are urgent warning signs for chest pain?",
    #"What are common causes of acute shortness of breath?",
    #"What red flags should be considered for severe headache?",
    #"What information is relevant for evaluating abdominal pain with fever?",
    "What is the protocol for managing sepsis in a critical care unit?",
    "What are the common symptoms for appendicitis, and can it be cured via medicine? If not, what surgical procedure should be followed to treat it?",
    "What are the effective treatments or solutions for addressing sudden patchy hair loss, commonly seen as localized bald spots on the scalp, and what could be the possible causes behind it?",
    "What treatments are recommended for a person who has sustained a physical injury to brain tissue, resulting in temporary or permanent impairment of brain function?",

]


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
    run_retrieval_demo()


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


def run_generation_graph_demo(llm: Any) -> None:
    """Run the ADR-007 Phase 1 graph over the four canonical questions."""

    config = get_config()
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


if __name__ == "__main__":
    main()
