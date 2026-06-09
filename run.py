from pathlib import Path
from dotenv import load_dotenv

from src.frontline_clinical_rag.pipeline import create_hybrid_retriever

load_dotenv()


CLINICAL_QUESTIONS = [
    "What are urgent warning signs for chest pain?",
    "What are common causes of acute shortness of breath?",
    "What red flags should be considered for severe headache?",
    "What information is relevant for evaluating abdominal pain with fever?",
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
    print("Initializing Hierarchical Retriever...")
    hierarchical_retriever = create_hybrid_retriever(strategy="hierarchical")

    print("Initializing Recursive Retriever...")
    recursive_retriever = create_hybrid_retriever(strategy="recursive")

    for question in CLINICAL_QUESTIONS:
        print(f"\n" + "="*100)
        print(f"QUESTION: {question}")
        print("="*100)

        # Hierarchical Results
        print("\n[ STRATEGY: HIERARCHICAL ]")
        h_docs = hierarchical_retriever.invoke(question)
        for index, doc in enumerate(h_docs, start=1):
            print(f"\n  ({index}) {_format_source(doc.metadata)}")
            # Show first 400 chars for comparison
            content = doc.page_content.strip().replace("\n", " ")
            print(f"      {content[:400]}...")

        # Recursive Results
        print("\n[ STRATEGY: RECURSIVE ]")
        r_docs = recursive_retriever.invoke(question)
        for index, doc in enumerate(r_docs, start=1):
            print(f"\n  ({index}) {_format_source(doc.metadata)}")
            content = doc.page_content.strip().replace("\n", " ")
            print(f"      {content[:400]}...")


if __name__ == "__main__":
    main()
