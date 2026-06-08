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
    title = metadata.get("source_title") or metadata.get("source") or "Unknown source"
    hierarchy = metadata.get("section_hierarchy") or []
    section = " > ".join(hierarchy) if isinstance(hierarchy, list) else str(hierarchy)
    page = metadata.get("page_number") or metadata.get("page") or "unknown"
    return f"Source: {title} - Section: {section or 'unknown'} - Page {page}"


def main() -> None:
    retriever = create_hybrid_retriever()
    for question in CLINICAL_QUESTIONS:
        print(f"\n## Question: {question}")
        for index, doc in enumerate(retriever.invoke(question), start=1):
            print(f"\n[{index}] {_format_source(doc.metadata)}")
            print(doc.page_content[:700].replace("\n", " "))


if __name__ == "__main__":
    main()
