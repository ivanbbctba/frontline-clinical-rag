import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_medical_documents():
    """Basic medical documents for testing chunkers."""
    return [
        Document(
            page_content="Heart Failure\n\nHeart failure is a complex clinical syndrome...",
            metadata={"source": "test_cardiology.pdf", "page": 42}
        ),
        Document(
            page_content="Black Box Warning: This drug may cause severe hepatotoxicity.",
            metadata={"source": "test_pharmacology.pdf", "page": 15}
        ),
        Document(
            page_content="Diabetes Mellitus\n\nType 2 diabetes is characterized by insulin resistance.",
            metadata={"source": "test_endocrinology.pdf", "page": 88}
        ),
    ]


@pytest.fixture
def simple_text_document():
    return Document(
        page_content="This is a simple medical note without any special formatting.",
        metadata={"source": "simple_note.pdf", "page": 1}
    )