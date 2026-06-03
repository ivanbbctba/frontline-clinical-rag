from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FRONTLINE_",
        extra="ignore",
    )

    # Grok API
    grok_api_key: str
    grok_model: str = "grok-3-beta"

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"

    # Hardware acceleration
    embedding_device: str = "cpu"  # "cpu", "cuda", or "mps"

    # Data locations
    raw_data_path: str = "data/raw"

    # FAISS
    vector_store_path: str = "data/vector_store/faiss_index"

    # Chunking (from ADR-002)
    chunk_size: int = 800
    chunk_overlap: int = 200

    # Heading detection (used by HierarchicalMedicalChunker)
    max_heading_length: int = 140

    # Medical RAG specific
    max_tokens: int = 1024
    temperature: float = 0.0

    @property
    def vector_store_dir(self) -> Path:
        return Path(self.vector_store_path).parent

    @property
    def faiss_index_path(self) -> Path:
        return Path(self.vector_store_path)


settings = Settings()
