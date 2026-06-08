"""Central configuration for frontline-clinical-rag.

Uses pydantic-settings to load from .env with FRONTLINE_ prefix.
All tunable parameters for the system should live here.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FRONTLINE_",
        extra="ignore",
    )

    # === Core / Generation ===
    grok_api_key: str
    grok_model: str = "grok-3-beta"
    max_tokens: int = 1024
    temperature: float = 0.0

    # === Embeddings & Ingestion (ADR-002 / ADR-003) ===
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"  # cpu | cuda | mps
    raw_data_path: str = "data/raw"
    vector_store_path: str = "data/vector_store/faiss_index"
    chunk_size: int = 800
    chunk_overlap: int = 200
    max_heading_length: int = 140

    # === Retrieval / Hybrid Retriever (ADR-004) ===
    # These values feed MetadataAwareHybridRetriever via dependency injection.
    # Override in .env for different environments or experiments.
    retriever_k_final: int = 5
    retriever_k_dense: int = 8
    retriever_k_sparse: int = 8
    retriever_rrf_k: int = 60
    retriever_boost_factors: dict[str, float] = Field(
        default_factory=lambda: {"warning": 1.7, "table": 1.25}
    )
    retriever_safety_warning_levels: list[str] = Field(
        default_factory=lambda: ["black_box", "boxed_warning"]
    )
    retriever_safety_query_terms: list[str] = Field(
        default_factory=lambda: [
            "warning",
            "contraindication",
            "black box",
            "adverse",
            "risk",
            "toxicity",
        ]
    )
    retriever_safety_downweight_factor: float = 0.55
    # Example .env override for the dict (valid JSON):
    # FRONTLINE_RETRIEVER_BOOST_FACTORS='{"warning": 2.0, "table": 1.3}'

    @property
    def vector_store_dir(self) -> Path:
        return Path(self.vector_store_path).parent

    @property
    def faiss_index_path(self) -> Path:
        return Path(self.vector_store_path)


settings = Settings()
