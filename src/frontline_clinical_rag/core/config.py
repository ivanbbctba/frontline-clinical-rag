"""
core/config.py

Centralized, validated configuration for the Frontline Clinical RAG system.

This is the single source of truth for all settings (embeddings, vector store,
retrieval strategy, paths, API keys, safety flags, etc.).

Why this design:
- Avoids scattered os.getenv / magic strings across the codebase.
- Enables easy overrides for testing (in-memory stores, mock embeddings).
- Supports .env files + environment variables with clear precedence.
- Nested structure mirrors the architecture (embedding, retrieval, vector_store).
- Pydantic v2 gives automatic validation, type coercion, and nice error messages.

Clinical note:
Configuration for a medical RAG system must make safety and reproducibility
explicit. We expose flags that will later control guardrails and evaluation mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """LLM provider settings.

    Generation is intentionally not wired into ADR-005 yet, but the chosen
    provider order is configured here so future chain assembly remains
    explicit and reproducible.
    """

    provider: Literal["local", "xai"] = Field(
        "local", validation_alias=AliasChoices("LLM_PROVIDER", "provider"), description="Primary LLM provider"
    )
    fallback_provider: Literal["xai"] = Field(
        "xai", validation_alias=AliasChoices("LLM_FALLBACK_PROVIDER", "fallback_provider"), description="Fallback LLM provider"
    )
    local_base_url: str = Field(
        "http://localhost:11434", validation_alias=AliasChoices("LLM_LOCAL_BASE_URL", "local_base_url"), description="Local OpenAI-compatible LLM URL"
    )
    local_model_name: str = Field("llama3.1", validation_alias=AliasChoices("LLM_LOCAL_MODEL_NAME", "local_model_name"), description="Local LLM model name")
    xai_base_url: str = Field("https://api.x.ai/v1", validation_alias=AliasChoices("LLM_XAI_BASE_URL", "xai_base_url"), description="xAI API base URL")
    xai_model_name: str = Field("grok-3-mini", validation_alias=AliasChoices("LLM_XAI_MODEL_NAME", "xai_model_name"), description="xAI fallback model")
    api_key: str | None = Field(None, validation_alias=AliasChoices("LLM_API_KEY", "api_key"), description="xAI API key")

    model_config = SettingsConfigDict(
        env_prefix="LLM_", env_file=".env", extra="ignore"
    )


class OpenAIConfig(BaseSettings):
    """OpenAI-compatible provider settings retained for compatibility."""

    api_key: str | None = Field(None, description="OpenAI API key")
    base_url: str | None = Field(
        None, description="Optional custom base URL (e.g. Azure or local proxy)"
    )
    organization: str | None = Field(None)

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_", env_file=".env", extra="ignore"
    )


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""

    provider: Literal["openai", "local", "voyage"] = Field(
        "local", validation_alias=AliasChoices("EMBEDDING_PROVIDER", "provider"), description="Embedding provider"
    )
    model_name: str = Field("BAAI/bge-m3", validation_alias=AliasChoices("EMBEDDING_MODEL_NAME", "model_name"), description="Model identifier")
    dimensions: int = Field(1024, validation_alias=AliasChoices("EMBEDDING_DIMENSIONS", "dimensions"), description="Output vector dimension")
    batch_size: int = Field(100, validation_alias=AliasChoices("EMBEDDING_BATCH_SIZE", "batch_size"), description="Batch size for embedding calls")
    device: str = Field("cpu", validation_alias=AliasChoices("EMBEDDING_DEVICE", "device"), description="Device for local embedding models")

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_", env_file=".env", extra="ignore"
    )


class VectorStoreConfig(BaseSettings):
    """Vector database / retrieval index settings."""

    backend: Literal["chroma", "faiss", "weaviate"] = Field(
        "faiss", description="Vector store implementation"
    )
    persist_directory: str = Field(
        "data/vector_store/faiss_index",
        validation_alias="path",
        description="Directory for persisted index (git-ignored in production)",
    )
    collection_name: str = Field(
        "merck_manual_clinical",
        description="Collection / index name inside the vector store",
    )
    distance_metric: str = Field(
        "cosine", description="Distance function used by the store"
    )

    model_config = SettingsConfigDict(
        env_prefix="VECTOR_STORE_", env_file=".env", extra="ignore"
    )

    @field_validator("persist_directory")
    @classmethod
    def ensure_directory_exists(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class RetrievalConfig(BaseSettings):
    """Retrieval strategy and quality parameters."""

    top_k: int = Field(5, validation_alias="k_final", description="Number of documents to retrieve")
    dense_top_k: int = Field(8, validation_alias="k_dense", description="Number of dense results to retrieve")
    sparse_top_k: int = Field(8, validation_alias="k_sparse", description="Number of sparse results to retrieve")
    use_hybrid: bool = Field(True, description="Whether to use full hybrid retrieval")
    rrf_k: int = Field(60, description="Reciprocal Rank Fusion constant")
    metadata_boosting: dict[str, float] = Field(
        default_factory=lambda: {
            "section_hierarchy": 1.2,
            "has_warning": 1.5,
            "page_range_boost": 1.0,
            "warning": 1.7,
            "table": 1.25,
        },
        description="Configurable field-level boosting for metadata-aware retrieval",
    )
    safety_warning_levels: list[str] = Field(
        default_factory=lambda: ["black_box", "boxed_warning"]
    )
    safety_query_terms: list[str] = Field(
        default_factory=lambda: [
            "warning",
            "contraindication",
            "black box",
            "adverse",
            "risk",
            "toxicity",
        ]
    )
    safety_downweight_factor: float = Field(0.55)
    chunk_overlap_for_context: int = Field(200)

    model_config = SettingsConfigDict(
        env_prefix="RETRIEVER_", env_file=".env", extra="ignore"
    )


class SafetyConfig(BaseSettings):
    """Clinical safety and guardrail configuration."""

    require_citation: bool = Field(True)
    max_context_tokens: int = Field(8000)
    enable_refusal: bool = Field(True)
    medical_disclaimer: str = Field(
        "This is an AI-assisted decision support tool. All information must be verified against primary sources and clinical judgment. Not a substitute for professional medical advice."
    )

    model_config = SettingsConfigDict(
        env_prefix="SAFETY_", env_file=".env", extra="ignore"
    )


class AppConfig(BaseSettings):
    """Root configuration object — the single source of truth."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig, alias="retriever")
    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
    )
    merck_pdf_path: Path = Field(
        default=Path("data/raw/Merk medical_diagnosis_manual-1-1999.pdf"),
        description="Path to the full Merck Manual PDF (large file, do not commit)",
    )
    raw_data_path: Path = Field(default=Path("data/raw"))
    chunk_size: int = Field(800)
    chunk_overlap: int = Field(200)
    max_heading_length: int = Field(140)

    model_config = SettingsConfigDict(
        env_prefix="FRONTLINE_",
        env_file=".env",
        env_nested_delimiter="_",
        extra="ignore",
        arbitrary_types_allowed=True,
    )

    @field_validator("merck_pdf_path")
    @classmethod
    def validate_pdf_exists(cls, v: Path) -> Path:
        if not v.exists():
            pass
        return v


_config: AppConfig | None = None


def get_config(reload: bool = False) -> AppConfig:
    """Return the singleton validated configuration."""

    global _config
    if _config is None or reload:
        _config = AppConfig()
    return _config


def reset_config() -> None:
    """Helper for tests to force re-initialization with new environment."""

    global _config
    _config = None
