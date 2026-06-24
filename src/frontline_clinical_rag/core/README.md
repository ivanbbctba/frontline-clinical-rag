# Core Package

This package owns the centralized configuration contract for the Frontline Clinical RAG system.

The goal is to keep runtime settings, safety thresholds, retrieval defaults, evaluation controls, and project paths in one validated place instead of spreading environment lookups or magic constants across feature packages. Packages such as `retrieval`, `pipeline`, `safety`, and `evaluation` should import configuration from `src.frontline_clinical_rag.core.config` rather than creating package-local config modules.

## What Lives Here

- `config.py` defines the typed Pydantic settings models for the application.
- `__init__.py` marks the package as the home for shared core utilities and configuration.

The main public entry point is `get_config()`, which returns the singleton `AppConfig` instance used by the rest of the project.

## Configuration Sections

`AppConfig` groups settings by system concern:

- `llm`: local and xAI-compatible LLM provider settings.
- `openai`: OpenAI-compatible provider settings retained for compatibility.
- `embedding`: embedding provider, model, dimensions, batch size, and device.
- `vector_store`: vector database backend, persistence path, collection name, and distance metric.
- `retrieval`: retrieval strategy, top-k values, hybrid retrieval controls, metadata boosts, and safety-aware retrieval terms.
- `safety`: clinical guardrail settings, including citation requirements, disclaimer text, and low-confidence threshold.
- `evaluation`: ADR-009 evaluation run controls and deterministic metric thresholds.
- Project paths and ingestion defaults, including `project_root`, `merck_pdf_path`, raw data location, chunk size, overlap, and heading limits.

## Environment Loading

Settings are loaded through Pydantic Settings with `.env` support and environment-variable overrides.

Examples:

```bash
FRONTLINE_RETRIEVER_FORCE_REBUILD_INDEX=true
LLM_PROVIDER=xai
LLM_API_KEY=...
EMBEDDING_PROVIDER=local
SAFETY_LOW_CONFIDENCE_THRESHOLD=0.65
EVALUATION_LOW_CONFIDENCE_THRESHOLD=0.65
```

Nested application settings use the `FRONTLINE_` prefix and `_` nested delimiter where applicable. Individual settings classes also define domain-specific prefixes such as `LLM_`, `EMBEDDING_`, `VECTOR_STORE_`, `RETRIEVER_`, `SAFETY_`, and `EVALUATION_`.

## Python Usage

```python
from src.frontline_clinical_rag.core.config import EvaluationConfig, get_config

config = get_config()

retrieval_strategy = config.retrieval.strategy
low_confidence_threshold = config.safety.low_confidence_threshold

evaluation_config = EvaluationConfig.from_app_config(
    config,
    run_mode="harness_integration",
    strategies=[retrieval_strategy],
)
```

Use `reset_config()` in tests when environment variables are changed and the singleton must be rebuilt:

```python
from src.frontline_clinical_rag.core.config import get_config, reset_config

reset_config()
config = get_config(reload=True)
```

## Evaluation Configuration

Evaluation settings intentionally live in `core.config`, not in `src.frontline_clinical_rag.evaluation`.

ADR-009 metrics need retrieval and safety thresholds that are already application-level configuration. `EvaluationConfig.from_app_config()` derives evaluation defaults from `AppConfig` so the evaluation harness can remain external to production graph logic without becoming a second source of configuration truth.

## Guardrails

- Do not add package-local config modules in feature packages when a setting belongs to the application.
- Do not call `os.getenv()` directly from production modules for settings that should be validated here.
- Do not duplicate safety, retrieval, or evaluation thresholds outside `core.config`.
- Do not store secrets in source files; use environment variables or a local `.env` file.
- Keep settings typed and validated so configuration failures surface early.