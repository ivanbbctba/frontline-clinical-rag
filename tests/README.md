# Tests

This folder contains tests for the `frontline-clinical-rag` project.

## Running Tests

```bash
# Run all tests
pipenv run pytest

# Run only ingestion tests
pipenv run pytest tests/ingestion/ -v

# Run with coverage (after installing pytest-cov)
pipenv run pytest tests/ingestion/ --cov=src.frontline_clinical_rag.ingestion