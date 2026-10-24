"""Explicit opt-in for tests that mutate a shared local model daemon."""

import os

import pytest
import requests


@pytest.fixture(scope="module")
def skip_if_no_ollama():
    """A reachable operator daemon alone never authorizes model mutation."""
    if os.environ.get("RUN_OLLAMA_MODEL_MUTATION_TESTS") != "1":
        pytest.skip("Live model mutation requires explicit opt-in and a disposable Ollama daemon")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
    except requests.RequestException:
        pytest.skip("Disposable Ollama daemon is not accessible")
    if response.status_code != 200:
        pytest.skip("Disposable Ollama daemon is not running on localhost:11434")
