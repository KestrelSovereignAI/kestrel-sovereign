"""An ordinary test invocation cannot opt itself into host model mutation."""

import pytest

from tests.shared.live_ollama import skip_if_no_ollama


@pytest.mark.parametrize("configured", [None, "0"])
def test_live_model_mutation_requires_explicit_opt_in(monkeypatch, configured):
    if configured is None:
        monkeypatch.delenv("RUN_OLLAMA_MODEL_MUTATION_TESTS", raising=False)
    else:
        monkeypatch.setenv("RUN_OLLAMA_MODEL_MUTATION_TESTS", configured)
    with pytest.raises(pytest.skip.Exception, match="explicit opt-in"):
        skip_if_no_ollama.__wrapped__()
