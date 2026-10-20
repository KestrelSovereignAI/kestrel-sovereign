"""ExecutionFence requires observation, not mutation, of a backend label."""

import inspect
from typing import get_type_hints

from kestrel_sovereign.execution_custody import ExecutionFence


def test_backend_label_is_a_read_only_protocol_requirement():
    """Frozen host fences must be substitutable without making them mutable."""
    backend = inspect.getattr_static(ExecutionFence, "backend_type", None)

    assert isinstance(backend, property), (
        "the protocol must not require a writable label"
    )
    assert backend.fset is None
    assert backend.fget is not None
    assert get_type_hints(backend.fget)["return"] is str
