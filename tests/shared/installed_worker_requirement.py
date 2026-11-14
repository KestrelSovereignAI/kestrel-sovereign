"""Optional source tests, mandatory installed-worker proof when requested.

Like the Runpod acquisition convention, source CI skips absent optional
components. An installed-worker job sets REQUIRE_ENV=1 and must fail instead.
This checks availability only; the installed job owns wheel/RECORD provenance.
"""
import importlib
from importlib import resources
import os

import pytest

REQUIRE_ENV = "KESTREL_REQUIRE_INSTALLED_WORKER_TESTS"


def installed_worker_required():
    value = os.environ.get(REQUIRE_ENV, "")
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    raise pytest.UsageError(f"{REQUIRE_ENV} must be '1' or '0', got {value!r}")


def unavailable(reason):
    if installed_worker_required():
        pytest.fail(f"required installed-worker tests: {reason}", pytrace=False)
    pytest.skip(f"optional installed-worker tests: {reason}")


def worker_module(name, *, required_callables=()):
    installed_worker_required()  # A misspelled required flag never means optional.
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as error:
        # A broken transitive dependency is a real failure, never an absence skip.
        if error.name is None or not (name == error.name or name.startswith(error.name + ".")):
            raise
        unavailable(f"{name} is not installed")
    for attribute in required_callables:
        if not callable(getattr(module, attribute, None)):
            unavailable(f"{name}.{attribute} is unavailable")
    return module


def worker_program(name):
    installed_worker_required()
    path = resources.files("kestrel_sovereign.visual_computer").joinpath("_worker_programs", name)
    if not path.is_file():
        unavailable(f"installed Core worker resource _worker_programs/{name} is absent")
    return path  # No source-program fallback, copy or sys.modules overlay.
