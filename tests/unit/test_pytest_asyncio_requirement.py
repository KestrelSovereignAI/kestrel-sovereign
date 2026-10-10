"""A run without pytest-asyncio stops before its first test (#3522).

With ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`` and no ``-p asyncio``, the suite used
to pass every sync test and then error at the first async fixture, so a
targeted gate read "N passed" over files it never reached.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import tests.conftest as root


def _config_without_pytest_asyncio():
    return SimpleNamespace(
        pluginmanager=SimpleNamespace(is_registered=lambda plugin: False)
    )


def test_this_run_has_pytest_asyncio(request):
    root._require_pytest_asyncio(request.config)


def test_a_run_without_pytest_asyncio_is_refused_with_the_plugins_to_name(
    monkeypatch,
):
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    with pytest.raises(pytest.UsageError, match="requires the pytest-asyncio") as exc:
        root._require_pytest_asyncio(_config_without_pytest_asyncio())

    assert "-p asyncio -p anyio" in str(exc.value)


def test_the_refusal_names_no_autoload_flag_the_run_did_not_set(monkeypatch):
    monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)

    with pytest.raises(pytest.UsageError) as exc:
        root._require_pytest_asyncio(_config_without_pytest_asyncio())

    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" not in str(exc.value)


def test_the_root_conftest_refuses_before_configuring_anything():
    """Mutating the wiring away must fail here."""
    with pytest.raises(pytest.UsageError, match="requires the pytest-asyncio"):
        root.pytest_configure(_config_without_pytest_asyncio())
