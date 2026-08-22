"""Environment loading belongs to a process entry point, not a constructor (#2896).

``LLMService.__init__`` used to open with ``load_dotenv()``. Because
``load_dotenv`` defaults to ``override=False`` it looked harmless — it protects
keys that are *set*. It does not protect keys that were deliberately **unset**,
so any code that decided to run without a variable had that decision reversed
the next time something lazily constructed an LLM service (RAG chunking,
inception, embedding, per-agent boot). ``load_sovereign_trust_root`` refuses
when an explicit path and ``KESTREL_SOVEREIGN_TRUST_ROOT_PATH`` disagree, so a
governance operation could fail because an unrelated embedding call reloaded
the operator's real trust root.

The other half is where the file came from. A bare ``load_dotenv()`` resolves
via ``find_dotenv()``, which walks up from the **calling module's own file** —
``kestrel_sovereign/llm/service.py`` — so it picked up whatever ``.env`` sat
above the package: the repo root in a source clone, and anything above
``site-packages`` in a pip install. Never the agent's home, which is the only
directory whose ``.env`` this process has any business reading.

:func:`kestrel_sovereign.paths.load_project_env` is the replacement: a *named*
home, python-dotenv's own parser, and ``setdefault`` semantics.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from kestrel_sovereign import paths


@pytest.fixture(autouse=True)
def restore_environ():
    """``load_project_env`` mutates through ``os.environ.setdefault``, which
    ``monkeypatch`` does not track — so a key it sets survives the test and
    leaks into the rest of the session. This file is the reference example for
    the helper, so it should not model the hazard it exists to document."""
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty project home that is also the current directory.

    Pinning ``KESTREL_HOME`` matters: without it the resolver walks up from CWD
    looking for a marker file and finds the dev repo above ``tmp_path``.
    """
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    paths.reset_cache()
    yield tmp_path
    paths.reset_cache()


# ---------------------------------------------------------------------------
# The regression: a constructor must not load the environment
# ---------------------------------------------------------------------------


@pytest.fixture
def dotenv_loads(monkeypatch):
    """Record every dotenv file that gets written into ``os.environ``.

    Hooks ``DotEnv.set_as_environment_variables`` rather than ``load_dotenv``
    because ``from dotenv import load_dotenv`` binds the function into the
    importing module at import time — patching the name in ``dotenv.main``
    afterwards would not be seen by the very call this test exists to catch.
    Every path into the environment goes through this method.
    """
    import dotenv.main as dotenv_main

    loaded: list[str] = []
    original = dotenv_main.DotEnv.set_as_environment_variables

    def _record(self, *args, **kwargs):
        loaded.append(str(self.dotenv_path))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        dotenv_main.DotEnv, "set_as_environment_variables", _record
    )
    return loaded


def test_constructing_an_llm_service_loads_no_dotenv_file(home, dotenv_loads):
    from kestrel_sovereign.llm.service import LLMService

    LLMService()

    assert dotenv_loads == [], (
        "Constructing an LLMService loaded a .env into the process "
        f"environment: {dotenv_loads}. Environment loading belongs at a "
        "process entry point (#2896)."
    )


def test_constructing_an_llm_service_cannot_resurrect_an_unset_variable(
    home, monkeypatch
):
    """The consequence that matters, stated in domain terms.

    ``find_dotenv`` is pointed at a real file here so the failure is
    deterministic. Left to its own resolution it walks up from
    ``llm/service.py``, which finds the repo's ``.env`` in a source clone and
    usually nothing at all in a git worktree or on CI — so the *absence* of a
    file, rather than the fix, is what would make this pass. Naming the file
    removes that: whatever ``find_dotenv`` points at, constructing a service
    must not read it into the environment.
    """
    import dotenv.main as dotenv_main

    (home / ".env").write_text(
        "KESTREL_SOVEREIGN_TRUST_ROOT_PATH=/operators/real/root.did.json\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dotenv_main, "find_dotenv", lambda *a, **k: str(home / ".env")
    )
    # The decision this process made, and which nothing downstream may reverse.
    monkeypatch.delenv("KESTREL_SOVEREIGN_TRUST_ROOT_PATH", raising=False)

    from kestrel_sovereign.llm.service import LLMService

    LLMService()

    assert "KESTREL_SOVEREIGN_TRUST_ROOT_PATH" not in os.environ, (
        "Constructing an LLMService put a deliberately-unset variable back "
        "into the environment (#2896)."
    )


# ---------------------------------------------------------------------------
# The replacement: load_project_env
# ---------------------------------------------------------------------------


def test_load_project_env_fills_in_a_variable_the_process_lacks(home, monkeypatch):
    (home / ".env").write_text("KESTREL_TEST_TOKEN=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("KESTREL_TEST_TOKEN", raising=False)

    paths.load_project_env(home)

    assert os.environ["KESTREL_TEST_TOKEN"] == "from-dotenv"


def test_load_project_env_never_overrides_an_exported_value(home, monkeypatch):
    """An exported value is an operator decision this process already made."""
    (home / ".env").write_text("KESTREL_TEST_TOKEN=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("KESTREL_TEST_TOKEN", "exported")

    paths.load_project_env(home)

    assert os.environ["KESTREL_TEST_TOKEN"] == "exported"


def test_load_project_env_skips_excluded_keys(home, monkeypatch):
    """``kestrel create`` excludes the data key so a stale persisted value
    cannot mask an exported⇄persisted custody conflict (#2468)."""
    (home / ".env").write_text(
        "KESTREL_DATA_KEY=persisted\nKESTREL_TEST_TOKEN=from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)
    monkeypatch.delenv("KESTREL_TEST_TOKEN", raising=False)

    paths.load_project_env(home, exclude=("KESTREL_DATA_KEY",))

    assert "KESTREL_DATA_KEY" not in os.environ
    assert os.environ["KESTREL_TEST_TOKEN"] == "from-dotenv"


def test_load_project_env_reads_the_named_home_not_the_current_directory(
    tmp_path, monkeypatch
):
    """The whole point of the move: the file is chosen by the caller.

    ``find_dotenv()`` walks up from the *calling module's* file, which is how a
    bare ``load_dotenv()`` in ``llm/service.py`` reached the repo root instead
    of any agent's home.
    """
    target_home = tmp_path / "home"
    target_home.mkdir()
    (target_home / ".env").write_text(
        "KESTREL_TEST_TOKEN=from-the-named-home\n", encoding="utf-8"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text(
        "KESTREL_TEST_TOKEN=from-the-wrong-place\n", encoding="utf-8"
    )
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("KESTREL_TEST_TOKEN", raising=False)

    paths.load_project_env(target_home)

    assert os.environ["KESTREL_TEST_TOKEN"] == "from-the-named-home"


def test_load_project_env_is_a_no_op_when_the_home_has_no_dotenv(home, monkeypatch):
    monkeypatch.delenv("KESTREL_TEST_TOKEN", raising=False)
    before = dict(os.environ)

    paths.load_project_env(home)

    assert dict(os.environ) == before


# ---------------------------------------------------------------------------
# Entry points that were getting their environment by accident
#
# Removing the constructor-time load is only safe if every process that
# depended on it now loads deliberately. These two did: they build an
# LLMService and an agent, and nothing else on their path touches `.env`.
# `KESTREL_DATA_KEY` in particular has no `.env` fallback anywhere in the key
# hierarchy, so losing it is not a degraded mode — it is a decryption failure
# reported as a decryption failure, for what is an env-loading regression.
# ---------------------------------------------------------------------------


def _dotenv_home(tmp_path, monkeypatch, **values):
    lines = "".join(f"{k}={v}\n" for k, v in values.items())
    (tmp_path / ".env").write_text(lines, encoding="utf-8")
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    paths.reset_cache()
    for key in values:
        monkeypatch.delenv(key, raising=False)
    return tmp_path


def test_kestrel_shell_loads_the_home_environment(tmp_path, monkeypatch):
    """`kestrel shell <agent>` falls back to an in-process agent when no server
    is running — the path the "no server running" message points users at."""
    from kestrel_sovereign import cli

    _dotenv_home(tmp_path, monkeypatch, KESTREL_TEST_TOKEN="from-dotenv")
    monkeypatch.chdir(tmp_path)

    # Stop right after the load; the rest of the command needs a real agent.
    class _Stop(Exception):
        pass

    def _boom(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(cli.MultiAgentConfig, "load", staticmethod(_boom))

    with pytest.raises(_Stop):
        cli.cmd_shell(SimpleNamespace(name="Test"))

    assert os.environ.get("KESTREL_TEST_TOKEN") == "from-dotenv"


def test_kestrel_embeddings_loads_the_home_environment(tmp_path, monkeypatch):
    """`reindex` builds an LLMService and re-embeds stored content, so it needs
    provider credentials and the data key to decrypt what it re-embeds."""
    from kestrel_sovereign import cli_embeddings

    _dotenv_home(tmp_path, monkeypatch, KESTREL_TEST_TOKEN="from-dotenv")
    monkeypatch.chdir(tmp_path)

    captured = {}

    def _capture(args):
        captured["token"] = os.environ.get("KESTREL_TEST_TOKEN")
        return ("stop here", None, None)

    monkeypatch.setattr(cli_embeddings, "_resolve_db_target", _capture)

    cli_embeddings.run(SimpleNamespace(embeddings_command="audit"))

    assert captured.get("token") == "from-dotenv", (
        "the environment was not loaded before the command read it"
    )
