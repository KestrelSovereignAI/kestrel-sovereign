"""Every demo must guard against running on a live instance (issue #1974).

A demo (or the TEMPLATE new demos are copied from) that mutates a live instance
can corrupt real data — see #1973. This statically enforces that each
`demos/<name>/demo.cjs` calls `assertIsolatedDemoTarget` before mutating and never
defaults its base URL to the live port (8888), so the guard can't be forgotten.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_SCRIPTS = sorted((PROJECT_ROOT / "demos").glob("*/demo.cjs"))


def test_demo_scripts_exist():
    assert DEMO_SCRIPTS, "no demos/*/demo.cjs found — glob or layout changed"


@pytest.mark.parametrize("demo", DEMO_SCRIPTS, ids=lambda p: p.parent.name)
def test_demo_calls_isolation_guard(demo: Path):
    source = demo.read_text(encoding="utf-8")
    assert "assertIsolatedDemoTarget(request, BASE_URL" in source, (
        f"{demo.parent.name}/demo.cjs must call assertIsolatedDemoTarget(request, BASE_URL, apiKey) "
        "in beforeAll before any mutation (issue #1974)."
    )


@pytest.mark.parametrize("demo", DEMO_SCRIPTS, ids=lambda p: p.parent.name)
def test_demo_env_guard_runs_before_credentials(demo: Path):
    # getApiKey hits /api/auth/key (can mint/persist a key on a live host), so the
    # credential-free env check must run first (issue #1974).
    source = demo.read_text(encoding="utf-8")
    env_at = source.find("assertIsolatedDemoEnv(BASE_URL)")
    key_at = source.find("getApiKey(request, BASE_URL)")
    assert env_at != -1, f"{demo.parent.name}/demo.cjs must call assertIsolatedDemoEnv(BASE_URL)."
    assert key_at != -1, f"{demo.parent.name}/demo.cjs must fetch its API key via getApiKey."
    assert env_at < key_at, (
        f"{demo.parent.name}/demo.cjs must call assertIsolatedDemoEnv(BASE_URL) BEFORE getApiKey "
        "so a live-instance run aborts before any credentialed side effect (issue #1974)."
    )


@pytest.mark.parametrize("demo", DEMO_SCRIPTS, ids=lambda p: p.parent.name)
def test_demo_does_not_default_to_live_port(demo: Path):
    source = demo.read_text(encoding="utf-8")
    assert "localhost:8888" not in source, (
        f"{demo.parent.name}/demo.cjs must not default BASE_URL to the live port 8888; "
        "use the demo port 8900 (issue #1973/#1974)."
    )
