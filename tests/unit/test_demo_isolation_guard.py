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
def test_demo_does_not_default_to_live_port(demo: Path):
    source = demo.read_text(encoding="utf-8")
    assert "localhost:8888" not in source, (
        f"{demo.parent.name}/demo.cjs must not default BASE_URL to the live port 8888; "
        "use the demo port 8900 (issue #1973/#1974)."
    )
