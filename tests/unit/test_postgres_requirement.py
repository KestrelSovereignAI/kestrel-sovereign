"""The unit tier runs its PostgreSQL cases in CI, and cannot skip them (#3381).

Every dual-backend test skips its PostgreSQL case when no URL is set. The unit
job had no PostgreSQL, so all of those cases skipped and the tier reported
green — including the PostgreSQL paths #2527's crash lived on. These tests pin
both halves of the fix: the job provides PostgreSQL and demands it, and the
guard turns a skipped PostgreSQL case into a failure.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from tests.shared import postgres_requirement as guard
from tests.utils.ci_budget import CI_WORKFLOW
from tests.utils.postgres_schema import with_search_path


def _item(**params):
    return SimpleNamespace(callspec=SimpleNamespace(params=params))


def _skipped(reason="TEST_POSTGRES_URL is not set"):
    return SimpleNamespace(
        skipped=True,
        outcome="skipped",
        longrepr=("tests/unit/test_x.py", 12, f"Skipped: {reason}"),
    )


@pytest.fixture
def required(monkeypatch):
    monkeypatch.setenv(guard.REQUIRE_ENV, "1")


# ---------------------------------------------------------------------------
# Session gate
# ---------------------------------------------------------------------------


def test_a_required_run_without_the_url_refuses_to_start():
    with pytest.raises(pytest.UsageError, match=guard.URL_ENV):
        guard.check_session({guard.REQUIRE_ENV: "1"})


def test_the_gate_asks_for_the_url_every_fixture_reads():
    """DATABASE_URL alone is not enough: some fixtures read only the test URL."""
    with pytest.raises(pytest.UsageError):
        guard.check_session(
            {guard.REQUIRE_ENV: "1", "DATABASE_URL": "postgresql://x/y"}
        )


def test_a_required_run_with_the_url_starts():
    guard.check_session({guard.REQUIRE_ENV: "1", guard.URL_ENV: "postgresql://x/y"})


@pytest.mark.parametrize("value", ["", "0"])
def test_an_unrequired_run_starts_without_a_url(value):
    guard.check_session({guard.REQUIRE_ENV: value})
    guard.check_session({})


@pytest.mark.parametrize("value", ["yes", "true", "2"])
def test_an_unrecognised_requirement_is_refused_not_read_as_off(value):
    with pytest.raises(pytest.UsageError, match=guard.REQUIRE_ENV):
        guard.check_session({guard.REQUIRE_ENV: value})


# ---------------------------------------------------------------------------
# Per-case conversion
# ---------------------------------------------------------------------------


def test_a_skipped_postgres_case_fails_when_required(required):
    report = _skipped("PostgreSQL not available: connection refused")

    guard.fail_skipped_postgres_case(_item(db_backend="postgres"), report)

    assert report.outcome == "failed"
    assert "connection refused" in report.longrepr
    assert guard.REQUIRE_ENV in report.longrepr


def test_any_fixture_parametrized_with_postgres_counts(required):
    """Not only ``db_backend``: #3380's suite has its own ``db`` fixture."""
    report = _skipped()

    guard.fail_skipped_postgres_case(_item(db="postgres", facade="host"), report)

    assert report.outcome == "failed"


def test_a_skipped_sqlite_sibling_is_left_alone(required):
    """A SQLite case skipping because its claim needs PostgreSQL is by design."""
    report = _skipped("concurrent Stop transactions need PostgreSQL")

    guard.fail_skipped_postgres_case(_item(db_backend="sqlite"), report)

    assert report.outcome == "skipped"


def test_an_unparametrized_skip_is_left_alone(required):
    report = _skipped()

    guard.fail_skipped_postgres_case(SimpleNamespace(), report)

    assert report.outcome == "skipped"


def test_a_postgres_case_may_still_skip_when_not_required(monkeypatch):
    monkeypatch.delenv(guard.REQUIRE_ENV, raising=False)
    report = _skipped()

    guard.fail_skipped_postgres_case(_item(db_backend="postgres"), report)

    assert report.outcome == "skipped"


def test_an_xfail_is_not_converted(required):
    report = _skipped()
    report.wasxfail = "known"

    guard.fail_skipped_postgres_case(_item(db_backend="postgres"), report)

    assert report.outcome == "skipped"


def test_a_non_string_parameter_is_never_compared_loosely(required):
    class Loud:
        def __eq__(self, other):  # pragma: no cover - must not be called
            raise AssertionError("compared a non-string parameter")

    report = _skipped()

    guard.fail_skipped_postgres_case(_item(value=Loud()), report)

    assert report.outcome == "skipped"


def test_the_guard_is_wired_into_the_root_conftest():
    """Mutating the wiring away must fail here, not only in CI."""
    import tests.conftest as root

    assert root._fail_skipped_postgres_case is guard.fail_skipped_postgres_case
    assert root._check_postgres_requirement is guard.check_session
    assert hasattr(root, "pytest_runtest_makereport")


# ---------------------------------------------------------------------------
# CI wiring
# ---------------------------------------------------------------------------


def _unit_job() -> dict:
    return yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]["unit-tests"]


def _unit_step() -> dict:
    steps = [
        step
        for step in _unit_job()["steps"]
        if "pytest tests/unit/" in str(step.get("run", ""))
    ]
    assert len(steps) == 1, steps
    return steps[0]


def test_the_unit_job_provides_postgres():
    service = _unit_job()["services"]["postgres"]
    assert "pg_isready" in service["options"], "no health check: tests could race it"
    port = str(service["ports"][0]).split(":")[0]
    assert f"@localhost:{port}/" in _unit_step()["env"][guard.URL_ENV]


def test_the_unit_step_requires_its_postgres_cases():
    assert str(_unit_step()["env"][guard.REQUIRE_ENV]) == "1"


def test_the_unit_step_does_not_move_code_onto_postgres():
    """Only tests read TEST_POSTGRES_URL; the application URLs stay unset."""
    env = _unit_step()["env"]
    for name in ("DATABASE_URL", "KESTREL_DATABASE_URL"):
        assert name not in env
        assert name not in (_unit_job().get("env") or {})


# ---------------------------------------------------------------------------
# Disposable-schema helper
# ---------------------------------------------------------------------------


def test_with_search_path_replaces_an_existing_one_and_keeps_other_options():
    url = "postgresql://u:p@h:5433/db?sslmode=disable&search_path=public"

    scoped = with_search_path(url, "hold_parity_abc")

    assert scoped == (
        "postgresql://u:p@h:5433/db?sslmode=disable&search_path=hold_parity_abc"
    )
