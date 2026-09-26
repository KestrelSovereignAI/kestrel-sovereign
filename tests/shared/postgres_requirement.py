"""Make the PostgreSQL leg of dual-backend tests fail instead of skip (#3381).

Dual-backend tests skip their PostgreSQL case when no database URL is set,
which is right on a laptop without PostgreSQL and wrong in CI: a CI job that
loses its URL (a renamed variable, a dropped ``env:`` key, a service that
never came up) turns every PostgreSQL case into a skip and still reports
green. That is how #2527's PostgreSQL defect went through every gate.

A job that provides PostgreSQL sets ``KESTREL_REQUIRE_POSTGRES_TESTS=1``.
Then:

* the session refuses to start unless ``TEST_POSTGRES_URL`` is set — the one
  variable every dual-backend fixture honors (some read only it); and
* a test whose parameters select PostgreSQL (``db_backend``, or any fixture
  or argument parametrized with the value ``"postgres"``) is reported as
  failed if it skips, carrying the skip reason. A connection failure that a
  fixture turns into a skip is thereby a failure too.

Only the PostgreSQL case is covered. A SQLite case that skips because its
claim needs PostgreSQL is left alone: that skip is the test's design, and
its PostgreSQL sibling still runs.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

REQUIRE_ENV = "KESTREL_REQUIRE_POSTGRES_TESTS"
URL_ENV = "TEST_POSTGRES_URL"
POSTGRES_PARAM = "postgres"


def postgres_required(environ: dict | None = None) -> bool:
    """Whether this run must execute every PostgreSQL case."""

    env = os.environ if environ is None else environ
    value = env.get(REQUIRE_ENV, "")
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    # A typo must not read as "not required": that would restore exactly the
    # silent skip this guard exists to remove.
    raise pytest.UsageError(f"{REQUIRE_ENV} must be '1' or '0', got {value!r}")


def check_session(environ: dict | None = None) -> None:
    """Refuse a required run that has no PostgreSQL URL."""

    env = os.environ if environ is None else environ
    if postgres_required(env) and not env.get(URL_ENV):
        raise pytest.UsageError(
            f"{REQUIRE_ENV}=1 but {URL_ENV} is not set: every PostgreSQL test "
            "case would skip. Point it at the job's PostgreSQL service."
        )


def is_postgres_case(item: Any) -> bool:
    """Whether *item* is the PostgreSQL case of a parametrized test."""

    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False
    return any(
        isinstance(value, str) and value == POSTGRES_PARAM
        for value in callspec.params.values()
    )


def fail_skipped_postgres_case(item: Any, report: Any) -> None:
    """Turn *report* into a failure if it skips a required PostgreSQL case."""

    if not report.skipped or hasattr(report, "wasxfail"):
        return
    if not postgres_required() or not is_postgres_case(item):
        return
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else ""
    report.outcome = "failed"
    report.longrepr = (
        f"PostgreSQL case skipped with {REQUIRE_ENV}=1 (#3381). Skip reason: {reason}"
    )
