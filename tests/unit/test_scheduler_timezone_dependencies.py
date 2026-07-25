"""Base-install contracts for scheduler time-zone evaluation."""

import builtins
from datetime import datetime, timezone

from kestrel_sovereign.features.scheduler.cron import next_run


def test_utc_and_iana_cron_do_not_import_optional_pandas_or_phoenix(monkeypatch):
    """The scheduler must use stdlib ``zoneinfo`` (and base ``tzdata`` on Windows).

    This deliberately rejects imports from optional data-science/observability
    extras while evaluating both the default UTC path and a non-UTC IANA zone.
    """

    original_import = builtins.__import__
    blocked_roots = {"pandas", "phoenix", "arize_phoenix", "pydantic_ai"}

    def import_without_optional(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in blocked_roots:
            raise AssertionError(f"scheduler time-zone evaluation imported optional {name!r}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_optional)
    after = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    assert next_run("0 13 * * *", after, "UTC") == datetime(
        2026, 1, 1, 13, 0, tzinfo=timezone.utc
    )
    assert next_run("0 7 * * *", after, "America/Chicago") == datetime(
        2026, 1, 1, 13, 0, tzinfo=timezone.utc
    )
