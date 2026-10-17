"""One stored UTC instant decays equally as SQLite text or PostgreSQL datetime."""

from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.storage.memory_retriever import calculate_decay


@pytest.mark.parametrize(
    "form", ["native-naive", "native-aware", "sqlite-naive", "sqlite-aware"]
)
def test_native_memory_timestamp_does_not_become_neutral_fallback(form):
    old = datetime.now(timezone.utc) - timedelta(days=2000)
    values = {
        "native-naive": old.replace(tzinfo=None),
        "native-aware": old,
        "sqlite-naive": old.replace(tzinfo=None).isoformat(),
        "sqlite-aware": old.isoformat(),
    }
    expected = 0.5 ** (2000 / (30 * 1.02))
    assert calculate_decay(values[form], importance=0.01) == pytest.approx(
        expected, rel=1e-6, abs=0
    )
