"""Native success/error controls for the SQLite observer-retirement fixture."""

import sqlite3
from contextlib import closing

import pytest

from tests.shared.sqlite_retirement import (
    owned_sqlite_observers as owned_sqlite_observers,
)

pytestmark = pytest.mark.usefixtures("owned_sqlite_observers")


@pytest.mark.parametrize("body_error", [False, True])
def test_original_observer_closes_after_success_or_body_error(tmp_path, body_error):
    path = tmp_path / "observer.db"
    with closing(sqlite3.connect(path)) as observer, observer:
        observer.execute("CREATE TABLE effects(id INTEGER)")
        observer.execute("INSERT INTO effects VALUES(1)")

    original = ValueError("observer body failure")
    if body_error:
        with pytest.raises(ValueError) as caught:
            with closing(sqlite3.connect(path)) as observer, observer:
                assert observer.execute("SELECT id FROM effects").fetchone() == (1,)
                raise original
        assert caught.value is original
    else:
        with closing(sqlite3.connect(path)) as observer, observer:
            assert observer.execute("SELECT id FROM effects").fetchone() == (1,)
