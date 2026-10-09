"""Transaction-command recognition respects native SQL lexical boundaries."""

import pytest

from kestrel_sovereign.storage.db.interface import QueryError
from kestrel_sovereign.storage.db.transaction_control import reject_transaction_control


@pytest.mark.parametrize("dialect", ["sqlite", "postgres"])
@pytest.mark.parametrize(
    "command", ["COMMIT", "END", "ROLLBACK", "BEGIN", "SAVEPOINT p", "RELEASE p"]
)
@pytest.mark.parametrize(
    "prefix", ["", ";; --comment\r\n", "/* comment; */\n", "SELECT 'not; COMMIT'; "]
)
def test_owned_transaction_commands_are_refused(dialect, command, prefix):
    with pytest.raises(QueryError, match="transaction control"):
        reject_transaction_control(prefix + command + ";", dialect=dialect)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT $$BEGIN; COMMIT; END$$; SELECT 1",
        "SELECT $tag$BEGIN; COMMIT; END$tag$; SELECT 1",
        "SELECT $😀$ '; COMMIT; END $😀$; SELECT 1",
        "DO $fn$ BEGIN PERFORM 1; END; $fn$;",
        "SELECT E'escaped\\'; COMMIT; still-string';",
        "SELECT 'COMMIT' AS \"BEGIN;END\"; -- ROLLBACK\n SELECT 1;",
        "SELECT 1; /* outer /* COMMIT; */ END; */ SELECT 2;",
        "CREATE FUNCTION f() RETURNS integer LANGUAGE SQL BEGIN ATOMIC SELECT 1; END; SELECT 2;",
        "CREATE OR REPLACE FUNCTION f() RETURNS integer BEGIN /* gap */ ATOMIC SELECT CASE WHEN true THEN 1 ELSE 2 END; END;",
        "CREATE PROCEDURE f() LANGUAGE SQL BEGIN ATOMIC SELECT 'END; COMMIT;'; END;",
    ],
)
def test_postgres_body_strings_and_comments_are_not_transaction_commands(sql):
    reject_transaction_control(sql, dialect="postgres")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT $$COMMIT;$$; COMMIT;",
        "SELECT $😀$ '$😀$; COMMIT;",
        "DO $fn$ BEGIN PERFORM 1; END; $fn$; COMMIT;",
        "SELECT 1; /* outer /* END; */ inner */ COMMIT;",
        "SELECT foo$tag$ FROM proof; COMMIT; SELECT '$tag$';",
        "SELECT 1; PREPARE TRANSACTION 'owned';",
        "SELECT 1; START TRANSACTION;",
        "SELECT 1; ABORT;",
        "SELECT E'escaped\\'; still-string'; COMMIT;",
        "SELECT E'a'\n'b\\'x';\nSELECT 'c\\';\nCOMMIT;",
        "SELECT E'a' -- continuation\n'b\\'x'; SELECT 'c\\'; COMMIT;",
        "SELECT E'a' /* newline\n comment */ 'b\\'x'; SELECT 'c\\'; COMMIT;",
        "CREATE FUNCTION f() RETURNS integer LANGUAGE SQL BEGIN ATOMIC SELECT 1; END; COMMIT;",
        "CREATE FUNCTION f() RETURNS integer LANGUAGE SQL BEGIN ATOMIC SELECT CASE WHEN true THEN 1 ELSE 2 END; END; ROLLBACK;",
        "CREATE PROCEDURE f() LANGUAGE SQL BEGIN ATOMIC SELECT 1; END; BEGIN;",
    ],
)
def test_postgres_control_after_opaque_body_or_nested_comment_is_refused(sql):
    with pytest.raises(QueryError, match="transaction control"):
        reject_transaction_control(sql, dialect="postgres")


def test_sqlite_trigger_body_is_one_native_statement():
    reject_transaction_control(
        "CREATE TRIGGER copied AFTER INSERT ON proof BEGIN INSERT INTO proof VALUES ('END;'); END;",
        dialect="sqlite",
    )
