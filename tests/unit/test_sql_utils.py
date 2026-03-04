"""Tests for kestrel_sovereign.sql_utils — SQL identifier validation."""

import pytest

from kestrel_sovereign.sql_utils import safe_table_name, safe_column_name


class TestSafeTableName:
    """Verify safe_table_name accepts valid identifiers and rejects bad ones."""

    def test_simple_name(self):
        assert safe_table_name("conversations") == "conversations"

    def test_underscore_prefix(self):
        assert safe_table_name("_private") == "_private"

    def test_name_with_digits(self):
        assert safe_table_name("table2") == "table2"

    def test_mixed_case(self):
        assert safe_table_name("ConversationHistory") == "ConversationHistory"

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="Invalid table name"):
            safe_table_name("")

    def test_rejects_leading_digit(self):
        with pytest.raises(ValueError, match="Invalid table name"):
            safe_table_name("1table")

    def test_rejects_sql_injection_semicolon(self):
        with pytest.raises(ValueError, match="Invalid table name"):
            safe_table_name("users; DROP TABLE users")

    def test_rejects_dash(self):
        with pytest.raises(ValueError, match="Invalid table name"):
            safe_table_name("my-table")

    def test_rejects_space(self):
        with pytest.raises(ValueError, match="Invalid table name"):
            safe_table_name("my table")

    def test_rejects_parentheses(self):
        with pytest.raises(ValueError, match="Invalid table name"):
            safe_table_name("users()")

    def test_rejects_quotes(self):
        with pytest.raises(ValueError, match="Invalid table name"):
            safe_table_name("users'--")

    def test_rejects_dot(self):
        with pytest.raises(ValueError, match="Invalid table name"):
            safe_table_name("schema.table")


class TestSafeColumnName:
    """Verify safe_column_name accepts valid identifiers and rejects bad ones."""

    def test_simple_column(self):
        assert safe_column_name("content") == "content"

    def test_underscore_column(self):
        assert safe_column_name("created_at") == "created_at"

    def test_rowid(self):
        assert safe_column_name("rowid") == "rowid"

    def test_rejects_injection_attempt(self):
        with pytest.raises(ValueError, match="Invalid column name"):
            safe_column_name("col = 1 OR 1=1 --")

    def test_rejects_star(self):
        with pytest.raises(ValueError, match="Invalid column name"):
            safe_column_name("*")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="Invalid column name"):
            safe_column_name("")
