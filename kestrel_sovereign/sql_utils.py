"""Shared SQL safety utilities.

Provides validation for dynamic identifiers (table names, column names)
that must be interpolated into SQL strings. This prevents SQL injection
when parameterized placeholders cannot be used (e.g., table names in
SQLite do not support parameter binding).
"""

import re

_VALID_SQL_IDENTIFIER = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def safe_table_name(name: str) -> str:
    """Validate and return a safe table name for SQL interpolation.

    Raises:
        ValueError: If the name contains characters outside [a-zA-Z0-9_]
                    or does not start with a letter or underscore.
    """
    if not _VALID_SQL_IDENTIFIER.match(name):
        raise ValueError(f"Invalid table name: {name!r}")
    return name


def safe_column_name(name: str) -> str:
    """Validate and return a safe column name for SQL interpolation.

    Raises:
        ValueError: If the name contains characters outside [a-zA-Z0-9_]
                    or does not start with a letter or underscore.
    """
    if not _VALID_SQL_IDENTIFIER.match(name):
        raise ValueError(f"Invalid column name: {name!r}")
    return name
