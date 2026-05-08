"""Database backend interface — sovereign-side re-export.

The canonical `DatabaseBackend` ABC lives in `kestrel_sdk.storage.database`
(SDK 0.10+) so feature packages outside this repo (notably
`kestrel-feature-entities` and `kestrel-castle`) can develop against a
stable SDK contract without depending on `kestrel_sovereign`.

This module preserves the existing
`from kestrel_sovereign.storage.db.interface import DatabaseBackend`
import path used throughout sovereign — it now re-exports the SDK ABC and
error classes verbatim.
"""

from kestrel_sdk.storage.database import (
    ConnectionError,
    DatabaseBackend,
    DatabaseError,
    Params,
    QueryError,
    Row,
    TransactionError,
)

__all__ = [
    "DatabaseBackend",
    "DatabaseError",
    "ConnectionError",
    "QueryError",
    "TransactionError",
    "Params",
    "Row",
]
