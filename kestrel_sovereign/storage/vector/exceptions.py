"""Errors raised by the generic vector-search backends."""


class VectorSearchError(Exception):
    """Raised when a vector-search backend cannot service a request.

    Examples:
        - Required filter column missing from a ``knn()`` filter dict
        - Embedding column is not nullable but no embedding was provided
        - Backend-specific failures (numpy missing for PurePython,
          pgvector extension missing for PgVector, etc.)

    Callers should treat this as a 4xx-style error (the request was
    malformed or unsupported), not a 5xx (the storage layer itself is
    healthy).
    """
