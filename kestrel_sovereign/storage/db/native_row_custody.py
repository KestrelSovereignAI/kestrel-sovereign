"""Closed native row profiles; ordinary current owners never gain late privilege."""

from kestrel_sovereign.execution_custody import ExecutionAuthorityError

from .original_result_custody import _original_result_profile, _require_original_result


def _current_step(owner):
    from kestrel_sovereign.storage.current_visual_step import _CurrentVisualStepMutation

    return type(owner) is _CurrentVisualStepMutation


def _current_row(owner):
    return _current_step(owner)


def _native_row_profile(owner, *, role="primary"):
    if _current_step(owner):
        if role == "owners":
            return "visual_computer_owners", {"owner": "TEXT", "generation": "BIGINT"}
        if role == "requests":
            return "visual_computer_requests", {
                "owner": "TEXT",
                "request_id": "TEXT",
                "session_key": "TEXT",
                "payload": "BLOB",
                "revision": "BIGINT",
                "released": "BIGINT",
                "retired": "BIGINT",
            }
        if role == "primary":
            return "visual_computer_actions", {
                "owner": "TEXT",
                "session_key": "TEXT",
                "action_id": "TEXT",
                "payload": "BLOB",
                "pending": "BIGINT",
            }
    elif role == "primary":
        return _original_result_profile(owner)
    raise ExecutionAuthorityError("Exact closed native relation required")


def _require_native_row(owner, backend, *, admission=False):
    _native_row_profile(owner)
    if _current_row(owner):
        if admission:
            owner.require_admission(backend)
        else:
            owner.require_original(backend)
    else:
        _require_original_result(owner, backend, admission=admission)


def _step_row_queries(owner):
    if not _current_step(owner):
        raise ExecutionAuthorityError("Exact current step program required")
    from .sqlite import SQLiteBackend

    sqlite = type(owner._backend) is SQLiteBackend
    table = (
        "main.visual_computer_actions"
        if sqlite
        else owner._postgres_relation.qualified_name
    )
    requests = (
        "main.visual_computer_requests"
        if sqlite
        else owner._aux_relations[1].qualified_name
    )
    parent = f"EXISTS (SELECT 1 FROM {requests} r WHERE r.owner = a.owner AND r.session_key = a.session_key AND r.revision = ? AND r.released = 0 AND r.retired = 0)"
    if owner._mode == "claim":
        # No unscoped lookup: repeat original owner explicitly in fixed values.
        query = (
            f"INSERT INTO {table} (owner, session_key, action_id, payload, pending) "
            f"SELECT ?, ?, ?, ?, 1 WHERE ? > (SELECT COUNT(*) FROM {table} WHERE owner = ?) "
            f"AND EXISTS (SELECT 1 FROM {requests} WHERE session_key = ? AND revision = ? "
            "AND owner = ? AND released = 0 AND retired = 0) "
            "ON CONFLICT (owner, session_key, action_id) DO NOTHING"
        )
        match = (
            f"SELECT 1 FROM {table} WHERE owner = ? AND session_key = ? AND action_id = ? "
            "AND payload = ? AND ? > 0 AND owner = ? AND session_key = ? AND ? >= 0 AND owner = ? AND pending = 1"
        )
        return query, match
    if owner._mode != "update":
        raise ExecutionAuthorityError("Unknown current step program")
    cast = "?" if sqlite else "CAST(? AS BYTEA)"
    return (
        f"UPDATE {table} AS a SET payload = ?, pending = ? WHERE owner = ? AND session_key = ? AND action_id = ? AND payload = ? AND pending = 1 AND {parent}",
        f"SELECT 1 FROM {table} WHERE payload = ? AND pending = ? AND owner = ? AND session_key = ? AND action_id = ? AND {cast} IS NOT NULL AND ? >= 0",
    )
