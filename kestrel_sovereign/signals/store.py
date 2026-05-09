"""signal_log persistence.

Privacy is first-class here, not bolted on later (per SIGNAL_DISPATCHER.md
§"Logging & Privacy"):

- UNTRUSTED raw payloads are NEVER stored. Only digest + redacted summary.
- TRUSTED payloads are stored only if the source's RedactionPolicy opts in.
- Caller identifier is redacted by default to role/scope.
- Causation chain is stored as digest; full chain is reconstructible from
  individual entries via signal_id joins.
- Retention is per-source via `registration.retention_days`; a sweep ACTION
  signal handles deletion.

The redaction policy runs BEFORE the row hits the database. There is no
encrypted-at-rest fallback; if a redaction policy is too permissive, raw
data lands in the log. Review policies carefully.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from kestrel_sdk.signals import (
    CausationFrame,
    Signal,
    SignalResult,
    SourceRegistration,
    Trust,
)
from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.storage.db.interface import DatabaseBackend

logger = logging.getLogger(__name__)


def _digest(data: Any) -> str:
    """Stable sha256 over a JSON-serializable value. Used for payload,
    artifact, action_result, and causation chain digests."""
    serialized = json.dumps(
        data, sort_keys=True, default=_json_default, ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return str(obj)


def _serialize_chain(chain: list[CausationFrame]) -> list[dict]:
    return [
        {
            "agent_id": f.agent_id,
            "source": f.source,
            "signal_id": f.signal_id,
            "turn_id": f.turn_id,
            "depth": f.depth,
            "emitted_at": f.emitted_at.isoformat(),
        }
        for f in chain
    ]


# Truncation marker appended when `_truncate_to_bytes` shortens text.
# ASCII so its byte length equals its character length; budgeted INTO
# the cap so the returned string (suffix included) is guaranteed
# encode("utf-8") <= max_bytes.
_TRUNCATION_SUFFIX = "...(truncated)"
_TRUNCATION_SUFFIX_BYTES = len(_TRUNCATION_SUFFIX.encode("utf-8"))


def _truncate_to_bytes(text: str, max_bytes: int) -> str:
    """Truncate ``text`` so its UTF-8 byte length is at most
    ``max_bytes`` (suffix included). UTF-8 boundary safe — a multi-byte
    codepoint that would straddle the cut is dropped via
    ``errors='ignore'`` rather than raising or producing invalid UTF-8.

    The suffix is budgeted INSIDE the cap (not added on top), so
    callers and tests can rely on the byte invariant. Caught in #907
    review P2: the prior `len(summary)` truncation counted Python
    characters and the suffix was added on top, both of which let
    non-ASCII text exceed the documented byte cap.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    body_budget = max_bytes - _TRUNCATION_SUFFIX_BYTES
    if body_budget <= 0:
        # Cap too small to fit even the suffix. Return the suffix
        # truncated to fit; rare/pathological case (cap < ~14 bytes).
        return _TRUNCATION_SUFFIX.encode("utf-8")[:max_bytes].decode(
            "utf-8", errors="ignore"
        )
    truncated_body = encoded[:body_budget].decode("utf-8", errors="ignore")
    return truncated_body + _TRUNCATION_SUFFIX


class SignalLogStore(UnifiedStoreBase):
    """Backend-agnostic signal log persistence."""

    TABLE = "signal_log"

    def __init__(self, backend: DatabaseBackend):
        super().__init__(backend)

    async def initialize(self) -> None:
        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        json_type = self.json_type()

        await self._backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                kind TEXT NOT NULL,
                mode TEXT NOT NULL,
                urgency TEXT NOT NULL,
                dedupe_key TEXT,
                target_agent TEXT NOT NULL,
                session_id TEXT,
                caller_redacted TEXT,
                visibility TEXT NOT NULL,
                dispatched_at {ts_type} {ts_default},
                completed_at {ts_type},
                status TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                turn_id TEXT,
                artifact_digest TEXT,
                action_result_digest TEXT,
                payload_digest TEXT NOT NULL,
                payload_redacted TEXT,
                payload_raw {json_type},
                result_summary TEXT,
                causation_chain_digest TEXT NOT NULL,
                error TEXT,
                retention_until {ts_type} NOT NULL
            )
        """)

        # Phase 7 of #889: result_summary column added so UI consumers
        # of the signal_completed SSE event get a bounded body. Sources
        # opt in by setting `SourceRegistration.result_summary`.
        # Existing databases get the column via this additive ALTER
        # (silently ignored if already applied).
        await self._additive_alter("result_summary TEXT")

        # kestrel-sovereign#1137 chunk 1C — constitutional-injection
        # forensics. Each is an additive ALTER following the same
        # silently-skip-on-already-applied pattern as `result_summary`.
        # See docs/architecture/CONSTITUTION_INJECTION.md v1.4 §5.
        # All columns are NULL for ACTION/ARTIFACT signals (no system
        # prompt, no constitution applied) and for legacy entries that
        # predate the migration.
        await self._additive_alter("constitution_hash TEXT")
        await self._additive_alter("doctrine_bundle_hash TEXT")
        # echo_canary_status: 'verified' | 'missing' | 'not_required'
        await self._additive_alter("echo_canary_status TEXT")
        # injected_clauses_json / dropped_clauses_json: JSON list of
        # clause names. NULL for ACTION/ARTIFACT (no system prompt was
        # built); empty list for COGNITION dispatches that injected
        # nothing or dropped nothing. Kept as TEXT (JSON-validated at
        # write time) for SQLite/Postgres parity.
        await self._additive_alter("injected_clauses_json TEXT")
        await self._additive_alter("dropped_clauses_json TEXT")

        # Indexes for the queries we expect: by source, by target_agent,
        # by status (for failure dashboards), and by retention_until (sweep).
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.TABLE}_source "
            f"ON {self.TABLE}(source)"
        )
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.TABLE}_target "
            f"ON {self.TABLE}(target_agent)"
        )
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.TABLE}_status "
            f"ON {self.TABLE}(status)"
        )
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.TABLE}_retention "
            f"ON {self.TABLE}(retention_until)"
        )
        # kestrel-sovereign#1137 chunk 1C — auditor query: "all dispatches
        # under constitution X." Partial index keeps the index small (most
        # rows are pre-#1137 or non-COGNITION and have NULL constitution_hash).
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.TABLE}_constitution_hash "
            f"ON {self.TABLE}(constitution_hash) "
            f"WHERE constitution_hash IS NOT NULL"
        )
        logger.info(f"SignalLogStore initialized ({self._backend.backend_type})")

    async def _additive_alter(self, column_def: str) -> None:
        """Apply an additive ALTER TABLE ADD COLUMN, silently skipping
        when the column already exists.

        Centralized helper so future additive migrations can mirror the
        Phase 7 #889 / #1137 chunk 1C pattern without copy-pasting the
        try/except. The ALTER is applied via a normal ``execute`` so it
        commits even on backends that auto-commit DDL.
        """
        try:
            await self._backend.execute(
                f"ALTER TABLE {self.TABLE} ADD COLUMN {column_def}"
            )
        except Exception as e:
            msg = str(e).lower()
            if "duplicate column" in msg or "already exists" in msg:
                return  # additive migration already applied; idempotent
            logger.warning(
                "Additive ALTER (%s) failed (proceeding without it): %s",
                column_def,
                e,
            )

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    async def append(
        self,
        signal: Signal,
        registration: SourceRegistration,
        result: SignalResult,
        *,
        constitution_hash: Optional[str] = None,
        doctrine_bundle_hash: Optional[str] = None,
        echo_canary_status: Optional[str] = None,
        injected_clauses: Optional[list[str]] = None,
        dropped_clauses: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Persist a dispatch outcome. Payload redaction runs HERE
        (the dispatcher does not see the redacted form before this
        call); per-source result UI summarization also runs here so
        the bounded body is stored AND returned for the SSE payload.

        Args:
            signal / registration / result: the dispatch context.
            constitution_hash: kestrel-sovereign#1137 chunk 1C — the
                operative constitution_hash for this dispatch (NULL for
                ACTION/ARTIFACT signals or pre-#1137 entries).
            doctrine_bundle_hash: same, for the operative doctrine bundle.
            echo_canary_status: 'verified' | 'missing' | 'not_required'
                per CONSTITUTION_INJECTION.md §3 receipt semantics.
            injected_clauses: ordered list of clause names that ended up
                in the system prompt (NULL for ACTION/ARTIFACT). Empty
                list means "system prompt was assembled but contained
                nothing trackable" — distinct from NULL ("no system
                prompt path").
            dropped_clauses: list of clause names dropped by the
                priority-ordered truncation (NULL when no truncation
                happened or no system prompt path).

        Returns the `result_summary` text the source produced (or None
        when the source didn't set `result_summary` on its
        registration). The caller (dispatcher) uses the return value
        to populate the `signal_completed` SSE payload.
        """
        from kestrel_sdk.signals import MAX_RESULT_SUMMARY_BYTES

        policy = registration.log_redaction
        assert policy is not None, "Registry should reject sources without a policy"

        # Payload redaction (third-party data we don't trust): the
        # incoming signal payload — webhook bodies, A2A metadata, cron
        # args. Raw UNTRUSTED stays in memory until this point and
        # never touches the wire if the policy is honest.
        try:
            payload_redacted = policy.summarize(signal.payload)
        except Exception as e:
            logger.exception(
                "RedactionPolicy.summarize raised for source '%s'; "
                "storing only digest. Fix the policy.",
                signal.source,
            )
            payload_redacted = f"<redaction failed: {type(e).__name__}>"

        # Phase 7 of #889: per-source UI summarization of the OUTPUT
        # body. Different concern from payload redaction — the result
        # is the bird's own output (artifact/action_result/cognition
        # return), bounded for SSE bandwidth and signal_log row size.
        # Sources opt in by setting registration.result_summary; default
        # None means UI consumers see metadata-only signal_completed
        # events. Hard-capped at MAX_RESULT_SUMMARY_BYTES regardless,
        # as defense in depth against a misconfigured callback.
        result_summary: Optional[str] = None
        result_body = (
            result.artifact if result.artifact is not None else result.action_result
        )
        if registration.result_summary is not None and result_body is not None:
            try:
                summary = registration.result_summary(result_body)
            except Exception as e:
                logger.exception(
                    "result_summary callback raised for source '%s'; "
                    "result_summary will be a placeholder. Fix the callback.",
                    signal.source,
                )
                summary = f"<result_summary failed: {type(e).__name__}>"
            if not isinstance(summary, str):
                summary = str(summary) if summary is not None else ""
            result_summary = _truncate_to_bytes(
                summary, MAX_RESULT_SUMMARY_BYTES
            )

        store_raw = (
            registration.trust == Trust.TRUSTED and policy.store_raw_trusted
        )
        payload_raw_json = (
            json.dumps(signal.payload, default=_json_default) if store_raw else None
        )

        caller_redacted: Optional[str] = None
        if signal.caller is not None:
            caller_redacted = (
                "<redacted>" if policy.redact_caller_identifier else signal.caller
            )

        retention_until = datetime.now(timezone.utc) + timedelta(
            days=registration.retention_days
        )

        artifact_digest = (
            _digest(result.artifact) if result.artifact is not None else None
        )
        action_result_digest = (
            _digest(result.action_result)
            if result.action_result is not None
            else None
        )

        # kestrel-sovereign#1137 chunk 1C — serialize clause lists as
        # JSON-validated TEXT for SQLite/Postgres parity. None lists stay
        # NULL in the column; explicit empty lists serialize to "[]".
        injected_clauses_json: Optional[str] = (
            json.dumps(injected_clauses) if injected_clauses is not None else None
        )
        dropped_clauses_json: Optional[str] = (
            json.dumps(dropped_clauses) if dropped_clauses is not None else None
        )

        await self._backend.execute(
            f"""
            INSERT INTO {self.TABLE} (
                id, source, kind, mode, urgency, dedupe_key,
                target_agent, session_id, caller_redacted, visibility,
                dispatched_at, completed_at, status, duration_ms,
                turn_id, artifact_digest, action_result_digest,
                payload_digest, payload_redacted, payload_raw,
                result_summary,
                causation_chain_digest, error, retention_until,
                constitution_hash, doctrine_bundle_hash, echo_canary_status,
                injected_clauses_json, dropped_clauses_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal.id,
                signal.source,
                signal.kind,
                signal.mode.value,
                signal.urgency.value,
                signal.dedupe_key,
                signal.target_agent,
                signal.session_id,
                caller_redacted,
                signal.visibility.value,
                self.to_timestamp_param(signal.arrived_at),
                self.to_timestamp_param(datetime.now(timezone.utc)),
                result.status.value,
                result.duration_ms,
                result.turn_id,
                artifact_digest,
                action_result_digest,
                _digest(signal.payload),
                payload_redacted,
                payload_raw_json,
                result_summary,
                _digest(_serialize_chain(signal.causation_chain)),
                result.error,
                self.to_timestamp_param(retention_until),
                constitution_hash,
                doctrine_bundle_hash,
                echo_canary_status,
                injected_clauses_json,
                dropped_clauses_json,
            ),
        )

        return result_summary

    # ------------------------------------------------------------------
    # Retention sweep
    # ------------------------------------------------------------------

    async def purge_expired(self, *, now: Optional[datetime] = None) -> int:
        """Delete rows past their `retention_until`. Called by the
        retention-sweep ACTION signal (registered separately)."""
        cutoff = now if now is not None else datetime.now(timezone.utc)
        return await self._backend.execute(
            f"DELETE FROM {self.TABLE} WHERE retention_until < ?",
            (self.to_timestamp_param(cutoff),),
        )
