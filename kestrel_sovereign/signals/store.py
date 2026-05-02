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
        try:
            await self._backend.execute(
                f"ALTER TABLE {self.TABLE} ADD COLUMN result_summary TEXT"
            )
        except Exception as e:
            msg = str(e).lower()
            if "duplicate column" in msg or "already exists" in msg:
                pass  # additive migration already applied
            else:
                logger.warning(
                    "result_summary ALTER failed (proceeding without it): %s", e,
                )

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
        logger.info(f"SignalLogStore initialized ({self._backend.backend_type})")

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    async def append(
        self,
        signal: Signal,
        registration: SourceRegistration,
        result: SignalResult,
    ) -> Optional[str]:
        """Persist a dispatch outcome. Payload redaction runs HERE
        (the dispatcher does not see the redacted form before this
        call); per-source result UI summarization also runs here so
        the bounded body is stored AND returned for the SSE payload.

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
            if len(summary) > MAX_RESULT_SUMMARY_BYTES:
                summary = summary[:MAX_RESULT_SUMMARY_BYTES] + "...(truncated)"
            result_summary = summary

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

        await self._backend.execute(
            f"""
            INSERT INTO {self.TABLE} (
                id, source, kind, mode, urgency, dedupe_key,
                target_agent, session_id, caller_redacted, visibility,
                dispatched_at, completed_at, status, duration_ms,
                turn_id, artifact_digest, action_result_digest,
                payload_digest, payload_redacted, payload_raw,
                result_summary,
                causation_chain_digest, error, retention_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
