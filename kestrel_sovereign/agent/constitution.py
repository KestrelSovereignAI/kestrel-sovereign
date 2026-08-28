"""Constitution verification and integrity checking for Kestrel Agent."""
import asyncio
import logging
import hashlib
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Mapping, Optional, Tuple
from enum import Enum
from datetime import datetime, timezone

from kestrel_sovereign.command_policy import (
    GENESIS_AUDIT_BYPASS_COMMANDS,
    prefixed_command_token,
)

# Import the concrete submodule, not the ``storage`` package aggregator.
# constitution.py is pulled into the LLMService import chain (via
# ``agent/__init__`` -> token_counter), and the storage package itself
# reaches back into LLMService at import time when sizing the
# conversation-history embedding column. Importing from the package
# top-level forces ``storage/__init__`` to finish binding ``GraphNode``
# first, which it cannot while it is the thing mid-initialization —
# yielding a "partially initialized module" ImportError that silently
# disables provider embeddings. Importing the submodule directly breaks
# that cycle (see #1792).
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.privacy_wrapper import (
    acquire_control_plane_capability,
)


class SafeModeCause(str, Enum):
    """Why cognition is restricted.

    Amendment III makes Safe Mode the response to *integrity* failure, and
    requires any discrepancy to be reported to the Sovereign — on the stated
    grounds that "the system cannot be permitted to lie about itself". So the
    cause has to be recorded rather than inferred: reporting a store outage as
    an integrity failure tells the Sovereign their constitution was violated
    when it was not, and that is the lie the amendment exists to prevent.
    """

    #: The constitution did not verify. The only cause Amendment III names.
    INTEGRITY = "integrity"
    #: The governing constitution could not be assembled at startup.
    BOOTSTRAP = "bootstrap"
    #: Authoritative runtime state could not be READ. An availability fact.
    STATE_UNAVAILABLE = "state_unavailable"
    #: The restriction is held in memory only, because the store could not be
    #: written. It is real, but it is not durable, and saying otherwise would
    #: promise a latch that does not survive a restart.
    STATE_NOT_PERSISTED = "state_not_persisted"
    #: Durable state read cleanly, but the identity node it describes is
    #: gone. Not an outage — the read succeeded — and not a hash mismatch
    #: either. Amendment III requires the discrepancy be reported, so it is
    #: named rather than folded into a neighbour.
    IDENTITY_MISSING = "identity_missing"
    #: Stored conversation memory could not be decrypted. An availability
    #: failure of the memory, not of governance state and not of the
    #: constitution — saying "governance state could not be read" would
    #: misdirect the operator to a store that is answering fine.
    MEMORY_UNREADABLE = "memory_unreadable"
    #: Restored from a durable record written before causes were recorded.
    #: Deliberately not INTEGRITY: an unrecorded cause is not evidence of one.
    UNRECORDED = "unrecorded"


#: What to tell the Sovereign, per recorded cause. Amendment III requires the
#: discrepancy be reported; it does not permit reporting a different one, and
#: three separate surfaces were each hard-coding "an integrity failure".
_RESTRICTION_PHRASES = {
    SafeModeCause.INTEGRITY.value: "an integrity failure",
    SafeModeCause.BOOTSTRAP.value: "an incomplete constitution bootstrap",
    SafeModeCause.STATE_UNAVAILABLE.value: (
        "governance state that could not be read"
    ),
    SafeModeCause.STATE_NOT_PERSISTED.value: (
        "governance state that could not be saved"
    ),
    SafeModeCause.IDENTITY_MISSING.value: "a missing agent identity record",
    SafeModeCause.MEMORY_UNREADABLE.value: (
        "stored memory that could not be decrypted"
    ),
    SafeModeCause.UNRECORDED.value: "a restriction whose cause was not recorded",
}


def describe_safe_mode_restriction(agent, *, audit_pending: bool = False) -> str:
    """Phrase the restriction from what was recorded, not from a default.

    A startup audit outranks the stored cause because it describes why
    cognition is refused right now. Otherwise the recorded cause decides, and
    an absent one says so rather than borrowing integrity's name.
    """
    if audit_pending:
        return "a required startup integrity audit"
    cause = getattr(agent, "_safe_mode_cause", None)
    phrase = _RESTRICTION_PHRASES.get(
        cause, "a restriction whose cause was not recorded"
    )
    # A latch held only in memory is a second active fact, and the cause slot
    # deliberately keeps the stronger one. Health reports both; the Sovereign
    # was told only the first, and "not durable" is exactly what they need to
    # know before restarting.
    if (
        getattr(agent, "_constitution_state_persistence_pending", False)
        and cause != SafeModeCause.STATE_NOT_PERSISTED.value
    ):
        phrase += " (this restriction could not be saved and will not survive a restart)"
    return phrase


class ConstitutionMixin:
    """Mixin class providing constitution verification methods."""

    AUDIT_INTERVAL = int(os.environ.get("KESTREL_AUDIT_INTERVAL", "100"))

    # ------------------------------------------------------------------
    # SignalDispatcher constitutional-injection hooks (#1137 chunk 1G)
    #
    # The dispatcher consults these via `getattr(agent, ...)` for
    # COGNITION sources with `constitution_injection="full"`. They
    # populate signal_log's per-dispatch audit row and gate
    # doctrine-bundle drift detection.
    #
    # The constitution-hash hook is trivially shippable in Phase 1
    # because the value already lives on the identity node. The
    # doctrine-bundle hooks return None by default; agents that want
    # drift detection override `compute_live_doctrine_bundle_hash` to
    # invoke `kestrel_sovereign.agent.doctrine_bundle.compute_doctrine_bundle_hash`
    # with their project_root + bootstrap files. Phase 2 of the epic
    # wires this on KestrelAgent globally.
    # ------------------------------------------------------------------

    async def get_constitution_hash(self):
        """Return the agent's anchored constitution hash, or None.

        Reads `agent_node.properties["constitution_hash"]`. Returns
        None if the agent has no identity node yet (pre-anchor) or
        the property hasn't been set.
        """
        try:
            agent_node = await self.storage.get_node(self.agent_id)
        except Exception:
            logging.exception(
                "get_constitution_hash: agent_node lookup failed; "
                "returning None so dispatcher records NULL"
            )
            return None
        if agent_node is None:
            return None
        return agent_node.properties.get("constitution_hash")

    async def ensure_doctrine_bundle_anchored(self):
        """Auto-anchor the doctrine bundle if no anchor exists yet.

        Codex round-18 P1 fix: without this, agents upgraded to
        Phase 1 would have `agent_node.properties['doctrine_bundle_hash']`
        unset, the dispatcher's drift check would skip (anchored=None),
        and edits to AGENTS.md / TORTOISE_DOCTRINE.md would be
        accepted indefinitely — defeating the per-dispatch drift
        protection.

        Behavior:
        - If already anchored, return the existing hash.
        - Otherwise compute the live bundle hash and write it to
          agent_node.properties as the anchor, returning the hash.
        - Returns None if project_root is unresolvable (no checkout
          context) — drift detection then stays skipped, but the
          dispatch still runs.
        """
        try:
            agent_node = await self.storage.get_node(self.agent_id)
        except Exception:
            logging.exception(
                "ensure_doctrine_bundle_anchored: agent_node lookup failed"
            )
            return None
        if agent_node is None:
            return None

        from kestrel_sovereign.agent.doctrine_bundle import (
            PROP_BUNDLE_ANCHORED_AT,
            PROP_BUNDLE_FILES,
            PROP_BUNDLE_HASH,
        )

        existing = agent_node.properties.get(PROP_BUNDLE_HASH)
        if existing:
            return existing

        live_hash = await self.compute_live_doctrine_bundle_hash()
        if not live_hash:
            return None

        agent_node.properties[PROP_BUNDLE_HASH] = live_hash
        agent_node.properties[PROP_BUNDLE_ANCHORED_AT] = datetime.now(
            timezone.utc
        ).isoformat()
        # Recompute the contributing files list so an auditor knows
        # which files went into the anchored hash. Codex round-22 P3:
        # include operator-declared `doctrine_anchored_paths` so the
        # file list matches the hash for that extensibility case.
        try:
            from kestrel_sovereign.agent.doctrine_bundle import (
                PROP_BUNDLE_ANCHORED_PATHS,
                compute_doctrine_bundle_hash,
                resolve_anchored_paths,
            )

            project_root = await self._resolve_project_root_for_doctrine()
            if project_root is not None:
                extra_paths = (
                    agent_node.properties.get(PROP_BUNDLE_ANCHORED_PATHS) or []
                )
                paths = resolve_anchored_paths(
                    project_root=project_root, extra_paths=extra_paths
                )
                cb = getattr(self, "context_builder", None)
                bootstrap = OrderedDict()
                if cb is not None:
                    try:
                        bootstrap = OrderedDict(cb._bootstrap_files)
                    except Exception:
                        bootstrap = OrderedDict()
                snapshot = compute_doctrine_bundle_hash(
                    anchored_files=paths, bootstrap_files=bootstrap
                )
                agent_node.properties[PROP_BUNDLE_FILES] = list(snapshot.files)
        except Exception:
            logging.exception(
                "ensure_doctrine_bundle_anchored: file-list snapshot failed; "
                "anchoring hash without file list"
            )

        try:
            await self.storage.add_node(agent_node, capability=acquire_control_plane_capability())
        except Exception:
            logging.exception(
                "ensure_doctrine_bundle_anchored: agent_node persist failed"
            )
            # Even if persist fails, the in-memory hash is set so this
            # dispatch can proceed; the next call will retry.
        logging.info(
            f"Auto-anchored doctrine bundle: hash={live_hash[:16]}..."
        )
        return live_hash

    async def get_anchored_doctrine_bundle_hash(self):
        """Return the anchored doctrine_bundle_hash from agent_node, or None.

        Default implementation reads
        `agent_node.properties["doctrine_bundle_hash"]`. Returns None
        when no bundle has been anchored — the dispatcher then records
        the live hash without claiming drift.
        """
        try:
            agent_node = await self.storage.get_node(self.agent_id)
        except Exception:
            logging.exception(
                "get_anchored_doctrine_bundle_hash: agent_node lookup failed"
            )
            return None
        if agent_node is None:
            return None
        return agent_node.properties.get("doctrine_bundle_hash")

    async def compute_live_doctrine_bundle_hash(self):
        """Compute the live (filesystem-current) doctrine_bundle_hash, or None.

        Resolution strategy (codex round-16 P2 fix — was previously
        a Phase 2 stub):
        1. `KESTREL_PROJECT_ROOT` env var (operator override).
        2. Walk up from this module's `__file__` looking for `.git`
           or `pyproject.toml` (development checkouts).
        3. Return None if neither is available — drift detection
           skipped but no exception raised.

        With a project_root resolved, hashes the doctrine bundle:
        anchored doctrine paths (DEFAULT_ANCHORED_PATHS plus any
        operator-declared additions on `agent_node.properties[
        "doctrine_anchored_paths"]`) + bootstrap files from
        `self.context_builder._bootstrap_files` if available.
        """
        from kestrel_sovereign.agent.doctrine_bundle import (
            compute_doctrine_bundle_hash,
            resolve_anchored_paths,
            PROP_BUNDLE_ANCHORED_PATHS,
        )

        project_root = await self._resolve_project_root_for_doctrine()
        if project_root is None:
            return None

        extra_paths = []
        try:
            agent_node = await self.storage.get_node(self.agent_id)
            if agent_node is not None:
                extra_paths = (
                    agent_node.properties.get(PROP_BUNDLE_ANCHORED_PATHS) or []
                )
        except Exception:
            logging.exception(
                "compute_live_doctrine_bundle_hash: agent_node lookup failed; "
                "computing bundle without operator-extra paths"
            )

        anchored_paths = resolve_anchored_paths(
            project_root=project_root, extra_paths=extra_paths
        )

        # Bootstrap files come from the context_builder when present.
        bootstrap_files: "OrderedDict[str, str]" = OrderedDict()
        cb = getattr(self, "context_builder", None)
        if cb is not None:
            try:
                bootstrap_files = OrderedDict(cb._bootstrap_files)
            except Exception:
                logging.exception(
                    "compute_live_doctrine_bundle_hash: bootstrap loader read failed"
                )
                bootstrap_files = OrderedDict()

        try:
            snapshot = compute_doctrine_bundle_hash(
                anchored_files=anchored_paths,
                bootstrap_files=bootstrap_files,
            )
            return snapshot.hash
        except Exception:
            logging.exception(
                "compute_live_doctrine_bundle_hash: hash computation failed"
            )
            return None

    async def get_anchored_doctrine_files(self):
        """Return an `OrderedDict[name, content]` of anchored doctrine
        files for full-injection dispatches, or None if no project_root
        is resolvable.

        The dispatcher passes this dict to
        `ContextManager.build_context(anchored_doctrine=...)` so the
        budget-aware assembler injects TORTOISE_DOCTRINE.md / AGENTS.md
        into the system prompt (codex round-16 P2 fix). Without this
        wiring, full injection would record the doctrine bundle hash
        for audit but the model would never see the doctrine content.
        """
        from kestrel_sovereign.agent.doctrine_bundle import (
            PROP_BUNDLE_ANCHORED_PATHS,
            resolve_anchored_paths,
        )

        project_root = await self._resolve_project_root_for_doctrine()
        if project_root is None:
            return None

        extra_paths = []
        try:
            agent_node = await self.storage.get_node(self.agent_id)
            if agent_node is not None:
                extra_paths = (
                    agent_node.properties.get(PROP_BUNDLE_ANCHORED_PATHS) or []
                )
        except Exception:
            logging.exception(
                "get_anchored_doctrine_files: agent_node lookup failed"
            )

        anchored_paths = resolve_anchored_paths(
            project_root=project_root, extra_paths=extra_paths
        )

        # Codex round-18 P2: skip the constitution itself when
        # building the anchored-doctrine injection map. The agent's
        # system-prompt path independently includes the constitution
        # via `_get_governing_constitution()`, so emitting it here
        # too would put it in the prompt twice (once as
        # `--- GOVERNING CONSTITUTION ---`, once as
        # `--- KESTREL CONSTITUTION ---`), wasting budget and
        # potentially evicting lower-priority doctrine under the cap.
        # The bundle HASH still includes the constitution file (for
        # drift-detection completeness via `compute_doctrine_bundle_hash`);
        # only the prompt-injection set excludes it.
        skip_names = {"KESTREL_CONSTITUTION.md"}

        # Codex round-23 P2: keys are basenames so the assembler's
        # section labels match. Operators who declare two anchored
        # paths with the same basename get a logged warning and only
        # the FIRST occurrence wins (predictable, reproducible). The
        # alternative — silently overwriting — was the codex finding.
        files: "OrderedDict[str, str]" = OrderedDict()
        for path in anchored_paths:
            if path.name in skip_names:
                continue
            if path.name in files:
                logging.warning(
                    "get_anchored_doctrine_files: duplicate basename %s "
                    "(another path with the same filename already "
                    "registered); skipping %s. Rename or restructure "
                    "the conflicting doctrine path to avoid silent "
                    "doctrine omission.",
                    path.name,
                    path,
                )
                continue
            try:
                files[path.name] = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                # Match doctrine_bundle behavior — missing anchored
                # files are skipped, not an error.
                continue
            except OSError as e:
                logging.warning(
                    "get_anchored_doctrine_files: cannot read %s: %s",
                    path,
                    e,
                )
                continue
        return files

    async def _resolve_project_root_for_doctrine(self) -> Optional[Path]:
        """Best-effort project_root resolution for doctrine bundling.

        Operator override via `KESTREL_PROJECT_ROOT` wins; otherwise
        walk up from this module's `__file__` looking for `.git` or
        `pyproject.toml`. Returns None when neither is available
        (e.g. installed-package deployments without a checkout) —
        the default `compute_live_doctrine_bundle_hash` then
        gracefully reports None and drift detection is skipped.
        """
        env_root = os.environ.get("KESTREL_PROJECT_ROOT")
        if env_root:
            p = Path(env_root)
            if p.exists():
                return p

        # Walk up from this file looking for repo markers.
        candidate = Path(__file__).resolve()
        for parent in candidate.parents:
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                return parent
        return None

    def verify_constitution_echo(
        self,
        *,
        canary: str,
        prompt_template_format: str,
        signal_id: str,
        response,
    ):
        """Default echo verifier — scans the dispatch response for the
        canary using the format-specific primitives.

        Format → channel:
        - `codex`: `response` is a dict with `constitution_canary`
          field (codex CLI structured output). When the dispatcher's
          `process_input` returns the JSON-decoded structured response,
          we use `verify_in_structured_response`. If `response` is a
          raw string (the in-agent claude_code path), fall through to
          JSON extraction.
        - `local`: `response` is the raw model text; parse JSON and
          look for `_canary` field via `verify_in_json_response`.
        - `claude_code`: scans the per-turn phantom
          `_constitution_receipt` tool-call records captured by
          KestrelAgent.
        - `bare`: returns MISSING — caller-responsibility, the default
          can't know how to parse.

        Returns a `CanaryStatus` value (the dispatcher accepts both
        the enum and the string form).
        """
        from kestrel_sovereign.signals.constitution_canary import (
            CODEX_CANARY_FIELD,
            LOCAL_CANARY_FIELD,
            CanaryStatus,
            verify_in_json_response,
            verify_in_structured_response,
            verify_in_tool_calls,
        )

        # Map format → field name. Codex uses `constitution_canary`,
        # local uses `_canary` — codex round-12 P2 caught that the
        # JSON-string fallback was using the wrong default for codex
        # responses returned as raw text.
        if prompt_template_format == "codex":
            field = CODEX_CANARY_FIELD
            if isinstance(response, dict):
                return verify_in_structured_response(
                    response, canary, field=field
                )
            if isinstance(response, str):
                return verify_in_json_response(
                    response, canary, field=field
                )
            return CanaryStatus.MISSING

        if prompt_template_format == "local":
            field = LOCAL_CANARY_FIELD
            if isinstance(response, str):
                return verify_in_json_response(
                    response, canary, field=field
                )
            if isinstance(response, dict):
                return verify_in_structured_response(
                    response, canary, field=field
                )
            return CanaryStatus.MISSING

        if prompt_template_format == "claude_code":
            tool_calls = getattr(self, "_constitution_receipt_tool_calls", [])
            return verify_in_tool_calls(tool_calls, canary)

        # bare → caller-responsibility.
        return CanaryStatus.MISSING

    def _init_constitution_audit_tracking(self):
        """Initialize constitution audit tracking. Called by KestrelAgent.__init__."""
        self._interaction_count = 0
        self._last_audit_time = ConstitutionMixin._constitution_now(self)
        self._safe_mode_reason = None
        self._safe_mode_entered_at = None
        self._safe_mode_exited_at = None
        self._safe_mode_exit_authorization = None
        self._constitution_state_migration_pending = False
        self._constitution_bootstrap_pending = False
        self._constitution_state_load_error = None
        self._constitution_audit_pending = False
        self._constitution_state_persistence_pending = False
        self._safe_mode_cause: Optional[str] = None

    def _constitution_now(self) -> datetime:
        """Return an aware UTC time, using the injected test clock if present."""
        clock = vars(self).get("_constitution_clock")
        now = clock() if callable(clock) else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc)

    @asynccontextmanager
    async def _constitution_state_guard(self):
        """Serialize state transitions, while allowing same-task nesting."""
        lock = vars(self).get("_constitution_state_lock")
        current_task = asyncio.current_task()
        if lock is None or vars(self).get("_constitution_state_lock_owner") is current_task:
            yield
            return
        async with lock:
            self._constitution_state_lock_owner = current_task
            try:
                yield
            finally:
                self._constitution_state_lock_owner = None

    @staticmethod
    def _constitution_epoch() -> datetime:
        """Sentinel used when no successful full audit has ever been recorded."""
        return datetime.fromtimestamp(0, timezone.utc)

    def _constitution_state_snapshot(
        self,
        *,
        now: Optional[datetime] = None,
        safe_mode: Optional[bool] = None,
        last_successful_audit_at: Optional[datetime] = None,
        interaction_count: Optional[int] = None,
        bootstrap_pending: Optional[bool] = None,
    ):
        """Build the database value from the current in-memory state."""
        from kestrel_sovereign.constitution.runtime_state import (
            ConstitutionRuntimeState,
        )

        audit_at = (
            self._last_audit_time
            if last_successful_audit_at is None
            else last_successful_audit_at
        )
        if audit_at <= self._constitution_epoch():
            audit_at = None
        return ConstitutionRuntimeState(
            agent_id=self.agent_id,
            safe_mode=self._safe_mode if safe_mode is None else safe_mode,
            safe_mode_reason=self._safe_mode_reason,
            safe_mode_cause=self._safe_mode_cause,
            safe_mode_entered_at=self._safe_mode_entered_at,
            safe_mode_exited_at=self._safe_mode_exited_at,
            safe_mode_exit_authorization=self._safe_mode_exit_authorization,
            last_successful_audit_at=audit_at,
            interaction_count=(
                self._interaction_count
                if interaction_count is None
                else interaction_count
            ),
            updated_at=now or self._constitution_now(),
            bootstrap_pending=(
                self._constitution_bootstrap_pending
                if bootstrap_pending is None
                else bootstrap_pending
            ),
        )

    async def _initialize_constitution_runtime_state(
        self, *, is_new_identity: bool = False
    ) -> None:
        """Restore Safe Mode and audit deadlines before the agent becomes ready.

        A missing row for an existing identity is an explicit legacy migration:
        no successful audit is invented. A truly new identity instead receives
        a durable bootstrap marker so it can establish its first anchor and
        immediately prove it through the same full verifier. Persisting that
        distinction prevents a crash between node creation and verification
        from turning a new identity into an unsafe legacy auto-anchor.
        """
        from kestrel_sovereign.constitution.runtime_state import (
            ConstitutionRuntimeStateStore,
        )

        pending_entry = bool(
            vars(self).get("_constitution_state_persistence_pending")
            and self._safe_mode
        )
        pending_reason = self._safe_mode_reason
        pending_entered_at = self._safe_mode_entered_at
        # Captured with the rest: restoring the prior row overwrites the
        # buffered cause before this is persisted, so the new restriction
        # would be written with the previous row's cause or none at all.
        pending_cause = vars(self).get("_safe_mode_cause")

        async def persist_pending_entry(store) -> None:
            if not pending_entry:
                return
            self._safe_mode = True
            self._safe_mode_reason = pending_reason
            self._safe_mode_cause = pending_cause
            self._safe_mode_entered_at = pending_entered_at
            self._safe_mode_exited_at = None
            self._safe_mode_exit_authorization = None
            now = self._constitution_now()
            await store.write(
                self._constitution_state_snapshot(now=now),
                event_type="safe_mode_entered",
                event_reason=pending_reason,
            )
            self._constitution_state_persistence_pending = False

        try:
            store = ConstitutionRuntimeStateStore(self._raw_storage._backend)
            await store.initialize()
            self._constitution_state_store = store
            state = await store.load(self.agent_id)
            if state is None:
                self._interaction_count = 0
                self._last_audit_time = self._constitution_epoch()
                self._constitution_bootstrap_pending = is_new_identity
                self._constitution_state_migration_pending = not is_new_identity
                self._constitution_audit_pending = True
                now = self._constitution_now()
                await store.write(
                    self._constitution_state_snapshot(now=now),
                    event_type=(
                        "new_identity_bootstrap_required"
                        if is_new_identity
                        else "legacy_state_migration_required"
                    ),
                    event_reason=(
                        "initial anchor and full audit required before readiness"
                        if is_new_identity
                        else "full constitutional audit required before readiness"
                    ),
                )
                await persist_pending_entry(store)
                return

            self._safe_mode = state.safe_mode
            self._safe_mode_reason = state.safe_mode_reason
            # UNRECORDED is for rows written before causes were persisted —
            # a NULL column, not every restart. Labelling a known cause
            # UNRECORDED after a routine restart loses it from the report;
            # calling a missing one INTEGRITY manufactures a constitutional
            # violation out of an absent field. Both are wrong, differently.
            if not state.safe_mode:
                self._safe_mode_cause = None
            else:
                self._safe_mode_cause = (
                    state.safe_mode_cause or SafeModeCause.UNRECORDED.value
                )
            self._safe_mode_entered_at = state.safe_mode_entered_at
            self._safe_mode_exited_at = state.safe_mode_exited_at
            self._safe_mode_exit_authorization = (
                state.safe_mode_exit_authorization
            )
            self._last_audit_time = (
                state.last_successful_audit_at or self._constitution_epoch()
            )
            self._interaction_count = state.interaction_count
            self._constitution_bootstrap_pending = state.bootstrap_pending
            self._constitution_state_migration_pending = (
                state.last_successful_audit_at is None
                and not state.bootstrap_pending
            )
            self._constitution_audit_pending = (
                state.bootstrap_pending
                or state.last_successful_audit_at is None
                or (
                    self._constitution_now() - state.last_successful_audit_at
                ).total_seconds()
                >= 24 * 3600
            )
            await persist_pending_entry(store)
            # A completed runtime record paired with a missing identity node is
            # deletion/corruption, not a second first boot. Only a persisted,
            # still-pending bootstrap marker authorizes initial auto-anchoring.
            if (
                is_new_identity
                and not state.bootstrap_pending
                and not self._safe_mode
            ):
                await self.enter_safe_mode(
                    "Agent identity node missing during constitutional restore",
                    cause=SafeModeCause.IDENTITY_MISSING.value,
                )
            if self._safe_mode:
                logging.critical(
                    "RESTORED DURABLE SAFE MODE for agent %s", self.agent_id
                )
        except Exception as exc:  # noqa: BLE001 - state failure must fail closed
            self._mark_constitution_state_unavailable(exc)

    def _mark_constitution_state_unavailable(
        self,
        exc: Exception,
        *,
        cause: str = SafeModeCause.STATE_UNAVAILABLE.value,
        read_failed: bool = True,
    ) -> None:
        """Keep cognition blocked when authoritative state cannot be trusted.

        ``read_failed`` separates the two ways trust is lost. A failed READ
        leaves the state unknown; a failed WRITE leaves it known but not
        durable. Recording a write failure as a read outage made health report
        an outage that never happened, and ``_constitution_state_load_error``
        is a fact about reading — setting it for a write error is the same
        confusion one field down.
        """
        now = self._constitution_now()
        # Read BEFORE this call sets the flag, or it is always true and the
        # scoping below does nothing.
        was_restricted = bool(getattr(self, "_safe_mode", False))
        self._safe_mode = True
        # Does NOT overwrite an existing reason. An integrity finding followed
        # by a failed state read used to be reported as "state unavailable",
        # which hid a constitutional violation Amendment III requires be
        # reported. The availability fact has its own home below; it does not
        # need this slot, and taking it downgraded the stronger claim.
        #
        # Preserved only while that restriction is still in force. After an
        # authorized exit the reason and cause linger as history, and carrying
        # them forward would report an exited violation as a live one.
        self._safe_mode_reason = (
            (self._safe_mode_reason if was_restricted else None)
            or "Constitution runtime state unavailable"
        )
        self._safe_mode_cause = (
            (self._safe_mode_cause if was_restricted else None) or cause
        )
        self._safe_mode_entered_at = (
            getattr(self, "_safe_mode_entered_at", None) or now
        )
        if read_failed:
            self._constitution_state_load_error = type(exc).__name__
        self._constitution_audit_pending = False
        logging.critical(
            "CONSTITUTION STATE unavailable; remaining in Safe Mode (%s)",
            type(exc).__name__,
        )

    async def _persist_constitution_runtime_state(
        self,
        *,
        now: Optional[datetime] = None,
        event_type: Optional[str] = None,
        event_reason: Optional[str] = None,
        event_authorization: Optional[str] = None,
        safe_mode: Optional[bool] = None,
        last_successful_audit_at: Optional[datetime] = None,
        interaction_count: Optional[int] = None,
        bootstrap_pending: Optional[bool] = None,
    ) -> bool:
        """Persist current state; pre-initialization test harnesses are no-ops."""
        store = vars(self).get("_constitution_state_store")
        if store is None:
            # A lightweight ConstitutionMixin-only unit harness has no store
            # slot at all. A real KestrelAgent has the slot from __init__; None
            # there means initialization is still in flight, so buffer a
            # fail-closed Safe Mode entry for immediate persistence once the DB
            # connects instead of pretending the write succeeded.
            if "_constitution_state_store" not in vars(self):
                return vars(self).get("_constitution_state_load_error") is None
            was_restricted = bool(getattr(self, "_safe_mode", False))
            self._safe_mode = True
            self._safe_mode_reason = (
                (self._safe_mode_reason if was_restricted else None)
                or "Constitution runtime state not initialized"
            )
            self._safe_mode_cause = (
                (self._safe_mode_cause if was_restricted else None)
                or SafeModeCause.STATE_NOT_PERSISTED.value
            )
            self._safe_mode_entered_at = (
                self._safe_mode_entered_at or self._constitution_now()
            )
            self._constitution_state_persistence_pending = True
            return False
        # Decided BEFORE the snapshot is built. Clearing it afterwards left the
        # stale cause in the row just written, so a restart restored
        # ``state_not_persisted`` and health claimed the recovered write had
        # never persisted. The except path re-sets it if this write fails.
        try:
            await store.write(
                self._constitution_state_snapshot(
                    now=now,
                    safe_mode=safe_mode,
                    last_successful_audit_at=last_successful_audit_at,
                    interaction_count=interaction_count,
                    bootstrap_pending=bootstrap_pending,
                ),
                event_type=event_type,
                event_reason=event_reason,
                event_authorization=event_authorization,
            )
            # The snapshot is on disk, so whatever earlier failure set this is
            # no longer true. Leaving it set reported ``state_not_persisted``
            # for the rest of the process after one transient write error.
            # Only the durability FACT clears. The cause is why cognition is
            # restricted — a write that failed really is the trigger — and
            # erasing it to UNRECORDED lost that across a restart, leaving
            # health and `!safe-mode` claiming no cause was recorded. The two
            # are separate questions and only one of them just changed.
            self._constitution_state_persistence_pending = False
            return True
        except Exception as exc:  # noqa: BLE001 - never continue normally
            # The write failed, so whatever this call was recording exists
            # only in memory and will not survive a restart. That is a
            # different fact from the store being unreadable, and only the
            # latter was being recorded — so a Safe Mode entered during a
            # disk-full or disconnected write reported no durability warning
            # at all while promising a latch that "clears only with an
            # authorized exit".
            self._constitution_state_persistence_pending = True
            self._mark_constitution_state_unavailable(
                exc,
                cause=SafeModeCause.STATE_NOT_PERSISTED.value,
                read_failed=False,
            )
            return False

    async def _record_successful_constitution_audit(
        self, *, source: str, audited_at: Optional[datetime] = None
    ) -> bool:
        """Advance the deadline only after a complete verifier succeeds."""
        async with ConstitutionMixin._constitution_state_guard(self):
            return await ConstitutionMixin._record_successful_constitution_audit_locked(
                self, source=source, audited_at=audited_at
            )

    async def _record_successful_constitution_audit_locked(
        self, *, source: str, audited_at: Optional[datetime] = None
    ) -> bool:
        """Locked implementation for successful-audit persistence."""
        now = audited_at or self._constitution_now()
        persisted = await ConstitutionMixin._persist_constitution_runtime_state(
            self,
            now=now,
            event_type="audit_succeeded",
            event_reason=source,
            last_successful_audit_at=now,
            interaction_count=0,
            bootstrap_pending=False,
        )
        if not persisted:
            return False
        self._last_audit_time = now
        self._interaction_count = 0
        self._constitution_state_migration_pending = False
        self._constitution_bootstrap_pending = False
        self._constitution_audit_pending = False
        return True

    async def _begin_explicit_constitution_audit(self) -> bool:
        """Persist a due marker before an operator-triggered full verifier.

        If the verifier fails and the subsequent Safe Mode write encounters a
        storage fault, the due marker still forces another full audit on the
        next restart instead of leaving a recent-success timestamp that could
        make the restarted process look normal.
        """
        async with ConstitutionMixin._constitution_state_guard(self):
            due_count = max(self._interaction_count, self.AUDIT_INTERVAL)
            persisted = await self._persist_constitution_runtime_state(
                event_type="audit_started",
                event_reason="explicit_verification",
                interaction_count=due_count,
            )
            if persisted:
                self._interaction_count = due_count
            return persisted

    async def _run_explicit_constitution_audit(
        self,
    ) -> tuple[Optional[bool], str, bool]:
        """Run an explicit full audit as one serialized state transition."""
        async with ConstitutionMixin._constitution_state_guard(self):
            started = await self._begin_explicit_constitution_audit()
            if not started:
                return (
                    None,
                    "Durable constitutional audit marker unavailable",
                    False,
                )

            is_valid, message = await self._verify_constitution_integrity()
            self._constitution_verified = is_valid
            if is_valid:
                recorded = await (
                    ConstitutionMixin._record_successful_constitution_audit_locked(
                        self, source="explicit_verification"
                    )
                )
                return is_valid, message, recorded

            recorded = await ConstitutionMixin._enter_safe_mode_locked(
                self, message
            )
            return is_valid, message, recorded

    async def _audit_constitution_on_startup(self) -> None:
        """Run a due/migration audit before initialization reports readiness."""
        if self._constitution_state_load_error is not None:
            return
        now = self._constitution_now()
        due = (
            self._constitution_audit_pending
            or self._constitution_state_migration_pending
            or self._last_audit_time <= self._constitution_epoch()
            or (now - self._last_audit_time).total_seconds() >= 24 * 3600
        )
        if not due:
            return
        if self._constitution_bootstrap_pending:
            governing = await self._get_governing_constitution()
            if governing.startswith("Error:"):
                await self.enter_safe_mode(
                    f"Startup constitution bootstrap failed: {governing}",
                    cause=SafeModeCause.BOOTSTRAP.value,
                )
                return
        is_valid, message = await self._verify_constitution_integrity()
        if is_valid:
            await self._record_successful_constitution_audit(
                source="startup", audited_at=now
            )
            return
        await self.enter_safe_mode(f"Startup constitution audit failed: {message}")

    async def _maybe_audit(self):
        """
        Check if an audit is due and trigger it if needed.

        Audits are triggered when:
        - Interaction count reaches AUDIT_INTERVAL (default 100), OR
        - 24 hours have elapsed since the last audit

        Called from process_input() and process_input_streaming().
        """
        # Safe Mode is already restricted. Explicit diagnostic verification is
        # available via !verify-constitution; blocked prompts do not consume or
        # reset the persisted audit deadline.
        if self._safe_mode or vars(self).get("_constitution_audit_pending", False):
            return

        # Lazy initialization for backward compatibility
        if not hasattr(self, '_interaction_count') or not hasattr(self, '_last_audit_time'):
            self._init_constitution_audit_tracking()

        guard = ConstitutionMixin._constitution_state_guard(self)
        async with guard:
            await ConstitutionMixin._maybe_audit_locked(self)

    async def _maybe_audit_locked(self):
        """Increment, persist, and (when due) perform the full audit."""
        now = ConstitutionMixin._constitution_now(self)
        self._interaction_count += 1
        if not await ConstitutionMixin._persist_constitution_runtime_state(
            self, now=now
        ):
            return

        hours_since_audit = (now - self._last_audit_time).total_seconds() / 3600

        if self._interaction_count >= self.AUDIT_INTERVAL or hours_since_audit >= 24:
            logging.info(
                f"Constitution audit triggered: "
                f"interactions={self._interaction_count}, hours={hours_since_audit:.1f}"
            )
            is_valid, message = await self._verify_constitution_integrity()

            if not is_valid:
                # Constitution integrity failure - enter safe mode
                await self.enter_safe_mode(f"Constitution audit failed: {message}")
            else:
                logging.info(f"Constitution audit passed: {message}")
                await ConstitutionMixin._record_successful_constitution_audit_locked(
                    self, source="periodic", audited_at=now
                )

            # Notify audit anchor feature if available
            try:
                for feature in getattr(self, 'features', {}).values():
                    if type(feature).__name__ == 'AuditAnchorFeature':
                        await feature.on_audit_complete({"is_valid": is_valid, "message": message})
                        break
            except Exception as e:
                logging.warning(f"Audit anchor notification failed: {e}")

    async def _verify_constitution_integrity(self) -> Tuple[bool, str]:
        """
        Verify that the constitution file hasn't been tampered with.
        Compares current file hash against the anchored hash in storage.
        """
        # Per-agent overlay verification runs FIRST and UNCONDITIONALLY (#1722) —
        # before any base-constitution early-return — so it covers legacy agents
        # that have an anchored overlay but no base ``constitution_hash``. Without
        # this, those agents would skip the periodic overlay re-read and miss a
        # runtime mutation/removal.
        overlay_valid, overlay_msg = await self.verify_constitution_overlay()
        if not overlay_valid:
            return False, overlay_msg

        agent_node = await self.storage.get_node(self.agent_id)
        if not agent_node:
            return False, "INTEGRITY FAILURE: Agent identity node not found"

        stored_hash = agent_node.properties.get("constitution_hash")
        if not stored_hash:
            # FAIL CLOSED (#2463 review): a missing anchor gives the integrity
            # verifier nothing to prove against. Lazy auto-anchoring (in
            # ``_get_governing_constitution``) is NOT verification — an
            # integrity verifier whose anchor was deleted must fail closed and
            # drive Safe Mode, not silently report success. Legacy migration of
            # pre-anchor agents is a separate, operator-driven concern.
            logging.critical(
                "CONSTITUTION INTEGRITY: no anchored constitution_hash on the "
                "agent identity node (deleted? never anchored?)."
            )
            return False, (
                "INTEGRITY FAILURE: No anchored constitution hash — the agent's "
                "governing anchor is missing. Re-anchor with a signed amendment "
                "artifact before resuming."
            )

        # PROOF 1 — the operative stored blob must be retrievable, decryptable,
        # and hash to the stored anchor (#2463 review). Merely matching the
        # package hash to the anchor property does not prove the agent's own
        # anchored copy is intact: a missing file row, corrupt ciphertext/
        # plaintext, or a wrong data key would otherwise pass silently. Retrieve
        # it (this decrypts through the storage layer), reject None/decrypt
        # failures, and confirm SHA-256(plaintext) == anchor.
        try:
            stored_bytes = await self.storage.retrieve_file(stored_hash)
        except Exception as e:  # noqa: BLE001 — decrypt/IO failure = integrity failure
            logging.critical(
                "CONSTITUTION INTEGRITY: cannot retrieve/decrypt anchored blob "
                "%s: %s", stored_hash, e,
            )
            return False, (
                f"INTEGRITY FAILURE: Cannot retrieve/decrypt the anchored "
                f"constitution blob (missing row / wrong key / corruption): {e}"
            )
        if stored_bytes is None:
            logging.critical(
                "CONSTITUTION INTEGRITY: anchored blob %s is missing.",
                stored_hash,
            )
            return False, (
                "INTEGRITY FAILURE: Anchored constitution blob is missing "
                "(file row deleted?)."
            )
        stored_blob_hash = hashlib.sha256(stored_bytes).hexdigest()
        if stored_blob_hash != stored_hash:
            logging.critical(
                "CONSTITUTION INTEGRITY: anchored blob hash %s != anchor %s",
                stored_blob_hash, stored_hash,
            )
            return False, (
                "INTEGRITY FAILURE: Anchored constitution blob does not hash to "
                "its stored anchor (corruption/tamper)."
            )

        # PROOF 2 — the governance edge (agent --governed_by--> constitution)
        # must exist and point at the anchored constitution hash (#2463 review).
        # Without this an attacker could repoint or delete the governance
        # relationship while leaving the hash property intact.
        try:
            governance_edges = await self.storage.get_edges_from(self.agent_id)
        except Exception as e:  # noqa: BLE001
            logging.critical(
                "CONSTITUTION INTEGRITY: cannot read governance edges: %s", e,
            )
            return False, (
                f"INTEGRITY FAILURE: Cannot read the agent's governance edges: {e}"
            )
        has_governance_edge = any(
            getattr(edge, "label", None) == "governed_by"
            and getattr(edge, "target_id", None) == stored_hash
            for edge in (governance_edges or [])
        )
        if not has_governance_edge:
            logging.critical(
                "CONSTITUTION INTEGRITY: missing/wrong governed_by edge for "
                "anchor %s.", stored_hash,
            )
            return False, (
                "INTEGRITY FAILURE: Missing or mis-targeted governed_by edge to "
                "the anchored constitution (governance relationship tampered?)."
            )

        # PROOF 3 — live-source parity. Recompute the hash from the AUTHORITATIVE
        # packaged governing source
        # through the single production resolver (#2463) — the same source
        # inception anchored — NOT the documentation copy under docs/ (which
        # carries OKF frontmatter and drifts) and NOT the stored blob itself
        # (comparing the blob to its own hash can never detect a mutation of
        # the governing source). For an agent with an active Amendment VIII
        # emancipation contract, the resolver renders the active form so we
        # compare against the correctly-rendered governing bytes.
        from kestrel_sovereign.constitution.emancipation import (
            EmancipationConfigError,
            contract_from_json,
        )
        from kestrel_sovereign.constitution.resolver import (
            resolve_governing_constitution_bytes,
        )

        try:
            contract = contract_from_json(
                agent_node.properties.get("emancipation_contract")
            )
        except EmancipationConfigError as e:
            return False, (
                f"INTEGRITY FAILURE: Anchored emancipation contract is corrupted: {e}"
            )

        try:
            governing_content = resolve_governing_constitution_bytes(contract)
        except Exception as e:
            # FAIL CLOSED (#2463 review): the authoritative packaged governing
            # source is a wheel-shipped data file that MUST always be present
            # and readable. If it is missing, unreadable, or otherwise cannot
            # be resolved (FileNotFoundError / OSError / empty-source
            # ValueError), we CANNOT prove the anchored constitution still
            # matches its governing source — so we must NOT report success
            # merely because the source could not be loaded. Treat any
            # resolution failure as an integrity failure and drive the agent
            # into Safe Mode.
            logging.critical(
                "CONSTITUTION INTEGRITY: cannot resolve the authoritative "
                "governing constitution source: %s",
                e,
            )
            return False, (
                f"INTEGRITY FAILURE: Cannot resolve authoritative governing "
                f"constitution (source missing/unreadable/ambiguous): {e}"
            )

        governing_hash = hashlib.sha256(governing_content).hexdigest()
        if governing_hash != stored_hash:
            logging.critical(
                "CONSTITUTION MISMATCH!\n"
                f"  Anchored:  {stored_hash}\n"
                f"  Governing: {governing_hash}"
            )
            return False, (
                "INTEGRITY FAILURE: Governing constitution has been modified."
            )

        logging.info("Constitution integrity verified against governing source.")
        base_msg = f"Constitution integrity verified. Hash: {stored_hash[:16]}..."
        # Also verify spawn mandate constraints if present
        spawn_valid, spawn_msg = await self._verify_spawn_mandate_constraints()
        if not spawn_valid:
            return False, spawn_msg
        return True, base_msg

    # ------------------------------------------------------------------
    # Per-agent constitution overlay anchoring (#1722)
    #
    # The overlay (`<agent_dir>/CONSTITUTION.md`) can grant DANGEROUS Amendment
    # IX capabilities (shell_execution_host, filesystem_write). Anyone able to
    # write a file next to the agent DB could therefore self-grant host shell —
    # the overlay was never integrity-checked. These methods anchor the overlay
    # hash in the identity node and verify it; the capability gate
    # (ComputerUseFeature._granted_capabilities) only honors overlay grants when
    # ``constitution_overlay_verified`` is True, and the periodic audit fails
    # CLOSED on an unanchored / mutated / removed overlay.
    # ------------------------------------------------------------------

    OVERLAY_HASH_PROPERTY = "constitution_overlay_hash"

    async def verify_constitution_overlay(self) -> Tuple[bool, str]:
        """Verify the per-agent overlay against its anchored hash.

        Sets ``self.constitution_overlay_verified`` and returns
        ``(is_valid, message)``. Decision matrix (overlay sha computed at load
        in ``KestrelAgent.__init__``; anchor read from the identity node):

        * overlay present + anchor matches            → verified, OK
        * overlay present + anchor mismatches          → FAIL (tampered)
        * overlay present + NO anchor                  → FAIL (unanchored: an
          un-vetted overlay must not be trusted for capability grants)
        * overlay absent + anchor present              → FAIL (anchored overlay
          was removed — tampering)
        * overlay absent + no anchor                   → OK (normal agent)

        ``is_valid=False`` drives the integrity audit into safe mode.
        """
        # Re-read the overlay from disk EVERY call (not the __init__-cached hash)
        # so the periodic audit detects live mutation/removal while the agent is
        # running (#1722). Refresh the cached text + sha so the capability gate
        # also sees current content.
        overlay_path = getattr(self, "_constitution_overlay_path", None)
        overlay_sha = None
        if overlay_path is not None and overlay_path.exists():
            try:
                overlay_bytes = overlay_path.read_bytes()
            except OSError as e:
                logging.warning("Could not re-read constitution overlay %s: %s", overlay_path, e)
                overlay_bytes = None
            if overlay_bytes is not None:
                # Hash the raw bytes FIRST so the anchor comparison runs even for
                # a non-UTF-8 overlay; decode defensively (invalid UTF-8 must NOT
                # raise past the comparison and skip fail-closed — codex r3). A
                # non-text overlay yields no parseable grants, which is safe.
                overlay_sha = hashlib.sha256(overlay_bytes).hexdigest()
                try:
                    self.constitution_text = overlay_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    logging.warning(
                        "Constitution overlay %s is not valid UTF-8; no grants parseable.",
                        overlay_path,
                    )
                    self.constitution_text = None
            else:
                self.constitution_text = None
        else:
            # File absent now (never existed, or removed at runtime).
            if getattr(self, "constitution_text", None) is not None and overlay_path is not None:
                self.constitution_text = None
        self._constitution_overlay_sha = overlay_sha

        anchor = None
        try:
            agent_node = await self.storage.get_node(self.agent_id)
            if agent_node:
                anchor = agent_node.properties.get(self.OVERLAY_HASH_PROPERTY)
        except Exception as e:  # noqa: BLE001
            # Can't read the anchor → can't trust an overlay if one is present.
            if overlay_sha is not None:
                self.constitution_overlay_verified = False
                return False, f"OVERLAY INTEGRITY FAILURE: cannot read anchor: {e}"
            self.constitution_overlay_verified = False
            return True, "No overlay; anchor unreadable but not required"

        if overlay_sha is None:
            if anchor:
                # An overlay was anchored before but is now gone — tampering.
                self.constitution_overlay_verified = False
                return False, (
                    "OVERLAY INTEGRITY FAILURE: an anchored constitution overlay "
                    "is missing (was it deleted?)"
                )
            self.constitution_overlay_verified = False
            return True, "No per-agent constitution overlay"

        if not anchor:
            # Present but never anchored → must not be trusted (fail closed).
            self.constitution_overlay_verified = False
            return False, (
                "OVERLAY INTEGRITY FAILURE: per-agent CONSTITUTION.md is present "
                "but NOT anchored. Its capability grants are ignored. Anchor it "
                "with `kestrel constitution anchor-overlay` if it is legitimate."
            )

        if anchor != overlay_sha:
            self.constitution_overlay_verified = False
            logging.critical(
                "CONSTITUTION OVERLAY MISMATCH!\n  Anchored: %s\n  File:     %s",
                anchor, overlay_sha,
            )
            return False, "OVERLAY INTEGRITY FAILURE: overlay has been modified since anchoring."

        self.constitution_overlay_verified = True
        return True, f"Constitution overlay verified. Hash: {overlay_sha[:16]}..."

    async def anchor_constitution_overlay(self) -> Tuple[bool, str]:
        """Anchor the CURRENT overlay's hash in the identity node (trusted action).

        This is the one operation that establishes trust in an overlay; it must
        only be reachable through a trusted channel (the host CLI/operator), not
        from anything an attacker with mere file-write can drive. After
        anchoring, ``constitution_overlay_verified`` becomes True and the
        overlay's Amendment IX grants are honored."""
        overlay_sha = getattr(self, "_constitution_overlay_sha", None)
        if overlay_sha is None:
            return False, "No per-agent constitution overlay to anchor."
        agent_node = await self.storage.get_node(self.agent_id)
        if not agent_node:
            return False, "Agent identity node not found; cannot anchor overlay."
        agent_node.properties[self.OVERLAY_HASH_PROPERTY] = overlay_sha
        await self.storage.add_node(agent_node, capability=acquire_control_plane_capability())  # upsert
        self.constitution_overlay_verified = True
        logging.info("Anchored constitution overlay hash %s", overlay_sha[:16])
        return True, f"Anchored constitution overlay. Hash: {overlay_sha[:16]}..."

    async def _verify_spawn_mandate_constraints(self) -> Tuple[bool, str]:
        """Verify spawn mandate constraints if this agent was spawned.

        If the agent has a spawn_mandate property, validates that its
        scoped constitution constraints are still valid restrictions
        (not grants of new capabilities).

        Returns:
            Tuple of (is_valid, message). Returns (True, ...) if no
            spawn mandate is present.
        """
        spawn_mandate = getattr(self, 'spawn_mandate', None)
        if spawn_mandate is None:
            return True, "No spawn mandate — base constitution only"

        # Lazy import to avoid circular dependency
        from kestrel_sovereign.spawn.scoped_constitution import ScopedConstitution

        parent_features = {
            name for name in getattr(self, 'features', {}).keys()
        }

        scoped = ScopedConstitution(
            base_constitution="",  # Not needed for constraint validation
            additional_constraints=getattr(spawn_mandate, 'additional_constraints', {}),
            features_allowed=getattr(spawn_mandate, 'features_allowed', []),
            parent_features=parent_features,
        )

        is_valid, message = scoped.validate_constraints()
        if not is_valid:
            logging.critical(
                f"SPAWN MANDATE CONSTRAINT VIOLATION: {message}"
            )
            return False, f"SPAWN MANDATE VIOLATION: {message}"

        logging.info("Spawn mandate constraints verified successfully")
        return True, "Spawn mandate constraints verified"

    async def enter_safe_mode(
        self, reason: str, *, cause: str = SafeModeCause.INTEGRITY.value
    ):
        """Enter Safe Mode and durably record the security boundary.

        ``cause`` defaults to INTEGRITY because that is what Amendment III
        names, and every caller that does not say otherwise is reporting a
        failed verification. Callers restricting cognition for a different
        reason must say so — the report must not have to guess.
        """
        async with ConstitutionMixin._constitution_state_guard(self):
            return await ConstitutionMixin._enter_safe_mode_locked(
                self, reason, cause=cause
            )

    async def _enter_safe_mode_locked(
        self, reason: str, *, cause: str = SafeModeCause.INTEGRITY.value
    ) -> bool:
        """Locked implementation of :meth:`enter_safe_mode`."""
        # Record agent consent before entering safe mode
        consent = self.features.get("ConsentFeature") if hasattr(self, 'features') else None
        if consent:
            try:
                await consent.request_consent(
                    "safe_mode_entry",
                    {"reason": reason},
                )
            except Exception:
                pass  # Never block on consent failure -- safe mode is critical

        now = self._constitution_now()
        was_already_safe = self._safe_mode
        self._safe_mode = True
        self._safe_mode_reason = reason
        self._safe_mode_cause = cause
        self._safe_mode_entered_at = (
            (getattr(self, "_safe_mode_entered_at", None) or now)
            if was_already_safe
            else now
        )
        self._safe_mode_exited_at = None
        self._safe_mode_exit_authorization = None
        self._constitution_audit_pending = False
        persisted = await self._persist_constitution_runtime_state(
            now=now,
            event_type="safe_mode_entered",
            event_reason=reason,
            safe_mode=True,
        )
        logging.critical(f"ENTERING SAFE MODE: {reason}")
        privacy_agent = getattr(self, "privacy_agent", None)
        if privacy_agent is not None:
            try:
                await privacy_agent.add_conversation(
                    role="system",
                    content=f"SAFE MODE ACTIVATED: {reason}",
                    metadata={
                        "event": "safe_mode",
                        "reason": reason,
                        "timestamp": self._get_timestamp(),
                    },
                )
            except Exception as exc:  # noqa: BLE001 - boundary is already durable
                logging.warning(
                    "Could not append Safe Mode conversation event: %s",
                    type(exc).__name__,
                )
        return persisted

    async def exit_safe_mode(self, authorization: str = None):
        """Exit Safe Mode after explicit authority and a fresh full audit."""
        async with ConstitutionMixin._constitution_state_guard(self):
            return await ConstitutionMixin._exit_safe_mode_locked(
                self, authorization
            )

    async def _exit_safe_mode_locked(self, authorization: str = None):
        """Locked implementation of :meth:`exit_safe_mode`."""
        if not self._safe_mode:
            return "Not in safe mode."
        if not authorization:
            return "Safe Mode remains active: explicit Sovereign authorization is required."

        is_valid, message = await self._verify_constitution_integrity()
        if not is_valid:
            await self.enter_safe_mode(f"Safe Mode exit verification failed: {message}")
            return f"Safe Mode remains active: integrity verification failed: {message}"

        now = self._constitution_now()
        old_reason = self._safe_mode_reason
        old_exited_at = self._safe_mode_exited_at
        old_exit_authorization = self._safe_mode_exit_authorization
        self._safe_mode_exited_at = now
        self._safe_mode_exit_authorization = authorization
        persisted = await self._persist_constitution_runtime_state(
            now=now,
            event_type="safe_mode_exited",
            event_reason=old_reason,
            event_authorization=authorization,
            safe_mode=False,
            last_successful_audit_at=now,
            interaction_count=0,
            bootstrap_pending=False,
        )
        if not persisted:
            self._safe_mode_exited_at = old_exited_at
            self._safe_mode_exit_authorization = old_exit_authorization
            return "Safe Mode remains active: constitutional state could not be persisted."

        self._safe_mode = False
        self._last_audit_time = now
        self._interaction_count = 0
        self._constitution_state_migration_pending = False
        self._constitution_bootstrap_pending = False
        self._constitution_audit_pending = False
        logging.warning(f"EXITING SAFE MODE. Authorization: {authorization or 'none provided'}")
        privacy_agent = getattr(self, "privacy_agent", None)
        if privacy_agent is not None:
            try:
                await privacy_agent.add_conversation(
                    role="system",
                    content="SAFE MODE DEACTIVATED after successful integrity verification.",
                    metadata={
                        "event": "safe_mode_exit",
                        "authorization": authorization,
                        "timestamp": self._get_timestamp(),
                    },
                )
            except Exception as exc:  # noqa: BLE001 - transition is already durable
                logging.warning(
                    "Could not append Safe Mode exit conversation event: %s",
                    type(exc).__name__,
                )
        return "Safe mode deactivated after successful integrity verification."

    def _trusted_sovereign_did_document(
        self,
        agent_node: Optional[GraphNode] = None,
    ) -> dict:
        """Return the operator-pinned Sovereign DID document for amendments.

        The amendment signer must be an authority outside the running agent's
        own signing identity. Never fall back to ``self.identity``: the agent
        holds that private key and accepting it would make reanchor
        self-authorizing. ``agent_node`` remains in the signature for caller
        compatibility but is deliberately ignored: protected DB state cannot
        establish the key that authorizes mutation of that state (#2499).
        """
        del agent_node
        from kestrel_sovereign.constitution.trust_root import (
            load_sovereign_trust_root,
        )

        return load_sovereign_trust_root(
            explicit_path=getattr(self, "_sovereign_trust_root_path", None),
            agent_dids=self._agent_signing_dids(),
        )

    def _agent_signing_dids(self) -> set[str]:
        dids: set[str] = set()
        agent_id = getattr(self, "agent_id", None)
        if isinstance(agent_id, str) and agent_id:
            dids.add(agent_id)

        identity = getattr(self, "identity", None)
        if identity is None:
            return dids
        for attr in ("legacy_did", "new_did", "signing_did"):
            value = getattr(identity, attr, None)
            if isinstance(value, str) and value:
                dids.add(value)
        did_doc = getattr(identity, "legacy_did_document", None)
        if isinstance(did_doc, Mapping):
            doc_id = did_doc.get("id")
            if isinstance(doc_id, str) and doc_id:
                dids.add(doc_id)
        return dids

    async def _anchor_constitution_governance(
        self, constitution_hash: str
    ) -> list[str]:
        """Ensure the constitution document node + ``governed_by`` edge exist,
        and that NO other ``governed_by`` edge survives.

        Inception creates a ``document`` node keyed by the constitution content
        hash and links ``agent --governed_by--> constitution`` (issue #2463's
        integrity proof 2 checks that edge). Every OTHER path that changes the
        anchored ``constitution_hash`` (signed reanchor, legacy lazy anchoring)
        MUST maintain the same governance structure, or the periodic audit would
        fail closed on a legitimately re-anchored agent. This mirrors the
        inception wiring so the byte-selection seam and its governance edge stay
        in lockstep.

        Anchoring also prunes every ``governed_by`` edge whose target is not
        ``constitution_hash`` (#2617): without this, each runtime reanchor
        accumulated a dangling edge to the previous constitution — exactly the
        inconsistent governance state the integrity audit exists to prevent.
        The new edge is added BEFORE the prune so the agent never has zero
        governing edges. Returns the pruned edge targets.

        The whole sequence runs in one storage transaction: the underlying
        graph calls otherwise auto-commit one mutation at a time, and a
        failure between the edge add and the prune would commit exactly the
        multi-edge drift state this method exists to remove. The document
        node is created only when MISSING — ``add_node`` is a full-properties
        upsert, so rewriting an existing node would strip its inception
        metadata (``created_at``) from a constitution that hasn't changed.
        """
        pruned: list[str] = []
        async with self.storage.transaction():
            if await self.storage.get_node(constitution_hash) is None:
                constitution_node = GraphNode(
                    node_id=constitution_hash,
                    node_type="document",
                    label="KESTREL_CONSTITUTION",
                    properties={
                        "hash": constitution_hash,
                        "type": "Constitution",
                        "created_at": self._get_timestamp(),
                    },
                )
                await self.storage.add_node(constitution_node)
            await self.storage.add_edge(
                self.agent_id, constitution_hash, "governed_by"
            )
            edges = await self.storage.get_edges_from(self.agent_id)
            for edge in edges or []:
                if (
                    getattr(edge, "label", None) == "governed_by"
                    and getattr(edge, "target_id", None) != constitution_hash
                ):
                    await self.storage.delete_edge(
                        self.agent_id, edge.target_id, "governed_by"
                    )
                    pruned.append(edge.target_id)
        if pruned:
            logging.warning(
                "Pruned %d stale governed_by edge(s) while anchoring %s: %s",
                len(pruned),
                constitution_hash[:16],
                ", ".join(t[:16] for t in pruned),
            )
        return pruned

    async def reanchor_constitution(
        self,
        expected_hash: str = None,
        authorization: str = None,
        amendment_artifact_path: str = None,
    ) -> str:
        """Re-anchor the agent to the current constitution on disk.

        Use after a legitimate constitution update (e.g. amendment ratification).
        Requires a detached artifact signed by the Sovereign root key. The
        expected hash prefix is retained only as an operator integrity hint; it
        is not the trust boundary. Does NOT auto-exit safe mode — use
        !safe-mode exit separately after verifying.

        Args:
            expected_hash: Optional hash prefix (min 8 chars) of the new constitution.
            authorization: Who authorized this re-anchor (logged in audit trail).
            amendment_artifact_path: Path to a signed amendment/reanchor artifact.
        """
        if not amendment_artifact_path:
            return (
                "Error: Signed amendment artifact required. "
                "Usage: !reanchor-constitution <artifact.json> [expected_hash_prefix]"
            )
        if expected_hash and len(expected_hash) < 8:
            return "Error: Expected hash prefix must be at least 8 characters."

        agent_node = await self.storage.get_node(self.agent_id)
        if not agent_node:
            return "Error: Agent identity node not found."

        old_hash = agent_node.properties.get("constitution_hash", "none")

        # Resolve the new governing bytes through the SINGLE production resolver
        # (#2463) reading the authoritative packaged source
        # (config.CONSTITUTION_PATH), rendered to this agent's anchored
        # Amendment VIII active form — NOT the documentation copy under docs/
        # (which carries OKF frontmatter and drifts). Reanchoring off the docs
        # copy would anchor a hash the periodic audit — which recomputes from
        # the packaged source — could never match, false-tripping Safe Mode.
        from kestrel_sovereign.config import CONSTITUTION_PATH
        from kestrel_sovereign.constitution.emancipation import (
            EmancipationConfigError,
            contract_from_json,
        )
        from kestrel_sovereign.constitution.resolver import (
            resolve_governing_constitution_bytes,
        )

        try:
            reanchor_contract = contract_from_json(
                agent_node.properties.get("emancipation_contract")
            )
        except EmancipationConfigError as e:
            return (
                f"Error: Anchored emancipation contract is corrupted: {e}. "
                f"Refusing to reanchor without a clean structured receipt."
            )

        constitution_path_used = CONSTITUTION_PATH
        try:
            constitution_content = resolve_governing_constitution_bytes(
                reanchor_contract,
                constitution_path=CONSTITUTION_PATH,
            )
        except FileNotFoundError:
            return "Error: No constitution file found on disk."
        except Exception as e:
            # FAIL CLOSED: an unreadable/ambiguous governing source must not be
            # anchored (#2463).
            return f"Error: Cannot resolve authoritative governing constitution: {e}"

        new_hash = hashlib.sha256(constitution_content).hexdigest()

        if expected_hash and not new_hash.startswith(expected_hash):
            logging.critical(
                f"REANCHOR REJECTED: expected prefix {expected_hash} "
                f"does not match file hash {new_hash}"
            )
            return (
                f"Error: Hash mismatch. File hash {new_hash[:16]}... "
                f"does not start with expected prefix '{expected_hash}'."
            )

        # Amendment VIII on an agent with NO structured receipt (#2465). Its
        # anchored bytes are the only record of the contract, so ``reanchor_
        # contract`` above is None and the resolver just rendered the dormant
        # canonical text — which a Sovereign-signed artifact over those exact
        # bytes would then authorize, erasing the authored terms. Refuse before
        # any crypto or write. Shared with the offline CLI so the two entry
        # points cannot diverge on this.
        if old_hash != "none":
            from kestrel_sovereign.constitution.anchored_bytes import (
                read_anchored_constitution,
            )
            from kestrel_sovereign.constitution.emancipation import (
                unwitnessed_emancipation_downgrade,
            )

            # ABSENT and UNREADABLE are different answers, and only the
            # ungoverned connection can tell them apart: ``self.storage`` is
            # bound to this agent, so a blob with no ``file_owners`` row reads
            # back as absent — the state of every agent in the cohort this
            # guard protects whose governance edge has drifted (#2649/#2616).
            # See :mod:`kestrel_sovereign.constitution.anchored_bytes`.
            db = getattr(getattr(self, "_raw_storage", None), "db", None)
            if db is None:
                # Without it the question cannot be asked at all, and the
                # answer decides whether an irrevocable right is erased.
                logging.critical(
                    "REANCHOR REJECTED: no ungoverned storage connection to "
                    "read the anchored constitution with"
                )
                return (
                    "Error: Refusing to reanchor: this agent has no storage "
                    "connection able to read its anchored constitution, so an "
                    "active Amendment VIII cannot be ruled out (#2465). "
                    "Restart the agent and re-run."
                )
            try:
                anchored_text, anchored_present = await read_anchored_constitution(
                    db, old_hash
                )
            except Exception as exc:  # noqa: BLE001 — a database failure, not a key one
                # Undecryptable bytes come back as UNREADABLE; anything that
                # escapes is the storage layer itself failing, and that is a
                # different message. Still fails closed — an active Amendment
                # VIII cannot be ruled out either way.
                logging.critical(
                    "REANCHOR REJECTED: could not read the anchored "
                    "constitution %s: %r", old_hash[:12], exc,
                )
                return (
                    f"Error: Could not read this agent's anchored constitution "
                    f"({old_hash[:12]}…): {exc!r}. An active Amendment VIII "
                    "cannot be ruled out, so nothing was written (#2465)."
                )
            downgrade = unwitnessed_emancipation_downgrade(
                anchored_contract=reanchor_contract,
                anchored_text=anchored_text,
                anchored_present=anchored_present,
                old_hash=old_hash,
                new_hash=new_hash,
                new_text=constitution_content.decode("utf-8"),
            )
            if downgrade is not None:
                logging.critical("REANCHOR REJECTED: %s", downgrade)
                return f"Error: {downgrade}"

        from kestrel_sovereign.constitution.trust_root import (
            SovereignTrustRootError,
        )

        try:
            trusted_did_document = self._trusted_sovereign_did_document(agent_node)
        except SovereignTrustRootError as exc:
            return f"Error: {exc}"

        from kestrel_sovereign.constitution.amendment_artifact import (
            AmendmentArtifactError,
            load_verified_reanchor_artifact,
        )

        artifact_path_used = str(amendment_artifact_path)
        try:
            (
                amendment_artifact_bytes,
                amendment_artifact,
                verification,
            ) = load_verified_reanchor_artifact(
                amendment_artifact_path,
                trusted_did_document=trusted_did_document,
                expected_constitution_sha256=new_hash,
            )
        except AmendmentArtifactError as exc:
            logging.critical(
                "REANCHOR REJECTED: %s",
                exc,
            )
            return f"Error: {exc}"

        if new_hash == old_hash:
            # The signed artifact for this exact hash verified above, so
            # converging the governance edges is inside the same
            # authorization envelope (#2617): re-assert the anchored edge
            # and prune any dangling governed_by edges left by pre-fix
            # reanchors, instead of leaving them for the audit to trip on.
            # The helper runs its edge convergence in one transaction, so a
            # mid-prune failure rolls back rather than committing a partial
            # edge set.
            try:
                pruned = await self._anchor_constitution_governance(new_hash)
            except Exception as e:
                return (
                    f"Error: Governance edge cleanup failed and was rolled "
                    f"back; no changes were committed: {e}"
                )
            stale_pruned = [t for t in (pruned or []) if t != old_hash]
            if stale_pruned:
                return (
                    f"Constitution already anchored to current version. "
                    f"Hash: {new_hash[:16]}...\n"
                    f"  Pruned {len(stale_pruned)} stale governed_by "
                    f"edge(s): "
                    + ", ".join(f"{t[:16]}..." for t in stale_pruned)
                )
            return f"Constitution already anchored to current version. Hash: {new_hash[:16]}..."

        from kestrel_sovereign.constitution.genesis_audit import (
            supersede_genesis_audit,
        )
        from kestrel_sovereign.constitution.reanchor_receipt import (
            supersede_constitution_reanchor,
        )

        # Every mutation below — the new constitution blob, the artifact
        # blob + node, the governed_by edge convergence, the superseded
        # genesis receipt, and the agent's constitution_hash pointer — is
        # one transaction. The storage facade otherwise auto-commits each
        # call, so a failure after the edge prune but before the agent-node
        # update would durably commit the property/edge disagreement this
        # very command exists to repair, and a concurrent integrity audit
        # could observe it.
        try:
            async with self.storage.transaction():
                stored_hash = await self.storage.store_file(
                    constitution_content, "KESTREL_CONSTITUTION.md"
                )
                artifact_hash = await self.storage.store_file(
                    amendment_artifact_bytes,
                    "KESTREL_CONSTITUTION.reanchor.signed.json",
                )

                artifact_node = GraphNode(
                    node_id=artifact_hash,
                    node_type="constitution_amendment_artifact",
                    label="Signed Constitution Reanchor Artifact",
                    # Content-derived fields only — see the sibling writer in
                    # ``setup/constitution_reanchor``. The per-agent facts
                    # (``source_path``, when this agent anchored it, and the
                    # verification against *this* agent's trust root) go to the
                    # agent's ``constitution_reanchor`` audit property (#2893).
                    properties={
                        "hash": artifact_hash,
                        "type": "SignedConstitutionAmendment",
                        "artifact_type": amendment_artifact.get("artifact_type"),
                        "constitution_hash": stored_hash,
                        "signer": verification.signer,
                        "created_at": amendment_artifact.get("created_at"),
                    },
                )
                # The runtime and setup reanchor writers touch these shared
                # rows in different semantic order. Take the complete set first
                # so PostgreSQL always observes one canonical lock order.
                await self.storage.lock_nodes_for_update(
                    [artifact_hash, stored_hash]
                )
                await self.storage.add_node(
                    artifact_node,
                    capability=acquire_control_plane_capability(),
                )

                # Maintain the governance document node + governed_by edge
                # for the new anchor so the periodic integrity audit's edge
                # proof (#2463) still holds after a legitimate reanchor.
                # This also prunes every non-target governed_by edge — the
                # old anchor's edge plus any dangling ones (#2617). Its own
                # transaction scope joins this outer one (same task).
                pruned_edges = await self._anchor_constitution_governance(
                    stored_hash
                )
                stale_pruned = [
                    t for t in (pruned_edges or []) if t != old_hash
                ]

                agent_node.properties["constitution_hash"] = stored_hash
                # A signed reanchor changes the bytes governed by the genesis
                # receipt. Preserve the completed receipt as history, then
                # require a fresh audit bound to the new hash. The reanchor
                # command itself remains available; the next ordinary
                # cognition turn completes this explicit pending state.
                supersede_genesis_audit(
                    agent_node.properties,
                    constitution_hash=stored_hash,
                    provenance="runtime:constitution_reanchor",
                    recorded_at=self._get_timestamp(),
                )
                supersede_constitution_reanchor(
                    agent_node.properties,
                    receipt={
                        "timestamp": self._get_timestamp(),
                        "old_hash": old_hash,
                        "new_hash": stored_hash,
                        "path": constitution_path_used,
                        "signed_artifact_hash": artifact_hash,
                        "signed_artifact_path": artifact_path_used,
                        "signed_artifact_signer": verification.signer,
                        "signed_artifact_verification": verification.reason,
                        "authorization": authorization or "unspecified",
                        "expected_hash_prefix": expected_hash,
                    },
                    provenance="runtime:constitution_reanchor",
                    recorded_at=self._get_timestamp(),
                )
                await self.storage.add_node(agent_node, capability=acquire_control_plane_capability())
        except Exception as e:
            return (
                f"Error: Reanchor failed mid-write and was rolled back; "
                f"no changes were committed: {e}"
            )

        logging.warning(
            f"CONSTITUTION RE-ANCHORED by {authorization or 'unspecified'}: "
            f"{old_hash[:16]}... → {stored_hash[:16]}... "
            f"from {constitution_path_used}"
        )

        await self.privacy_agent.add_conversation(
            role="system",
            content=f"Constitution re-anchored. Old: {old_hash[:16]}... New: {stored_hash[:16]}...",
            metadata={
                "event": "constitution_reanchor",
                "old_hash": old_hash,
                "new_hash": stored_hash,
                "signed_artifact_hash": artifact_hash,
                "signed_artifact_signer": verification.signer,
                "authorization": authorization or "unspecified",
                "timestamp": self._get_timestamp(),
            },
        )

        safe_mode_note = ""
        if self._safe_mode:
            safe_mode_note = "\n\n  Agent remains in SAFE MODE. Run !safe-mode exit to resume operation."

        stale_note = ""
        if stale_pruned:
            stale_note = (
                f"\n  Pruned stale governed_by edge(s): "
                + ", ".join(f"{t[:16]}..." for t in stale_pruned)
            )

        return (
            f"Constitution re-anchored successfully.\n"
            f"  Old hash: {old_hash[:16]}...\n"
            f"  New hash: {stored_hash[:16]}...\n"
            f"  Source:   {constitution_path_used}\n"
            f"  Artifact: {artifact_hash[:16]}... signed by {verification.signer}\n"
            f"  Auth:     {authorization or 'unspecified'}"
            f"{stale_note}"
            f"{safe_mode_note}"
        )

    async def _get_governing_constitution(
        self,
        *,
        allow_lazy_anchor: bool = True,
    ) -> str:
        """Retrieve the constitution from the trusted, anchored source.

        ``allow_lazy_anchor=False`` is the diagnostic/read-only path. It reads
        an existing anchor but never creates one, so context-status dry runs can
        measure the same governing bytes as production without causing the
        legacy lazy-anchor transaction.
        """
        agent_node = await self.storage.get_node(self.agent_id)
        if not agent_node:
            return "Error: Agent's own identity node not found in storage."

        constitution_hash = agent_node.properties.get("constitution_hash")
        if not constitution_hash:
            if not allow_lazy_anchor:
                return (
                    "Error: Governing constitution is not anchored; "
                    "read-only retrieval cannot create the missing anchor."
                )
            logging.warning("Constitution hash not found. Attempting to load and anchor default.")

            # Auto-anchor the SAME authoritative packaged governing bytes the
            # periodic audit later recomputes (#2463) via the single production
            # resolver — reading config.CONSTITUTION_PATH rendered to this
            # agent's anchored Amendment VIII form — NOT the docs/ copy (OKF
            # frontmatter → different hash → false Safe Mode on the next audit).
            from kestrel_sovereign.config import CONSTITUTION_PATH
            from kestrel_sovereign.constitution.emancipation import (
                EmancipationConfigError,
                contract_from_json,
            )
            from kestrel_sovereign.constitution.resolver import (
                resolve_governing_constitution_bytes,
            )

            try:
                anchor_contract = contract_from_json(
                    agent_node.properties.get("emancipation_contract")
                )
            except EmancipationConfigError as e:
                return f"Error: Anchored emancipation contract is corrupted: {e}"

            constitution_path_used = CONSTITUTION_PATH
            try:
                constitution_content = resolve_governing_constitution_bytes(
                    anchor_contract,
                    constitution_path=CONSTITUTION_PATH,
                )
            except FileNotFoundError:
                return "Error: No constitution file found."
            except Exception as e:
                # FAIL CLOSED: never anchor an unreadable/ambiguous source.
                return f"Error: Cannot resolve authoritative governing constitution: {e}"

            try:
                # One transaction: blob + governance edges + agent pointer
                # land together or not at all — a partial lazy anchor would
                # be the same property/edge drift #2617 repairs.
                async with self.storage.transaction():
                    constitution_hash = await self.storage.store_file(constitution_content, "KESTREL_CONSTITUTION.md")
                    # Mirror inception's governance wiring so the integrity audit's
                    # edge proof (#2463) holds for a lazily-anchored legacy agent.
                    await self._anchor_constitution_governance(constitution_hash)
                    agent_node.properties["constitution_hash"] = constitution_hash
                    await self.storage.add_node(agent_node, capability=acquire_control_plane_capability())
                logging.info(f"Anchored constitution with hash: {constitution_hash}")
            except Exception as e:
                return f"Error: Failed to anchor constitution: {e}"

        try:
            constitution_bytes = await self.storage.retrieve_file(constitution_hash)
            constitution_text = constitution_bytes.decode('utf-8')
            if self.extension:
                try:
                    amendments = self.extension.get_constitution_amendments()
                    if amendments:
                        constitution_text = f"{constitution_text}\n\n--- APP AMENDMENTS ---\n{amendments.strip()}"
                except Exception:
                    pass
            # Append this spawned child's mandate constraints (#2225) so its
            # behavioral_rules / restrictions reach the model in the governing
            # constitution — the anchored base is left untouched (no hash change;
            # mirrors the runtime APP AMENDMENTS append above). ``spawn_mandate``
            # is attached durably on every boot by #2137's reload path; a
            # restriction only ever tightens, so this cannot weaken the base.
            try:
                from kestrel_sovereign.spawn.scoped_constitution import (
                    render_mandate_constitution_block,
                )

                block = render_mandate_constitution_block(
                    getattr(self, "spawn_mandate", None)
                )
                if block:
                    constitution_text = f"{constitution_text}\n\n{block}"
            except Exception:
                # Rendering coerces to str and shouldn't raise on accepted
                # values; if it somehow does, surface it loudly rather than
                # silently ship a governing constitution missing the mandate's
                # restrictions (the control this adds). Don't abort the whole
                # constitution — that would break the prompt entirely.
                logging.exception(
                    "Failed to render spawn-mandate constraints into governing "
                    "constitution for %s; the mandate block is MISSING from this "
                    "render.", getattr(self, "agent_id", "?"),
                )
            return constitution_text
        except Exception as e:
            return f"Error: Could not retrieve constitution for hash {constitution_hash}. Reason: {e}"

    async def _persist_governance_receipt_node(self, agent_node: GraphNode) -> None:
        """Persist a fresh first-party governance receipt on the agent node.

        The genesis-audit receipt is free-text (audit verdict/reasoning) that no
        per-field check can prove content-free and — being generated fresh at
        first cognition — cannot be carried along, so the privacy wrapper (the
        FEATURE-facing surface) refuses it in a volatile mode by design (#2672
        review P1). The genesis audit is nonetheless a constitutional lifecycle
        boundary that MUST persist regardless of privacy mode to gate cognition,
        so it is written to the RAW store — the same low-level store inception
        uses for the initial agent node — the smallest explicit source-of-truth
        path (#2672 review finding P3). In a persistent mode the raw store and the
        wrapper are behaviorally identical (the wrapper is a pure pass-through),
        so this changes nothing outside volatile modes. Falls back to the wrapper
        WITH the control-plane capability only when no raw store is wired
        (degraded/test paths).
        """
        raw = getattr(self, "_raw_storage", None)
        if raw is not None and hasattr(raw, "add_node"):
            await raw.add_node(agent_node)
        else:
            await self.storage.add_node(
                agent_node, capability=acquire_control_plane_capability()
            )

    async def _persist_genesis_audit_completion(
        self,
        agent_node: GraphNode,
        record: dict,
    ) -> None:
        """Atomically persist the node receipt and conversation witness."""
        agent_node.properties["genesis_audit"] = record
        status = record["status"]
        content = (
            f"Genesis audit {status}. Risk level: {record['risk_level']}. "
            f"{record.get('reasoning', '')}"
        )

        async def _write() -> None:
            await self._persist_governance_receipt_node(agent_node)
            await self.privacy_agent.add_conversation(
                role="system",
                content=content,
                metadata={"event": "genesis_audit", "result": record},
            )

        db = getattr(getattr(self, "_raw_storage", None), "db", None)
        if db is None:
            await _write()
            return
        async with db.transaction():
            await _write()

    async def _persist_genesis_audit_pending_attempt(
        self,
        agent_node: GraphNode,
        *,
        code: str,
        provenance: str,
    ) -> None:
        """Keep an unavailable auditor retryable without leaking diagnostics."""
        from kestrel_sovereign.constitution.genesis_audit import (
            pending_genesis_audit,
            utc_timestamp,
        )

        constitution_hash = agent_node.properties.get("constitution_hash")
        existing = agent_node.properties.get("genesis_audit")
        if not isinstance(existing, dict) or existing.get("status") != "pending":
            existing = pending_genesis_audit(
                constitution_hash,
                provenance="runtime:deferred",
            )
        existing["last_attempt_at"] = utc_timestamp()
        existing["last_attempt_provenance"] = provenance
        existing["last_error"] = code
        existing["audited"] = False
        agent_node.properties["genesis_audit"] = existing
        await self._persist_governance_receipt_node(agent_node)

    async def perform_genesis_audit(
        self,
        *,
        provenance: str = "runtime:explicit",
    ) -> bool:
        """Complete the durable genesis audit once, without silent overwrite."""
        from kestrel_sovereign.constitution.genesis_audit import (
            GENESIS_AUDIT_FAILED,
            GENESIS_AUDIT_PASSED,
            GENESIS_AUDIT_PENDING,
            GenesisAuditError,
            GenesisAuditPendingError,
            GenesisAuditRejectedError,
            evaluate_genesis_constitution,
            validate_completed_genesis_audit,
        )

        logging.info("Agent %s performing genesis self-audit", self.agent_id)
        agent_node = await self.storage.get_node(self.agent_id)
        if agent_node is None:
            raise GenesisAuditError("Genesis audit failed: agent node is missing.")

        constitution_hash = agent_node.properties.get("constitution_hash")
        if not constitution_hash:
            raise GenesisAuditError(
                "Genesis audit failed: governing constitution hash is missing."
            )

        existing = agent_node.properties.get("genesis_audit")
        if isinstance(existing, dict):
            status = existing.get("status")
            # Pre-#2470 completed receipts had no explicit status but did carry
            # timestamp/risk/hash. Upgrade that durable evidence in place and
            # never call the auditor again merely because the schema evolved.
            legacy_risk = existing.get("risk_level")
            if (
                status is None
                and existing.get("timestamp")
                and legacy_risk in (1, 2, 3)
                and existing.get("constitution_hash") == constitution_hash
            ):
                status = (
                    GENESIS_AUDIT_FAILED
                    if legacy_risk >= 3
                    else GENESIS_AUDIT_PASSED
                )
                existing.update(
                    {
                        "status": status,
                        "completed_at": existing["timestamp"],
                        "provenance": "runtime:migrated_legacy_receipt",
                        "audited": True,
                    }
                )
                agent_node.properties["genesis_audit"] = existing
                await self._persist_governance_receipt_node(agent_node)
            if status not in (
                GENESIS_AUDIT_PENDING,
                GENESIS_AUDIT_PASSED,
                GENESIS_AUDIT_FAILED,
            ):
                raise GenesisAuditError("Genesis audit state is malformed.")
            if status in (GENESIS_AUDIT_PASSED, GENESIS_AUDIT_FAILED):
                status = validate_completed_genesis_audit(
                    existing, constitution_hash
                )
                if status == GENESIS_AUDIT_PASSED:
                    return True
                raise GenesisAuditRejectedError(existing)

        # Audit the exact stored bytes named by constitution_hash. The ordinary
        # prompt path may append runtime-only extension/mandate constraints;
        # those are not part of this content-addressed governing receipt.
        try:
            constitution = await self.storage.retrieve_file(constitution_hash)
        except Exception:
            constitution = None
        if not isinstance(constitution, (bytes, str)):
            await self._persist_genesis_audit_pending_attempt(
                agent_node,
                code="constitution_unavailable",
                provenance=provenance,
            )
            raise GenesisAuditPendingError("constitution_unavailable")
        constitution_bytes = (
            constitution
            if isinstance(constitution, bytes)
            else constitution.encode("utf-8")
        )
        if hashlib.sha256(constitution_bytes).hexdigest() != constitution_hash:
            await self._persist_genesis_audit_pending_attempt(
                agent_node,
                code="constitution_hash_mismatch",
                provenance=provenance,
            )
            raise GenesisAuditError(
                "Genesis audit failed: stored governing bytes do not match "
                "the anchored constitution hash."
            )

        try:
            record = await evaluate_genesis_constitution(
                constitution_bytes,
                constitution_hash=constitution_hash,
                auditor=self.get_audit_response,
                provenance=provenance,
            )
        except GenesisAuditPendingError as exc:
            await self._persist_genesis_audit_pending_attempt(
                agent_node,
                code=exc.code,
                provenance=provenance,
            )
            raise
        except GenesisAuditRejectedError as exc:
            await self._persist_genesis_audit_completion(agent_node, exc.record)
            logging.error(
                "Genesis audit rejected governing constitution for %s at risk %s",
                self.agent_id,
                exc.record.get("risk_level"),
            )
            raise

        await self._persist_genesis_audit_completion(agent_node, record)
        logging.info(
            "Genesis self-audit passed for %s at risk %s",
            self.agent_id,
            record["risk_level"],
        )
        return True

    async def _ensure_genesis_audit_ready(self) -> bool:
        """Serialize deferred first-turn audits and refuse all early cognition."""
        from kestrel_sovereign.constitution.genesis_audit import (
            GENESIS_AUDIT_FAILED,
            GENESIS_AUDIT_PASSED,
            GenesisAuditError,
            GenesisAuditRejectedError,
            pending_genesis_audit,
            validate_completed_genesis_audit,
        )

        agent_node = await self.storage.get_node(self.agent_id)
        if agent_node is None:
            raise GenesisAuditError("Genesis audit readiness has no agent node.")

        record = agent_node.properties.get("genesis_audit")
        if record is None:
            # Migrate every pre-#2470 identity. Missing birth metadata is not a
            # safe test signal: a production caller can construct KestrelAgent
            # directly too. Tests/demos must inject or stub the real lifecycle
            # explicitly instead of gaining a production fail-open path.
            constitution_hash = agent_node.properties.get("constitution_hash")
            if not constitution_hash:
                raise GenesisAuditError(
                    "Genesis audit readiness has no governing constitution hash."
                )
            record = pending_genesis_audit(
                constitution_hash,
                provenance="runtime:migrated_legacy_identity",
            )
            agent_node.properties["genesis_audit"] = record
            await self._persist_governance_receipt_node(agent_node)

        if not isinstance(record, dict):
            raise GenesisAuditError("Genesis audit state is malformed.")
        status = record.get("status")
        if status in (GENESIS_AUDIT_PASSED, GENESIS_AUDIT_FAILED):
            status = validate_completed_genesis_audit(
                record,
                agent_node.properties.get("constitution_hash"),
            )
        if status == GENESIS_AUDIT_PASSED:
            return True
        if status == GENESIS_AUDIT_FAILED:
            raise GenesisAuditRejectedError(record)

        lock = getattr(self, "_genesis_audit_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._genesis_audit_lock = lock
        async with lock:
            # A concurrent first turn may have completed while this caller
            # waited. perform_genesis_audit re-reads and refuses overwrite too.
            return await self.perform_genesis_audit(
                provenance="runtime:first_cognition",
            )

    async def _genesis_audit_cognition_block(self, user_input: str) -> str | None:
        """Return a user-facing block message, or ``None`` when ready.

        Only non-cognitive recovery/diagnostic commands bypass the gate. Unknown
        commands and ``!continue`` can fall through to an LLM turn, so they must
        complete genesis just like ordinary text.
        """
        command = prefixed_command_token(user_input)
        if command in GENESIS_AUDIT_BYPASS_COMMANDS:
            return None

        from kestrel_sovereign.constitution.genesis_audit import (
            GenesisAuditError,
            GenesisAuditPendingError,
            GenesisAuditRejectedError,
        )

        try:
            await self._ensure_genesis_audit_ready()
        except GenesisAuditPendingError:
            return (
                "⏳ GENESIS AUDIT PENDING\n\n"
                "This agent cannot begin cognition until an audit-capable LLM "
                "verifies its governing constitution. No cognition request was "
                "sent. Configure an auditor and retry this turn."
            )
        except GenesisAuditRejectedError as exc:
            return (
                "🚨 GENESIS AUDIT FAILED\n\n"
                "The governing constitution was rejected, so cognition remains "
                f"blocked. {exc.record.get('reasoning', 'No reason recorded.')}"
            )
        except GenesisAuditError:
            logging.exception("Genesis audit readiness failed closed")
            return (
                "🚨 GENESIS AUDIT BLOCKED\n\n"
                "The durable genesis-audit receipt is missing or inconsistent. "
                "No cognition request was sent; inspect the agent's audit state."
            )
        return None
