"""
Kestrel Security - Hierarchical Permission Storage.

This module provides SQLite-backed storage for tool permissions with:
- Feature → Tool hierarchy
- Session-scoped overrides
- Rollup state calculation for UI
- Audit logging
"""

import aiosqlite
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sovereign.audit_time import utc_now_iso
from kestrel_sovereign.features.security.args_summary import (
    _MAX_REPAIR_TRIM,
    mask_sensitive,
    remask_summary,
    repair_unparseable_summary,
)

#: The audit-search tool's own name. Its rows are excluded from every search
#: (see ``_match_predicate``) because the security hook records the call, query
#: and all, BEFORE the tool body runs. Defined here rather than in the feature
#: so the exclusion and the ``@tool(name=...)`` cannot drift apart.
SEARCH_TOOL_NAME = "security_audit_search"

#: ``action`` value for a feature-as-subagent DISPATCH row, as opposed to the
#: ``tool_execution`` rows its inner tool calls write. ``SecurityHook`` runs on
#: PRE_SUBAGENT_CALL as well as PRE_TOOL_USE, and until #3107 both wrote
#: "tool_execution", so a dispatch envelope was indistinguishable from the work
#: it requested. The distinction is what lets a read-back exclude requests and
#: keep actions.
SUBAGENT_DISPATCH_ACTION = "subagent_dispatch"


def fold_query(text):
    """Canonicalise a QUERY for matching: decode escapes, casefold. No masking.

    Split from :func:`fold_stored_summary` because the two answer different
    questions and sharing one function was a defect (#3107 review round 8). The
    stored summary is redacted before it becomes searchable; a query must not
    be, or an ordinary search for ``monkey``, ``password reset`` or ``API key
    rotation`` folds to the empty string, becomes the LIKE pattern ``%%``, and
    matches every row in the table.
    """
    if not text:
        return text
    return _decode_unicode_escapes(text).casefold()


def fold_stored_summary(text):
    """Decode JSON escaping and casefold, so a query and a stored summary can
    be compared as the text a human wrote (#3107).

    ``args_summary`` holds ``json.dumps`` output with the default
    ``ensure_ascii=True``, so "Échec" is persisted as ``\\u00c9chec``. Neither
    a literal comparison nor SQLite's ASCII-only ``LOWER`` can match that
    against a query of "échec". Decoding first makes both sides the same kind
    of thing before either is folded.

    A truncated summary is not valid JSON — ``summarize_args`` cuts at 500
    characters mid-structure — and those are exactly the long issue bodies the
    motivating case searches. Folding the raw text there would leave every
    escape undecoded and reintroduce the bug one row-shape over, so the
    fallback repairs the cut JSON and parses it, which decodes the escapes
    inside it the same way the parseable branch does; a prose-only row (no
    JSON at all) is folded as written. Returning nothing for every unparseable
    row is not an option either: it would silently shrink the corpus the
    caller believes it searched — only a row whose JSON cannot be repaired is
    withheld, because it cannot be masked.
    """
    if not isinstance(text, str):
        # A BLOB (the column has TEXT affinity, but bytes are stored as
        # bytes) would raise inside the registered scalar function, and a
        # raised scalar fails the WHOLE query — one such row and every search
        # errors permanently, the same hazard round 5 closed for a lone
        # surrogate. Nothing here can read it, so it matches nothing.
        return ""
    if not text:
        return text
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        # Truncated past parsing. Repair the cut JSON and mask it structurally
        # — the same masker and the same walk the parseable branch below
        # applies — rather than scan the raw text: a scanner has to know every
        # value shape (a container, an escaped nested string, a quoted key in
        # prose) and rounds 13–16 each found one it did not. A row whose JSON
        # cannot be repaired cannot be masked, so it is not searchable; a
        # prose-only row has no key position and folds as it is (#3107).
        repaired = repair_unparseable_summary(text)
        if repaired is None:
            return ""
        prefix, masked = repaired
        folded = prefix if masked is None else prefix + " " + _flatten_json(masked)
        return folded.casefold().encode("utf-8", "replace").decode("utf-8")

    if isinstance(parsed, (dict, list)):
        # MASK BEFORE FOLDING. Masking only on the way out closes the display
        # and leaves the MATCH open: a caller could compare the query against a
        # raw legacy secret and read it back one character at a time from
        # hit/no-hit, while every returned row dutifully showed ***MASKED***
        # (#3107 review round 7). The searchable projection has to be the
        # masked one, so matching and display are the same text.
        # READ path: a nested payload cut inside a row that parses needs the
        # repair slack the unparseable branch already has, or its secret
        # stays in the searchable projection (round 20 review).
        parsed = mask_sensitive(parsed, repair_slack=_MAX_REPAIR_TRIM)
        # Walk the VALUES rather than re-serializing: json.dumps would put back
        # the standard escapes (\", \n, doubled backslashes) that summarize_args
        # introduced, so `say "hello"`, multiline text and Windows paths would
        # not match their own stored form. ensure_ascii=False happens to fix
        # \uXXXX and nothing else.
        # Same hazard round 5 closed on the unparseable branch: json.dumps
        # happily stores a lone surrogate (a cut emoji in model-emitted
        # arguments), and handing one back to SQLite from inside the scalar
        # function raises — which fails the WHOLE query, so one poisoned row
        # made every search return an error, permanently. Replace here, at
        # the one place the value crosses back into SQLite.
        return _flatten_json(parsed).casefold().encode("utf-8", "replace").decode("utf-8")
    if isinstance(parsed, str):
        # A row that is itself a JSON string may carry a JSON-encoded payload
        # one level up from where the walker sees it (round 21 review); the
        # same masker, the same read-path slack.
        parsed = mask_sensitive(parsed, repair_slack=_MAX_REPAIR_TRIM)
        return parsed.casefold().encode("utf-8", "replace").decode("utf-8")
    return str(parsed).casefold()


def _flatten_json(value):
    """Concatenate every key and string value, decoded, for matching."""
    parts = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                parts.append(str(k))
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        else:
            # A string value that is itself a JSON document serialized by the
            # caller (an args_json, an HTTP payload) still carries literal
            # ``\\u2014``/``\\u00c9`` after the OUTER row is decoded, while
            # fold_query decodes the query side — neither spelling matched
            # (round 22 review). Decode the leaf so both sides are the same
            # kind of thing; only ``\\uXXXX``, so the round-10 ``\\b`` bug
            # stays closed.
            parts.append(_decode_unicode_escapes(node) if isinstance(node, str) else str(node))

    walk(value)
    return " ".join(parts)


_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
_SURROGATE_PAIR = re.compile(
    r"\\u(d[89ab][0-9a-fA-F]{2})\\u(d[c-f][0-9a-fA-F]{2})", re.IGNORECASE
)


def _decode_unicode_escapes(text):
    """Decode only ``\\uXXXX`` (and surrogate pairs) in a QUERY.

    The stored side of a parseable row comes out of ``json.loads`` already
    decoded, so a query must be treated the same way — decoding the standard
    escapes too turned ``\\b`` in a grep the agent just ran into a backspace,
    and the search for that exact command reported a false absence, with a
    message telling the caller absence is weak evidence (#3107 round 10).
    The stored side of a TRUNCATED row is now repaired and parsed too, so the
    same decoding applies to both sides of every comparison.
    """
    def _one(match):
        code = int(match.group(1), 16)
        if 0xD800 <= code <= 0xDFFF:
            return match.group(0)
        return chr(code)

    paired = _SURROGATE_PAIR.sub(
        lambda m: chr(
            0x10000
            + ((int(m.group(1), 16) - 0xD800) << 10)
            + (int(m.group(2), 16) - 0xDC00)
        ),
        text,
    )
    return _UNICODE_ESCAPE.sub(_one, paired)


logger = logging.getLogger(__name__)


class UnknownFeatureError(ValueError):
    """Raised when a permission write targets a feature with no registered
    permission-tree group under any casing/alias variant.

    Surfacing this (instead of logging a false success) is what stops
    ``set_feature_permission`` from silently no-oping on an alias spelling
    that resolves to no tree group (F253).
    """


class PermissionLevel(Enum):
    """
    Permission levels for tools.

    - ALLOW: Always allow the tool to execute
    - AUTO: Auto-approve after earlier constitutional/honesty/security hooks
      have not blocked the call
    - DENY: Always deny the tool execution
    - ALWAYS_ASK: Ask every time, even when global Auto mode is enabled
    - ASK: Ask for user approval each time (default for new tools)
    - SESSION: Allow for the current session only (not persisted)
    """
    ALLOW = "allow"
    AUTO = "auto"
    DENY = "deny"
    ALWAYS_ASK = "always_ask"
    ASK = "ask"
    SESSION = "session"


def assert_sdk_permission_level_parity() -> None:
    """Fail closed unless SDK declarations match Sovereign enforcement.

    Feature packages declare defaults with the SDK enum while this module owns
    persistence and enforcement.  Comparing both names *and* values prevents a
    newly-added or renamed SDK value from being accepted by only part of the
    security stack.
    """
    from kestrel_sdk.features import PermissionLevel as SDKPermissionLevel

    sovereign = {level.name: level.value for level in PermissionLevel}
    sdk = {level.name: level.value for level in SDKPermissionLevel}
    enforcement = set(_LEVEL_RANK)
    composition = set(_CONTRIBUTED_TIGHTENING)
    hardening = set(_HARDENED_PRESERVED_LEVELS)
    if (
        sdk != sovereign
        or enforcement != set(PermissionLevel)
        or composition != set(PermissionLevel)
        or hardening != set(PermissionLevel)
    ):
        raise RuntimeError(
            "SDK/Sovereign permission vocabulary mismatch; refusing feature "
            "permission defaults "
            f"(sdk={sdk!r}, sovereign={sovereign!r}, "
            f"enforced={sorted(level.value for level in enforcement)!r}, "
            f"composed={sorted(level.value for level in composition)!r}, "
            f"hardened={sorted(level.value for level in hardening)!r})"
        )


@dataclass
class ToolPermission:
    """Permission configuration for a single tool."""
    feature_name: str
    tool_name: str
    level: PermissionLevel
    reason: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class FeaturePermissions:
    """
    Aggregated permissions for a feature with rollup state.

    The rollup_state property calculates the aggregate state of all
    tools in this feature, used for UI display.
    """
    feature_name: str
    tools: List[ToolPermission]

    @property
    def rollup_state(self) -> str:
        """
        Calculate rollup state from children.

        Returns:
            - "allow_all": All tools are ALLOW
            - "auto_all": All tools are AUTO
            - "deny_all": All tools are DENY
            - "ask_all": All tools are ASK
            - "session_all": All tools are SESSION
            - "mixed": Tools have different settings
        """
        if not self.tools:
            return "ask_all"  # Default

        levels = {t.level for t in self.tools}

        if len(levels) == 1:
            level = list(levels)[0]
            return f"{level.value}_all"

        return "mixed"


def _snake_case(name: str) -> str:
    """Convert PascalCase to snake_case the same way `Feature.tool_name` does.
    Used at the storage boundary to look up legacy rows under both casings.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _pascal_case(name: str) -> str:
    """Best-effort inverse of ``_snake_case`` for the common snake form
    ``word_word_word``: capitalize each underscored part and concatenate.

    Lossy: e.g. ``m_c_p_agent`` → ``MCPAgent`` (correct round-trip) and
    ``mcp_agent`` → ``McpAgent`` (NOT the original class name, but still
    fine — that class name doesn't exist in the codebase anyway). Used
    only for the lookup-time variant search; we don't write under this
    form, so any oddness stays at read-time only."""
    if "_" not in name:
        return name
    parts = [p for p in name.split("_") if p]
    if not parts:
        return name
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _name_variants(name: str) -> set[str]:
    """Return the set of feature_name strings to try at lookup time:
    the input, its snake form, and a candidate PascalCase form. See the
    detailed rationale on `_lookup_rows` for why we accept both."""
    return {name, _snake_case(name), _pascal_case(name)}


# DENY-wins ordering for casing-variant resolution. An explicit DENY anywhere
# in the row set is treated as the operator's last word — a stale ALLOW under
# a legacy casing must not silently re-enable a tool the operator has since
# blocked under the canonical casing. ALWAYS_ASK similarly survives stale
# grants because it is the "prompt before irreversible work" tier; ALLOW/AUTO
# outrank ASK/SESSION; SESSION outranks ASK because SESSION is a deliberate
# per-session grant whereas ASK is the "no row at all" default. See codex
# review #1427 P1 — without the DENY priority, security regressions sneak in
# via the mixed-case DB state this normalization layer was added to handle.
_LEVEL_RANK = {
    PermissionLevel.DENY: 100,
    PermissionLevel.ALWAYS_ASK: 90,
    PermissionLevel.ALLOW: 4,
    PermissionLevel.AUTO: 3,
    PermissionLevel.SESSION: 2,
    PermissionLevel.ASK: 1,
}

# Static-rail composition is a separate concern from the legacy-row winner
# policy above. ALLOW/AUTO/SESSION/ASK are operational modes, not a general
# restrictiveness order: global auto mode can promote three of them, while
# static ALLOW entries intentionally migrate selected unattended tools. Only
# ALWAYS_ASK and DENY are hard rails with a strict ordering. Listing every
# declaration keeps the vocabulary closed when the SDK enum grows.
_CONTRIBUTED_TIGHTENING = {
    PermissionLevel.ALLOW: None,
    PermissionLevel.AUTO: None,
    PermissionLevel.SESSION: None,
    PermissionLevel.ASK: None,
    PermissionLevel.ALWAYS_ASK: PermissionLevel.ALWAYS_ASK,
    PermissionLevel.DENY: PermissionLevel.DENY,
}


def compose_restrictive_permission(
    sovereign_level: PermissionLevel,
    declared_level: PermissionLevel,
) -> PermissionLevel:
    """Compose a Sovereign baseline or rail with a declaration, fail closed.

    Ordinary declaration modes cannot replace a Sovereign-owned feature
    baseline or per-tool override. A contributed ALWAYS_ASK or DENY may tighten
    it, and DENY remains the strongest rail. This policy is intentionally not a
    numeric ordering: the operational modes have different migration and
    auto-mode semantics.
    """

    if set(_CONTRIBUTED_TIGHTENING) != set(PermissionLevel):
        raise RuntimeError(
            "Permission restrictiveness vocabulary is incomplete; refusing "
            "to compose static and contributed defaults"
        )
    tightening = _CONTRIBUTED_TIGHTENING[declared_level]
    if sovereign_level is PermissionLevel.DENY or tightening is None:
        return sovereign_level
    if tightening is PermissionLevel.DENY:
        return PermissionLevel.DENY
    if sovereign_level is PermissionLevel.ALWAYS_ASK:
        return PermissionLevel.ALWAYS_ASK
    return tightening


# Hardened registration normally preserves levels at least as restrictive as
# its target.  Static ALLOW entries are deliberate migrations for unattended
# reconciliation, so they retain their historic behavior: stale ASK/AUTO/
# SESSION defaults are upgraded while explicit ALWAYS_ASK and DENY operator
# rails survive.  Listing every target fails closed if the enum grows.
_HARDENED_PRESERVED_LEVELS = {
    PermissionLevel.ALLOW: frozenset({
        PermissionLevel.ALLOW,
        PermissionLevel.ALWAYS_ASK,
        PermissionLevel.DENY,
    }),
    PermissionLevel.AUTO: frozenset({
        PermissionLevel.AUTO,
        PermissionLevel.SESSION,
        PermissionLevel.ASK,
        PermissionLevel.ALWAYS_ASK,
        PermissionLevel.DENY,
    }),
    PermissionLevel.SESSION: frozenset({
        PermissionLevel.SESSION,
        PermissionLevel.ASK,
        PermissionLevel.ALWAYS_ASK,
        PermissionLevel.DENY,
    }),
    PermissionLevel.ASK: frozenset({
        PermissionLevel.ASK,
        PermissionLevel.ALWAYS_ASK,
        PermissionLevel.DENY,
    }),
    PermissionLevel.ALWAYS_ASK: frozenset({
        PermissionLevel.ALWAYS_ASK,
        PermissionLevel.DENY,
    }),
    PermissionLevel.DENY: frozenset({PermissionLevel.DENY}),
}

_AUTO_MODE_EXEMPT_LEVELS = {
    PermissionLevel.DENY,
    PermissionLevel.ALWAYS_ASK,
}


def _resolve_levels(levels: List["PermissionLevel"]) -> "PermissionLevel":
    """Resolve a list of (feature_name casing-variant) row levels for the same
    logical (feature, tool) pair. DENY is a hard stop; otherwise pick the
    most permissive grant."""
    return max(levels, key=lambda level: _LEVEL_RANK.get(level, 0))


# Kept as a public alias for the test suite — the previous name read more
# naturally for the non-DENY case. Use ``_resolve_levels`` for new callers
# so the DENY-wins behavior is unambiguous from the name.
_most_permissive = _resolve_levels


class PermissionStore:
    """
    SQLite-backed hierarchical permission storage.

    Features:
    - Persistent storage of permission settings
    - Session-scoped overrides (not persisted)
    - Audit logging of permission decisions
    - Hierarchical tree retrieval for UI

    Example:
        store = PermissionStore("kestrel_prime.db")
        await store.initialize()

        # Set permission
        await store.set_permission("ModelAgent", "list_models", PermissionLevel.ALLOW)

        # Get permission (checks session overrides first)
        level = await store.get_permission("WalletAgent", "send_payment")
        # Returns: PermissionLevel.ASK (default)

        # Get full tree for UI
        tree = await store.get_permission_tree()
    """

    def __init__(self, db_path: str):
        """
        Initialize the permission store.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._session_overrides: Dict[str, PermissionLevel] = {}
        # Tool names that are feature-as-subagent DISPATCH entries — see
        # :meth:`mark_dispatch_entry`. Repopulated on every boot by feature
        # registration, so it is always current without a schema column.
        self._dispatch_entries: set = set()
        # Marked this boot but not yet persisted — flushed on the next read so
        # registration stays synchronous and cheap.
        self._dispatch_entries_dirty: set = set()
        # Flush tasks are held here so a running flush is never garbage
        # collected mid-await (asyncio keeps only weak references).
        self._dispatch_flush_tasks: set = set()
        # Global Auto has two tiers backing one effective state:
        #   - _global_auto_session: in-memory, cleared on session reset.
        #   - _global_auto_always:  persisted in security_global_settings,
        #     rehydrated on initialize(), and survives session resets and
        #     server restarts until the operator explicitly turns it off.
        # ``_global_auto_mode`` (the property below) is the effective OR of the
        # two, so all existing read sites keep working unchanged.
        self._global_auto_session = False
        self._global_auto_always = False
        self._initialized = False
        # Bidirectional alias map populated by ``migrate_legacy_feature_aliases``.
        # Used by ``_lookup_rows`` to translate between the snake-case alias an
        # operator/integration might use (``computer_use``) and the canonical
        # class name (``ComputerUseFeature``) — covers the non-derived aliases
        # that ``_snake_case``/``_pascal_case`` can't recover (codex review
        # #1427 P2).
        self._feature_alias_to_class: Dict[str, str] = {}
        self._feature_class_to_alias: Dict[str, str] = {}

    @property
    def _global_auto_mode(self) -> bool:
        """Effective global Auto: enabled if either tier is active."""
        return self._global_auto_session or self._global_auto_always

    async def initialize(self) -> None:
        """Create database tables if they don't exist."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS security_permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    level TEXT NOT NULL DEFAULT 'ask',
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(feature_name, tool_name)
                )
            """)

            # Names that have ever denoted a feature-as-subagent DISPATCH
            # entry (#3107). Durable and append-only on purpose: the in-memory
            # set is rebuilt from CURRENTLY LOADED features, so a feature that
            # is removed or renamed takes its name out of the exclusion while
            # its historical envelope rows stay in the table — and text from an
            # old subagent REQUEST then reads as a prior attempt. The audit log
            # outlives the feature list, so the filter has to as well.
            await db.execute("""
                CREATE TABLE IF NOT EXISTS security_dispatch_entries (
                    tool_name TEXT PRIMARY KEY,
                    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS security_global_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS security_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT,
                    tool_name TEXT,
                    action TEXT,
                    decision TEXT,
                    user_choice TEXT,
                    args_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Index for faster permission lookups
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_security_permissions_lookup
                ON security_permissions(feature_name, tool_name)
            """)

            # Index for audit log queries
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_security_audit_created
                ON security_audit_log(created_at DESC)
            """)
            # NOTE (F092): legacy rows written by the old CURRENT_TIMESTAMP
            # default are deliberately NOT rewritten. AuditHasher hashes each
            # row's RAW created_at, so mutating those bytes would make every
            # pre-existing cryptographic anchor covering them fail verification.
            # Correctness is instead achieved without mutation: new rows are
            # written canonical (log_decision), and the audit_anchor reader
            # normalizes timestamps for comparison while preserving raw bytes.

            # Sovereign-curated auto-approve allowlist (the "Approve-and-
            # remember" store). Operator-seeded rules live in kestrel.toml;
            # these are the rows the Sovereign added via the Mews approval
            # panel and can revoke by deleting the row. See
            # kestrel_sovereign/security/auto_approve.py.
            await db.execute("""
                CREATE TABLE IF NOT EXISTS auto_approve_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT,
                    pattern TEXT NOT NULL,
                    repo_scope TEXT NOT NULL DEFAULT '',
                    added_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(agent, pattern, repo_scope)
                )
            """)

            # Full, immutable audit row for every auto-approved invocation.
            # security_audit_log lacks agent_did/command/exit_code columns;
            # this is the "no silent runs" record the constitution requires.
            # Two-phase: a row is inserted at approve-time (exit_code NULL)
            # and finalized with the real exit code once the tool returns.
            await db.execute("""
                CREATE TABLE IF NOT EXISTS auto_approve_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_did TEXT,
                    agent_name TEXT,
                    feature_name TEXT,
                    tool_name TEXT,
                    command TEXT,
                    pattern TEXT,
                    repo_scope TEXT,
                    rule_source TEXT,
                    decision TEXT NOT NULL DEFAULT 'auto_approved',
                    exit_code INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_auto_approve_audit_created
                ON auto_approve_audit(created_at DESC)
            """)

            await db.commit()

        await self._load_global_auto_always()

        self._initialized = True
        logger.info("PermissionStore initialized")

    async def migrate_legacy_feature_aliases(
        self,
        aliases: Dict[str, str],
    ) -> int:
        """One-time consolidation of legacy snake_case/alias rows into
        ``feature.name`` (PascalCase) rows.

        Different code paths historically wrote ``security_permissions``
        under different feature_name conventions: the subagent path used
        ``feature.name`` (class name, e.g. ``ComputerUseFeature``); the
        direct-tool path used ``feature.tool_name`` (snake-case alias,
        e.g. ``computer_use``); the operator's ``!security-set`` call used
        whatever the user typed. The orchestrator now normalizes to
        PascalCase (#1427), but agents already in operation have months of
        snake-rowed grants the operator intended to keep.

        This migration copies the more permissive of any (alias, tool)
        row pair into the (PascalCase, tool) row, preserving operator
        intent. The alias rows themselves are kept (audit trail) so a
        rollback can re-read them; lookups now resolve via the canonical
        PascalCase row.

        Args:
            aliases: Mapping of ``feature.tool_name`` → ``feature.name``,
                built from the loaded feature registry. The caller (agent
                init) supplies this — the store has no knowledge of which
                Feature classes are loaded.

        Returns:
            Number of (feature, tool) pairs whose canonical row was
            upserted from a more-permissive alias row.
        """
        if not aliases:
            return 0
        # Persist for runtime lookup so a post-startup write under the alias
        # (``set_permission("computer_use", ...)`` from an operator or
        # integration) still resolves on canonical-name reads.
        for alias, canonical in aliases.items():
            if alias and canonical and alias != canonical:
                self._feature_alias_to_class[alias] = canonical
                self._feature_class_to_alias[canonical] = alias
        upserts = 0
        async with aiosqlite.connect(self.db_path) as db:
            for alias, canonical in aliases.items():
                if alias == canonical:
                    continue
                cursor = await db.execute(
                    "SELECT tool_name, level FROM security_permissions "
                    "WHERE feature_name = ?",
                    (alias,),
                )
                alias_rows = await cursor.fetchall()
                if not alias_rows:
                    continue
                for tool_name, raw_level in alias_rows:
                    try:
                        alias_level = PermissionLevel(raw_level)
                    except ValueError:
                        continue
                    cursor = await db.execute(
                        "SELECT level FROM security_permissions "
                        "WHERE feature_name = ? AND tool_name = ?",
                        (canonical, tool_name),
                    )
                    row = await cursor.fetchone()
                    if row:
                        try:
                            existing = PermissionLevel(row[0])
                        except ValueError:
                            existing = PermissionLevel.ASK
                        winner = _most_permissive([existing, alias_level])
                        if winner == existing:
                            continue
                        await db.execute(
                            "UPDATE security_permissions SET level = ?, "
                            "updated_at = CURRENT_TIMESTAMP "
                            "WHERE feature_name = ? AND tool_name = ?",
                            (winner.value, canonical, tool_name),
                        )
                    else:
                        await db.execute(
                            "INSERT INTO security_permissions "
                            "(feature_name, tool_name, level, reason) "
                            "VALUES (?, ?, ?, ?)",
                            (
                                canonical,
                                tool_name,
                                alias_level.value,
                                f"Migrated from legacy alias '{alias}' (#1427)",
                            ),
                        )
                    upserts += 1
            await db.commit()
        if upserts:
            logger.info(
                "Permission alias migration: %d row(s) consolidated into "
                "canonical PascalCase rows.", upserts,
            )
        return upserts

    async def get_permission(
        self,
        feature_name: str,
        tool_name: str
    ) -> PermissionLevel:
        """
        Get permission level for a tool.

        Checks in order:
        1. Session overrides (in-memory)
        2. Persistent storage
        3. Default (ASK)

        Args:
            feature_name: Name of the feature
            tool_name: Name of the tool

        Returns:
            PermissionLevel for the tool
        """
        key = f"{feature_name}.{tool_name}"

        # Session override takes priority. In global Auto mode, explicit
        # DENY and ALWAYS_ASK remain hard policy rails; everything else can
        # flow through Auto.
        if key in self._session_overrides:
            level = self._session_overrides[key]
            if self._global_auto_mode and level not in _AUTO_MODE_EXEMPT_LEVELS:
                return PermissionLevel.AUTO
            return level

        # Check persistent storage.
        #
        # The DB has historically accumulated a mix of casings for the same
        # logical feature: PascalCase (`TaskFeature.respond_to_a2a_task`,
        # written from the subagent-dispatch path and the operator
        # ``!security-set`` tool) and snake_case (`task_feature.…`, written
        # from the older direct-tool dispatch path before #1427's
        # normalization). Either form may be the source of truth on a given
        # row, so look up BOTH variants and resolve any tie by preferring
        # the more permissive level — that matches the operator's intent:
        # if they ever granted ALLOW under either name, the tool is allowed
        # (rather than re-asking just because the bookkeeping shifted).
        # The orchestrator's new lookup canonicalizes to PascalCase, so
        # future writes converge on the class-name form; this fallback
        # exists for backward compatibility with the mixed-case state on
        # already-running agents (#1427 sibling).
        rows = await self._lookup_rows(feature_name, tool_name)
        if rows:
            best = _most_permissive(rows)
            if self._global_auto_mode and best not in _AUTO_MODE_EXEMPT_LEVELS:
                return PermissionLevel.AUTO
            return best

        # Default for unregistered tools.
        # Demo servers (KESTREL_DEMO_SERVER=1) auto-allow — _register_all_tools
        # only catches sub-tools (web_search), not feature-as-subagent
        # invocations (web_search_feature, the snake-cased class name from
        # Feature.tool_name). Without this, the modal still pops for
        # subagent-level calls even though every sub-tool is ALLOW. Demo
        # subjects that aren't security shouldn't have to chase that.
        import os as _os
        if _os.environ.get("KESTREL_DEMO_SERVER", "").lower() in ("1", "true", "yes"):
            return PermissionLevel.ALLOW
        if self._global_auto_mode:
            return PermissionLevel.AUTO
        return PermissionLevel.ASK

    async def _lookup_rows(
        self,
        feature_name: str,
        tool_name: str,
    ) -> List[PermissionLevel]:
        """Look up `security_permissions` rows for both PascalCase and
        snake_case forms of ``feature_name`` (e.g. ``TaskFeature`` and
        ``task_feature``) and return all matching `PermissionLevel` values.

        See ``get_permission`` for the rationale — the DB has accumulated
        rows under both casings over time and we want operator grants to
        survive the orchestrator-side normalization to PascalCase (#1427).
        Also queries any registered feature alias for the canonical name
        (and vice versa) so non-derived pairs like ``computer_use`` ↔
        ``ComputerUseFeature`` resolve symmetrically (codex review #1427 P2).
        """
        # Snake/Pascal casing plus registered cross-form aliases (covers
        # non-derived pairs that _name_variants can't reproduce, e.g.
        # ``computer_use`` ↔ ``ComputerUseFeature``).
        names = self.feature_name_variants(feature_name)
        placeholders = ",".join(["?"] * len(names))
        query = (
            f"SELECT level FROM security_permissions "
            f"WHERE feature_name IN ({placeholders}) AND tool_name = ?"
        )
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, (*names, tool_name))
            rows = await cursor.fetchall()
        levels: List[PermissionLevel] = []
        for row in rows:
            try:
                levels.append(PermissionLevel(row[0]))
            except ValueError:
                # Stale level value from a removed enum variant — ignore.
                continue
        return levels

    def feature_name_variants(self, feature_name: str) -> set[str]:
        """Return every feature_name casing/alias variant that resolves to the
        same logical feature as ``feature_name``.

        This is the same resolution ``_lookup_rows`` applies at read time —
        snake/Pascal casing plus registered cross-form aliases (e.g.
        ``computer_use`` <-> ``ComputerUseFeature``). Exposed so callers that
        need to validate a feature_name against the registered permission tree
        (e.g. ``SecurityFeature.set_permission``) accept the same names the
        store itself honors, rather than only the exact canonical spelling.
        """
        names = _name_variants(feature_name)
        canonical = self._feature_alias_to_class.get(feature_name)
        if canonical:
            names.add(canonical)
        alias = self._feature_class_to_alias.get(feature_name)
        if alias:
            names.add(alias)
        return names

    async def set_permission(
        self,
        feature_name: str,
        tool_name: str,
        level: PermissionLevel,
        scope: str = "always",
        reason: Optional[str] = None,
    ) -> None:
        """
        Set permission for a tool.

        Args:
            feature_name: Name of the feature
            tool_name: Name of the tool
            level: Permission level to set
            scope: "always" (persistent), "session" (in-memory only), "once" (no storage)
            reason: Optional reason for the permission change
        """
        key = f"{feature_name}.{tool_name}"

        if scope == "session":
            self._session_overrides[key] = level
            logger.info(f"Set session permission: {key} = {level.value}")

        elif scope == "always":
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO security_permissions
                    (feature_name, tool_name, level, reason, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(feature_name, tool_name)
                    DO UPDATE SET level = excluded.level,
                                  reason = excluded.reason,
                                  updated_at = CURRENT_TIMESTAMP
                """, (feature_name, tool_name, level.value, reason))
                await db.commit()

            logger.info(f"Set persistent permission: {key} = {level.value}")

        # scope == "once" - no storage, just allow this execution

    async def set_feature_permission(
        self,
        feature_name: str,
        level: PermissionLevel,
        reason: Optional[str] = None,
    ) -> None:
        """
        Set all tools in a feature to the same level (bulk update).

        Args:
            feature_name: Name of the feature
            level: Permission level to set for all tools
            reason: Optional reason for the permission change

        Raises:
            UnknownFeatureError: If no permission-tree group matches
                ``feature_name`` under any casing/alias variant. Previously
                this path silently no-oped on an alias spelling (e.g.
                ``task_feature`` for ``TaskFeature``) yet still logged success,
                leaving the operator's intended control absent (F253).
        """
        # Resolve the target group with the SAME casing/alias resolution the
        # read path uses, so an alias spelling (``task_feature``) matches the
        # canonical tree group (``TaskFeature``) instead of finding nothing.
        variants = self.feature_name_variants(feature_name)
        tree = await self.get_permission_tree()
        feature = next((f for f in tree if f.feature_name in variants), None)

        if feature is None:
            # No group matched — refuse loudly rather than confirm a control
            # that was never written.
            raise UnknownFeatureError(
                f"Unknown feature '{feature_name}': no registered permission "
                "group matches under any casing/alias variant. Nothing was "
                "persisted."
            )

        # Write rows under the CANONICAL spelling the tree exposes, not the
        # (possibly aliased) spelling the caller passed.
        canonical_name = feature.feature_name
        for tool in feature.tools:
            await self.set_permission(
                canonical_name,
                tool.tool_name,
                level,
                scope="always",
                reason=reason
            )

        logger.info(f"Set feature permission: {canonical_name} = {level.value}")

    async def get_permission_tree(self) -> List[FeaturePermissions]:
        """
        Get full hierarchical permission tree for UI.

        Returns all features and their tools with current permission levels,
        including rollup states for display.

        Returns:
            List of FeaturePermissions with tools and rollup states
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT feature_name, tool_name, level, reason, created_at, updated_at
                FROM security_permissions
                ORDER BY feature_name, tool_name
            """)
            rows = await cursor.fetchall()

        # Group by feature
        features_dict: Dict[str, List[ToolPermission]] = {}

        for row in rows:
            feature_name = row["feature_name"]
            tool = ToolPermission(
                feature_name=feature_name,
                tool_name=row["tool_name"],
                level=PermissionLevel(row["level"]),
                reason=row["reason"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

            # Apply session override if present
            key = f"{feature_name}.{tool.tool_name}"
            if key in self._session_overrides:
                tool.level = self._session_overrides[key]

            if feature_name not in features_dict:
                features_dict[feature_name] = []
            features_dict[feature_name].append(tool)

        return [
            FeaturePermissions(feature_name=name, tools=tools)
            for name, tools in features_dict.items()
        ]

    def mark_dispatch_entry(self, tool_name: str) -> None:
        """Record that ``tool_name`` is a feature-as-subagent DISPATCH entry.

        The audit table cannot tell a dispatch envelope from a tool call by
        inspection, and it needs to: an envelope records what a task ASKED FOR,
        the inner rows record what was DONE, and a read-back that confuses them
        reports a request as prior work (#3107).

        Labelling by hook event was not enough. ``orchestrator_engine`` fires
        PRE_SUBAGENT_CALL for the dispatch AND a second PRE_TOOL_USE around
        ``execute_as_subagent`` with the same arguments — deliberately, so
        PRE_TOOL_USE-only hooks see chat-path and inline-path dispatches alike
        (PR #1385). Only the first carries the dispatch event, so a rule keyed
        on the event misses the second by construction.

        This is keyed on the NAME instead, which both rows share and which the
        feature itself declares at registration. In memory rather than a
        column: registration runs every boot, so the set is always current,
        and a column would be a second source of truth for something the
        registry already knows.
        """
        if tool_name:
            self._dispatch_entries.add(tool_name)
            self._dispatch_entries_dirty.add(tool_name)
            # Durability cannot wait for the first search. A feature that
            # writes envelope rows, is then removed, and whose process restarts
            # before anyone searches would lose the name entirely — and its
            # historical REQUESTS would start reading as prior actions, which
            # is the failure this table exists to prevent. Flushed on a task so
            # registration stays synchronous; the read path still syncs, so a
            # lost task degrades to the old behaviour rather than an error.
            try:
                task = asyncio.get_running_loop().create_task(
                    self.sync_dispatch_entries()
                )
                self._dispatch_flush_tasks.add(task)
                task.add_done_callback(self._dispatch_flush_tasks.discard)
            except RuntimeError:
                # No loop (sync construction in a test): the read path flushes.
                pass

    @property
    def dispatch_entries(self) -> frozenset:
        """Tool names registered as feature-as-subagent dispatch entries."""
        return frozenset(self._dispatch_entries)

    async def sync_dispatch_entries(self) -> frozenset:
        """Persist names marked this boot, then return the FULL durable set.

        The union matters, not this boot's list: a feature removed or renamed
        since the rows were written is absent from the live feature map but its
        envelope rows are still in the audit log, and an exclusion built only
        from what is loaded now would stop covering them.
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Snapshot, then discard only what was written: a name marked
            # while the executemany/commit awaits were in flight is still
            # dirty and goes out with the next flush. Clearing the whole set
            # dropped it, and a feature removed before the next boot would then
            # have its envelope rows read as prior actions.
            pending = sorted(self._dispatch_entries_dirty)
            if pending:
                await db.executemany(
                    "INSERT OR IGNORE INTO security_dispatch_entries "
                    "(tool_name) VALUES (?)",
                    [(name,) for name in pending],
                )
                await db.commit()
                self._dispatch_entries_dirty.difference_update(pending)
            cursor = await db.execute(
                "SELECT tool_name FROM security_dispatch_entries"
            )
            rows = await cursor.fetchall()
        self._dispatch_entries.update(r[0] for r in rows)
        return frozenset(self._dispatch_entries)

    async def register_tool(
        self,
        feature_name: str,
        tool_name: str,
        default_level: PermissionLevel = PermissionLevel.ASK,
        *,
        hardened: bool = False,
    ) -> None:
        """
        Register a tool with default permission (if not already registered).

        Called when features are loaded to ensure all tools are in the tree.

        Args:
            feature_name: Name of the feature
            tool_name: Name of the tool
            default_level: Default permission level (default: ASK)
            hardened: When True, ``default_level`` is an enforced static or
                contributed rail rather than a first-registration default. In
                addition to the usual insert-if-missing, incompatible existing
                rows for this tool (across every casing/alias variant) are
                migrated to it. Explicit harder operator rails are preserved;
                static ALLOW migrations retain their reviewed upgrade policy.
                This is what closes the
                F203 upgrade gap (#2093): an agent that already persisted a
                permissive ``ALLOW`` row for a destructive memory tool under
                the old feature-level default would otherwise keep bypassing
                approval forever, because plain registration is INSERT-OR-IGNORE
                and ``get_permission`` returns persisted rows before defaults.
        """
        async with aiosqlite.connect(self.db_path) as db:
            if hardened:
                # Force-upgrade incompatible pre-existing rows (any casing /
                # alias variant) while preserving the reviewed harder choices
                # for this target. The explicit table avoids reusing the
                # unrelated legacy-alias winner ranking.
                try:
                    preserved = [
                        level.value
                        for level in _HARDENED_PRESERVED_LEVELS[default_level]
                    ]
                except KeyError as exc:
                    raise RuntimeError(
                        "Permission hardening vocabulary is incomplete; "
                        f"cannot register {default_level!r}"
                    ) from exc
                names = sorted(self.feature_name_variants(feature_name))
                name_placeholders = ",".join(["?"] * len(names))
                preserved_placeholders = ",".join(["?"] * len(preserved))
                await db.execute(
                    f"""
                    UPDATE security_permissions
                    SET level = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE feature_name IN ({name_placeholders})
                      AND tool_name = ?
                      AND level NOT IN ({preserved_placeholders})
                    """,
                    (default_level.value, *names, tool_name, *preserved),
                )
            await db.execute("""
                INSERT OR IGNORE INTO security_permissions
                (feature_name, tool_name, level)
                VALUES (?, ?, ?)
            """, (feature_name, tool_name, default_level.value))
            await db.commit()

    async def log_decision(
        self,
        feature_name: str,
        tool_name: str,
        action: str,
        decision: str,
        user_choice: Optional[str] = None,
        args_summary: Optional[str] = None,
    ) -> None:
        """
        Log a permission decision to the audit log.

        Args:
            feature_name: Name of the feature
            tool_name: Name of the tool
            action: Action type (e.g., "tool_execution", "permission_change")
            decision: Decision made (e.g., "allowed", "denied", "user_approved")
            user_choice: User's choice if they were asked (once/session/always)
            args_summary: Summary of tool arguments (truncated for privacy)
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Write an explicit canonical UTC ISO-8601 stamp rather than relying
            # on the column's CURRENT_TIMESTAMP default (which is space-separated,
            # offset-less, and sorts incorrectly against the ISO timestamps other
            # audit sources emit — F092).
            await db.execute("""
                INSERT INTO security_audit_log
                (feature_name, tool_name, action, decision, user_choice, args_summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (feature_name, tool_name, action, decision, user_choice,
                  args_summary, utc_now_iso()))
            await db.commit()

    async def get_audit_log(self, limit: int = 50) -> List[Dict]:
        """
        Get recent audit log entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of audit log entries (most recent first)
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # Order by the autoincrement id (insertion order ≈ chronological)
            # rather than created_at: legacy rows may carry a space-separated
            # timestamp that sorts incorrectly against ISO rows (F092), and id
            # ordering is format-agnostic and stable.
            cursor = await db.execute("""
                SELECT feature_name, tool_name, action, decision,
                       user_choice, args_summary, created_at
                FROM security_audit_log
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = await cursor.fetchall()

        return [
            {
                "feature": row["feature_name"],
                "tool": row["tool_name"],
                "action": row["action"],
                "decision": row["decision"],
                "user_choice": row["user_choice"],
                # Re-masked here too: this is the store's OTHER read path over
                # the same column (/api/security/audit), and "one door in the
                # store" is only true if both doors mask (round 17 review).
                "args_summary": remask_summary(row["args_summary"]),
                "timestamp": row["created_at"],
            }
            for row in rows
        ]

    def _match_predicate(
        self,
        needle: str,
        *,
        tool_name: Optional[str],
        days: Optional[int],
    ) -> Tuple[str, List[Any]]:
        """Build the WHERE clause for a match, and its parameters.

        One builder, so the page and its headroom row (the breadth signal
        ``search_audit_log`` reads instead of a count query) come from the
        same predicate and cannot disagree.
        """
        def _escape_like(text: str) -> str:
            """Neutralise LIKE metacharacters in a literal."""
            return (
                text.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )

        def _like_prefix(text: str) -> str:
            """Match ``text`` and anything suffixed to it, e.g. ``.outcome``."""
            return _escape_like(text) + "%"

        def _like(text: str) -> str:
            # LIKE wildcards in the needle would silently widen the match — a
            # caller searching for a literal "%" must not get everything.
            return "%" + _escape_like(text) + "%"

        # Two problems with matching this column, and one answer to both.
        #
        # ``summarize_args`` persists ``json.dumps(...)`` output, and json.dumps
        # defaults to ensure_ascii=True — so a title stored from "Échec — café"
        # is on disk as "\u00c9chec \u2014 caf\u00e9". A LIKE built from the
        # literal query finds nothing for exactly the natural-language
        # fragments a caller reaches for; both real filings behind #3107
        # carried an em dash. And SQLite's built-in LOWER folds ASCII only, so
        # matching the ESCAPED form case-insensitively does not work either:
        # "É" and "é" escape to \u00c9 and \u00e9, which differ in a
        # character LOWER will not touch.
        #
        # So neither side is compared raw. ``py_fold`` (registered on the
        # connection) decodes the JSON escaping and casefolds, and both the
        # column and the needle go through it. Measured 2026-09-03 at about
        # 0.7 s for a full scan of 86,000 rows with an eighth of them
        # truncated (repair costs more than a parse), and under 2 s if every
        # row is — this table has no index for LIKE anyway.
        clauses = [
            (
                "(py_fold(args_summary) LIKE ? ESCAPE '\\' "
                "OR py_fold(tool_name) LIKE ? ESCAPE '\\')"
            ),
            # A dispatch is a REQUEST, not an action. ``SecurityHook`` also
            # runs on PRE_SUBAGENT_CALL, so a feature-as-subagent dispatch
            # writes a row carrying the whole task text — including, when the
            # dispatch is what reached this very tool, the search phrase
            # itself. Left in, the enclosing call returns as prior work and a
            # novel search reads as already done. Excluding dispatch envelopes
            # generally is the honest rule rather than a special case: what a
            # task ASKED for is not evidence of what was DONE, and the inner
            # tool rows are what record that.
            # COALESCE on every exclusion: the schema declares action and
            # tool_name without NOT NULL, and under three-valued logic
            # `NULL <> x`, `NULL NOT LIKE x` and `NULL NOT IN (...)` are all
            # NULL — the row silently leaves the corpus the caller believes it
            # searched (round 21 review). A NULL is "not the excluded value".
            "COALESCE(action, '') <> ?",
            # A search must never return its own act of searching. The security
            # hook writes its PRE_TOOL_USE row — carrying this very query inside
            # ``args_summary`` — BEFORE the tool body runs, so without this the
            # current invocation always matches itself: the no-match branch is
            # unreachable in production and a brand-new search reads as prior
            # work. That is exactly the failure this tool exists to prevent,
            # manufactured by the tool. Prefix match takes the paired
            # ``.outcome`` row with it.
            # ESCAPE is not optional here: the pattern escapes the literal
            # underscores in the tool name, and without declaring the escape
            # character SQLite reads "\\_" as backslash-then-wildcard and the
            # exclusion silently matches nothing.
            "COALESCE(tool_name, '') NOT LIKE ? ESCAPE '\\'",
        ]
        folded = _like(fold_query(needle))
        params: List[Any] = [
            folded, folded, SUBAGENT_DISPATCH_ACTION,
            # Escaped: SEARCH_TOOL_NAME contains underscores, and an unescaped
            # LIKE would treat each as "any character" — excluding unrelated
            # tools such as `security-audit-search-index` and making them
            # unfindable even with an exact tool_name filter. Same wildcard bug
            # already fixed for the query, still present in the exclusion.
            _like_prefix(SEARCH_TOOL_NAME),
        ]

        # ...and by NAME, which is what actually covers the case. The
        # orchestrator fires PRE_SUBAGENT_CALL for a dispatch AND a second
        # PRE_TOOL_USE around ``execute_as_subagent`` carrying the same
        # arguments, so only the first row gets the dispatch ACTION. Both name
        # the feature's dispatch entry, and that is the fact both share.
        dispatch_names = sorted(self._dispatch_entries)
        if dispatch_names:
            placeholders = ", ".join("?" for _ in dispatch_names)
            clauses.append(f"COALESCE(tool_name, '') NOT IN ({placeholders})")
            params.extend(dispatch_names)

        if tool_name:
            clauses.append("tool_name = ?")
            params.append(tool_name)

        if days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
            # created_at is ISO ("...T...+00:00") today, but legacy rows are
            # space-separated and offset-less (F092). Comparing the two forms as
            # text is wrong in a way that silently DROPS rows: " " (32) sorts
            # below "T" (84), so EVERY legacy row on the cutoff's own date
            # compares below an ISO cutoff regardless of the time it carries.
            # Normalize both to "YYYY-MM-DD HH:MM:SS" first.
            clauses.append("substr(replace(created_at, 'T', ' '), 1, 19) >= ?")
            params.append(cutoff.strftime("%Y-%m-%d %H:%M:%S"))

        return " AND ".join(clauses), params

    async def search_audit_log(
        self,
        query: str,
        *,
        tool_name: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 20,
    ) -> Tuple[List[Dict], bool]:
        """Return audit rows whose recorded call matches ``query`` (#3107).

        The sibling of :meth:`get_audit_log`, and deliberately not a variant of
        it. ``get_audit_log`` answers "what happened recently" and is what the
        operator's ``/api/security/audit`` page renders. This answers a
        different question — *"have I already done this?"* — which a listing
        cannot answer at all: the write that matters may be a hundred rows back,
        and the only handle the caller has on it is a description of the thing
        itself.

        That difference is what makes returning ``args_summary`` here
        defensible where returning it from an unbounded listing was not, and it
        holds only while the query is a description. ``limit`` is therefore
        fetched with ONE row of headroom and the caller is told whether that
        row existed. A separate COUNT could pass at the bound while the page
        query — on its own connection, so its own SQLite snapshot — saw more
        rows and returned their arguments anyway. One statement cannot
        disagree with itself.

        **A row is an AUTHORIZATION, not a completion.** The security hook logs
        at ``PRE_TOOL_USE``, so a row records that the call was allowed to run —
        including ``auto_denied`` and timed-out attempts. Callers must read
        ``decision`` rather than treating presence as proof the work happened.

        Args:
            query: Substring to look for. Must be non-empty after stripping.
            tool_name: Restrict to one tool (exact match).
            days: Only rows from the last N days.
            limit: Maximum rows returned, newest first.
        """
        needle = (query or "").strip()
        if not needle:
            return [], False

        # Durable union, not this boot's registrations: an envelope written by
        # a feature since removed must still be excluded.
        await self.sync_dispatch_entries()

        where, params = self._match_predicate(
            needle, tool_name=tool_name, days=days,
        )
        # One row of headroom: whether it came back IS the "too broad"
        # signal, read from the same statement rather than a racing COUNT.
        params.append(int(limit) + 1)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # The fold has to happen inside the query: SQLite cannot express
            # it, so Python is registered as a scalar function on this
            # connection. Deterministic, so SQLite may cache per-value.
            await db.create_function(
                "py_fold", 1, fold_stored_summary, deterministic=True
            )
            # id DESC for the same reason get_audit_log uses it: legacy rows
            # carry a space-separated timestamp that sorts incorrectly against
            # the ISO ones (F092), and id ordering is format-agnostic.
            cursor = await db.execute(
                f"""
                SELECT feature_name, tool_name, action, decision,
                       user_choice, args_summary, created_at
                FROM security_audit_log
                WHERE {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                tuple(params),
            )
            rows = await cursor.fetchall()

        has_more = len(rows) > int(limit)
        rows = rows[: int(limit)]

        return [
            {
                "feature": row["feature_name"],
                "tool": row["tool_name"],
                "action": row["action"],
                "decision": row["decision"],
                "user_choice": row["user_choice"],
                # Re-masked HERE, in the store's own read path: the searchable
                # projection is masked (fold_stored_summary), and the row that
                # comes back must be the same text, whoever calls this. A
                # re-mask that lived only in the tool left the store
                # disagreeing with itself about its own rows, and the next
                # caller (an endpoint, another feature) would have leaked a
                # legacy row the tool had been hiding (round 14 review).
                # get_audit_log applies the same re-mask.
                "args_summary": remask_summary(row["args_summary"]),
                "timestamp": row["created_at"],
            }
            for row in rows
        ], has_more

    # ------------------------------------------------------------------
    # Auto-approve allowlist (Sovereign-curated, revocable)
    # ------------------------------------------------------------------

    async def list_auto_approve_rules(self) -> List[Dict]:
        """Return the dynamic (DB-backed) auto-approve rules."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT id, agent, pattern, repo_scope, added_by, created_at
                FROM auto_approve_rules ORDER BY created_at DESC
            """)
            rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "agent": r["agent"],
                "pattern": r["pattern"],
                "repo_scope": r["repo_scope"],
                "added_by": r["added_by"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def add_auto_approve_rule(
        self,
        *,
        pattern: str,
        repo_scope: str = "",
        agent: Optional[str] = None,
        added_by: Optional[str] = None,
    ) -> None:
        """Add (or no-op if duplicate) a Sovereign-curated allowlist rule.

        This is what the Mews "Approve-and-remember" button calls. Revoke
        by deleting the row (:meth:`remove_auto_approve_rule`).
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR IGNORE INTO auto_approve_rules
                (agent, pattern, repo_scope, added_by)
                VALUES (?, ?, ?, ?)
            """, (agent, pattern, repo_scope, added_by))
            await db.commit()
        logger.info(
            "auto_approve: remembered rule (agent=%s, repo=%s) added_by=%s",
            agent or "*", repo_scope or "*", added_by or "?",
        )

    async def remove_auto_approve_rule(self, rule_id: int) -> bool:
        """Revoke a dynamic rule. Returns True if a row was deleted."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM auto_approve_rules WHERE id = ?", (rule_id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def log_auto_approve(
        self,
        *,
        agent_did: Optional[str],
        agent_name: Optional[str],
        feature_name: str,
        tool_name: str,
        command: str,
        pattern: str,
        repo_scope: str,
        rule_source: str,
    ) -> int:
        """Insert the phase-1 audit row; return its id for finalization.

        The row is written *before* the tool runs so an auto-approved
        invocation can never execute without an audit trail. ``exit_code``
        is filled in by :meth:`finalize_auto_approve` once the tool exits.
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO auto_approve_audit
                (agent_did, agent_name, feature_name, tool_name, command,
                 pattern, repo_scope, rule_source, decision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'auto_approved')
            """, (
                agent_did, agent_name, feature_name, tool_name, command,
                pattern, repo_scope, rule_source,
            ))
            await db.commit()
            return int(cursor.lastrowid)

    async def finalize_auto_approve(
        self, audit_id: int, exit_code: int
    ) -> None:
        """Stamp the real exit code + completion time on a phase-1 row."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE auto_approve_audit
                SET exit_code = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ? AND completed_at IS NULL
            """, (exit_code, audit_id))
            await db.commit()

    async def get_auto_approve_audit(self, limit: int = 50) -> List[Dict]:
        """Recent auto-approve audit rows (most recent first)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT id, agent_did, agent_name, feature_name, tool_name,
                       command, pattern, repo_scope, rule_source, decision,
                       exit_code, created_at, completed_at
                FROM auto_approve_audit ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    def clear_session_overrides(self) -> None:
        """Clear session-scoped overrides and the session tier of global Auto.

        The persistent ("always") tier deliberately survives a session reset —
        that is the whole point of Always Auto. After this call the effective
        global Auto state falls back to whatever was persisted.
        """
        count = len(self._session_overrides)
        self._session_overrides.clear()
        self._global_auto_session = False
        logger.info(
            "Cleared %d session overrides; session-tier global Auto disabled "
            "(persistent Auto=%s)",
            count,
            self._global_auto_always,
        )

    def set_global_auto_mode(self, enabled: bool) -> None:
        """Enable or disable the *session* tier of global Auto mode.

        In-memory only; cleared on session reset. For the persistent tier use
        :meth:`set_global_auto_mode_scope`.
        """
        self._global_auto_session = bool(enabled)
        logger.warning(
            "Session-tier global security Auto mode %s",
            "enabled" if self._global_auto_session else "disabled",
        )

    async def set_global_auto_mode_scope(self, scope: str) -> None:
        """Set the effective global Auto tier.

        ``off``     — disable both tiers and clear the persisted flag.
        ``session`` — enable the in-memory tier; clear any persisted flag
                      (an explicit choice of "this session" downgrades Always).
        ``always``  — persist the flag so it survives session resets and
                      server restarts until explicitly turned off.
        """
        if scope not in ("off", "session", "always"):
            raise ValueError(
                f"Invalid global Auto scope '{scope}'. Use: off, session, always"
            )

        if scope == "off":
            self._global_auto_session = False
            self._global_auto_always = False
            await self._delete_global_auto_always()
        elif scope == "session":
            self._global_auto_session = True
            self._global_auto_always = False
            await self._delete_global_auto_always()
        else:  # always
            self._global_auto_always = True
            await self._persist_global_auto_always()

        logger.warning("Global security Auto mode set to scope=%s", scope)

    def get_global_auto_mode(self) -> bool:
        """Return whether global Auto mode is effectively enabled (either tier)."""
        return self._global_auto_mode

    def get_global_auto_mode_scope(self) -> str:
        """Return the effective global Auto tier: 'always', 'session', or 'off'."""
        if self._global_auto_always:
            return "always"
        if self._global_auto_session:
            return "session"
        return "off"

    async def _load_global_auto_always(self) -> None:
        """Rehydrate the persistent global Auto tier from storage."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT value FROM security_global_settings WHERE key = ?",
                ("global_auto_always",),
            )
            row = await cursor.fetchone()
        self._global_auto_always = bool(row) and row[0] == "1"
        if self._global_auto_always:
            logger.warning("Persistent global security Auto mode rehydrated (always)")

    async def _persist_global_auto_always(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO security_global_settings (key, value, updated_at)
                VALUES (?, '1', CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET value = '1',
                                               updated_at = CURRENT_TIMESTAMP
                """,
                ("global_auto_always",),
            )
            await db.commit()

    async def _delete_global_auto_always(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM security_global_settings WHERE key = ?",
                ("global_auto_always",),
            )
            await db.commit()

    def __repr__(self) -> str:
        return (
            f"PermissionStore(db={self.db_path}, "
            f"session_overrides={len(self._session_overrides)}, "
            f"global_auto_mode={self._global_auto_mode})"
        )
