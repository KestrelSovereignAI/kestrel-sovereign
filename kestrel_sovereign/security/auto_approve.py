"""Scoped auto-approve policy for gated tool invocations.

This is the keystone that lets a sovereign agent (Emma) close her own
dispatch loop without the Sovereign typing approvals on the CLI. A
``computer_use.shell`` request that matches a Sovereign-curated regex
allowlist, scoped to a specific agent and repo, is auto-approved at the
single approval chokepoint
(:meth:`ApprovalQueue.request_approval`) instead of stalling on a human.

Constitutional invariant (Article I): this expands Emma's *authority*, not
her *autonomy*. Every auto-approved action is

  (a) scoped to a specific pattern the Sovereign explicitly added,
  (b) audited immutably (command, agent DID, timestamp, exit code) —
      *no silent runs*, and
  (c) revocable by removing the pattern.

Rules come from two unioned sources:

  * **Operator seed** — ``[[security.auto_approve.shell]]`` tables in
    ``kestrel.toml``. Baseline patterns the operator ships with.
  * **Sovereign-curated** — the ``auto_approve_rules`` DB table, populated
    by the "Approve-and-remember" button in the Mews approval panel and
    revocable by deleting the row.

The matcher itself is pure and DB-free; the dynamic rules and the audit
sink live on :class:`PermissionStore` (which already owns the agent DB).
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kestrel_sovereign.features.security.permissions import PermissionStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoApproveRule:
    """A single allowlist entry.

    ``pattern`` is an anchored-or-unanchored regex matched (via
    :func:`re.search`) against the derived command string. ``repo_scope``
    is both audit metadata *and* a defence-in-depth guard: the scope
    string must literally appear in the command, so a pattern can never be
    coerced into acting on a different repo. ``agent`` (when set) restricts
    the rule to one agent by name; ``None`` means any agent.
    """

    pattern: str
    repo_scope: str
    audit: bool = True
    agent: Optional[str] = None
    source: str = "seed"  # "seed" | "db"

    def compiled(self) -> Optional[re.Pattern[str]]:
        try:
            return re.compile(self.pattern)
        except re.error as exc:  # pragma: no cover - operator misconfig
            logger.warning(
                "auto_approve: skipping invalid pattern %r: %s",
                self.pattern,
                exc,
            )
            return None


@dataclass(frozen=True)
class AutoApproveMatch:
    """The rule that authorised a request, carried back to the queue."""

    rule: AutoApproveRule
    command: str


def derive_command(
    feature_name: str, tool_name: str, tool_args: Dict[str, Any]
) -> Optional[str]:
    """Reduce a tool's args to the single string the allowlist matches.

    Returns ``None`` for tool shapes the allowlist does not govern (so the
    request falls through to normal human approval).
    """
    fname = (feature_name or "").lower()
    if fname == "computer_use" and tool_name == "shell":
        argv = tool_args.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            return shlex.join(str(a) for a in argv)
        cmd = tool_args.get("command")
        return str(cmd) if cmd else None
    # NOTE: compute.run_script is deliberately NOT auto-approvable here.
    # A regex allowlist matches a command *string*; a signed script's
    # executable content is fetched by id and is not in tool_args, so any
    # name+purpose-based key would let a different script body run
    # unreviewed (codex review P1, epic #1290). Signed-script approval
    # therefore stays fully human-gated. The only auto-approve surface is
    # computer_use.shell, whose exit code IS finalized in
    # computer_use._audit_run — so there is no unfinalized-audit path.
    return None


_FREEFORM_FLAGS = {
    "--title", "--body", "-t", "-b", "-F", "--body-file", "-m",
    "--message", "-c", "--comment",
}


def suggest_rule_from_command(command: str) -> tuple[str, str]:
    """Derive a conservative ``(pattern, repo_scope)`` for "remember".

    "Approve-and-remember" must not learn an over-broad rule. We anchor a
    regex on the *fixed prefix* of the command — every token up to (but
    not including) the first free-form value flag (``--title``,
    ``--body``, …) — and escape it literally. So approving::

        gh issue create -R OWNER/REPO --title "x" --body "y"

    remembers exactly ``^gh\\ issue\\ create\\ \\-R\\ OWNER/REPO`` — it
    will auto-approve future ``gh issue create`` calls *against that repo*
    regardless of title/body, and nothing else. ``repo_scope`` is pulled
    from a ``-R``/``--repo`` ``owner/name`` token for the audit + the
    defence-in-depth guard.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    prefix: List[str] = []
    repo_scope = ""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _FREEFORM_FLAGS:
            break
        prefix.append(tok)
        if tok in ("-R", "--repo") and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            prefix.append(nxt)
            if "/" in nxt:
                repo_scope = nxt
            i += 2
            continue
        i += 1
    if not prefix:
        prefix = tokens[:1]
    pattern = "^" + re.escape(" ".join(prefix))
    return pattern, repo_scope


class AutoApprovePolicy:
    """Evaluates a request against the unioned seed + DB allowlist."""

    def __init__(
        self,
        seed_rules: List[AutoApproveRule],
        permission_store: Optional["PermissionStore"] = None,
    ) -> None:
        self._seed_rules = list(seed_rules)
        self._permission_store = permission_store

    @classmethod
    def from_config(
        cls,
        security_cfg: Optional[Dict[str, Any]],
        permission_store: Optional["PermissionStore"] = None,
    ) -> "AutoApprovePolicy":
        """Build from the ``[security]`` section of ``kestrel.toml``.

        Expected shape::

            [[security.auto_approve.shell]]
            pattern = "^gh issue (create|comment) -R KestrelSovereignAI/kestrel-sovereign"
            repo_scope = "KestrelSovereignAI/kestrel-sovereign"
            audit = true
            agent = "Emma"            # optional; omit for any agent
        """
        rules: List[AutoApproveRule] = []
        auto = ((security_cfg or {}).get("auto_approve") or {})
        for entry in auto.get("shell", []) or []:
            try:
                rules.append(
                    AutoApproveRule(
                        pattern=str(entry["pattern"]),
                        repo_scope=str(entry.get("repo_scope", "")),
                        audit=bool(entry.get("audit", True)),
                        agent=(
                            str(entry["agent"])
                            if entry.get("agent")
                            else None
                        ),
                        source="seed",
                    )
                )
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "auto_approve: ignoring malformed seed entry %r: %s",
                    entry,
                    exc,
                )
        if rules:
            logger.info(
                "AutoApprovePolicy loaded %d seed rule(s) from kestrel.toml",
                len(rules),
            )
        return cls(rules, permission_store)

    async def _all_rules(self) -> List[AutoApproveRule]:
        rules = list(self._seed_rules)
        if self._permission_store is not None:
            try:
                for row in await self._permission_store.list_auto_approve_rules():
                    rules.append(
                        AutoApproveRule(
                            pattern=str(row["pattern"]),
                            repo_scope=str(row.get("repo_scope", "")),
                            audit=True,
                            agent=row.get("agent") or None,
                            source="db",
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - never block approvals
                logger.warning(
                    "auto_approve: failed to load dynamic rules: %s",
                    exc,
                    exc_info=True,
                )
        return rules

    async def evaluate(
        self,
        *,
        agent_name: Optional[str],
        feature_name: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Optional[AutoApproveMatch]:
        """Return the matching rule, or ``None`` to fall through to human.

        Match requires, for some rule: agent-scope OK *and* the
        ``repo_scope`` string present in the command *and* the regex
        matching the command. First match wins.
        """
        command = derive_command(feature_name, tool_name, tool_args)
        if not command:
            return None
        for rule in await self._all_rules():
            if rule.agent is not None and rule.agent != agent_name:
                continue
            if rule.repo_scope and rule.repo_scope not in command:
                continue
            regex = rule.compiled()
            if regex is None:
                continue
            if regex.search(command):
                logger.info(
                    "auto_approve: %s matched %s rule (agent=%s, repo=%s)",
                    tool_name,
                    rule.source,
                    rule.agent or "*",
                    rule.repo_scope or "*",
                )
                return AutoApproveMatch(rule=rule, command=command)
        return None
