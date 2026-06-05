"""Test-evidence verification for the Agent/Talon review loop.

Running tests is the evidence gate the Agent/Talon review loop turns on,
so it deserves a first-class, deterministic result vocabulary instead of
ad-hoc shell calls whose failures are ambiguous. This module is that
vocabulary plus the small orchestrator that produces it.

Issue #1542: the review loop must distinguish, precisely, between

  * ``passed``            — the command ran and exited 0
  * ``failed``            — the command ran and exited non-zero
  * ``blocked_by_policy`` — operator policy / approval layer refused, or
                            no user ever decided (timeout / cancel). This
                            is NOT a user denial.
  * ``blocked_by_user``   — a user explicitly denied at the approval
                            prompt (the approval record says so).
  * ``blocked_by_sandbox``— the execution environment refused to run the
                            command (e.g. sandbox/permission refusal),
                            distinct from policy.
  * ``tooling_error``     — the command could not run for a tooling
                            reason (binary missing, timeout, exception).
  * ``not_run``           — not attempted.

The deliberate split between ``blocked_by_policy`` and ``blocked_by_user``
encodes the acceptance criterion that a sandbox/approval-layer rejection
must not be reported as a user denial unless the approval record
explicitly says the user denied it. ``classify_denial`` is the single
home for that rule; it reads only the approval queue's own
``(approved, scope)`` contract (see
``features/security/approval_queue.py``).

This layer owns *reviewer-side* audited execution and result reporting.
It intentionally does not live in RestartCoordinator — restart/update is
only the deployment primitive; implementation and review workflows own
test evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, ClassVar, Mapping, Optional, Sequence

# Allowlisted project test command prefixes. A command whose normalized
# form starts with one of these is considered a safe, pre-authorized
# project test invocation and runs without an approval prompt. Anything
# else is approval-gated (the reviewer can still vouch for it, but the
# block is recorded precisely). Keep this conservative: it is an
# allowlist of *test runners*, not a general shell escape hatch.
DEFAULT_TEST_ALLOWLIST: tuple[str, ...] = (
    "uv run pytest",
    "uv run python -m pytest",
    "uv run python run_tests.py",
    "uv run ./run_tests.py",
    "python -m pytest",
    "pytest",
    "./run_tests.py",
    "run_tests.py",
    "npx playwright test",
)

# How many trailing characters of stdout/stderr to retain in evidence.
# Enough to show the failing assertion / traceback tail without dragging
# a whole test log into the audit record or a PR comment.
_TAIL_CHARS = 2000


class VerificationState(str, Enum):
    """Observed outcome of a single verification command.

    String-valued so it serializes cleanly into signal payloads, audit
    rows, and PR/review comments without a custom encoder.
    """

    PASSED = "passed"
    FAILED = "failed"
    BLOCKED_BY_POLICY = "blocked_by_policy"
    BLOCKED_BY_USER = "blocked_by_user"
    BLOCKED_BY_SANDBOX = "blocked_by_sandbox"
    TOOLING_ERROR = "tooling_error"
    NOT_RUN = "not_run"


# Severity ordering for rolling many command results up into one overall
# state. Lower index = surfaced first. "Did any test fail" dominates;
# "everything passed" is the floor. ``not_run`` sits just above passed so
# an evidence set with nothing attempted does not masquerade as a pass.
_SEVERITY_ORDER: tuple[VerificationState, ...] = (
    VerificationState.FAILED,
    VerificationState.TOOLING_ERROR,
    VerificationState.BLOCKED_BY_SANDBOX,
    VerificationState.BLOCKED_BY_USER,
    VerificationState.BLOCKED_BY_POLICY,
    VerificationState.NOT_RUN,
    VerificationState.PASSED,
)


def normalize_command(command: str) -> str:
    """Collapse internal whitespace so allowlist matching is stable."""
    return " ".join(str(command).split())


def is_allowlisted(
    command: str,
    allowlist: Sequence[str] = DEFAULT_TEST_ALLOWLIST,
) -> bool:
    """True if ``command`` is a pre-authorized project test invocation.

    Match is prefix-based on the whitespace-normalized command so
    ``uv run pytest tests/unit -q`` matches the ``uv run pytest`` entry
    while ``uv run python deploy.py`` does not match anything.
    """
    norm = normalize_command(command)
    for prefix in allowlist:
        p = normalize_command(prefix)
        if norm == p or norm.startswith(p + " "):
            return True
    return False


def _tail(text: str, limit: int = _TAIL_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[-limit:]


@dataclass(frozen=True)
class CommandExecution:
    """Raw result of trying to run a command.

    Produced by an injected executor so the classification logic stays
    pure and unit-testable. ``ran`` is the load-bearing field: it records
    whether the process actually executed (and therefore whether
    ``returncode`` is meaningful) versus whether something stopped it
    before it ran.
    """

    ran: bool
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: Optional[int] = None
    # Set when the process could not run for a tooling reason (binary
    # missing, timeout, unexpected exception). Mutually informative with
    # ``ran=False``.
    error: Optional[str] = None
    # Set when the *sandbox/execution environment* refused to run the
    # command (permission refusal), as opposed to a missing binary.
    sandbox_denied: bool = False


@dataclass(frozen=True)
class TestCommandResult:
    """One command's verification evidence."""

    # Not a pytest test class despite the name — keep the collector away.
    __test__: ClassVar[bool] = False

    command: str
    state: VerificationState
    exit_code: Optional[int] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_ms: Optional[int] = None
    allowlisted: bool = False
    # Human-readable precise reason. For the blocked_* / tooling states
    # this carries the sub-reason (e.g. "approval request timed out
    # before a user decided") so attribution survives even within a
    # single state bucket.
    summary: str = ""

    @property
    def is_pass(self) -> bool:
        return self.state is VerificationState.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "state": self.state.value,
            "exit_code": self.exit_code,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "duration_ms": self.duration_ms,
            "allowlisted": self.allowlisted,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class CIStatus:
    """CI status as reported by the implementation side (Talon/PR).

    The reviewer does not run CI; it records what CI reported so merge
    notes can cite it. ``checks`` holds individual check identifiers
    (name + conclusion + url) when available.
    """

    state: str = "unknown"  # passed | failed | pending | unknown
    summary: str = ""
    url: str = ""
    checks: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "summary": self.summary,
            "url": self.url,
            "checks": [dict(c) for c in self.checks],
        }

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> Optional["CIStatus"]:
        if not data:
            return None
        checks_raw = data.get("checks") or ()
        checks = tuple(dict(c) for c in checks_raw if isinstance(c, Mapping))
        return cls(
            state=str(data.get("state", "unknown")),
            summary=str(data.get("summary", "")),
            url=str(data.get("url", "")),
            checks=checks,
        )


@dataclass
class VerificationEvidence:
    """Aggregated test evidence for a review/merge decision."""

    results: list[TestCommandResult] = field(default_factory=list)
    ci_status: Optional[CIStatus] = None
    # Free-form reviewer note, e.g. "local tests could not run; CI is the
    # remaining hard gate."
    note: str = ""

    @property
    def overall_state(self) -> VerificationState:
        if not self.results:
            return VerificationState.NOT_RUN
        present = {r.state for r in self.results}
        for state in _SEVERITY_ORDER:
            if state in present:
                return state
        return VerificationState.NOT_RUN

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.is_pass for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_state": self.overall_state.value,
            "all_passed": self.all_passed,
            "results": [r.to_dict() for r in self.results],
            "ci_status": self.ci_status.to_dict() if self.ci_status else None,
            "note": self.note,
        }

    def to_markdown(self) -> str:
        """Render evidence for inclusion in review/merge notes.

        Output is intentionally a self-contained ``## Test Evidence``
        block so it can be dropped straight into a PR comment or merge
        note (acceptance criterion: merge/review notes include test
        evidence, not just source-review assertions).
        """
        lines = ["## Test Evidence", ""]
        lines.append(f"**Overall:** `{self.overall_state.value}`")
        lines.append("")
        if self.results:
            lines.append("| Command | Result | Exit | Notes |")
            lines.append("| --- | --- | --- | --- |")
            for r in self.results:
                exit_disp = "—" if r.exit_code is None else str(r.exit_code)
                note = (r.summary or "").replace("\n", " ").replace("|", "\\|")
                if len(note) > 160:
                    note = note[:157] + "..."
                cmd = r.command.replace("|", "\\|")
                lines.append(
                    f"| `{cmd}` | `{r.state.value}` | {exit_disp} | {note} |"
                )
        else:
            lines.append("_No local verification commands were run._")
        lines.append("")
        if self.ci_status is not None:
            ci = self.ci_status
            ci_line = f"**CI:** `{ci.state}`"
            if ci.summary:
                ci_line += f" — {ci.summary}"
            if ci.url:
                ci_line += f" ([details]({ci.url}))"
            lines.append(ci_line)
            for check in ci.checks:
                name = check.get("name", "check")
                concl = check.get("conclusion", check.get("status", "?"))
                url = check.get("url", "")
                suffix = f" ([link]({url}))" if url else ""
                lines.append(f"  - `{name}`: `{concl}`{suffix}")
            lines.append("")
        if self.note:
            lines.append(f"> {self.note}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def classify_execution(
    command: str,
    execution: CommandExecution,
    *,
    allowlisted: bool,
) -> TestCommandResult:
    """Map a raw :class:`CommandExecution` to a verification result."""
    if execution.sandbox_denied:
        return TestCommandResult(
            command=command,
            state=VerificationState.BLOCKED_BY_SANDBOX,
            allowlisted=allowlisted,
            summary=execution.error
            or "execution environment refused to run the command (sandbox)",
        )
    if not execution.ran:
        return TestCommandResult(
            command=command,
            state=VerificationState.TOOLING_ERROR,
            allowlisted=allowlisted,
            summary=execution.error or "command did not run (tooling error)",
        )
    state = (
        VerificationState.PASSED
        if execution.returncode == 0
        else VerificationState.FAILED
    )
    summary = (
        "exited 0"
        if state is VerificationState.PASSED
        else f"exited {execution.returncode}"
    )
    return TestCommandResult(
        command=command,
        state=state,
        exit_code=execution.returncode,
        stdout_tail=_tail(execution.stdout),
        stderr_tail=_tail(execution.stderr),
        duration_ms=execution.duration_ms,
        allowlisted=allowlisted,
        summary=summary,
    )


def classify_denial(command: str, scope: Optional[str]) -> TestCommandResult:
    """Attribute an approval denial precisely, per the #1542 rule.

    Reads only the approval queue's ``scope`` contract, where the scope
    returned alongside ``approved=False`` carries the denial's provenance:

      * ``user_denied`` — a human pressed deny via the deny tool /
        ``!security-deny`` (``SecurityFeature.deny_request`` submits this
        scope, and ``ApprovalQueue.request_approval`` returns it). This is
        the canonical real *user* denial path.
      * ``once`` / ``session`` / ``always`` with ``approved=False`` — a
        human denied through the web UI ``/approve`` endpoint, which only
        accepts those scopes. Also a genuine user denial.
      * ``denied`` — an operator/auto policy DENY. ``request_approval``
        early-returns this *without* a human ever being asked, so it is
        NOT a user denial (this is the exact collision #1542 fixed: the
        deny tool used to also emit ``denied``).
      * ``timeout`` / ``cancelled`` / ``cancelled_all`` mean no user ever
        decided — emphatically not a user denial.

    Anything we cannot positively attribute to the user is reported as
    ``blocked_by_policy`` so a sandbox/approval-layer rejection is never
    mislabeled as the user saying no.
    """
    s = (scope or "").strip().lower()
    if s in ("user_denied", "once", "session", "always"):
        return TestCommandResult(
            command=command,
            state=VerificationState.BLOCKED_BY_USER,
            allowlisted=False,
            summary="user explicitly denied the command at the approval prompt",
        )
    if s == "timeout":
        return TestCommandResult(
            command=command,
            state=VerificationState.BLOCKED_BY_POLICY,
            allowlisted=False,
            summary=(
                "approval request timed out before a user decided "
                "(not a user denial)"
            ),
        )
    if s in ("cancelled", "cancelled_all"):
        return TestCommandResult(
            command=command,
            state=VerificationState.BLOCKED_BY_POLICY,
            allowlisted=False,
            summary=(
                "approval request was cancelled before a user decided "
                "(not a user denial)"
            ),
        )
    return TestCommandResult(
        command=command,
        state=VerificationState.BLOCKED_BY_POLICY,
        allowlisted=False,
        summary=(
            "blocked by operator policy / approval layer "
            "(not a user denial)"
        ),
    )


# Injected dependency signatures. Keeping these as plain callables means
# the verifier carries no feature/agent coupling and is fully unit
# testable.
Executor = Callable[..., Awaitable[CommandExecution]]
# Returns (approved, scope) like ApprovalQueue.request_approval, or None
# when there is no approval mechanism available at all (fail-closed).
Approver = Callable[[str], Awaitable[Optional[tuple[bool, str]]]]


class TalonVerifier:
    """Runs verification commands and produces precise result states.

    Allowlisted commands run directly. Non-allowlisted commands are
    approval-gated; the approval outcome is attributed precisely via
    :func:`classify_denial`. The verifier never executes a non-allowlisted
    command that was not approved.
    """

    def __init__(
        self,
        execute: Executor,
        approve: Optional[Approver] = None,
        allowlist: Sequence[str] = DEFAULT_TEST_ALLOWLIST,
    ) -> None:
        self._execute = execute
        self._approve = approve
        self._allowlist = tuple(allowlist)

    async def verify_command(
        self, command: str, *, timeout: int = 600
    ) -> TestCommandResult:
        command = command.strip()
        if not command:
            return TestCommandResult(
                command=command,
                state=VerificationState.TOOLING_ERROR,
                summary="empty command",
            )

        allowlisted = is_allowlisted(command, self._allowlist)
        if not allowlisted:
            if self._approve is None:
                return TestCommandResult(
                    command=command,
                    state=VerificationState.BLOCKED_BY_POLICY,
                    allowlisted=False,
                    summary=(
                        "command is not on the test allowlist and no "
                        "approval mechanism is available; not run "
                        "(fail-closed, not a user denial)"
                    ),
                )
            decision = await self._approve(command)
            if decision is None:
                return TestCommandResult(
                    command=command,
                    state=VerificationState.BLOCKED_BY_POLICY,
                    allowlisted=False,
                    summary=(
                        "approval mechanism unavailable; non-allowlisted "
                        "command not run (fail-closed, not a user denial)"
                    ),
                )
            approved, scope = decision
            if not approved:
                return classify_denial(command, scope)

        execution = await self._execute(command, timeout=timeout)
        return classify_execution(command, execution, allowlisted=allowlisted)

    async def verify_commands(
        self,
        commands: Sequence[str],
        *,
        timeout: int = 600,
        ci_status: Optional[CIStatus] = None,
        note: str = "",
    ) -> VerificationEvidence:
        results: list[TestCommandResult] = []
        for command in commands:
            results.append(await self.verify_command(command, timeout=timeout))
        return VerificationEvidence(
            results=results, ci_status=ci_status, note=note
        )
