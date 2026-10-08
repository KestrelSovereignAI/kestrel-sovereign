"""Refuse a deploy restart that would boot agents into constitution Safe Mode.

A deploy can change the governing constitution: a package upgrade carrying new
constitution text, or a revision that edits it. Every agent stays anchored to
the old hash, so restarting onto that code puts each of them in constitution
Safe Mode until a Sovereign reanchor ceremony runs (#3517). Nothing about that
was visible before the restart. Every deploy path therefore asks this module
first:

- ``kestrel update``, before it changes anything, against the revision its
  ``git pull --ff-only`` will land on;
- ``kestrel restart``, against the installed code;
- the restart coordinator, before it spawns a restart and, for an
  ``update_then_restart`` request, against the fetched revision before the
  update checks it out.

The comparison is doctor's constitution drift check: the same read-only
governance readings and the same canonical resolver the startup integrity
audit uses. It reads the agent databases directly, so it never needs the host
to be up.

What a deploy changes is the *package*. A descriptor-selected external source
is operator configuration the deploy does not touch, so it is read from disk.
The incoming bytes go through the resolver of the code doing the checking; a
revision that changes the resolver itself is checked again by ``kestrel
restart`` once installed, before anything restarts.
"""

from __future__ import annotations

import functools
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from kestrel_sovereign.doctor import (
    ConstitutionAnchorVerdict,
    DoctorReport,
    _check_constitution_drift,
    _read_agent_governance,
    runtime_env,
)
from kestrel_sovereign.multi_agent.config import (
    MULTI_AGENT_CONFIG_FILENAME,
    MultiAgentConfig,
)

#: Where the Sovereign reanchor (adoption) procedure is documented.
ADOPTION_RUNBOOK = "docs/architecture/security/SOVEREIGN_TRUST_ROOT.md"

#: Exit status of a ``kestrel update`` / ``kestrel restart`` this gate refused.
#: Distinct from the core-state codes ``kestrel update`` also returns (2-4).
CONSTITUTION_ADOPTION_REQUIRED = 5

#: The operator flag that restarts anyway, for an operator about to run the
#: adoption ceremony.
OVERRIDE_FLAG = "--allow-constitution-safe-mode"

#: Verdicts the startup integrity audit turns into Safe Mode.
_SAFE_MODE_STATUSES = frozenset({"mismatch", "audit_failure"})

_GIT_TIMEOUT_SECONDS = 60


class ConstitutionAdoptionError(RuntimeError):
    """The gate could not establish what the code about to run would produce.

    Deliberately not an ``OSError``/``ValueError``: doctor's readers turn those
    into per-agent findings, and a gate that cannot read the incoming revision
    has learned nothing about any agent.
    """


@dataclass(frozen=True)
class ConstitutionAdoptionCheck:
    """One gate evaluation: a verdict per agent the restart would boot."""

    verdicts: tuple[ConstitutionAnchorVerdict, ...] = ()
    #: What the governing hashes were computed from, as the operator reads it.
    code_label: str = "the installed code"

    @property
    def blocking(self) -> tuple[ConstitutionAnchorVerdict, ...]:
        """The agents the restart would boot into constitution Safe Mode."""
        return tuple(v for v in self.verdicts if v.status in _SAFE_MODE_STATUSES)

    @property
    def unverified(self) -> tuple[ConstitutionAnchorVerdict, ...]:
        """The agents whose anchor could not be read, so were not compared."""
        return tuple(v for v in self.verdicts if v.status == "unverified")

    @property
    def safe(self) -> bool:
        return not self.blocking


def check_constitution_adoption(
    project_dir: Path,
    *,
    agent_names: Iterable[str] | None = None,
    packaged_constitution: Callable[[], bytes | None] | None = None,
    code_label: str = "the installed code",
) -> ConstitutionAdoptionCheck:
    """Compare each agent's anchored hash with what the code about to run produces.

    ``agent_names`` limits the check to the agents a restart will boot; None
    means every local agent in the project's multi-agent registry, loaded the
    way the launcher loads it. ``packaged_constitution`` supplies the packaged
    constitution a deploy is about to install (see
    :func:`packaged_constitution_at`); None judges the code on disk. It is
    called at most once, and only when an agent governed by the package is
    actually compared.

    Read-only, and independent of the host: the anchors are read from the
    agent databases, and the governing source through the canonical resolver.

    Raises:
        ConstitutionAdoptionError: The registry cannot be loaded, or
            ``packaged_constitution`` could not produce the incoming bytes.
    """
    env = runtime_env(project_dir)
    try:
        multi_agent = MultiAgentConfig.load(
            project_dir / MULTI_AGENT_CONFIG_FILENAME, runtime_env=env
        )
    except (OSError, ValueError) as exc:
        raise ConstitutionAdoptionError(
            f"multi-agent configuration is invalid: {exc}"
        ) from exc

    local = multi_agent.get_local_agents()
    selected = (
        list(local) if agent_names is None else [n for n in agent_names if n in local]
    )
    readings = _read_agent_governance(
        multi_agent, project_dir, env, agent_names=frozenset(selected)
    )
    verdicts: dict[str, ConstitutionAnchorVerdict] = {}
    _check_constitution_drift(
        readings,
        DoctorReport(),
        env,
        packaged_constitution=(
            functools.cache(packaged_constitution)
            if packaged_constitution is not None
            else None
        ),
        verdicts=verdicts,
    )
    for name in selected:
        if name not in verdicts:
            # ``_read_agent_governance`` skips an agent with no anchor file.
            db_path = (project_dir / local[name].data_dir).resolve() / "kestrel_prime.db"
            verdicts[name] = ConstitutionAnchorVerdict(
                agent=name,
                status="unverified",
                detail=f"no kestrel_prime.db at {db_path}",
            )
    return ConstitutionAdoptionCheck(
        verdicts=tuple(verdicts[name] for name in selected),
        code_label=code_label,
    )


def packaged_constitution_at(repo_path: str | Path, revision: str) -> bytes | None:
    """The packaged constitution as checking out ``revision`` would leave it.

    Returns None when that checkout leaves the file on disk as it is: the
    checkout at ``repo_path`` does not supply the running package's
    constitution, or ``revision`` does not change it (so an uncommitted edit
    the checkout keeps is still what runs). Otherwise the bytes are what
    checkout writes, with the checkout's own filters applied.

    Raises:
        ConstitutionAdoptionError: git could not compare or read the revision,
            including a revision that has no packaged constitution at all.
    """
    from kestrel_sovereign.constitution.resolver import governing_constitution_path

    repo = Path(repo_path).resolve()
    packaged = Path(governing_constitution_path()).resolve()
    try:
        relpath = packaged.relative_to(repo).as_posix()
    except ValueError:
        return None

    compared = _git(repo, "diff", "--quiet", "HEAD", revision, "--", relpath)
    if compared.returncode == 0:
        return None
    if compared.returncode != 1:
        raise ConstitutionAdoptionError(
            f"cannot compare {relpath} at {revision} with HEAD in {repo}: "
            f"{_git_detail(compared)}"
        )
    shown = _git(repo, "cat-file", "--filters", f"{revision}:{relpath}")
    if shown.returncode != 0:
        raise ConstitutionAdoptionError(
            f"cannot read {relpath} at {revision} in {repo}: {_git_detail(shown)}"
        )
    return shown.stdout


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConstitutionAdoptionError(f"git is unavailable: {exc}") from exc


def _git_detail(result: subprocess.CompletedProcess) -> str:
    raw = result.stderr or result.stdout or b""
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    return text.strip() or f"git exited {result.returncode}"


def _describe(verdict: ConstitutionAnchorVerdict) -> str:
    if verdict.status == "mismatch":
        return (
            f"{verdict.agent}: anchored to {verdict.anchored_hash}; the "
            f"governing constitution ({verdict.governing_path}) hashes to "
            f"{verdict.governing_hash}"
        )
    anchored = (
        f" (anchored {verdict.anchored_hash})" if verdict.anchored_hash else ""
    )
    return (
        f"{verdict.agent}: its governing constitution cannot be produced"
        f"{anchored}: {verdict.detail}"
    )


def refusal_lines(check: ConstitutionAdoptionCheck) -> list[str]:
    """The operator-facing explanation of a blocking check, and how to adopt."""
    header = (
        f"Restarting onto {check.code_label} would boot "
        f"{len(check.blocking)} agent(s) into constitution Safe Mode:"
    )
    procedure = (
        (
            f'Adopt the new constitution first ("Reanchor procedure" in '
            f"{ADOPTION_RUNBOOK}):"
        ),
        (
            "  1. Sign a kestrel.constitution.reanchor.v1 artifact for each "
            "governing hash above with the Sovereign key the trust root pins."
        ),
        (
            "  2. Offline: install without restarting (`kestrel update "
            "--no-restart`), `kestrel terminate`, then for each agent run "
            "`kestrel constitution reanchor --agent-name <name> --force "
            "--signed-artifact <artifact> --trust-root <root>`, then "
            "`kestrel start`."
        ),
        (
            f"  3. Live: restart with {OVERRIDE_FLAG}, then run "
            "`!reanchor-constitution <artifact> <hash-prefix>` on each agent "
            "and leave Safe Mode with `!safe-mode exit`."
        ),
        (
            "  A governing source that cannot be produced (descriptor, trust "
            'root, pinned digest) is repaired under "Custom governing '
            'constitution sources" in the same document.'
        ),
    )
    return [
        header,
        *(f"  {_describe(verdict)}" for verdict in check.blocking),
        *procedure,
    ]


def refusal_reason(check: ConstitutionAdoptionCheck) -> str:
    """One-line terminal reason for a refused restart request."""
    agents = "; ".join(_describe(verdict) for verdict in check.blocking)
    return (
        f"constitution adoption required: restarting onto {check.code_label} "
        f"would boot agents into constitution Safe Mode ({agents}). A "
        f"Sovereign must reanchor them first; see {ADOPTION_RUNBOOK}."
    )


def unverified_lines(check: ConstitutionAdoptionCheck) -> list[str]:
    """A warning naming the agents the gate could not compare."""
    if not check.unverified:
        return []
    header = (
        f"Constitution anchor not verified for {len(check.unverified)} "
        f"agent(s); the gate cannot vouch for them (run `kestrel doctor`):"
    )
    return [header, *(f"  {v.agent}: {v.detail}" for v in check.unverified)]
