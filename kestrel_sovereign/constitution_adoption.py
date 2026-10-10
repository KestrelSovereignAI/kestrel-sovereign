"""Refuse a deploy restart that would boot agents into constitution Safe Mode.

A deploy can change the governing constitution: a package upgrade carrying new
constitution text, or a revision that edits it. Every agent stays anchored to
the old hash, so restarting onto that code puts each of them in constitution
Safe Mode until a Sovereign reanchor ceremony runs (#3517). Nothing about that
was visible before the restart. Every deploy path therefore asks this module
first:

- ``kestrel update``, before it changes anything, against the revision its
  ``git pull --ff-only`` will land on, for every local agent the shared
  package governs;
- ``kestrel restart`` (including ``kestrel update``'s restart step), against
  the installed code;
- the restart coordinator, before it spawns a restart and, for an
  ``update_then_restart`` request, against the fetched revision before the
  update checks it out and against the installed code once it has.

The comparison is doctor's constitution drift check: the same read-only
governance readings and the same canonical resolver the startup integrity
audit uses. It reads the agent databases directly, so it never needs the host
to be up.

What a deploy changes is its checkout: the packaged constitution, and any
descriptor-selected external source that checkout tracks (#3522). Both are
judged as the incoming revision leaves them. A source outside the checkout is
operator configuration the deploy does not touch, so it is read from disk. A
deploy that would replace the descriptor or trust root that *selects* the
source cannot be judged before it is installed, so it is refused.

Two checks, because the code that renders a constitution is part of what a
deploy replaces:

- Before an install, :func:`check_constitution_adoption` runs in the deploying
  process and judges the incoming constitution bytes with the resolver that
  process loaded: the only one there is before anything changes.
- Once code is installed, :func:`check_installed_constitution_adoption` judges
  it in a fresh interpreter launched the way the host is. A process that has
  imported ``kestrel_sovereign`` keeps that code after an install replaces it
  on disk, so a revision that changes how the resolver or the Amendment VIII
  rendering produces the governing bytes passes an in-process check while the
  host it restarts computes another hash. ``kestrel restart`` and the restart
  coordinator always judge the installed code this way.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from kestrel_sovereign.doctor import (
    ConstitutionAnchorVerdict,
    DeployedContent,
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

#: How the operator reads a check of the code installed on disk.
INSTALLED_CODE = "the installed code"

#: Verdicts the startup integrity audit turns into Safe Mode.
_SAFE_MODE_STATUSES = frozenset({"mismatch", "audit_failure"})

#: Every status a verdict can carry. A fresh interpreter answering with any
#: other status speaks a protocol this process cannot judge, so it refuses.
_VERDICT_STATUSES = frozenset({"match", "mismatch", "audit_failure", "unverified"})

_GIT_TIMEOUT_SECONDS = 60

#: Tree-entry modes ``git ls-tree`` reports for a directory and a link.
_GIT_TREE_MODE = b"040000"
_GIT_SYMLINK_MODE = b"120000"

#: Links one path resolution follows before it is a cycle, as Linux bounds it.
_MAX_SYMLINKS = 40

#: The module a fresh interpreter runs to check the installed code.
_FRESH_CHECK_MODULE = "kestrel_sovereign.constitution_adoption"

#: Version of the answer a fresh-interpreter check prints. The child is the
#: installed code, which may be newer than the process asking it.
_FRESH_CHECK_PROTOCOL = 1

#: Marks the answer line in the child's stdout, which imports may also write to.
_FRESH_CHECK_RESULT_PREFIX = "kestrel-constitution-adoption: "

#: Bound on the fresh-interpreter check: imports plus reading every agent's
#: anchor, each PostgreSQL probe already bounded by doctor.
_FRESH_CHECK_TIMEOUT_SECONDS = 180


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
    code_label: str = INSTALLED_CODE

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


def _load_registry(project_dir: Path, env: dict) -> MultiAgentConfig:
    """The project's multi-agent registry, loaded the way the launcher loads it."""
    try:
        return MultiAgentConfig.load(
            project_dir / MULTI_AGENT_CONFIG_FILENAME, runtime_env=env
        )
    except (OSError, ValueError) as exc:
        raise ConstitutionAdoptionError(
            f"multi-agent configuration is invalid: {exc}"
        ) from exc


def local_agents(project_dir: Path) -> dict:
    """The project's local agents, from the registry the checks judge.

    Raises:
        ConstitutionAdoptionError: The registry cannot be loaded.
    """
    return _load_registry(project_dir, runtime_env(project_dir)).get_local_agents()


def _select(local: dict, agent_names: Iterable[str] | None) -> list[str]:
    if agent_names is None:
        return list(local)
    return [name for name in agent_names if name in local]


def check_constitution_adoption(
    project_dir: Path,
    *,
    agent_names: Iterable[str] | None = None,
    deployed_content: DeployedContent | None = None,
    code_label: str = INSTALLED_CODE,
) -> ConstitutionAdoptionCheck:
    """Compare each agent's anchored hash with what this process's code produces.

    ``agent_names`` limits the check to the agents a restart will boot; None
    means every local agent in the project's multi-agent registry, loaded the
    way the launcher loads it. ``deployed_content`` supplies the bytes a
    deploy is about to leave at a governing path, packaged or external (see
    :func:`deployed_content_at`); None judges the files on disk. It is asked
    at most once per path: about a governing source only when an agent that
    source governs is actually compared, and about the descriptor and trust
    root of every selected agent that configures a descriptor.

    The bytes go through the resolver this process imported. Judging code that
    was installed after that import is :func:`check_installed_constitution_adoption`.

    Read-only, and independent of the host: the anchors are read from the
    agent databases, and the governing source through the canonical resolver.

    Raises:
        ConstitutionAdoptionError: The registry cannot be loaded,
            ``deployed_content`` could not produce the incoming bytes, or the
            deploy replaces a descriptor or trust root that selects a selected
            agent's governing source.
    """
    env = runtime_env(project_dir)
    multi_agent = _load_registry(project_dir, env)
    local = multi_agent.get_local_agents()
    selected = _select(local, agent_names)
    if deployed_content is not None:
        deployed_content = _deploy_reader(deployed_content)
        _refuse_replaced_source_selection(
            {name: local[name] for name in selected}, env, deployed_content
        )
    readings = _read_agent_governance(
        multi_agent, project_dir, env, agent_names=frozenset(selected)
    )
    verdicts: dict[str, ConstitutionAnchorVerdict] = {}
    _check_constitution_drift(
        readings,
        DoctorReport(),
        env,
        deployed_content=deployed_content,
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


def _deploy_reader(deployed_content: DeployedContent) -> DeployedContent:
    """``deployed_content``, asked once per path, failing as the gate fails.

    The drift check reads an ``OSError`` or ``ValueError`` as a property of
    the governing source, and an unreadable unpinned package as a skipped
    check. A failure to produce what the deploy leaves is neither: nothing
    has been learned about any agent, so it must refuse.
    """

    @functools.cache
    def read(path: Path) -> bytes | None:
        try:
            return deployed_content(path)
        except (OSError, ValueError) as exc:
            raise ConstitutionAdoptionError(
                f"cannot tell what the deploy leaves at {path}: {exc}"
            ) from exc

    return read


def _refuse_replaced_source_selection(
    agents: dict, env: dict, deployed_content: DeployedContent
) -> None:
    """Refuse a deploy that replaces what selects an agent's governing source.

    A source descriptor names the governing file and pins its digest; the
    trust root verifies the descriptor. The resolver reads both from disk, so
    a deploy that replaces either one changes which bytes govern in a way this
    check cannot judge before it is installed. A configuration that does not
    resolve is left to the drift check, which reports it per agent.

    Each file is asked about by every pathname that configures it, as
    configured: resolving one first would hide a link along it that the deploy
    retargets, leaving the old file to be judged.

    Raises:
        ConstitutionAdoptionError: The deploy replaces such a file, or a link
            or directory along a pathname that configures it.
    """
    from kestrel_sovereign.constitution.source_descriptor import (
        CONSTITUTION_SOURCE_DESCRIPTOR_ENV,
        ConstitutionSourceError,
        configured_source_descriptor_path,
    )
    from kestrel_sovereign.constitution.trust_root import SOVEREIGN_TRUST_ROOT_ENV

    def configured(*values) -> list[Path]:
        return [
            Path(value).expanduser()
            for value in values
            if value is not None and str(value).strip()
        ]

    for name, cfg in agents.items():
        try:
            descriptor = configured_source_descriptor_path(
                explicit_path=cfg.constitution_source_descriptor, environ=env
            )
        except ConstitutionSourceError:
            continue
        if descriptor is None:
            continue
        selecting = [
            ("source descriptor", path)
            for path in configured(
                cfg.constitution_source_descriptor,
                env.get(CONSTITUTION_SOURCE_DESCRIPTOR_ENV, "").strip(),
            )
        ] + [
            ("trust root", path)
            for path in configured(env.get(SOVEREIGN_TRUST_ROOT_ENV, "").strip())
        ]
        for role, path in selecting:
            if deployed_content(path) is not None:
                raise ConstitutionAdoptionError(
                    f"the deploy replaces {path}, the {role} that selects "
                    f"{name}'s governing constitution; which bytes would "
                    "govern cannot be judged before the deploy is installed"
                )


def check_installed_constitution_adoption(
    project_dir: Path,
    *,
    agent_names: Iterable[str] | None = None,
) -> ConstitutionAdoptionCheck:
    """:func:`check_constitution_adoption`, judged by the installed code.

    Runs the check in a fresh interpreter launched the way the host is: this
    process's interpreter, the project as its working directory, and the
    environment the launcher hands a spawned host. So the child imports the
    ``kestrel_sovereign`` the restarted host will import, and the resolver and
    Amendment VIII rendering that produce the governing hash are the installed
    ones, not the ones this process imported before an install replaced them.

    The registry is read here first so a project with none of the selected
    agents starts no interpreter. ``agent_names`` is as for
    :func:`check_constitution_adoption`.

    Raises:
        ConstitutionAdoptionError: The registry cannot be loaded, or the
            installed code could not answer: it failed to start or to check,
            timed out, or answered in a form this process cannot read.
    """
    # Absolute, because the child runs with the project as its working
    # directory and would resolve a relative path against it a second time.
    project_dir = Path(project_dir).absolute()
    env = runtime_env(project_dir)
    selected = _select(local_agents(project_dir), agent_names)
    if not selected:
        return ConstitutionAdoptionCheck(code_label=INSTALLED_CODE)
    command = [
        sys.executable, "-m", _FRESH_CHECK_MODULE,
        "--project-dir", str(project_dir),
    ]
    for name in selected:
        command += ["--agent", name]
    try:
        result = subprocess.run(
            command,
            cwd=str(project_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_FRESH_CHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConstitutionAdoptionError(
            f"checking {INSTALLED_CODE} timed out after "
            f"{_FRESH_CHECK_TIMEOUT_SECONDS}s"
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConstitutionAdoptionError(
            f"cannot start {sys.executable} to check {INSTALLED_CODE}: {exc}"
        ) from exc
    return ConstitutionAdoptionCheck(
        verdicts=_installed_verdicts(result, selected),
        code_label=INSTALLED_CODE,
    )


def _installed_verdicts(
    result: subprocess.CompletedProcess, selected: list[str]
) -> tuple[ConstitutionAnchorVerdict, ...]:
    """The verdicts a fresh-interpreter check answered with, validated.

    Anything short of exactly one known verdict per selected agent is a
    failure to check, never a pass: the child is newer code, and an answer
    this process misreads must not let a restart through.
    """
    answer = next(
        (
            line[len(_FRESH_CHECK_RESULT_PREFIX):]
            for line in reversed(result.stdout.splitlines())
            if line.startswith(_FRESH_CHECK_RESULT_PREFIX)
        ),
        None,
    )
    if result.returncode != 0 or answer is None:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise ConstitutionAdoptionError(
            f"{INSTALLED_CODE} could not be checked (exit {result.returncode})"
            + (f": {detail}" if detail else "")
        )
    try:
        payload = json.loads(answer)
        if payload.get("protocol") != _FRESH_CHECK_PROTOCOL:
            raise ValueError(f"unknown protocol {payload.get('protocol')!r}")
        if "error" in payload:
            raise ConstitutionAdoptionError(str(payload["error"]))
        verdicts = tuple(
            ConstitutionAnchorVerdict(**fields) for fields in payload["verdicts"]
        )
        unknown = {v.status for v in verdicts} - _VERDICT_STATUSES
        if unknown:
            raise ValueError(f"unknown verdict status {sorted(unknown)}")
        if sorted(v.agent for v in verdicts) != sorted(selected):
            raise ValueError(
                f"verdicts for {[v.agent for v in verdicts]}, asked about {selected}"
            )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ConstitutionAdoptionError(
            f"{INSTALLED_CODE} answered in a form this check cannot read: {exc}"
        ) from exc
    return verdicts


def deployed_content_at(repo_path: str | Path, revision: str) -> DeployedContent:
    """What checking out ``revision`` of ``repo_path`` would leave at a path.

    The returned function answers for one governing path: the packaged
    constitution, or a descriptor-selected external file that this checkout
    tracks (#3522). It returns None when the checkout leaves the file on disk
    as it is: the path is outside the checkout, or ``revision`` does not
    change it (so an uncommitted edit the checkout keeps is still what runs).
    Otherwise the bytes are what checkout writes, with the checkout's own
    filters applied.

    The path is the one configured, not its resolution: it is resolved here
    as the resolver resolves it, one component at a time, following every
    symbolic link on the way. A revision that changes any entry that walk
    passes through in this checkout changes which file the path names, which
    cannot be read from the revision as bytes: a link it follows, retargeted
    or replaced, or a directory it descends replaced by a link or a file. A
    directory whose contents change still names the same file, judged below.

    The function raises :class:`ConstitutionAdoptionError` when git could not
    compare or read the revision, including a revision that removes the file
    or changes such an entry.
    """
    repo = Path(repo_path).resolve()

    def tracked(path: Path) -> str | None:
        try:
            relpath = path.relative_to(repo).as_posix()
        except ValueError:
            return None
        return None if relpath == "." else relpath

    @functools.cache
    def entry(tree: str, relpath: str) -> tuple[bytes, bytes] | None:
        """The mode and object of ``relpath`` in ``tree``, or None if absent."""
        listed = _git(repo, "ls-tree", "-z", tree, "--", relpath)
        if listed.returncode != 0:
            raise ConstitutionAdoptionError(
                f"cannot read {relpath} at {tree} in {repo}: {_git_detail(listed)}"
            )
        wanted = os.fsencode(relpath)
        for record in listed.stdout.split(b"\0"):
            meta, _, name = record.partition(b"\t")
            if name == wanted:
                mode, _, oid = meta.split(b" ")
                return mode, oid
        return None

    def redirects(relpath: str, *, link: bool) -> bool:
        """Whether the revision changes which file a walk through here names."""
        before, after = entry("HEAD", relpath), entry(revision, relpath)
        if before == after:
            return False
        if link:
            return True
        # A directory stays a directory when only its contents change.
        return not all(e is None or e[0] == _GIT_TREE_MODE for e in (before, after))

    def content(path: Path) -> bytes | None:
        resolved, passed = _resolution_walk(Path(path))
        for location, link in passed:
            step = tracked(location)
            if step is not None and redirects(step, link=link):
                kind = "symbolic link" if link else "directory"
                raise ConstitutionAdoptionError(
                    f"{revision} changes the {kind} {step} in {repo}, which "
                    f"{path} passes through, so which file governs cannot be "
                    "judged before it is checked out"
                )
        relpath = tracked(resolved)
        if relpath is None:
            return None
        incoming = entry(revision, relpath)
        if incoming == entry("HEAD", relpath):
            return None
        if incoming is not None and incoming[0] == _GIT_SYMLINK_MODE:
            raise ConstitutionAdoptionError(
                f"{revision} leaves a symbolic link at {relpath} in {repo}, "
                "so which file governs cannot be judged before it is "
                "checked out"
            )
        shown = _git(repo, "cat-file", "--filters", f"{revision}:{relpath}")
        if shown.returncode != 0:
            raise ConstitutionAdoptionError(
                f"cannot read {relpath} at {revision} in {repo}: "
                f"{_git_detail(shown)}"
            )
        return shown.stdout

    return content


def _resolution_walk(path: Path) -> tuple[Path, list[tuple[Path, bool]]]:
    """Resolve ``path`` as the resolver will, noting every entry it passes.

    Returns the resolved path and, in order, each entry the resolution steps
    through before it: every symbolic link it follows and every directory it
    descends, at the entry's own location, with whether it is a link. A
    resolution of the configured path alone would hide the links it took.

    Raises:
        ConstitutionAdoptionError: The path names a cycle of links.
        OSError: A link could not be read.
    """
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    current = Path(path.anchor)
    pending = list(path.parts[1:])
    passed: list[tuple[Path, bool]] = []
    links = 0
    while pending:
        part = pending.pop(0)
        if part == "..":
            current = current.parent
            continue
        step = current / part
        if step.is_symlink():
            links += 1
            if links > _MAX_SYMLINKS:
                raise ConstitutionAdoptionError(
                    f"{path} follows more than {_MAX_SYMLINKS} symbolic links"
                )
            passed.append((step, True))
            target = Path(os.readlink(step))
            if target.is_absolute():
                current = Path(target.anchor)
                pending[:0] = target.parts[1:]
            else:
                pending[:0] = target.parts
            continue
        if pending:
            passed.append((step, False))
        current = step
    return current, passed


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
            "  2. Offline: `kestrel terminate` first, and stop any server "
            "started without `kestrel start` (a running agent's periodic "
            "integrity audit reads the constitution from disk, and the Safe "
            "Mode it enters persists), install without restarting "
            "(`kestrel update --no-restart`), then for each agent run "
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


def check_record(
    check: ConstitutionAdoptionCheck | None, *, error: str = ""
) -> dict:
    """What one evaluation of the gate decided, as a restart request records it.

    Every agent's verdict, matching or not, so a passed check is evidence
    that it ran and what it compared, not only the absence of a refusal
    (#3522). ``result`` is ``passed`` only when every agent was compared;
    ``passed_with_unverified`` names a pass over agents whose anchor could
    not be read. ``check`` None with ``error`` records a gate that could not
    establish what the code would govern by.
    """
    if check is None:
        return {"result": "unverifiable", "code": "", "error": error, "verdicts": []}
    if not check.safe:
        result = "refused"
    elif check.unverified:
        result = "passed_with_unverified"
    else:
        result = "passed"
    return {
        "result": result,
        "code": check.code_label,
        "error": "",
        "verdicts": [dataclasses.asdict(v) for v in check.verdicts],
    }


def unverified_lines(check: ConstitutionAdoptionCheck) -> list[str]:
    """A warning naming the agents the gate could not compare."""
    if not check.unverified:
        return []
    header = (
        f"Constitution anchor not verified for {len(check.unverified)} "
        f"agent(s); the gate cannot vouch for them (run `kestrel doctor`):"
    )
    return [header, *(f"  {v.agent}: {v.detail}" for v in check.unverified)]


def main(argv: list[str] | None = None) -> int:
    """Answer :func:`check_installed_constitution_adoption` from this interpreter.

    Prints one result line; the exit status is 0 whenever a result was
    printed, including a check that could not establish the governing source.
    """
    parser = argparse.ArgumentParser(
        prog=f"python -m {_FRESH_CHECK_MODULE}",
        description=(
            "Compare each agent's anchored constitution hash with what this "
            "interpreter's kestrel_sovereign governs it by (#3517)."
        ),
    )
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument(
        "--agent", action="append", default=[],
        help="An agent to check; repeat for more. Default: every local agent.",
    )
    args = parser.parse_args(argv)
    payload: dict = {"protocol": _FRESH_CHECK_PROTOCOL}
    try:
        check = check_constitution_adoption(
            args.project_dir, agent_names=args.agent or None
        )
    except ConstitutionAdoptionError as exc:
        payload["error"] = str(exc)
    else:
        payload["verdicts"] = [dataclasses.asdict(v) for v in check.verdicts]
    print(_FRESH_CHECK_RESULT_PREFIX + json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
