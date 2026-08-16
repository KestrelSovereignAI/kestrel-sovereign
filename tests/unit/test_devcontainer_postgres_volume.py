"""The devcontainer's PostgreSQL volume must be versioned with its server.

#2958 raised the PostgreSQL floor to 16, because the
``conversation_history.session_id`` backfill guards its cast with
``pg_input_is_valid(metadata, 'jsonb')`` (16+). Changing the image is only half
of that change, and the missing half fails in a way no test of the migration
can see.

A PostgreSQL data directory is not portable across a major. Pointed at a
cluster another major initialized, the server refuses to start at all —
*"database files are incompatible with server"* — and a container rebuild does
NOT remove a named volume, so the incompatible directory is still there on the
next attempt. In this compose file ``app`` waits on ``condition:
service_healthy``, so the result is not a degraded database: it is a
devcontainer that never comes up, for everyone who had one before the bump,
with the cause several layers below the symptom.

Versioning the volume name with the major is what makes the bump safe. The new
major initializes an empty directory of its own and the old volume stays on
disk, so nothing is destroyed and the previous data can still be dumped.

These tests exist because the coupling is invisible at the point of edit. The
next person to move this pin will change one line in a YAML file, see CI go
green — CI creates a fresh service container per run and so can never reproduce
this — and ship a wedge to every existing devcontainer. Failing here is the
only place that reaches them first.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE = REPO_ROOT / ".devcontainer" / "docker-compose.devcontainer.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: Where the server keeps its cluster inside the container. The mount that
#: lands here is the one that has to be versioned; the others (an init script,
#: for instance) are read-only and carry no cluster.
DATA_DIR = "/var/lib/postgresql/data"

#: The floor #2958 established. Asserted as a number rather than as the literal
#: tag so a move to 17 satisfies it and a silent slip back to 15 does not.
MINIMUM_MAJOR = 16


def _compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text())


def _image_major(image: str) -> int:
    """The server major in an image reference such as ``pgvector/pgvector:pg16``."""
    match = re.search(r":pg(\d+)\b", image)
    assert match, f"cannot read a PostgreSQL major out of image {image!r}"
    return int(match.group(1))


def _data_volume(service: dict) -> str:
    """The name of the named volume holding the cluster.

    Fails rather than returning ``None`` for a bind mount or a missing mount:
    both mean the question this module asks has no answer, and a test that
    quietly passes on a shape it cannot evaluate is worse than one that stops.
    """
    sources = [
        entry.split(":")[0]
        for entry in service.get("volumes", [])
        if entry.split(":")[1:2] == [DATA_DIR]
    ]
    assert len(sources) == 1, (
        f"expected exactly one mount at {DATA_DIR}, found {sources!r}"
    )
    source = sources[0]
    assert not source.startswith((".", "/")), (
        f"the cluster is on a bind mount ({source!r}); this module's reasoning "
        "about named-volume lifetime does not apply and needs revisiting"
    )
    return source


def test_the_cluster_volume_is_versioned_with_the_server_major():
    """The bug this file was written for: pg16 pointed at the pg15 volume.

    An unversioned name is not a style preference — it is the difference
    between a rebuild that starts fresh and one that hangs forever on a
    healthcheck the server cannot pass.
    """
    compose = _compose()
    postgres = compose["services"]["postgres"]
    major = _image_major(postgres["image"])
    volume = _data_volume(postgres)

    assert str(major) in volume, (
        f"postgres image is pg{major} but its data volume is named {volume!r}, "
        f"which does not carry the major. A rebuild would mount whatever "
        f"cluster the previous image left in {volume!r}; PostgreSQL refuses to "
        f"start on a cluster from another major, `app` waits on its health "
        f"forever, and no container rebuild clears a named volume. Name it for "
        f"the major (e.g. postgres{major}-data)."
    )


def test_the_versioned_volume_is_declared():
    """A mount naming an undeclared volume is a compose error, not a fallback.

    Renaming the mount and forgetting the ``volumes:`` block is the obvious way
    to half-apply the fix above, and it breaks at ``docker compose up`` — after
    the point where this test could have said why.
    """
    compose = _compose()
    volume = _data_volume(compose["services"]["postgres"])
    assert volume in (compose.get("volumes") or {}), (
        f"{volume!r} is mounted but not declared in the top-level volumes: block"
    )


def test_the_devcontainer_and_ci_agree_on_the_major():
    """One floor, two files. Skew here is a bug reproducible in only one of them.

    #2958's backfill needs 16+; a devcontainer below CI would fail the
    migration locally with CI green, and a devcontainer above it would hide a
    16-only mistake from the tier that gates merges.
    """
    devcontainer_major = _image_major(_compose()["services"]["postgres"]["image"])
    ci_major = _image_major(
        yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]["integration-tests"][
            "services"
        ]["postgres"]["image"]
    )

    assert devcontainer_major == ci_major, (
        f"devcontainer runs pg{devcontainer_major} and CI runs pg{ci_major}"
    )
    assert devcontainer_major >= MINIMUM_MAJOR, (
        f"#2958's session_id backfill needs `pg_input_is_valid(...,'jsonb')` "
        f"(PostgreSQL {MINIMUM_MAJOR}+); pg{devcontainer_major} would make that "
        "call an undefined function and the migration would abort the boot"
    )


DOCS = [
    Path(".devcontainer/README.md"),
    Path("docs/development/DEVCONTAINER_QUICKSTART.md"),
]

#: The container name the compose file fixes globally. Because it is fixed, it
#: is NOT a safe way to discover which project to operate on: in a second clone
#: or git worktree it resolves to whatever checkout happens to be running.
GLOBAL_CONTAINER_NAME = "kestrel-dev-postgres"


def _fenced_command_lines(text: str) -> list:
    """Lines inside ``` fences — i.e. what a reader can copy and run.

    The distinction matters: these documents *should* discuss ``down -v`` at
    length, because explaining why every short spelling of it is wrong is the
    whole point. What they must not do is put one where it can be pasted. A
    naive substring search over the whole file cannot tell those apart and
    fails on the explanation itself.
    """
    lines, inside = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            inside = not inside
            continue
        if inside:
            lines.append(line)
    return lines


def _unprefixable_volumes() -> set:
    """Declared volumes whose real Docker name cannot be read off this file.

    Compose prefixes a named volume with the project name — ``postgres16-data``
    is created as ``devcontainer_postgres16-data`` — unless the declaration
    pins an explicit ``name:``. The prefix is not a constant either: it
    defaults to the compose file's parent directory, and VS Code derives its
    own from the workspace path, so it differs per machine.

    A volume in this set therefore has no name a document may hand to
    ``docker volume rm``. Pin a ``name:`` and it leaves the set, because then
    the file does know what the volume is called.
    """
    declared = _compose().get("volumes") or {}
    return {
        volume
        for volume, spec in declared.items()
        if not (spec or {}).get("name")
    }


@pytest.mark.parametrize("doc", DOCS)
def test_no_document_ships_a_destructive_project_wide_command(doc: Path):
    """These docs must not hand over a copy-pasteable ``down -v``.

    Three spellings were tried and every one destroyed the wrong thing:

    * bare ``docker compose -f .devcontainer/… down -v`` — Compose derives the
      project from the compose file's own directory, so it targets
      ``devcontainer_*``: nothing under VS Code, or another project entirely.
    * ``-p`` with the project read from ``docker inspect kestrel-dev-postgres``
      — that container name is fixed globally, so in a second clone or git
      worktree it resolves to whatever checkout is running. The lookup succeeds
      while *this* checkout is stopped, and the ``down -v`` then deletes the
      other checkout's database.
    * naming the volumes directly — the prefix is not knowable from here.

    The failure is silent in the worst direction each time: either nothing is
    removed and the reader believes it was, or data that belongs to something
    else is destroyed. A reader who has to look up their own project cannot
    make that mistake by copy-paste, so the docs describe and do not prescribe.
    """
    offenders = [
        line.strip()
        for line in _fenced_command_lines((REPO_ROOT / doc).read_text())
        if "down -v" in line
    ]
    assert not offenders, (
        f"{doc} ships a destructive project-wide command: {offenders}. "
        "Every short spelling of this targets the wrong Compose project in "
        "some real setup — see this test's docstring. Describe how to find "
        "the volumes instead of handing over a command that deletes them. "
        "Discussing `down -v` in prose is fine and expected; putting one in a "
        "code fence is not."
    )


@pytest.mark.parametrize("doc", DOCS)
def test_no_document_derives_a_project_from_the_global_container_name(doc: Path):
    """``docker inspect kestrel-dev-postgres`` must not feed a destructive command.

    The compose file fixes ``container_name`` globally, so this lookup is not
    scoped to a checkout. With several clones or git worktrees on one machine —
    the normal case here — it returns whichever one is running, and using that
    project in ``down -v`` deletes that checkout's persistent volumes.
    """
    offenders = [
        line.strip()
        for line in _fenced_command_lines((REPO_ROOT / doc).read_text())
        if f"docker inspect {GLOBAL_CONTAINER_NAME}" in line
    ]
    assert not offenders, (
        f"{doc} derives a Compose project from the globally-fixed container "
        f"name: {offenders}. That resolves to whatever checkout is running, "
        "not necessarily this one."
    )


@pytest.mark.parametrize("doc", DOCS)
def test_no_document_removes_a_live_volume_by_its_compose_file_name(doc: Path):
    """The bug the test above replaced: a command that resolves to nothing.

    ``docker volume rm postgres16-data`` fails with *no such volume* while the
    live ``devcontainer_postgres16-data`` keeps running, and the earlier
    version of this test certified that command as correct because it compared
    against the logical name in the YAML rather than the name Docker creates.

    A prefixed reference is fine and is deliberately not matched here: the
    upgrade note points at ``<project>_postgres-data``, the superseded cluster,
    and tells the reader how to find their prefix. That instruction is correct
    precisely because it does not pretend to know the whole name.
    """
    text = (REPO_ROOT / doc).read_text()
    offenders = [
        (line.strip(), volume)
        for line in text.splitlines()
        if "docker volume rm" in line
        for volume in _unprefixable_volumes()
        # Only a BARE occurrence is wrong; ``<project>_postgres16-data`` and
        # ``$COMPOSE_PROJECT_NAME_postgres16-data`` both carry their prefix.
        if re.search(rf"(?:^|\s){re.escape(volume)}(?:\s|$)", line)
    ]
    assert not offenders, (
        f"{doc} tells the reader to run a command that removes nothing: "
        + "; ".join(
            f"{command!r} names {volume!r}, which Docker creates as "
            f"<project>_{volume}"
            for command, volume in offenders
        )
        + f". Use `{RESET_COMMAND}` instead."
    )
