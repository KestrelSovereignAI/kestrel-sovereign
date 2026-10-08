"""The constitution adoption gate on deploy restarts (#3517).

``kestrel update`` pulled a revision that changed the packaged constitution,
installed it and restarted; every agent, still anchored to the old hash, came
up in constitution Safe Mode. These tests pin the gate every deploy path now
asks first:

* the shared check compares each agent's anchored hash with what the code
  about to run produces — through the canonical resolver, read-only, without a
  running host;
* ``kestrel restart`` refuses before terminating anything, naming the agents
  and both hashes, unless ``--allow-constitution-safe-mode``;
* ``kestrel update`` judges the revision its pull will land on, before it
  changes anything.

The restart coordinator's use of the same check is pinned in
``test_restart_coordinator.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from kestrel_sovereign import cli, cli_lifecycle
from kestrel_sovereign.constitution_adoption import (
    ADOPTION_RUNBOOK,
    CONSTITUTION_ADOPTION_REQUIRED,
    ConstitutionAdoptionError,
    check_constitution_adoption,
    packaged_constitution_at,
    refusal_reason,
)
from tests.utils.constitution_anchor import (
    PACKAGED_CONSTITUTION_RELPATH,
    commit_constitution,
    git,
    origin_and_clone,
    seed_anchored_agents,
    sha256,
)

OLD = b"# Kestrel Constitution\nthe text every agent was anchored to\n"
NEW = b"# Kestrel Constitution\nthe text a deploy is about to install\n"


@pytest.fixture(autouse=True)
def _hermetic_governance(monkeypatch):
    """No ambient descriptor, trust root, or PostgreSQL backend.

    A developer host may export a trust root or descriptor; either changes
    which source governs, and so what these tests compare.
    """
    for name in (
        "KESTREL_CONSTITUTION_SOURCE_DESCRIPTOR_PATH",
        "KESTREL_SOVEREIGN_TRUST_ROOT_PATH",
        "KESTREL_DB_BACKEND",
        "KESTREL_DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def project(tmp_path):
    path = tmp_path / "project"
    path.mkdir()
    return path


def _install_package(monkeypatch, path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(path))
    return path


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """The installed packaged constitution, currently ``OLD``."""
    return _install_package(monkeypatch, tmp_path / "pkg" / "CONSTITUTION.md", OLD)


# ---------------------------------------------------------------------------
# The shared check
# ---------------------------------------------------------------------------


def test_an_anchored_fleet_is_safe(project, installed):
    seed_anchored_agents(project, {"Emma": sha256(OLD), "Kite": sha256(OLD)})

    check = check_constitution_adoption(project)

    assert check.safe
    assert [(v.agent, v.status) for v in check.verdicts] == [
        ("Emma", "match"),
        ("Kite", "match"),
    ]


def test_a_mismatch_names_the_agent_and_both_hashes(project, installed):
    installed.write_bytes(NEW)
    seed_anchored_agents(project, {"Emma": sha256(OLD), "Kite": sha256(NEW)})

    check = check_constitution_adoption(project)

    assert not check.safe
    [blocking] = check.blocking
    assert blocking.agent == "Emma"
    assert blocking.status == "mismatch"
    assert blocking.anchored_hash == sha256(OLD)
    assert blocking.governing_hash == sha256(NEW)
    reason = refusal_reason(check)
    assert "Emma" in reason and "Kite" not in reason
    assert sha256(OLD) in reason and sha256(NEW) in reason
    assert ADOPTION_RUNBOOK in reason


def test_the_incoming_package_is_judged_not_the_file_on_disk(project, installed):
    """A deploy is judged by what it will install, in both directions."""
    seed_anchored_agents(project, {"Emma": sha256(OLD)})

    incoming_new = check_constitution_adoption(
        project, packaged_constitution=lambda: NEW
    )
    assert [v.status for v in incoming_new.verdicts] == ["mismatch"]
    assert incoming_new.blocking[0].governing_hash == sha256(NEW)

    seed_anchored_agents(project, {"Emma": sha256(NEW)})
    assert check_constitution_adoption(
        project, packaged_constitution=lambda: NEW
    ).safe
    # ...and with no incoming change, the file on disk is what runs.
    assert not check_constitution_adoption(
        project, packaged_constitution=lambda: None
    ).safe


def test_incoming_bytes_are_produced_once_and_only_when_compared(project, installed):
    calls = []

    def incoming():
        calls.append(1)
        return NEW

    assert check_constitution_adoption(project, packaged_constitution=incoming).safe
    assert calls == [], "no agent was compared, so nothing needed fetching"

    seed_anchored_agents(project, {"Emma": sha256(NEW), "Kite": sha256(NEW)})
    assert check_constitution_adoption(project, packaged_constitution=incoming).safe
    assert calls == [1]


def test_incoming_bytes_the_audit_would_reject_are_blocking(project, installed):
    """A blank packaged constitution fails the audit for every agent."""
    seed_anchored_agents(project, {"Emma": sha256(OLD)})

    check = check_constitution_adoption(project, packaged_constitution=lambda: b"  \n")

    [blocking] = check.blocking
    assert blocking.status == "audit_failure"
    assert "empty" in blocking.detail


def test_a_failure_to_produce_incoming_bytes_is_not_an_agent_finding(
    project, installed
):
    seed_anchored_agents(project, {"Emma": sha256(OLD)})

    def unavailable():
        raise ConstitutionAdoptionError("git fetch failed")

    with pytest.raises(ConstitutionAdoptionError, match="git fetch failed"):
        check_constitution_adoption(project, packaged_constitution=unavailable)


def test_an_unreadable_anchor_is_unverified_not_blocking(project, installed):
    seed_anchored_agents(project, {"Emma": sha256(OLD)})
    (project / "agent_data" / "emma" / "kestrel_prime.db").unlink()

    check = check_constitution_adoption(project)

    assert check.safe
    [unverified] = check.unverified
    assert unverified.agent == "Emma"
    assert "kestrel_prime.db" in unverified.detail


def test_agent_names_limit_the_check(project, installed):
    installed.write_bytes(NEW)
    seed_anchored_agents(project, {"Emma": sha256(OLD), "Kite": sha256(NEW)})

    assert check_constitution_adoption(project, agent_names=["Kite"]).safe
    assert not check_constitution_adoption(project, agent_names=["Emma"]).safe
    assert check_constitution_adoption(project, agent_names=["nobody"]).verdicts == ()


def test_a_descriptor_selected_external_source_ignores_the_incoming_package(
    project, installed, tmp_path, monkeypatch
):
    """A deploy replaces the package, not an operator's external source."""
    from kestrel_sovereign.constitution.amendment_artifact import (
        did_document_from_legacy_public_key,
    )
    from kestrel_sovereign.constitution.source_descriptor import (
        build_legacy_signed_source_descriptor,
    )
    from kestrel_sovereign.security.crypto_suite import Secp256k1Suite

    external = tmp_path / "CUSTOM.md"
    external.write_bytes(b"# Custom Constitution\n")
    keypair = Secp256k1Suite().generate_keypair()
    signer = "did:pkh:eip155:1:0x0000000000000000000000000000000000003517"
    root = tmp_path / "root.did.json"
    root.write_text(
        json.dumps(did_document_from_legacy_public_key(signer, keypair.public_key))
    )
    descriptor = tmp_path / "source.signed.json"
    descriptor.write_text(
        json.dumps(
            build_legacy_signed_source_descriptor(
                signer_did=signer,
                source_kind="external",
                source_path=str(external),
                content_sha256=hashlib.sha256(external.read_bytes()).hexdigest(),
                private_key=keypair.private_key,
            )
        )
    )
    monkeypatch.setenv("KESTREL_SOVEREIGN_TRUST_ROOT_PATH", str(root))
    monkeypatch.setenv("KESTREL_CONSTITUTION_SOURCE_DESCRIPTOR_PATH", str(descriptor))
    seed_anchored_agents(project, {"Emma": sha256(external.read_bytes())})

    def incoming():
        raise AssertionError("the package does not govern this agent")

    check = check_constitution_adoption(project, packaged_constitution=incoming)

    assert [v.status for v in check.verdicts] == ["match"]


# ---------------------------------------------------------------------------
# What a revision of a checkout would install
# ---------------------------------------------------------------------------


def test_packaged_constitution_at_reads_what_a_revision_installs(tmp_path, monkeypatch):
    origin, clone = origin_and_clone(tmp_path, OLD)
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH",
        str(clone / PACKAGED_CONSTITUTION_RELPATH),
    )

    assert packaged_constitution_at(clone, "HEAD") is None

    commit_constitution(origin, NEW, "amend")
    git(clone, "fetch", "-q")
    assert packaged_constitution_at(clone, "origin/main") == NEW


def test_a_revision_that_leaves_the_constitution_alone_reads_as_no_change(
    tmp_path, monkeypatch
):
    origin, clone = origin_and_clone(tmp_path, OLD)
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH",
        str(clone / PACKAGED_CONSTITUTION_RELPATH),
    )
    (origin / "README.md").write_text("unrelated\n")
    git(origin, "add", "README.md")
    git(origin, "commit", "-q", "-m", "unrelated")
    git(clone, "fetch", "-q")

    assert packaged_constitution_at(clone, "origin/main") is None


def test_a_checkout_that_does_not_supply_the_package_changes_nothing(
    tmp_path, installed
):
    _, clone = origin_and_clone(tmp_path, NEW)

    assert packaged_constitution_at(clone, "HEAD~0") is None


def test_a_revision_without_a_packaged_constitution_is_refused(tmp_path, monkeypatch):
    origin, clone = origin_and_clone(tmp_path, OLD)
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH",
        str(clone / PACKAGED_CONSTITUTION_RELPATH),
    )
    git(origin, "rm", "-q", str(PACKAGED_CONSTITUTION_RELPATH))
    git(origin, "commit", "-q", "-m", "move it")
    git(clone, "fetch", "-q")

    with pytest.raises(ConstitutionAdoptionError, match="cannot read"):
        packaged_constitution_at(clone, "origin/main")


# ---------------------------------------------------------------------------
# kestrel restart
# ---------------------------------------------------------------------------


def _restart_args(**overrides):
    base = {"name": None, "force": False, "startup_timeout": 30}
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def lifecycle_calls(project, monkeypatch):
    """Point the CLI at ``project`` and record terminate/start."""
    calls = []
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    monkeypatch.setattr(
        cli, "cmd_terminate", lambda args: calls.append("terminate") or 0
    )
    monkeypatch.setattr(cli, "cmd_start", lambda args: calls.append("start") or 0)
    return calls


def test_restart_proceeds_when_every_agent_is_anchored(
    project, installed, lifecycle_calls
):
    seed_anchored_agents(project, {"Emma": sha256(OLD)})

    assert cli.cmd_restart(_restart_args()) == 0
    assert lifecycle_calls == ["terminate", "start"]


def test_restart_refuses_a_mismatch_before_terminating_anything(
    project, installed, lifecycle_calls, capsys
):
    installed.write_bytes(NEW)
    seed_anchored_agents(project, {"Emma": sha256(OLD), "Kite": sha256(OLD)})

    rc = cli.cmd_restart(_restart_args())

    assert rc == CONSTITUTION_ADOPTION_REQUIRED
    assert lifecycle_calls == []
    err = capsys.readouterr().err
    for text in ("Emma", "Kite", sha256(OLD), sha256(NEW), ADOPTION_RUNBOOK):
        assert text in err
    assert "--allow-constitution-safe-mode" in err
    assert "kestrel constitution reanchor" in err


def test_the_override_flag_restarts_deliberately(
    project, installed, lifecycle_calls, capsys
):
    installed.write_bytes(NEW)
    seed_anchored_agents(project, {"Emma": sha256(OLD)})

    rc = cli.cmd_restart(_restart_args(allow_constitution_safe_mode=True))

    assert rc == 0
    assert lifecycle_calls == ["terminate", "start"]
    assert "Safe Mode deliberately" in capsys.readouterr().err


def test_restarting_one_agent_judges_only_that_agent(
    project, installed, lifecycle_calls
):
    installed.write_bytes(NEW)
    seed_anchored_agents(project, {"Emma": sha256(OLD), "Kite": sha256(NEW)})

    assert cli.cmd_restart(_restart_args(name="Kite")) == 0
    assert lifecycle_calls == ["terminate", "start"]
    lifecycle_calls.clear()
    assert cli.cmd_restart(_restart_args(name="Emma")) == CONSTITUTION_ADOPTION_REQUIRED
    assert lifecycle_calls == []


def test_the_parser_offers_the_override_on_restart_and_update():
    parser = cli.build_parser()
    for command in (["restart"], ["update"]):
        assert parser.parse_args(command).allow_constitution_safe_mode is False
        assert parser.parse_args(
            command + ["--allow-constitution-safe-mode"]
        ).allow_constitution_safe_mode is True


# ---------------------------------------------------------------------------
# kestrel update
# ---------------------------------------------------------------------------


def _update_args(**overrides):
    base = {
        "name": None, "pull": True, "install": True, "features": False,
        "restart": True, "allow_dirty": False, "no_deps": False,
        "continue_on_error": False, "dry_run": False, "manifest": None,
        "force": False, "uv_sync": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def upstream(tmp_path, project, monkeypatch):
    """A checkout running ``OLD`` whose upstream now carries ``NEW``.

    The step helpers that would change the host are recorded instead of run;
    the gate's own git reads are real.
    """
    origin, clone = origin_and_clone(tmp_path, OLD)
    commit_constitution(origin, NEW, "amend the constitution")
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH",
        str(clone / PACKAGED_CONSTITUTION_RELPATH),
    )
    monkeypatch.setattr(cli, "_get_project_dir", lambda: project)
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: clone)
    steps = []
    monkeypatch.setattr(
        cli, "_run_git_pull", lambda _: steps.append("pull") or (0, "")
    )
    monkeypatch.setattr(
        cli,
        "_run_uv_pip_install_editable",
        lambda *a, **kw: steps.append("install") or (0, ""),
    )
    monkeypatch.setattr(
        cli, "cmd_restart", lambda args: steps.append(("restart", args)) or 0
    )
    return clone, steps


def _head(clone: Path) -> str:
    return git(clone, "rev-parse", "HEAD").strip()


def test_update_refuses_an_incoming_constitution_before_changing_anything(
    project, upstream, capsys
):
    clone, steps = upstream
    seed_anchored_agents(project, {"Emma": sha256(OLD)})
    head = _head(clone)

    rc = cli.cmd_update(_update_args())

    assert rc == CONSTITUTION_ADOPTION_REQUIRED
    assert steps == []
    assert _head(clone) == head
    assert (clone / PACKAGED_CONSTITUTION_RELPATH).read_bytes() == OLD
    err = capsys.readouterr().err
    for text in ("Emma", sha256(OLD), sha256(NEW), ADOPTION_RUNBOOK):
        assert text in err


def test_update_proceeds_when_agents_are_anchored_to_the_incoming_revision(
    project, upstream
):
    """The installed file differs from the anchor; the incoming one does not."""
    _, steps = upstream
    seed_anchored_agents(project, {"Emma": sha256(NEW)})

    assert cli.cmd_update(_update_args()) == 0

    assert [s if isinstance(s, str) else s[0] for s in steps] == [
        "pull", "install", "restart",
    ]
    assert steps[-1][1].allow_constitution_safe_mode is False


def test_update_override_proceeds_and_reaches_the_restart(project, upstream):
    _, steps = upstream
    seed_anchored_agents(project, {"Emma": sha256(OLD)})

    rc = cli.cmd_update(_update_args(allow_constitution_safe_mode=True))

    assert rc == 0
    assert steps[:2] == ["pull", "install"]
    assert steps[-1][1].allow_constitution_safe_mode is True


def test_update_without_a_restart_only_warns(project, upstream, capsys):
    """Installing without restarting is the offline ceremony's first step."""
    _, steps = upstream
    seed_anchored_agents(project, {"Emma": sha256(OLD)})

    rc = cli.cmd_update(_update_args(restart=False))

    assert rc == 0
    assert steps == ["pull", "install"]
    assert "next restart will be refused" in capsys.readouterr().err


def test_update_without_a_pull_judges_the_installed_code(project, upstream):
    _, steps = upstream
    seed_anchored_agents(project, {"Emma": sha256(OLD)})

    assert cli.cmd_update(_update_args(pull=False)) == 0
    assert [s if isinstance(s, str) else s[0] for s in steps] == [
        "install", "restart",
    ]


def test_dry_run_judges_the_last_fetch_without_fetching(
    project, upstream, monkeypatch
):
    clone, steps = upstream
    git(clone, "fetch", "-q")
    seed_anchored_agents(project, {"Emma": sha256(OLD)})
    monkeypatch.setattr(
        cli,
        "_run_git_fetch",
        lambda _: pytest.fail("a dry run must not fetch"),
    )

    assert cli.cmd_update(_update_args(dry_run=True)) == CONSTITUTION_ADOPTION_REQUIRED
    assert steps == []


def test_update_refuses_when_the_incoming_revision_cannot_be_read(
    project, upstream, monkeypatch, capsys
):
    _, steps = upstream
    seed_anchored_agents(project, {"Emma": sha256(OLD)})
    monkeypatch.setattr(cli, "_run_git_fetch", lambda _: (1, "network is down"))

    assert cli.cmd_update(_update_args()) == CONSTITUTION_ADOPTION_REQUIRED
    assert steps == []
    assert "network is down" in capsys.readouterr().err


def test_pull_target_follows_a_detached_checkout_to_its_reattach_branch(
    tmp_path,
):
    origin, clone = origin_and_clone(tmp_path, OLD)
    incoming = commit_constitution(origin, NEW, "amend")
    git(clone, "fetch", "-q")
    git(clone, "checkout", "-q", "--detach", "HEAD")

    assert cli_lifecycle._pull_target_revision(clone) == incoming


def test_pull_target_is_none_when_the_pull_brings_nothing(tmp_path):
    _, clone = origin_and_clone(tmp_path, OLD)
    assert cli_lifecycle._pull_target_revision(clone) is None

    # Local work ahead of the upstream: `git pull --ff-only` keeps HEAD.
    commit_constitution(clone, NEW, "local")
    git(clone, "fetch", "-q")
    assert cli_lifecycle._pull_target_revision(clone) is None
