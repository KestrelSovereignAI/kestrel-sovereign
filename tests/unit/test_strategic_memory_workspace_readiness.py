"""Tests for reading Talon's workspace readiness through its wait provider (#3548)."""

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement

from kestrel_sovereign.features.strategic_memory import (
    issue_selection,
    run_history,
    workspace_readiness,
)
from kestrel_sovereign.features.strategic_memory.workspace_readiness import (
    NO_TALON_WORKSPACE,
    TALON_WORKSPACE_UNUSABLE,
    WORKSPACE_READINESS_REQUIREMENT,
    TalonWorkspaceProviderOutdated,
    WorkspaceReadiness,
    WorkspaceReadinessUnreadable,
    talon_workspaces,
)
from kestrel_sovereign.waits.engine import WaitRegistry


def _unprovisioned(repo="o/r"):
    """``workspace_readiness()`` for a repository nobody has provisioned, as
    kestrel-feature-talon 0.2.13 answers it."""
    workspace = f"/srv/talon/projects/{repo.replace('/', '__')}"
    return {
        "repo": repo,
        "provisioned": False,
        "reason": "no_talon_workspace",
        "workspace": workspace,
        "next_step": f"talon_setup_workspace(repo='{repo}')",
        "detail": f"No talon workspace exists for {repo} at {workspace}.",
    }


class _Provider:
    """A ``talon`` wait provider answering ``workspace_readiness()``.

    ``answer`` is the report, or an exception to raise. Like the real
    provider's, the read is meant to be strictly read-only; ``poll()`` and
    ``active_handles()`` record a call in ``forbidden`` before failing.
    """

    signal = None

    def __init__(self, answer=None, kind="talon"):
        self.kind = kind
        self._answer = answer
        self.asked = []
        self.forbidden = []

    async def workspace_readiness(self, repo):
        self.asked.append(repo)
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer

    async def active_handles(self):
        self.forbidden.append("active_handles")
        raise AssertionError("workspace readiness must not enumerate jobs")

    async def poll(self, handle):
        self.forbidden.append(f"poll:{handle}")
        raise AssertionError("workspace readiness must not poll")


class _ProviderBefore0213:
    """kestrel-feature-talon 0.2.10-0.2.12: ``finished_runs()``, but no
    ``workspace_readiness()``."""

    kind = "talon"
    signal = None

    async def finished_runs(self):
        return {"complete": True, "runs": []}

    async def active_handles(self):
        return []

    async def poll(self, handle):
        raise AssertionError("workspace readiness must not poll")


def _agent(*providers):
    registry = WaitRegistry()
    for provider in providers:
        registry.register(provider)
    return SimpleNamespace(wait_registry=registry)


def test_the_workspace_vocabulary_matches_talons():
    """Copies of kestrel-feature-talon vocabulary, pinned here because core
    cannot import the package. Change both or neither."""
    # kestrel_feature_talon/wait_provider.py NO_TALON_WORKSPACE,
    # TALON_WORKSPACE_UNUSABLE
    assert workspace_readiness.NO_TALON_WORKSPACE == "no_talon_workspace"
    assert workspace_readiness.TALON_WORKSPACE_UNUSABLE == "talon_workspace_unusable"
    assert workspace_readiness.WORKSPACE_REFUSAL_REASONS == frozenset(
        {"no_talon_workspace", "talon_workspace_unusable"}
    )
    # kestrel_feature_talon/wait_provider.py TalonWaitable.workspace_readiness
    assert workspace_readiness.WORKSPACE_READINESS_METHOD == "workspace_readiness"
    # The same provider run history is read from.
    assert workspace_readiness.TALON_WAIT_KIND is run_history.TALON_WAIT_KIND
    # Selection reports Talon's reasons as Talon gives them.
    assert issue_selection.EXCLUDED_NO_TALON_WORKSPACE == "no_talon_workspace"
    assert issue_selection.EXCLUDED_TALON_WORKSPACE_UNUSABLE == "talon_workspace_unusable"


def test_the_requirement_is_the_first_release_with_workspace_readiness():
    """``TalonWaitable.workspace_readiness`` is absent at kestrel-feature-talon
    tag v0.2.12 and present at v0.2.13 (feature-talon #50). Core cannot import
    the package, so the floor is pinned here."""
    requirement = Requirement(WORKSPACE_READINESS_REQUIREMENT)

    assert requirement.name == "kestrel-feature-talon"
    assert not requirement.specifier.contains("0.2.12")
    assert requirement.specifier.contains("0.2.13")
    assert requirement.specifier.contains("0.3.0")
    assert TalonWorkspaceProviderOutdated.requirement == WORKSPACE_READINESS_REQUIREMENT


@pytest.mark.parametrize(
    "agent",
    [
        pytest.param(SimpleNamespace(), id="no-wait-registry"),
        pytest.param(_agent(), id="no-talon-provider"),
        pytest.param(
            _agent(_Provider(_unprovisioned(), kind="a2a")), id="another-kind-only"
        ),
    ],
)
def test_without_a_talon_provider_there_is_no_workspace_to_confirm(agent):
    """No Talon, no Talon workspace. A sibling kind is never asked: a peer
    must not be able to stand down a dispatch through a provider payload."""
    assert talon_workspaces(agent) is None


def test_a_provider_without_the_read_names_the_release_it_needs():
    with pytest.raises(TalonWorkspaceProviderOutdated) as raised:
        talon_workspaces(_agent(_ProviderBefore0213()))

    assert raised.value.requirement == "kestrel-feature-talon>=0.2.13"
    assert "kestrel-feature-talon>=0.2.13" in str(raised.value)
    assert "workspace_readiness()" in str(raised.value)
    assert "_ProviderBefore0213" in str(raised.value)


def test_resolving_the_read_reads_nothing():
    provider = _Provider({"repo": "o/r", "provisioned": True, "workspace": "/w"})

    assert talon_workspaces(_agent(provider)) is not None
    assert provider.asked == []


@pytest.mark.asyncio
async def test_a_provisioned_repository_is_ready():
    provider = _Provider({"repo": "o/r", "provisioned": True, "workspace": "/w/o__r"})

    readiness = await talon_workspaces(_agent(provider)).readiness("o/r")

    assert readiness == WorkspaceReadiness(
        repo="o/r", provisioned=True, workspace="/w/o__r"
    )
    assert provider.asked == ["o/r"]
    assert provider.forbidden == []


@pytest.mark.asyncio
async def test_the_read_is_given_the_spelling_the_run_will_be_given():
    """Talon derives the workspace path from the repository it is passed."""
    provider = _Provider({"repo": "Org/Repo", "provisioned": True, "workspace": "/w"})

    await talon_workspaces(_agent(provider)).readiness("Org/Repo")

    assert provider.asked == ["Org/Repo"]


@pytest.mark.asyncio
async def test_an_unprovisioned_repository_carries_talons_report():
    readiness = await talon_workspaces(_agent(_Provider(_unprovisioned()))).readiness(
        "o/r"
    )

    assert readiness == WorkspaceReadiness(
        repo="o/r",
        provisioned=False,
        reason=NO_TALON_WORKSPACE,
        workspace="/srv/talon/projects/o__r",
        next_step="talon_setup_workspace(repo='o/r')",
        detail="No talon workspace exists for o/r at /srv/talon/projects/o__r.",
    )


@pytest.mark.asyncio
async def test_an_unusable_workspace_carries_its_refusal():
    """Provisioning does not fix it, so Talon gives no workspace or next step."""
    provider = _Provider({
        "repo": "o/r",
        "provisioned": False,
        "reason": "talon_workspace_unusable",
        "detail": "talon runtime paths are not configured",
    })

    readiness = await talon_workspaces(_agent(provider)).readiness("o/r")

    assert readiness == WorkspaceReadiness(
        repo="o/r",
        provisioned=False,
        reason=TALON_WORKSPACE_UNUSABLE,
        detail="talon runtime paths are not configured",
    )


@pytest.mark.asyncio
async def test_a_synchronous_read_is_accepted():
    class _SyncProvider(_Provider):
        def workspace_readiness(self, repo):
            self.asked.append(repo)
            return self._answer

    provider = _SyncProvider(_unprovisioned())

    readiness = await talon_workspaces(_agent(provider)).readiness("o/r")

    assert readiness.reason == NO_TALON_WORKSPACE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer, says",
    [
        pytest.param(OSError("policy file unreadable"), "policy file unreadable", id="raises"),
        pytest.param(ValueError("needs a repository"), "needs a repository", id="value-error"),
        pytest.param(None, "returned NoneType, not a report", id="none"),
        pytest.param(["o/r"], "returned list, not a report", id="not-a-mapping"),
        pytest.param({"repo": "o/r"}, "did not say whether", id="no-provisioned"),
        pytest.param(
            {"repo": "o/r", "provisioned": "yes"}, "did not say whether", id="truthy-text"
        ),
        pytest.param(
            {"repo": "o/r", "provisioned": 1}, "did not say whether", id="truthy-int"
        ),
        pytest.param(
            {"repo": "o/r", "provisioned": False}, "does not know (None)", id="no-reason"
        ),
        pytest.param(
            {"repo": "o/r", "provisioned": False, "reason": "workspace_not_provisioned"},
            "does not know ('workspace_not_provisioned')",
            id="unknown-reason",
        ),
        pytest.param(
            {"repo": "o/r", "provisioned": False, "reason": ["no_talon_workspace"]},
            "does not know",
            id="unhashable-reason",
        ),
    ],
)
async def test_an_answer_outside_the_contract_is_unreadable(answer, says):
    """Neither a failed read nor an answer core cannot interpret confirms a
    workspace: a repository whose workspace is unknown is not provisioned."""
    provider = _Provider(answer)

    with pytest.raises(WorkspaceReadinessUnreadable, match="workspace_readiness") as raised:
        await talon_workspaces(_agent(provider)).readiness("o/r")

    assert says in str(raised.value)
    assert provider.forbidden == []


@pytest.mark.asyncio
async def test_an_answer_that_cannot_be_read_is_unreadable_not_a_crash():
    """A mapping that raises when read is still the provider's failure, and
    still confirms nothing."""

    class _Unreadable(Mapping):
        def __getitem__(self, key):
            raise RuntimeError("answer torn mid-read")

        def __iter__(self):
            return iter(("repo", "provisioned"))

        def __len__(self):
            return 2

    with pytest.raises(WorkspaceReadinessUnreadable, match="answer torn mid-read"):
        await talon_workspaces(_agent(_Provider(_Unreadable()))).readiness("o/r")

