"""Duplicate webhook names are reported where they are created (#3239).

#3216 made the shared dispatch router refuse an unprefixed ``/webhooks/{name}``
owned by more than one enabled agent — enforcement where the name is
consumed. Nothing reported the collision where the name was created: agent
B's ``webhooks_register`` succeeded for a name A already served, A's sender
started getting 404s identical to a typo, and the only signal was a host-log
line on the first refused delivery.

The host is the only party that sees every agent, so the report is host-
level: ``_report_webhook_name_collisions`` scans the enabled receivers of
every current agent, warns once per collided name at boot and at dynamic
onboarding (both are router mount passes), and is installed on each agent as
a scoped hook that ``webhooks_register`` consults (the third moment) and
that ``webhooks_list`` / ``webhooks_history`` render so BOTH owners can read
it — without any feature reaching across the tenancy boundary.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.features.webhooks.feature import WebhookFeature
from kestrel_sovereign.server import (
    _mount_feature_routers,
    _report_webhook_name_collisions,
)
from tests.unit.test_feature_route_lifecycle_gate import (
    API_KEY,
    _WebhookFeatureStub,
    _boot_multi_agent,
    _make_agent,
)
from tests.unit.test_webhooks_feature import _make_agent as _make_feature_agent


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def messages(self):
        return [r.getMessage() for r in self.records]

    def about(self, name: str):
        return [m for m in self.messages() if f"Webhook name '{name}'" in m]


@pytest.fixture
def host_log():
    """A handler on the server module's own logger: the app's boot
    reconfigures root logging, so ``caplog`` can miss these warnings."""
    target = logging.getLogger("kestrel_sovereign.server")
    handler = _Collect()
    target.addHandler(handler)
    try:
        yield handler
    finally:
        target.removeHandler(handler)


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------


def test_boot_reports_a_name_two_agents_own_and_installs_the_scoped_answer(host_log):
    os.environ["KESTREL_API_KEY"] = API_KEY
    agents = {
        "a": _make_agent({"WebhookFeature": _WebhookFeatureStub("deposit")}),
        "b": _make_agent({"WebhookFeature": _WebhookFeatureStub("deposit")}),
        "c": _make_agent({"WebhookFeature": _WebhookFeatureStub("alpha")}),
    }
    app, restore = _boot_multi_agent(agents)
    try:
        (message,) = host_log.about("deposit")
        assert "owned by 2 enabled receivers" in message
        assert "agents: a, b" in message
        assert "/api/agents/<agent>/webhooks/deposit" in message
        assert host_log.about("alpha") == []

        # Each owner holds the host's answer, scoped to its own names.
        assert agents["a"]._host_webhook_collisions(announce=False) == {
            "deposit": ["a", "b"]
        }
        assert agents["b"]._host_webhook_collisions(announce=False) == {
            "deposit": ["a", "b"]
        }
        assert agents["c"]._host_webhook_collisions(announce=False) == {}
        # The unscoped view is the host's.
        assert _report_webhook_name_collisions(app, announce=False) == {
            "deposit": ["a", "b"]
        }
    finally:
        restore()


def test_boot_with_one_owner_per_name_reports_nothing(host_log):
    os.environ["KESTREL_API_KEY"] = API_KEY
    agents = {
        "a": _make_agent({"WebhookFeature": _WebhookFeatureStub("deposit")}),
        "b": _make_agent({"WebhookFeature": _WebhookFeatureStub("alpha")}),
    }
    app, restore = _boot_multi_agent(agents)
    try:
        assert [m for m in host_log.messages() if "Webhook name" in m] == []
        assert agents["a"]._host_webhook_collisions(announce=False) == {}
        assert _report_webhook_name_collisions(app, announce=False) == {}
    finally:
        restore()


def test_a_disabled_owner_does_not_collide(host_log):
    """The scan is over ENABLED receivers, like the dispatch it explains."""
    os.environ["KESTREL_API_KEY"] = API_KEY
    disabled = _WebhookFeatureStub("deposit", enabled=False)
    agents = {
        "a": _make_agent({"WebhookFeature": _WebhookFeatureStub("deposit")}),
        "b": _make_agent({"WebhookFeature": disabled}),
    }
    app, restore = _boot_multi_agent(agents)
    try:
        assert host_log.about("deposit") == []
        assert agents["a"]._host_webhook_collisions(announce=False) == {}
        disabled.enabled = True
        assert agents["a"]._host_webhook_collisions(announce=False) == {
            "deposit": ["a", "b"]
        }
    finally:
        restore()


def test_two_receivers_of_one_agent_list_that_agent_twice(host_log):
    """A within-agent collision is refused on the prefixed form too (#3216);
    the report keeps one entry per owning receiver so it is visible."""
    os.environ["KESTREL_API_KEY"] = API_KEY
    agents = {
        "a": _make_agent(
            {
                "WebhookFeature": _WebhookFeatureStub("deposit"),
                "OtherWebhookFeature": _WebhookFeatureStub("deposit"),
            }
        ),
    }
    app, restore = _boot_multi_agent(agents)
    try:
        (message,) = host_log.about("deposit")
        assert "2 enabled receivers of one agent (a)" in message
        # The remedy mirrors the refusal (#3216): the agent-prefixed form is
        # refused too, so the log must not send the operator there.
        assert "agent-prefixed form are refused" in message
        assert "unregister all but one of them" in message
        assert "point each sender" not in message
        assert agents["a"]._host_webhook_collisions(announce=False) == {
            "deposit": ["a", "a"]
        }
    finally:
        restore()


# ---------------------------------------------------------------------------
# Dynamic onboarding
# ---------------------------------------------------------------------------


def test_onboarding_reports_the_collision_the_new_agent_brings(host_log):
    """Production order (review r1 P1): the onboarding hook mounts the
    newcomer's routers BEFORE the manager publishes it, so at its own mount
    pass the newcomer is absent from ``list_agents()``. The hook names it as
    a candidate under its routing name; the pass must announce and install
    from that, not from the (still unpublished) fleet."""
    os.environ["KESTREL_API_KEY"] = API_KEY
    agents = {"a": _make_agent({"WebhookFeature": _WebhookFeatureStub("deposit")})}
    app, restore = _boot_multi_agent(agents)
    try:
        assert host_log.about("deposit") == []

        newcomer = _make_agent({"WebhookFeature": _WebhookFeatureStub("deposit")})
        assert "b" not in agents  # not published yet: exactly the hook's moment
        _mount_feature_routers(app, agents=(newcomer,), candidates={"b": newcomer})

        (message,) = host_log.about("deposit")
        assert "agents: a, b" in message
        # The newcomer holds the answer already, before publication.
        assert newcomer._host_webhook_collisions(announce=False) == {
            "deposit": ["a", "b"]
        }

        # Publication commits; the incumbent's answer is live and sees it.
        agents["b"] = newcomer
        assert agents["a"]._host_webhook_collisions(announce=False) == {
            "deposit": ["a", "b"]
        }
        assert newcomer._host_webhook_collisions(announce=False) == {
            "deposit": ["a", "b"]
        }
    finally:
        restore()


def test_onboarding_a_newcomer_with_no_collision_reports_nothing(host_log):
    os.environ["KESTREL_API_KEY"] = API_KEY
    agents = {"a": _make_agent({"WebhookFeature": _WebhookFeatureStub("deposit")})}
    app, restore = _boot_multi_agent(agents)
    try:
        newcomer = _make_agent({"WebhookFeature": _WebhookFeatureStub("alpha")})
        _mount_feature_routers(app, agents=(newcomer,), candidates={"b": newcomer})
        assert [m for m in host_log.messages() if "Webhook name" in m] == []
        assert newcomer._host_webhook_collisions(announce=False) == {}
    finally:
        restore()


def test_the_scoped_hook_cannot_be_widened():
    """The feature asks only about its own names; the hook takes no scope."""
    os.environ["KESTREL_API_KEY"] = API_KEY
    agents = {
        "a": _make_agent({"WebhookFeature": _WebhookFeatureStub("deposit")}),
        "b": _make_agent({"WebhookFeature": _WebhookFeatureStub("alpha")}),
        "c": _make_agent({"WebhookFeature": _WebhookFeatureStub("alpha")}),
    }
    app, restore = _boot_multi_agent(agents)
    try:
        assert agents["a"]._host_webhook_collisions(announce=False) == {}
        with pytest.raises(TypeError):
            agents["a"]._host_webhook_collisions(scope=None)
        with pytest.raises(TypeError):
            agents["a"]._host_webhook_collisions(announce=False, candidates={})
    finally:
        restore()


def test_single_agent_boot_is_labelled_by_its_display_name(host_log):
    """The ``app.state.agent`` branch: no manager, no routing name."""
    from server import app

    os.environ["KESTREL_API_KEY"] = API_KEY
    solo = _make_agent(
        {
            "WebhookFeature": _WebhookFeatureStub("deposit"),
            "OtherWebhookFeature": _WebhookFeatureStub("deposit"),
        }
    )
    solo.agent_name = "Solo"
    original = (getattr(app.state, "agent", None), getattr(app.state, "agent_manager", None))
    app.state.agent = solo
    app.state.agent_manager = None
    _mount_feature_routers(app)
    try:
        (message,) = host_log.about("deposit")
        assert "of one agent (Solo)" in message
        assert solo._host_webhook_collisions(announce=False) == {"deposit": ["Solo", "Solo"]}
    finally:
        from kestrel_sovereign.server import _unmount_feature_routers

        _unmount_feature_routers(app)
        app.state.agent = original[0]
        app.state.agent_manager = original[1]


def test_scoped_hook_announces_only_when_asked(host_log):
    os.environ["KESTREL_API_KEY"] = API_KEY
    agents = {
        "a": _make_agent({"WebhookFeature": _WebhookFeatureStub("deposit")}),
        "b": _make_agent({"WebhookFeature": _WebhookFeatureStub("deposit")}),
    }
    app, restore = _boot_multi_agent(agents)
    try:
        assert len(host_log.about("deposit")) == 1  # the boot report
        agents["a"]._host_webhook_collisions(announce=False)
        assert len(host_log.about("deposit")) == 1
        agents["a"]._host_webhook_collisions(announce=True)
        assert len(host_log.about("deposit")) == 2
    finally:
        restore()


# ---------------------------------------------------------------------------
# Registration, through the real feature and the tools both owners read
# ---------------------------------------------------------------------------


async def _hosted_pair(tmp_path, sqlite_database_factory):
    """Two real WebhookFeatures on two hosted agents behind one manager,
    routers mounted on the real app so the host installs its hook."""
    db_a = await sqlite_database_factory(tmp_path / "a.db")
    agent_a = _make_feature_agent(db=db_a, agent_id="did:test:agent-emma")
    feat_a = WebhookFeature(agent_a)
    await feat_a.initialize()
    agent_a.features = {"WebhookFeature": feat_a}

    db_b = await sqlite_database_factory(tmp_path / "b.db")
    agent_b = _make_feature_agent(db=db_b, agent_id="did:test:agent-nellie")
    feat_b = WebhookFeature(agent_b)
    await feat_b.initialize()
    agent_b.features = {"WebhookFeature": feat_b}

    fleet = {"emma": agent_a, "nellie": agent_b}
    by_did = {agent.did: name for name, agent in fleet.items()}
    manager = MagicMock()
    manager.list_agents = MagicMock(side_effect=lambda: dict(fleet))
    manager.get_agent = MagicMock(side_effect=lambda name: fleet.get(name))
    manager.get_agent_name = MagicMock(side_effect=lambda did: by_did.get(did))
    agent_a._agent_manager = manager
    agent_b._agent_manager = manager
    return feat_a, feat_b, manager


@pytest.mark.asyncio
async def test_registration_is_reported_and_both_owners_can_read_it(
    tmp_path, sqlite_database_factory, host_log
):
    from server import app

    feat_a, feat_b, manager = await _hosted_pair(tmp_path, sqlite_database_factory)
    original = (getattr(app.state, "agent", None), getattr(app.state, "agent_manager", None))
    app.state.agent = None
    app.state.agent_manager = manager
    _mount_feature_routers(app)
    try:
        first = await feat_a.webhooks_register(
            name="deposit", auth_type="none", allow_unauthenticated=True
        )
        assert first.status.name == "OK", first
        assert "owners" not in first.data
        assert host_log.about("deposit") == []

        second = await feat_b.webhooks_register(
            name="deposit", auth_type="none", allow_unauthenticated=True
        )
        # The registering agent learns in the reply, and the reply is not a
        # clean OK: its unprefixed address is dead from this moment.
        assert second.status.name == "PARTIAL", second
        assert second.data["owners"] == ["emma", "nellie"]
        assert "NAME COLLISION" in second.error
        assert "/api/agents/nellie/webhooks/deposit" in second.error
        # The operator learns in the host log, at registration time.
        (message,) = host_log.about("deposit")
        assert "agents: emma, nellie" in message

        # The incumbent learns through the surfaces it reads.
        listing = await feat_a.webhooks_list()
        assert listing.data["collisions"] == {"deposit": ["emma", "nellie"]}
        assert "Name collision on this host" in listing.confirmation
        assert "agents: emma, nellie" in listing.confirmation
        # The reader's OWN agent-prefixed address, which does dispatch to it.
        assert "point each sender at /api/agents/emma/webhooks/deposit" in listing.confirmation
        history = await feat_a.webhooks_history()
        assert history.data["collisions"] == {"deposit": ["emma", "nellie"]}
        assert "Name collision on this host" in history.confirmation
        assert "point each sender at /api/agents/emma/webhooks/deposit" in history.confirmation
        assert "/api/agents/nellie/webhooks/deposit" in (await feat_b.webhooks_list()).confirmation

        # Reading is silent: only where a name is created does the host log.
        before = len(host_log.about("deposit"))
        await feat_a.webhooks_list()
        await feat_a.webhooks_history()
        await feat_b.webhooks_list()
        assert len(host_log.about("deposit")) == before

        # Live, not recorded: once the newcomer withdraws, nothing collides.
        removed = await feat_b.webhooks_remove("deposit")
        assert removed.status.name == "OK", removed
        assert (await feat_a.webhooks_list()).data["collisions"] == {}
        assert "collision" not in (await feat_a.webhooks_list()).confirmation
    finally:
        from kestrel_sovereign.server import _unmount_feature_routers

        _unmount_feature_routers(app)
        app.state.agent = original[0]
        app.state.agent_manager = original[1]


@pytest.mark.asyncio
async def test_a_second_name_on_the_same_agent_is_not_a_collision(
    tmp_path, sqlite_database_factory, host_log
):
    from server import app

    feat_a, feat_b, manager = await _hosted_pair(tmp_path, sqlite_database_factory)
    original = (getattr(app.state, "agent", None), getattr(app.state, "agent_manager", None))
    app.state.agent = None
    app.state.agent_manager = manager
    _mount_feature_routers(app)
    try:
        await feat_a.webhooks_register(name="deposit", auth_type="none", allow_unauthenticated=True)
        again = await feat_a.webhooks_register(name="refund", auth_type="none", allow_unauthenticated=True)
        assert again.status.name == "OK", again
        assert "owners" not in again.data
        assert host_log.messages() == []
        assert (await feat_a.webhooks_list()).data["collisions"] == {}
    finally:
        from kestrel_sovereign.server import _unmount_feature_routers

        _unmount_feature_routers(app)
        app.state.agent = original[0]
        app.state.agent_manager = original[1]


@pytest.mark.asyncio
async def test_a_standalone_agent_has_no_host_and_no_peers(tmp_path, sqlite_database_factory):
    """No hook installed (a single-agent boot, or a feature test's mock
    agent): the answer is empty, and a MagicMock's fabricated attribute is
    not mistaken for a host."""
    db = await sqlite_database_factory(tmp_path / "solo.db")
    feat = WebhookFeature(_make_feature_agent(db=db))
    await feat.initialize()
    from kestrel_sovereign.features.storage_access import installed_host_hook

    # The mock-safety claim itself (review r4 P2-3): the fabricated attribute
    # is not read as a hook at all — not merely swallowed by the guard.
    assert installed_host_hook(feat.agent, "_host_webhook_collisions") is None
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    target = logging.getLogger("kestrel_sovereign.features.webhooks.feature")
    target.addHandler(handler)
    try:
        reg = await feat.webhooks_register(name="deposit", auth_type="none", allow_unauthenticated=True)
        assert reg.status.name == "OK", reg
        assert "owners" not in reg.data
        assert (await feat.webhooks_list()).data["collisions"] == {}
        assert (await feat.webhooks_history()).data["collisions"] == {}
        assert not any("collision report failed" in m for m in records)
    finally:
        target.removeHandler(handler)


@pytest.mark.asyncio
async def test_a_host_answer_that_is_not_a_mapping_is_logged_not_trusted(tmp_path, sqlite_database_factory):
    """A host bug is named in the log and answered as "no collisions" —
    never as a collision, and never as a failed read (review r3 F3)."""
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    target = logging.getLogger("kestrel_sovereign.features.webhooks.feature")
    target.addHandler(handler)
    try:
        db = await sqlite_database_factory(tmp_path / "solo.db")
        agent = _make_feature_agent(db=db)
        agent._host_webhook_collisions = lambda **_kw: "deposit"
        feat = WebhookFeature(agent)
        await feat.initialize()
        listing = await feat.webhooks_list()
        assert listing.status.name == "OK"
        assert listing.data["collisions"] == {}
        assert any("not a mapping" in m for m in records)
    finally:
        target.removeHandler(handler)


@pytest.mark.asyncio
async def test_registration_colliding_with_this_agents_own_receiver_names_no_address(
    tmp_path, sqlite_database_factory, host_log
):
    """Two receivers of one agent (a second feature with a receiver): the
    agent-prefixed form is refused too, so the reply must say unregister,
    not point at that address."""
    from server import app

    feat_a, feat_b, manager = await _hosted_pair(tmp_path, sqlite_database_factory)
    feat_a.agent.features["OtherWebhookFeature"] = _WebhookFeatureStub("deposit")
    original = (getattr(app.state, "agent", None), getattr(app.state, "agent_manager", None))
    app.state.agent = None
    app.state.agent_manager = manager
    _mount_feature_routers(app)
    try:
        reg = await feat_a.webhooks_register(
            name="deposit", auth_type="none", allow_unauthenticated=True
        )
        assert reg.status.name == "PARTIAL", reg
        assert reg.data["owners"] == ["emma", "emma"]
        assert "of one agent (emma)" in reg.error
        assert "unregister all but one of them" in reg.error
        assert "point each sender" not in reg.error
        assert "/api/agents/emma/webhooks/deposit" not in reg.error
        assert "/api/agents/emma/webhooks/deposit" not in reg.confirmation  # r3 F4
        (message,) = host_log.about("deposit")
        assert "of one agent (emma)" in message
        # The surfaces the owner reads say the same thing (review r2 F1).
        for surface in (await feat_a.webhooks_list(), await feat_a.webhooks_history()):
            assert "of one agent (emma)" in surface.confirmation
            assert "unregister all but one of them" in surface.confirmation
            assert "point each sender" not in surface.confirmation
            assert "/api/agents/emma/webhooks/deposit" not in surface.confirmation
    finally:
        from kestrel_sovereign.server import _unmount_feature_routers

        _unmount_feature_routers(app)
        app.state.agent = original[0]
        app.state.agent_manager = original[1]


# ---------------------------------------------------------------------------
# The fourth moment: a runtime feature enable (review r1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enabling_a_feature_at_runtime_announces_a_collision():
    """A feature enabled through the host endpoint brings its webhook names
    live without a mount pass; the endpoint asks the installed hook to
    announce. Nothing is announced when the host installed no hook."""
    from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature
    from tests.unit.test_features_api import _lifecycle_agent
    from kestrel_sovereign.endpoints.features import _enable_feature_locked

    class ReceivingFeature(SDKFixtureFeature):
        contribution_prefix = "collision-announce-fixture"

    agent = _lifecycle_agent()
    feature = ReceivingFeature(agent)
    feature.enabled = False
    agent.features = {feature.name: feature}
    announced = []
    agent._host_webhook_collisions = lambda *, announce=True: announced.append(announce) or {}

    result = await _enable_feature_locked(agent, feature.name)
    assert result["status"] == "enabled"
    assert feature.enabled is True
    assert announced == [True]

    # No host, no hook: a fabricated attribute on a mock is not a host either.
    standalone = _lifecycle_agent()
    solo = ReceivingFeature(standalone)
    solo.enabled = False
    standalone.features = {solo.name: solo}
    result = await _enable_feature_locked(standalone, solo.name)
    assert result["status"] == "enabled"


# ---------------------------------------------------------------------------
# The one describer behind every surface (review r2 F1/F2)
# ---------------------------------------------------------------------------


def test_describer_cross_agent_points_at_the_prefixed_address():
    from kestrel_sovereign.features.webhooks.collision import describe_collision

    text = describe_collision("deposit", ["a", "b"])
    assert "owned by 2 enabled receivers on this host (agents: a, b)" in text
    assert "point each sender at /api/agents/<agent>/webhooks/deposit" in text
    own = describe_collision(
        "deposit", ["a", "b"], own_label="a", own_endpoint="/api/agents/a/webhooks/deposit"
    )
    assert "point each sender at /api/agents/a/webhooks/deposit" in own
    # An endpoint without the label it belongs to is never printed.
    anonymous = describe_collision("deposit", ["a", "b"], own_endpoint="/api/agents/a/webhooks/deposit")
    assert "/api/agents/a/webhooks/deposit" not in anonymous


def test_describer_within_one_agent_names_no_address():
    from kestrel_sovereign.features.webhooks.collision import describe_collision

    text = describe_collision("deposit", ["a", "a"], own_endpoint="/api/agents/a/webhooks/deposit")
    assert "of one agent (a)" in text
    assert "agent-prefixed form are refused" in text
    assert "unregister all but one of them" in text
    assert "/api/agents/a/webhooks/deposit" not in text
    assert "point each sender" not in text


def test_describer_mixed_ownership_says_both():
    """Review r2 F2: ``len(set(owners)) == 1`` was a proxy that mixed
    ownership defeated. Emma owns it twice, nellie once: the prefixed form
    resolves it for nellie's senders and is refused for emma's."""
    from kestrel_sovereign.features.webhooks.collision import describe_collision

    text = describe_collision("deposit", ["emma", "emma", "nellie"])
    assert "3 enabled receivers on this host (agents: emma, emma, nellie)" in text
    assert "for emma (2 of its own receivers) the agent-prefixed form is refused too" in text
    assert "all but one of those receivers must be unregistered" in text
    assert "senders of nellie use /api/agents/<agent>/webhooks/deposit" in text
    assert "point each sender" not in text


@pytest.mark.asyncio
async def test_mixed_ownership_is_described_the_same_on_every_surface(
    tmp_path, sqlite_database_factory, host_log
):
    from server import app

    feat_a, feat_b, manager = await _hosted_pair(tmp_path, sqlite_database_factory)
    feat_a.agent.features["OtherWebhookFeature"] = _WebhookFeatureStub("deposit")
    original = (getattr(app.state, "agent", None), getattr(app.state, "agent_manager", None))
    app.state.agent = None
    app.state.agent_manager = manager
    _mount_feature_routers(app)
    try:
        await feat_b.webhooks_register(name="deposit", auth_type="none", allow_unauthenticated=True)
        host_log.records.clear()
        reg = await feat_a.webhooks_register(
            name="deposit", auth_type="none", allow_unauthenticated=True
        )
        assert reg.data["owners"] == ["emma", "emma", "nellie"]
        surfaces = {
            "register": reg.error,
            "host log": host_log.about("deposit")[0],
            "emma list": (await feat_a.webhooks_list()).confirmation,
            "emma history": (await feat_a.webhooks_history()).confirmation,
            "nellie list": (await feat_b.webhooks_list()).confirmation,
        }
        for label, text in surfaces.items():
            assert "for emma (2 of its own receivers) the agent-prefixed form is refused too" in text, label
            assert "point each sender" not in text, label
            # Review r3 F1: emma's own prefixed form is refused, so no surface
            # — least of all emma's own — may hand it out.
            assert "/api/agents/emma/webhooks/deposit" not in text, label
        assert "senders of nellie use /api/agents/<agent>/webhooks/deposit" in surfaces["register"]
        assert "senders of nellie use /api/agents/nellie/webhooks/deposit" in surfaces["nellie list"]
        # Review r3 F4: the confirmation is worded after the consult.
        assert "/api/agents/emma/webhooks/deposit" not in reg.confirmation
        assert reg.data["agent_endpoint"] == "/api/agents/emma/webhooks/deposit"
    finally:
        from kestrel_sovereign.server import _unmount_feature_routers

        _unmount_feature_routers(app)
        app.state.agent = original[0]
        app.state.agent_manager = original[1]


# ---------------------------------------------------------------------------
# The onboarding moment, through the REAL host hook (review r2 F3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_real_onboarding_hook_names_the_newcomer_before_publication(
    monkeypatch, host_log
):
    """Not the mount pass called by hand: ``_onboard_host_registered_agent``
    itself, with a real ``AgentManager`` that has NOT published the newcomer
    — the order production runs it in. The hook must pass the candidate."""
    from types import SimpleNamespace

    from fastapi import FastAPI

    from kestrel_sovereign import server
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.multi_agent.config import MultiAgentConfig

    monkeypatch.delenv("KESTREL_HOST_URL", raising=False)
    monkeypatch.setenv("KESTREL_API_KEY", API_KEY)
    monkeypatch.setenv("KESTREL_A2A_TRANSPORT_KEY", "peer-transport-key")
    app = FastAPI()
    app.state.multi_agent_config = MultiAgentConfig.model_validate(
        {"host": {"bind": "0.0.0.0", "port": 8888}}
    )
    app.state.multi_agent_config_path = None
    manager = AgentManager()
    app.state.agent = None
    app.state.agent_manager = manager

    incumbent = SimpleNamespace(
        did="did:test:a", agent_id="did:test:a",
        features={"WebhookFeature": _WebhookFeatureStub("deposit")},
    )
    manager._register_agent("a", incumbent)
    newcomer = SimpleNamespace(
        did="did:test:b", agent_id="did:test:b",
        features={"WebhookFeature": _WebhookFeatureStub("deposit")},
    )
    assert "b" not in manager.list_agents()
    # Production order (agent_manager ~5448): the candidate is parked in the
    # manager's private onboarding registry, the hook runs, THEN
    # _register_agent publishes. The hook's own policy install requires the
    # first step; nothing here publishes the newcomer.
    manager._onboarding_agents["b"] = newcomer
    try:
        await server._onboard_host_registered_agent(app, manager, "b", newcomer)
        (message,) = host_log.about("deposit")
        assert "agents: a, b" in message
        assert newcomer._host_webhook_collisions(announce=False) == {"deposit": ["a", "b"]}
        assert "b" not in manager.list_agents()  # still unpublished: the hook's moment
        manager._register_agent("b", newcomer)
        manager._onboarding_agents.pop("b", None)
        assert incumbent._host_webhook_collisions(announce=False) == {"deposit": ["a", "b"]}
    finally:
        server._unmount_feature_routers(app)


# ---------------------------------------------------------------------------
# The announce never fails a committed lifecycle operation (review r2 F4/F5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_raising_hook_does_not_fail_a_committed_enable():
    from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature
    from tests.unit.test_features_api import _lifecycle_agent
    from kestrel_sovereign.endpoints.features import _enable_feature_locked

    class ReceivingFeature(SDKFixtureFeature):
        contribution_prefix = "collision-raise-fixture"

    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    target = logging.getLogger("kestrel_sovereign.endpoints.features")
    target.addHandler(handler)
    try:
        agent = _lifecycle_agent()
        feature = ReceivingFeature(agent)
        feature.enabled = False
        agent.features = {feature.name: feature}

        def broken(*, announce=True):
            raise RuntimeError("a peer's feature property blew up")

        agent._host_webhook_collisions = broken
        result = await _enable_feature_locked(agent, feature.name)
        assert result["status"] == "enabled"
        assert feature.enabled is True
        assert any("collision report at enable failed" in m and "blew up" in m for m in records)
    finally:
        target.removeHandler(handler)


@pytest.mark.asyncio
async def test_a_rollback_that_reactivates_a_feature_announces():
    """A failed disable/remove restores the formerly-enabled members without
    a mount pass: the fifth moment (review r2 F5)."""
    from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature
    from tests.unit.test_features_api import _lifecycle_agent
    from kestrel_sovereign.endpoints.features import _restore_feature_group

    class ReceivingFeature(SDKFixtureFeature):
        contribution_prefix = "collision-restore-fixture"

    agent = _lifecycle_agent()
    feature = ReceivingFeature(agent)
    feature.enabled = False  # torn down by the failed operation
    agent.features = {feature.name: feature}
    announced = []
    agent._host_webhook_collisions = lambda *, announce=True: announced.append(announce) or {}

    await _restore_feature_group(
        agent, ((feature.name, feature, True),), operation="disable"
    )
    assert feature.enabled is True
    assert announced == [True]


def test_describer_two_agents_each_with_two_receivers_is_not_one_agent():
    """Review r3 F2: ``duplicated and not single`` also matched two agents
    that each own the name twice; that is not "one agent" and no
    agent-prefixed form dispatches until each unregisters one."""
    from kestrel_sovereign.features.webhooks.collision import describe_collision

    text = describe_collision("deposit", ["a", "a", "b", "b"], own_label="a",
                              own_endpoint="/api/agents/a/webhooks/deposit")
    assert "of one agent" not in text
    assert "a through 2 of its own receivers, b through 2 of its own receivers" in text
    assert "every agent-prefixed form are refused" in text
    assert "/api/agents/a/webhooks/deposit" not in text


def test_describer_prints_the_readers_address_only_where_it_dispatches():
    from kestrel_sovereign.features.webhooks.collision import (
        describe_collision,
        prefixed_form_dispatches,
    )

    owners = ["emma", "emma", "nellie"]
    assert prefixed_form_dispatches(owners, "nellie") is True
    assert prefixed_form_dispatches(owners, "emma") is False
    assert prefixed_form_dispatches(owners, None) is False
    emma = describe_collision("deposit", owners, own_label="emma",
                              own_endpoint="/api/agents/emma/webhooks/deposit")
    assert "/api/agents/emma/webhooks/deposit" not in emma
    assert "senders of nellie use /api/agents/<agent>/webhooks/deposit" in emma
    nellie = describe_collision("deposit", owners, own_label="nellie",
                                own_endpoint="/api/agents/nellie/webhooks/deposit")
    assert "senders of nellie use /api/agents/nellie/webhooks/deposit" in nellie


@pytest.mark.asyncio
async def test_a_raising_host_report_does_not_fail_registration_or_reads(
    tmp_path, sqlite_database_factory
):
    """Review r3 F3: the registration consult is audit-class, like the
    host's enable/rollback announce. The webhook is live and persisted; the
    reply says so; the failure is logged, not returned."""
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    target = logging.getLogger("kestrel_sovereign.features.webhooks.feature")
    target.addHandler(handler)
    try:
        db = await sqlite_database_factory(tmp_path / "raise.db")
        agent = _make_feature_agent(db=db)

        def broken(*, announce=True):
            raise RuntimeError("a peer's feature property blew up")

        agent._host_webhook_collisions = broken
        feat = WebhookFeature(agent)
        await feat.initialize()
        reg = await feat.webhooks_register(
            name="deposit", auth_type="none", allow_unauthenticated=True
        )
        assert reg.status.name == "OK", reg
        assert "deposit" in feat.receiver.webhooks
        assert "owners" not in reg.data
        assert (await feat.webhooks_list()).status.name == "OK"
        assert (await feat.webhooks_history()).data["collisions"] == {}
        assert sum("collision report failed" in m and "blew up" in m for m in records) >= 3
    finally:
        target.removeHandler(handler)


# ---------------------------------------------------------------------------
# The property behind every sentence (review r4 P2-1/P2-2): a concrete
# address belongs only to an agent whose prefixed form dispatches, and only
# when no other agent could be meant; receiver counts are never invented.
# ---------------------------------------------------------------------------

import itertools
import re as _re

_ADDRESS = _re.compile(r"/api/agents/(?P<label>[^/<]+)/webhooks/deposit")


def _owner_shapes():
    labels = ("a", "b", "c")
    for counts in itertools.product(range(0, 4), repeat=len(labels)):
        owners = [label for label, n in zip(labels, counts) for _ in range(n)]
        if len(owners) > 1:
            yield owners


@pytest.mark.parametrize("owners", list(_owner_shapes()), ids=lambda o: "".join(o))
def test_describer_never_attaches_an_address_to_an_agent_it_cannot_dispatch_to(owners):
    from kestrel_sovereign.features.webhooks.collision import (
        collision_facts,
        describe_collision,
        prefixed_form_dispatches,
    )

    duplicated, single = collision_facts(owners)
    for reader in ("a", "b", "c", None):
        text = describe_collision(
            "deposit", owners, own_label=reader,
            own_endpoint=f"/api/agents/{reader}/webhooks/deposit" if reader else None,
        )
        for match in _ADDRESS.finditer(text):
            label = match.group("label")
            # A concrete address is the reader's own and dispatches to the
            # reader. Where the sentence names OTHER agents' senders (the
            # mixed case), the reader must be the only agent it could mean.
            assert label == reader, (owners, reader, text)
            assert prefixed_form_dispatches(owners, label), (owners, reader, text)
            if duplicated:
                assert single == [reader], (owners, reader, text)
        # Every receiver count named is the real one.
        for label, n in _re.findall(r"(\w) (?:through |\()(\d+) of its own receivers", text):
            assert int(n) == owners.count(label), (owners, text)
        assert "two of its own" not in text
        # A double owner is never told to unregister just one (unless one is all but one).
        if any(owners.count(label) > 2 for label in duplicated):
            assert "unregister one of them" not in text
