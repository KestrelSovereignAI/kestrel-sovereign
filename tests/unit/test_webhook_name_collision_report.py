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
        assert "unregister one of them" in message
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
        assert "'deposit' is also owned by: emma, nellie" in listing.confirmation
        history = await feat_a.webhooks_history()
        assert history.data["collisions"] == {"deposit": ["emma", "nellie"]}
        assert "Name collision on this host" in history.confirmation

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
    reg = await feat.webhooks_register(name="deposit", auth_type="none", allow_unauthenticated=True)
    assert reg.status.name == "OK", reg
    assert "owners" not in reg.data
    assert (await feat.webhooks_list()).data["collisions"] == {}
    assert (await feat.webhooks_history()).data["collisions"] == {}


@pytest.mark.asyncio
async def test_a_host_answer_that_is_not_a_mapping_is_a_host_bug(tmp_path, sqlite_database_factory):
    db = await sqlite_database_factory(tmp_path / "solo.db")
    agent = _make_feature_agent(db=db)
    agent._host_webhook_collisions = lambda **_kw: "deposit"
    feat = WebhookFeature(agent)
    await feat.initialize()
    with pytest.raises(RuntimeError, match="not a mapping"):
        await feat.webhooks_list()


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
        assert "within this agent" in reg.error
        assert "unregister one of them" in reg.error
        assert "point each sender" not in reg.error
        (message,) = host_log.about("deposit")
        assert "of one agent (emma)" in message
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
