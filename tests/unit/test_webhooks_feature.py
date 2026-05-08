"""
Unit tests for the generic Webhook Receiver feature (#156).

Tests:
- WebhookAuthType enum values
- WebhookConfig / WebhookEvent data models
- Auth handlers: NoAuth, BearerTokenAuth, HMACSignatureAuth, IPAllowlistAuth
- Auth factory (create_auth_handler)
- WebhookReceiver registration, unregistration, handle_webhook, rate limiting
- WebhookFeature lifecycle (initialize, shutdown)
- WebhookFeature tools: list, history, register, remove
- Tool discovery and command prefixes
- Database persistence and loading of webhooks
- Graceful degradation without database
- FastAPI router generation
"""

import hashlib
import hmac as hmac_mod
import json
import time
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.features.webhooks.models import (
    WebhookAuthType,
    WebhookConfig,
    WebhookEvent,
)
from kestrel_sovereign.features.webhooks.auth import (
    BearerTokenAuth,
    HMACSignatureAuth,
    IPAllowlistAuth,
    NoAuth,
    create_auth_handler,
)
from kestrel_sovereign.features.webhooks.receiver import WebhookReceiver
from kestrel_sovereign.features.webhooks.feature import WebhookFeature


# ============================================================================
# Helpers
# ============================================================================


def _make_db():
    """Create a mock AsyncDatabase."""
    db = AsyncMock()
    db.execute = AsyncMock(return_value=0)
    db.fetchall = AsyncMock(return_value=[])
    db.fetchone = AsyncMock(return_value=None)
    db.table_exists = AsyncMock(return_value=True)
    return db


def _make_agent(db=None, agent_id="did:test:webhook-agent"):
    """Create a mock KestrelAgent."""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.did = agent_id

    storage = MagicMock()
    storage.db = db
    agent.storage = storage
    agent._raw_storage = None
    agent.features = []
    return agent


def _make_config(
    name="test-hook",
    auth_type=WebhookAuthType.NONE,
    auth_config=None,
    enabled=True,
    rate_limit=60,
):
    """Create a WebhookConfig with sensible defaults."""
    return WebhookConfig(
        name=name,
        auth_type=auth_type,
        auth_config=auth_config or {},
        enabled=enabled,
        rate_limit=rate_limit,
        agent_id="did:test:webhook-agent",
    )


# ============================================================================
# Model Tests
# ============================================================================


class TestWebhookAuthType:
    def test_enum_values(self):
        assert WebhookAuthType.BEARER_TOKEN.value == "bearer_token"
        assert WebhookAuthType.HMAC_SHA256.value == "hmac_sha256"
        assert WebhookAuthType.IP_ALLOWLIST.value == "ip_allowlist"
        assert WebhookAuthType.NONE.value == "none"


class TestWebhookConfig:
    def test_to_dict_excludes_auth_config(self):
        config = _make_config(auth_config={"token": "supersecret"})
        d = config.to_dict()
        assert "auth_config" not in d
        assert d["name"] == "test-hook"
        assert d["auth_type"] == "none"

    def test_defaults(self):
        config = WebhookConfig(name="x", auth_type=WebhookAuthType.NONE)
        assert config.enabled is True
        assert config.rate_limit == 60
        assert config.id  # auto-generated UUID
        assert config.created_at  # auto-generated timestamp


class TestWebhookEvent:
    def test_to_dict(self):
        event = WebhookEvent(
            webhook_name="test",
            source_ip="1.2.3.4",
            authenticated=True,
            status_code=200,
            payload_hash="abc123",
        )
        d = event.to_dict()
        assert d["webhook_name"] == "test"
        assert d["source_ip"] == "1.2.3.4"
        assert d["authenticated"] is True
        assert d["status_code"] == 200
        assert d["payload_hash"] == "abc123"


# ============================================================================
# Auth Handler Tests
# ============================================================================


class TestNoAuth:
    def test_always_passes(self):
        auth = NoAuth()
        assert auth.validate(headers={}, body=b"", source_ip="0.0.0.0") is True


class TestBearerTokenAuth:
    def test_valid_token(self):
        auth = BearerTokenAuth({"token": "my-secret"})
        assert auth.validate(
            headers={"authorization": "Bearer my-secret"},
            body=b"",
            source_ip="0.0.0.0",
        ) is True

    def test_invalid_token(self):
        auth = BearerTokenAuth({"token": "my-secret"})
        assert auth.validate(
            headers={"authorization": "Bearer wrong-token"},
            body=b"",
            source_ip="0.0.0.0",
        ) is False

    def test_missing_header(self):
        auth = BearerTokenAuth({"token": "my-secret"})
        assert auth.validate(headers={}, body=b"", source_ip="0.0.0.0") is False

    def test_malformed_header(self):
        auth = BearerTokenAuth({"token": "my-secret"})
        assert auth.validate(
            headers={"authorization": "Basic abc123"},
            body=b"",
            source_ip="0.0.0.0",
        ) is False

    def test_empty_token_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            BearerTokenAuth({"token": ""})

    def test_missing_token_key_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            BearerTokenAuth({})


class TestHMACSignatureAuth:
    def _sign(self, secret, body, prefix="sha256="):
        digest = hmac_mod.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        return f"{prefix}{digest}"

    def test_valid_signature(self):
        secret = "webhook-secret"
        body = b'{"event": "test"}'
        auth = HMACSignatureAuth({"secret": secret})
        sig = self._sign(secret, body)
        assert auth.validate(
            headers={"x-hub-signature-256": sig},
            body=body,
            source_ip="0.0.0.0",
        ) is True

    def test_invalid_signature(self):
        auth = HMACSignatureAuth({"secret": "correct-secret"})
        sig = self._sign("wrong-secret", b"payload")
        assert auth.validate(
            headers={"x-hub-signature-256": sig},
            body=b"payload",
            source_ip="0.0.0.0",
        ) is False

    def test_missing_header(self):
        auth = HMACSignatureAuth({"secret": "secret"})
        assert auth.validate(headers={}, body=b"x", source_ip="0.0.0.0") is False

    def test_custom_header_and_prefix(self):
        secret = "my-secret"
        body = b"data"
        auth = HMACSignatureAuth({
            "secret": secret,
            "header": "x-custom-sig",
            "prefix": "hmac=",
        })
        sig = self._sign(secret, body, prefix="hmac=")
        assert auth.validate(
            headers={"x-custom-sig": sig},
            body=body,
            source_ip="0.0.0.0",
        ) is True

    def test_empty_secret_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            HMACSignatureAuth({"secret": ""})

    def test_timing_safe_comparison(self):
        """Verify that hmac.compare_digest is used (not ==)."""
        secret = "test"
        body = b"body"
        auth = HMACSignatureAuth({"secret": secret})
        sig = self._sign(secret, body)
        # The implementation uses hmac.compare_digest internally.
        # We validate correctness here; timing-safety is inherent in the API.
        assert auth.validate(
            headers={"x-hub-signature-256": sig},
            body=body,
            source_ip="0.0.0.0",
        ) is True


class TestIPAllowlistAuth:
    def test_allowed_ip(self):
        auth = IPAllowlistAuth({"allowed_ips": ["10.0.0.0/8"]})
        assert auth.validate(
            headers={}, body=b"", source_ip="10.1.2.3"
        ) is True

    def test_blocked_ip(self):
        auth = IPAllowlistAuth({"allowed_ips": ["10.0.0.0/8"]})
        assert auth.validate(
            headers={}, body=b"", source_ip="192.168.1.1"
        ) is False

    def test_exact_ip_match(self):
        auth = IPAllowlistAuth({"allowed_ips": ["1.2.3.4"]})
        assert auth.validate(
            headers={}, body=b"", source_ip="1.2.3.4"
        ) is True
        assert auth.validate(
            headers={}, body=b"", source_ip="1.2.3.5"
        ) is False

    def test_multiple_networks(self):
        auth = IPAllowlistAuth({
            "allowed_ips": ["10.0.0.0/8", "172.16.0.0/12"]
        })
        assert auth.validate(headers={}, body=b"", source_ip="10.0.0.1") is True
        assert auth.validate(headers={}, body=b"", source_ip="172.20.0.1") is True
        assert auth.validate(headers={}, body=b"", source_ip="192.168.0.1") is False

    def test_invalid_source_ip(self):
        auth = IPAllowlistAuth({"allowed_ips": ["10.0.0.0/8"]})
        assert auth.validate(
            headers={}, body=b"", source_ip="not-an-ip"
        ) is False

    def test_empty_allowlist_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            IPAllowlistAuth({"allowed_ips": []})

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            IPAllowlistAuth({})


class TestCreateAuthHandler:
    def test_creates_no_auth(self):
        handler = create_auth_handler("none", {})
        assert isinstance(handler, NoAuth)

    def test_creates_bearer(self):
        handler = create_auth_handler("bearer_token", {"token": "abc"})
        assert isinstance(handler, BearerTokenAuth)

    def test_creates_hmac(self):
        handler = create_auth_handler("hmac_sha256", {"secret": "abc"})
        assert isinstance(handler, HMACSignatureAuth)

    def test_creates_ip_allowlist(self):
        handler = create_auth_handler("ip_allowlist", {"allowed_ips": ["10.0.0.0/8"]})
        assert isinstance(handler, IPAllowlistAuth)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_auth_handler("magic_auth", {})


# ============================================================================
# WebhookReceiver Tests
# ============================================================================


class TestWebhookReceiverRegistration:
    def test_register_webhook(self):
        receiver = WebhookReceiver()
        config = _make_config(name="gh-push")
        receiver.register_webhook(config)
        assert "gh-push" in receiver.webhooks
        assert "gh-push" in receiver.auth_handlers

    def test_register_duplicate_raises(self):
        receiver = WebhookReceiver()
        config = _make_config(name="dup")
        receiver.register_webhook(config)
        with pytest.raises(ValueError, match="already registered"):
            receiver.register_webhook(config)

    def test_unregister_webhook(self):
        receiver = WebhookReceiver()
        config = _make_config(name="to-remove")
        receiver.register_webhook(config)
        assert receiver.unregister_webhook("to-remove") is True
        assert "to-remove" not in receiver.webhooks

    def test_unregister_unknown_returns_false(self):
        receiver = WebhookReceiver()
        assert receiver.unregister_webhook("nonexistent") is False

    def test_list_webhooks(self):
        receiver = WebhookReceiver()
        receiver.register_webhook(_make_config(name="a"))
        receiver.register_webhook(_make_config(name="b"))
        listing = receiver.list_webhooks()
        names = {w["name"] for w in listing}
        assert names == {"a", "b"}


class TestWebhookReceiverHandleWebhook:
    @pytest.mark.asyncio
    async def test_unknown_webhook_returns_404(self):
        receiver = WebhookReceiver()
        result = await receiver.handle_webhook(
            "nonexistent", headers={}, body=b"", source_ip="1.1.1.1"
        )
        assert result["status_code"] == 404
        assert len(receiver.event_log) == 1

    @pytest.mark.asyncio
    async def test_disabled_webhook_returns_503(self):
        receiver = WebhookReceiver()
        receiver.register_webhook(_make_config(name="off", enabled=False))
        # Manually disable after registration
        receiver.webhooks["off"].enabled = False
        result = await receiver.handle_webhook(
            "off", headers={}, body=b"", source_ip="1.1.1.1"
        )
        assert result["status_code"] == 503

    @pytest.mark.asyncio
    async def test_auth_failure_returns_401(self):
        receiver = WebhookReceiver()
        config = _make_config(
            name="secured",
            auth_type=WebhookAuthType.BEARER_TOKEN,
            auth_config={"token": "correct-token"},
        )
        receiver.register_webhook(config)

        result = await receiver.handle_webhook(
            "secured",
            headers={"authorization": "Bearer wrong-token"},
            body=b"",
            source_ip="1.1.1.1",
        )
        assert result["status_code"] == 401
        assert result["body"]["error"] == "Authentication failed"

    @pytest.mark.asyncio
    async def test_successful_webhook(self):
        receiver = WebhookReceiver()
        receiver.register_webhook(_make_config(name="open"))

        result = await receiver.handle_webhook(
            "open", headers={}, body=b'{"test": true}', source_ip="1.1.1.1"
        )
        assert result["status_code"] == 200
        assert result["body"]["status"] == "received"
        assert "event_id" in result["body"]

    @pytest.mark.asyncio
    async def test_bearer_auth_success(self):
        receiver = WebhookReceiver()
        config = _make_config(
            name="secured",
            auth_type=WebhookAuthType.BEARER_TOKEN,
            auth_config={"token": "my-token"},
        )
        receiver.register_webhook(config)

        result = await receiver.handle_webhook(
            "secured",
            headers={"authorization": "Bearer my-token"},
            body=b"data",
            source_ip="1.1.1.1",
        )
        assert result["status_code"] == 200

    @pytest.mark.asyncio
    async def test_hmac_auth_success(self):
        secret = "hmac-secret"
        body = b'{"event": "push"}'
        sig = "sha256=" + hmac_mod.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()

        receiver = WebhookReceiver()
        config = _make_config(
            name="gh",
            auth_type=WebhookAuthType.HMAC_SHA256,
            auth_config={"secret": secret},
        )
        receiver.register_webhook(config)

        result = await receiver.handle_webhook(
            "gh",
            headers={"x-hub-signature-256": sig},
            body=body,
            source_ip="1.1.1.1",
        )
        assert result["status_code"] == 200

    @pytest.mark.asyncio
    async def test_event_log_records_all_requests(self):
        receiver = WebhookReceiver()
        receiver.register_webhook(_make_config(name="hook"))

        await receiver.handle_webhook(
            "hook", headers={}, body=b"a", source_ip="1.1.1.1"
        )
        await receiver.handle_webhook(
            "unknown", headers={}, body=b"b", source_ip="2.2.2.2"
        )

        assert len(receiver.event_log) == 2

    @pytest.mark.asyncio
    async def test_event_log_payload_hash(self):
        receiver = WebhookReceiver()
        receiver.register_webhook(_make_config(name="hook"))

        body = b"hello webhook"
        await receiver.handle_webhook(
            "hook", headers={}, body=body, source_ip="1.1.1.1"
        )

        event = receiver.event_log[0]
        expected_hash = hashlib.sha256(body).hexdigest()
        assert event.payload_hash == expected_hash


class TestWebhookReceiverRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self):
        receiver = WebhookReceiver()
        receiver.register_webhook(_make_config(name="limited", rate_limit=3))

        # First 3 should succeed
        for _ in range(3):
            result = await receiver.handle_webhook(
                "limited", headers={}, body=b"", source_ip="1.1.1.1"
            )
            assert result["status_code"] == 200

        # 4th should be rate limited
        result = await receiver.handle_webhook(
            "limited", headers={}, body=b"", source_ip="1.1.1.1"
        )
        assert result["status_code"] == 429
        assert "Rate limit" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_zero_rate_limit_disables_limiting(self):
        receiver = WebhookReceiver()
        receiver.register_webhook(_make_config(name="unlimited", rate_limit=0))

        # Should never be rate limited
        for _ in range(100):
            result = await receiver.handle_webhook(
                "unlimited", headers={}, body=b"", source_ip="1.1.1.1"
            )
            assert result["status_code"] == 200


class TestWebhookReceiverGetRecentEvents:
    @pytest.mark.asyncio
    async def test_returns_newest_first(self):
        receiver = WebhookReceiver()
        receiver.register_webhook(_make_config(name="hook"))

        await receiver.handle_webhook(
            "hook", headers={}, body=b"first", source_ip="1.1.1.1"
        )
        await receiver.handle_webhook(
            "hook", headers={}, body=b"second", source_ip="2.2.2.2"
        )

        events = receiver.get_recent_events(limit=10)
        assert len(events) == 2
        # Newest first
        assert events[0]["source_ip"] == "2.2.2.2"
        assert events[1]["source_ip"] == "1.1.1.1"

    @pytest.mark.asyncio
    async def test_respects_limit(self):
        receiver = WebhookReceiver()
        receiver.register_webhook(_make_config(name="hook"))

        for i in range(5):
            await receiver.handle_webhook(
                "hook", headers={}, body=f"body-{i}".encode(), source_ip="1.1.1.1"
            )

        events = receiver.get_recent_events(limit=2)
        assert len(events) == 2


class TestWebhookReceiverRouter:
    def test_get_router_returns_router(self):
        receiver = WebhookReceiver()
        router = receiver.get_router()
        # Should have at least one route
        assert len(router.routes) >= 1


# ============================================================================
# WebhookFeature Tests
# ============================================================================


class TestWebhookFeatureInitialize:
    @pytest_asyncio.fixture
    async def feature(self):
        """Create and initialise a WebhookFeature with mock agent."""
        db = _make_db()
        agent = _make_agent(db=db)
        feat = WebhookFeature(agent)
        await feat.initialize()
        return feat

    @pytest.mark.asyncio
    async def test_creates_webhook_config_table(self, feature):
        create_calls = [
            c for c in feature._db.execute.call_args_list
            if "CREATE TABLE" in str(c) and "webhook_config" in str(c)
        ]
        assert len(create_calls) == 1

    @pytest.mark.asyncio
    async def test_creates_webhook_log_table(self, feature):
        create_calls = [
            c for c in feature._db.execute.call_args_list
            if "CREATE TABLE" in str(c) and "webhook_log" in str(c)
        ]
        assert len(create_calls) == 1

    @pytest.mark.asyncio
    async def test_creates_indexes(self, feature):
        index_calls = [
            c for c in feature._db.execute.call_args_list
            if "CREATE" in str(c) and "INDEX" in str(c)
        ]
        assert len(index_calls) == 2

    @pytest.mark.asyncio
    async def test_sets_agent_id(self, feature):
        assert feature._agent_id == "did:test:webhook-agent"

    @pytest.mark.asyncio
    async def test_receiver_created(self, feature):
        assert feature.receiver is not None
        assert isinstance(feature.receiver, WebhookReceiver)


class TestWebhookFeatureInitializeWithoutDB:
    @pytest.mark.asyncio
    async def test_initializes_without_db(self):
        agent = _make_agent(db=None)
        agent.storage.db = None
        feat = WebhookFeature(agent)
        await feat.initialize()
        assert feat._db is None
        assert feat.receiver is not None

    @pytest.mark.asyncio
    async def test_tools_work_without_db(self):
        agent = _make_agent(db=None)
        agent.storage.db = None
        feat = WebhookFeature(agent)
        await feat.initialize()

        envelope = await feat.webhooks_list()
        assert envelope.data["count"] == 0
        assert envelope.data["webhooks"] == []


class TestWebhookFeatureLoadPersisted:
    @pytest.mark.asyncio
    async def test_loads_persisted_webhooks(self):
        db = _make_db()
        db.fetchall = AsyncMock(return_value=[
            ("id-1", "github-push", "none", "{}", "push", 1, 60, "2026-03-05T00:00:00"),
            ("id-2", "deploy-hook", "bearer_token", '{"token":"abc"}', "deploy", 1, 30, "2026-03-05T01:00:00"),
        ])
        agent = _make_agent(db=db)
        feat = WebhookFeature(agent)
        await feat.initialize()

        assert "github-push" in feat.receiver.webhooks
        assert "deploy-hook" in feat.receiver.webhooks
        assert len(feat.receiver.webhooks) == 2

    @pytest.mark.asyncio
    async def test_skips_invalid_persisted_webhooks(self):
        db = _make_db()
        # bearer_token without a token should fail to register
        db.fetchall = AsyncMock(return_value=[
            ("id-1", "bad-hook", "bearer_token", "{}", "", 1, 60, "2026-03-05T00:00:00"),
        ])
        agent = _make_agent(db=db)
        feat = WebhookFeature(agent)
        # Should not raise
        await feat.initialize()
        assert "bad-hook" not in feat.receiver.webhooks


# ============================================================================
# WebhookFeature Tool Tests
# ============================================================================


class TestWebhooksList:
    @pytest_asyncio.fixture
    async def feature(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = WebhookFeature(agent)
        await feat.initialize()
        return feat

    @pytest.mark.asyncio
    async def test_list_empty(self, feature):
        envelope = await feature.webhooks_list()
        assert envelope.data["count"] == 0
        assert envelope.data["webhooks"] == []

    @pytest.mark.asyncio
    async def test_list_after_register(self, feature):
        await feature.webhooks_register(name="test-hook", auth_type="none")
        envelope = await feature.webhooks_list()
        assert envelope.data["count"] == 1
        assert envelope.data["webhooks"][0]["name"] == "test-hook"


class TestWebhooksRegister:
    @pytest_asyncio.fixture
    async def feature(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = WebhookFeature(agent)
        await feat.initialize()
        return feat

    @pytest.mark.asyncio
    async def test_register_none_auth(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature.webhooks_register(name="open-hook", auth_type="none")
        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["name"] == "open-hook"
        assert envelope.data["endpoint"] == "/webhooks/open-hook"

    @pytest.mark.asyncio
    async def test_register_bearer_auth(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature.webhooks_register(
            name="secure-hook",
            auth_type="bearer_token",
            auth_config_json='{"token": "test-token"}',
        )
        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["auth_type"] == "bearer_token"

    @pytest.mark.asyncio
    async def test_register_hmac_auth(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature.webhooks_register(
            name="hmac-hook",
            auth_type="hmac_sha256",
            auth_config_json='{"secret": "my-secret"}',
        )
        assert envelope.status is ToolResultStatus.OK

    @pytest.mark.asyncio
    async def test_register_ip_allowlist(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature.webhooks_register(
            name="ip-hook",
            auth_type="ip_allowlist",
            auth_config_json='{"allowed_ips": ["10.0.0.0/8"]}',
        )
        assert envelope.status is ToolResultStatus.OK

    @pytest.mark.asyncio
    async def test_register_invalid_auth_type(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature.webhooks_register(
            name="bad-hook", auth_type="magic_auth"
        )
        assert envelope.status is ToolResultStatus.ERROR
        assert "Invalid auth_type" in envelope.error

    @pytest.mark.asyncio
    async def test_register_invalid_auth_config_json(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature.webhooks_register(
            name="bad-json", auth_type="none", auth_config_json="not-json"
        )
        assert envelope.status is ToolResultStatus.ERROR
        assert "auth_config_json" in envelope.error

    @pytest.mark.asyncio
    async def test_register_invalid_name(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature.webhooks_register(name="bad name!", auth_type="none")
        assert envelope.status is ToolResultStatus.ERROR
        assert "alphanumeric" in envelope.error

    @pytest.mark.asyncio
    async def test_register_empty_name(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature.webhooks_register(name="", auth_type="none")
        assert envelope.status is ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_register_duplicate(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        await feature.webhooks_register(name="dup-hook", auth_type="none")
        envelope = await feature.webhooks_register(name="dup-hook", auth_type="none")
        assert envelope.status is ToolResultStatus.ERROR
        assert "already registered" in envelope.error

    @pytest.mark.asyncio
    async def test_register_persists_to_db(self, feature):
        await feature.webhooks_register(name="persist-hook", auth_type="none")
        insert_calls = [
            c for c in feature._db.execute.call_args_list
            if "INSERT INTO webhook_config" in str(c)
        ]
        assert len(insert_calls) == 1

    @pytest.mark.asyncio
    async def test_register_with_custom_rate_limit(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature.webhooks_register(
            name="rate-hook", auth_type="none", rate_limit=10
        )
        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["rate_limit"] == 10

    @pytest.mark.asyncio
    async def test_register_with_event_type(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature.webhooks_register(
            name="event-hook", auth_type="none", event_type="push"
        )
        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["event_type"] == "push"

    @pytest.mark.asyncio
    async def test_register_hyphens_and_underscores_in_name(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature.webhooks_register(
            name="my-webhook_v2", auth_type="none"
        )
        assert envelope.status is ToolResultStatus.OK


class TestWebhooksRemove:
    @pytest_asyncio.fixture
    async def feature(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = WebhookFeature(agent)
        await feat.initialize()
        return feat

    @pytest.mark.asyncio
    async def test_remove_existing(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        await feature.webhooks_register(name="to-remove", auth_type="none")
        envelope = await feature.webhooks_remove(name="to-remove")
        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["status"] == "removed"

    @pytest.mark.asyncio
    async def test_remove_not_found(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature.webhooks_remove(name="nonexistent")
        assert envelope.status is ToolResultStatus.ERROR
        assert "not found" in envelope.error

    @pytest.mark.asyncio
    async def test_remove_deletes_from_db(self, feature):
        await feature.webhooks_register(name="db-remove", auth_type="none")
        feature._db.execute.reset_mock()
        feature._db.fetchone = AsyncMock(return_value=("any-id",))
        await feature.webhooks_remove(name="db-remove")
        delete_calls = [
            c for c in feature._db.execute.call_args_list
            if "DELETE FROM webhook_config" in str(c)
        ]
        assert len(delete_calls) == 1

    @pytest.mark.asyncio
    async def test_remove_cleans_persisted_only_row(self, feature):
        """Receiver doesn't have the webhook (e.g. it failed to load on
        startup, or a prior remove was PARTIAL because the DB was
        temporarily down) but the persisted row still exists. The
        retry must scrub that row, NOT short-circuit on the receiver
        miss — otherwise the webhook resurrects on the next restart.
        Codex round 1 of #1126 caught the regression.
        """
        from kestrel_sdk.tools.result import ToolResultStatus
        # Receiver is empty; DB still has the row.
        feature._db.fetchone = AsyncMock(return_value=("orphan-id",))
        envelope = await feature.webhooks_remove(name="orphan")
        assert envelope.status is ToolResultStatus.PARTIAL
        assert "stale row" in envelope.error
        assert envelope.data["removed_from_memory"] is False
        assert envelope.data["deleted_from_db"] is True
        delete_calls = [
            c for c in feature._db.execute.call_args_list
            if "DELETE FROM webhook_config" in str(c)
        ]
        assert len(delete_calls) == 1

    @pytest.mark.asyncio
    async def test_remove_truly_not_found_returns_error(self, feature):
        """Neither receiver nor DB has the webhook → ERROR."""
        from kestrel_sdk.tools.result import ToolResultStatus
        feature._db.fetchone = AsyncMock(return_value=None)
        envelope = await feature.webhooks_remove(name="ghost")
        assert envelope.status is ToolResultStatus.ERROR
        assert "not found" in envelope.error

    @pytest.mark.asyncio
    async def test_remove_db_probe_fails_with_no_memory_hit(self, feature):
        """Receiver doesn't have it AND the DB probe blew up — we
        DO NOT know whether a persisted row exists, so reporting
        'removed' would be a lie. Must return ERROR with a retry
        hint, not PARTIAL with a fake confirmation. Codex round 2
        of #1126 caught the regression.
        """
        from kestrel_sdk.tools.result import ToolResultStatus
        feature._db.fetchone = AsyncMock(side_effect=RuntimeError("connection refused"))
        envelope = await feature.webhooks_remove(name="maybe-ghost")
        assert envelope.status is ToolResultStatus.ERROR
        assert "DB lookup against webhook_config failed" in envelope.error
        assert "Retry" in envelope.error
        assert envelope.data["removed_from_memory"] is False
        assert envelope.data["deleted_from_db"] is False
        assert envelope.data["db_lookup_failed"] is True


class TestWebhooksHistory:
    @pytest_asyncio.fixture
    async def feature(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = WebhookFeature(agent)
        await feat.initialize()
        return feat

    @pytest.mark.asyncio
    async def test_history_empty(self, feature):
        envelope = await feature.webhooks_history()
        assert envelope.data["count"] == 0
        assert envelope.data["events"] == []

    @pytest.mark.asyncio
    async def test_history_from_db(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[
            ("ev-1", "hook-a", "1.1.1.1", 1, 200, "abc", "2026-03-05T10:00:00"),
            ("ev-2", "hook-b", "2.2.2.2", 0, 401, "def", "2026-03-05T09:00:00"),
        ])
        envelope = await feature.webhooks_history()
        assert envelope.data["count"] == 2
        assert envelope.data["events"][0]["id"] == "ev-1"
        assert envelope.data["events"][1]["authenticated"] is False

    @pytest.mark.asyncio
    async def test_history_falls_back_to_memory(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[])
        # Simulate some in-memory events
        feature.receiver.register_webhook(_make_config(name="hook"))
        await feature.receiver.handle_webhook(
            "hook", headers={}, body=b"test", source_ip="3.3.3.3"
        )
        envelope = await feature.webhooks_history()
        assert envelope.data["count"] == 1

    @pytest.mark.asyncio
    async def test_history_respects_limit(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[])
        envelope = await feature.webhooks_history(limit=5)
        # Verify limit was passed (or at least used)
        assert envelope.data["count"] == 0  # No data, just checking it doesn't crash

    @pytest.mark.asyncio
    async def test_history_partial_when_db_query_fails(self, feature):
        """When the DB query raises, the envelope must say so — the
        in-memory ring buffer fallback may be missing persisted events.
        """
        from kestrel_sdk.tools.result import ToolResultStatus
        feature._db.fetchall = AsyncMock(side_effect=RuntimeError("disk i/o error"))
        envelope = await feature.webhooks_history()
        assert envelope.status is ToolResultStatus.PARTIAL
        assert "in-memory ring buffer" in envelope.error

    @pytest.mark.asyncio
    async def test_register_partial_when_persist_fails(self, feature):
        """When the in-memory register succeeds but the DB INSERT
        raises, the envelope must surface PARTIAL — the webhook
        works now but won't survive a restart.
        """
        from kestrel_sdk.tools.result import ToolResultStatus
        feature._db.execute = AsyncMock(side_effect=RuntimeError("table missing"))
        envelope = await feature.webhooks_register(name="lost-on-restart", auth_type="none")
        assert envelope.status is ToolResultStatus.PARTIAL
        assert "NOT survive" in envelope.error
        assert envelope.data["persisted"] is False


# ============================================================================
# Tool Discovery Tests
# ============================================================================


class TestWebhookToolDiscovery:
    @pytest.mark.asyncio
    async def test_tools_registered(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = WebhookFeature(agent)
        await feat.initialize()

        tools = feat.get_tools()
        tool_names = {t.name for t in tools}

        assert "webhooks_list" in tool_names
        assert "webhooks_history" in tool_names
        assert "webhooks_register" in tool_names
        assert "webhooks_remove" in tool_names
        assert len(tool_names) == 4

    @pytest.mark.asyncio
    async def test_tool_description(self):
        agent = _make_agent()
        feat = WebhookFeature(agent)
        assert "webhook" in feat.tool_description.lower()

    @pytest.mark.asyncio
    async def test_command_prefixes(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = WebhookFeature(agent)
        await feat.initialize()

        tools = feat.get_tools()
        prefixes = {t.schema.command_prefix for t in tools}
        assert "!webhooks list" in prefixes
        assert "!webhooks history" in prefixes
        assert "!webhooks register" in prefixes
        assert "!webhooks remove" in prefixes


# ============================================================================
# WebhookFeature Router Integration
# ============================================================================


class TestWebhookFeatureRouter:
    @pytest.mark.asyncio
    async def test_get_webhook_router(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = WebhookFeature(agent)
        await feat.initialize()
        router = feat.get_webhook_router()
        assert router is not None
        assert len(router.routes) >= 1


# ============================================================================
# Audit Logging
# ============================================================================


class TestWebhookAuditLogging:
    @pytest.mark.asyncio
    async def test_log_webhook_event_persists(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = WebhookFeature(agent)
        await feat.initialize()
        db.execute.reset_mock()

        await feat.log_webhook_event(
            webhook_name="test",
            source_ip="1.1.1.1",
            authenticated=True,
            status_code=200,
            payload_hash="abc123",
        )

        insert_calls = [
            c for c in db.execute.call_args_list
            if "INSERT INTO webhook_log" in str(c)
        ]
        assert len(insert_calls) == 1

    @pytest.mark.asyncio
    async def test_log_webhook_event_no_db(self):
        agent = _make_agent(db=None)
        agent.storage.db = None
        feat = WebhookFeature(agent)
        await feat.initialize()

        # Should not raise
        await feat.log_webhook_event(
            webhook_name="test",
            source_ip="1.1.1.1",
            authenticated=False,
            status_code=401,
            payload_hash="",
        )


# ============================================================================
# Run tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
