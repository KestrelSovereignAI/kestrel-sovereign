"""Contract tests for model-aware Codex reasoning-effort normalization."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.llm.codex_app_server import CodexAppServerError
from kestrel_sovereign.llm.codex_reasoning import (
    CodexReasoningCapability,
    CodexReasoningCapabilityState,
    normalize_codex_reasoning_effort,
    resolve_codex_reasoning_capability,
)


def _app_model(
    slug: str,
    efforts: tuple[str, ...],
    *,
    is_default: bool = False,
) -> dict:
    return {
        "id": slug,
        "model": slug,
        "isDefault": is_default,
        "supportedReasoningEfforts": [
            {"reasoningEffort": effort, "description": effort} for effort in efforts
        ],
    }


def _cached_model(
    slug: str,
    efforts: tuple[str, ...],
    *,
    include_metadata: bool = True,
) -> dict:
    row = {"slug": slug}
    if include_metadata:
        row["supported_reasoning_levels"] = [
            {"effort": effort, "description": effort} for effort in efforts
        ]
    return row


def _known(*efforts: str) -> CodexReasoningCapability:
    return CodexReasoningCapability(
        CodexReasoningCapabilityState.KNOWN,
        efforts=efforts,
        source="test catalog",
    )


def _unknown() -> CodexReasoningCapability:
    return CodexReasoningCapability(
        CodexReasoningCapabilityState.UNKNOWN,
        source="test catalog",
        detail="model metadata unavailable",
    )


def _no_effort() -> CodexReasoningCapability:
    return CodexReasoningCapability(
        CodexReasoningCapabilityState.NO_EFFORT,
        source="test catalog",
    )


_TEXT_TURN = [
    {"method": "item/agentMessage/delta", "params": {"delta": "ok"}},
    {
        "method": "item/completed",
        "params": {"item": {"type": "agentMessage", "text": "ok"}},
    },
    {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
]


class _ConfigAwareAppServer:
    def __init__(
        self,
        config: dict,
        models: tuple[dict, ...] = (),
        *,
        model_list_error: Exception | None = None,
        on_model_list=None,
    ):
        self.config = config
        self.models = models
        self.model_list_error = model_list_error
        self.on_model_list = on_model_list
        self.requests: list[tuple[str, dict]] = []
        self.thread_starts = 0
        self.started = False
        self.registered_handlers = {}

    async def ensure_started(self):
        self.started = True

    async def request(self, method, params=None, *, timeout=120):
        params = params or {}
        self.requests.append((method, params))
        if method == "config/read":
            return {"config": dict(self.config), "origins": {}}
        if method == "model/list":
            if self.model_list_error is not None:
                raise self.model_list_error
            if self.on_model_list is not None:
                self.on_model_list()
            return {"data": list(self.models), "nextCursor": None}
        if method == "thread/start":
            self.thread_starts += 1
            return {"thread": {"id": f"thr-{self.thread_starts}"}}
        if method == "turn/start":
            return {"turn": {"id": "turn-1"}}
        raise AssertionError(f"unexpected request: {method}")

    def register_server_request_handler(self, method, handler, *, thread_id=None):
        key = (method, thread_id)
        self.registered_handlers[key] = handler
        return lambda: self.registered_handlers.pop(key, None)

    def open_turn_sink(self, thread_id):
        return thread_id

    def close_turn_sink(self, thread_id):
        return None

    async def iter_turn_events(
        self,
        sink,
        *,
        idle_timeout=120,
        thread_id=None,
        cancel_token=None,
    ):
        for event in _TEXT_TURN:
            yield event


def _request_params(app, method: str) -> list[dict]:
    return [
        params for request_method, params in app.requests if request_method == method
    ]


class TestReasoningEffortPolicy:
    @pytest.mark.parametrize(
        "effort",
        ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"),
    )
    def test_public_vocabulary_passes_with_matching_ceiling(self, effort):
        assert (
            normalize_codex_reasoning_effort(
                effort,
                _known("low", "medium", "high", "xhigh", "max", "ultra"),
                model="gpt-current",
            )
            == effort
        )

    def test_absent_setting_needs_no_capability_evidence(self):
        assert (
            normalize_codex_reasoning_effort(
                None,
                _unknown(),
                model="gpt-5.5",
            )
            is None
        )

    @pytest.mark.parametrize("configured", ("max", "ultra"))
    def test_effort_above_model_ceiling_clamps_to_xhigh(self, configured):
        assert (
            normalize_codex_reasoning_effort(
                configured,
                _known("low", "medium", "high", "xhigh"),
                model="gpt-5.5",
            )
            == "xhigh"
        )

    @pytest.mark.parametrize("configured", ("", "maximum", "HIGH", 3, True))
    def test_unknown_or_wrong_type_fails_actionably(self, configured):
        with pytest.raises(
            CodexAppServerError,
            match=r"Invalid Codex model_reasoning_effort.*compatibility values",
        ):
            normalize_codex_reasoning_effort(
                configured,
                _known("low", "medium", "high", "xhigh"),
                model="gpt-5.5",
            )

    def test_directly_advertised_future_value_is_not_rejected(self):
        assert (
            normalize_codex_reasoning_effort(
                "super",
                _known("low", "medium", "super"),
                model="gpt-future",
            )
            == "super"
        )

    def test_unknown_capability_fails_closed(self):
        with pytest.raises(CodexAppServerError, match="ceiling could not be proved"):
            normalize_codex_reasoning_effort(
                "max",
                _unknown(),
                model="gpt-5.5",
            )

    def test_authoritative_no_effort_support_fails_distinctly(self):
        with pytest.raises(CodexAppServerError, match="explicitly advertises no"):
            normalize_codex_reasoning_effort(
                "high",
                _no_effort(),
                model="gpt-no-reasoning",
            )


class TestThreadStartEffortContract:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "requested_model",
            "configured_model",
            "configured_effort",
            "catalog_efforts",
            "expected_effort",
        ),
        (
            (
                "gpt-5.5",
                "gpt-5.6-sol",
                "max",
                ("low", "medium", "high", "xhigh"),
                "xhigh",
            ),
            (
                "gpt-5.6-sol",
                "gpt-5.6-sol",
                "max",
                ("low", "medium", "high", "xhigh", "max", "ultra"),
                "max",
            ),
            (
                "gpt-5.6-sol",
                "gpt-5.6-sol",
                "ultra",
                ("low", "medium", "high", "xhigh", "max", "ultra"),
                "ultra",
            ),
            (
                "auto",
                "gpt-5.5",
                "max",
                ("low", "medium", "high", "xhigh"),
                "xhigh",
            ),
            (
                "gpt-5.5",
                "gpt-5.5",
                None,
                ("low", "medium", "high", "xhigh"),
                None,
            ),
        ),
    )
    async def test_thread_start_uses_effective_config_and_model_ceiling(
        self,
        requested_model,
        configured_model,
        configured_effort,
        catalog_efforts,
        expected_effort,
    ):
        config = {"model": configured_model}
        if configured_effort is not None:
            config["model_reasoning_effort"] = configured_effort
        catalog = [_app_model(configured_model, catalog_efforts)]
        if requested_model not in ("auto", configured_model):
            catalog.append(_app_model(requested_model, catalog_efforts))
        app = _ConfigAwareAppServer(config, tuple(catalog))
        adapter = CodexAdapter()

        await adapter._ensure_thread(
            app,
            "session",
            requested_model,
            None,
            None,
        )

        expected_methods = ["config/read"]
        if configured_effort is not None:
            expected_methods.append("model/list")
        expected_methods.append("thread/start")
        assert [method for method, _ in app.requests] == expected_methods
        thread_params = app.requests[-1][1]
        if expected_effort is None:
            assert "model_reasoning_effort" not in thread_params["config"]
        else:
            assert thread_params["config"]["model_reasoning_effort"] == expected_effort

    @pytest.mark.asyncio
    async def test_invalid_effective_config_fails_before_thread_start(self):
        app = _ConfigAwareAppServer(
            {
                "model": "gpt-5.5",
                "model_reasoning_effort": "turbo",
            },
            (_app_model("gpt-5.5", ("low", "medium", "high", "xhigh")),),
        )
        adapter = CodexAdapter()

        with pytest.raises(CodexAppServerError, match="turbo"):
            await adapter._ensure_thread(
                app,
                "session",
                "gpt-5.5",
                None,
                None,
            )

        assert [method for method, _ in app.requests] == [
            "config/read",
            "model/list",
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "models",
        (
            (_app_model("gpt-other", ("low", "high")),),
            (_app_model("gpt-5.5", ()),),
            ({"id": "gpt-5.5", "model": "gpt-5.5"},),
        ),
        ids=("selected-model-missing", "explicit-no-support", "metadata-missing"),
    )
    async def test_unproved_or_absent_ceiling_fails_before_thread_start(
        self,
        models,
    ):
        app = _ConfigAwareAppServer(
            {
                "model": "gpt-5.5",
                "model_reasoning_effort": "max",
            },
            models,
        )
        adapter = CodexAdapter()

        with pytest.raises(CodexAppServerError, match="Cannot apply configured"):
            await adapter._ensure_thread(
                app,
                "session",
                "gpt-5.5",
                None,
                None,
            )

        assert [method for method, _ in app.requests] == [
            "config/read",
            "model/list",
        ]

    @pytest.mark.asyncio
    async def test_effort_change_invalidates_cached_thread(self):
        app = _ConfigAwareAppServer(
            {
                "model": "gpt-5.5",
                "model_reasoning_effort": "max",
            },
            (_app_model("gpt-5.5", ("low", "medium", "high", "xhigh")),),
        )
        adapter = CodexAdapter()

        await adapter._ensure_thread(app, "session", "gpt-5.5", None, None)
        app.config["model_reasoning_effort"] = "high"
        await adapter._ensure_thread(app, "session", "gpt-5.5", None, None)

        assert app.thread_starts == 2

    @pytest.mark.asyncio
    async def test_live_catalog_overrides_missing_or_stale_disk_cache(self):
        app = _ConfigAwareAppServer(
            {
                "model": "gpt-5.6-sol",
                "model_reasoning_effort": "max",
            },
            (_app_model("gpt-5.5", ("low", "medium", "high", "xhigh")),),
        )
        adapter = CodexAdapter()

        with patch.object(
            adapter,
            "_read_codex_models_cache",
            return_value=[
                _cached_model(
                    "gpt-5.5",
                    ("low", "medium", "high", "xhigh", "max", "ultra"),
                )
            ],
        ):
            await adapter._ensure_thread(
                app,
                "session",
                "gpt-5.5",
                None,
                None,
            )

        thread_params = app.requests[-1][1]
        assert thread_params["config"]["model_reasoning_effort"] == "xhigh"
        assert [method for method, _ in app.requests] == [
            "config/read",
            "model/list",
            "thread/start",
        ]

    @pytest.mark.asyncio
    async def test_live_catalog_resolves_app_server_default_on_cold_boot(self):
        app = _ConfigAwareAppServer(
            {"model_reasoning_effort": "max"},
            (
                _app_model(
                    "gpt-5.5",
                    ("low", "medium", "high", "xhigh"),
                    is_default=True,
                ),
            ),
        )
        adapter = CodexAdapter()

        with patch.object(adapter, "_read_codex_models_cache", return_value=[]):
            await adapter._ensure_thread(app, "session", "auto", None, None)

        assert app.requests[-1][1]["config"]["model_reasoning_effort"] == "xhigh"


class TestCapabilityEvidence:
    @pytest.mark.asyncio
    async def test_live_catalog_walks_pagination_before_disk_backup(self):
        requests = []

        class _PagedAppServer:
            async def request(self, method, params=None, *, timeout=120):
                params = params or {}
                requests.append((method, params))
                if method == "model/list" and "cursor" not in params:
                    return {
                        "data": [_app_model("gpt-other", ("low", "high"))],
                        "nextCursor": "page-2",
                    }
                if method == "model/list" and params.get("cursor") == "page-2":
                    return {
                        "data": [
                            _app_model(
                                "gpt-5.5",
                                ("low", "medium", "high", "xhigh"),
                            )
                        ],
                        "nextCursor": None,
                    }
                raise AssertionError((method, params))

        adapter = CodexAdapter()
        with patch.object(
            adapter,
            "_read_codex_models_cache",
            side_effect=AssertionError("disk cache should not be read"),
        ):
            capability = await resolve_codex_reasoning_capability(
                _PagedAppServer(),
                "gpt-5.5",
                adapter._read_codex_models_cache,
            )

        assert capability.state is CodexReasoningCapabilityState.KNOWN
        assert capability.efforts == ("low", "medium", "high", "xhigh")
        assert requests == [
            ("model/list", {"includeHidden": True, "limit": 100}),
            (
                "model/list",
                {"includeHidden": True, "limit": 100, "cursor": "page-2"},
            ),
        ]

    @pytest.mark.asyncio
    async def test_disk_cache_is_only_rpc_unavailability_backup(self):
        app = _ConfigAwareAppServer(
            {},
            model_list_error=CodexAppServerError("method unavailable"),
        )
        capability = await resolve_codex_reasoning_capability(
            app,
            "gpt-5.5",
            lambda: [
                _cached_model(
                    "gpt-5.5",
                    ("low", "medium", "high", "xhigh"),
                )
            ],
        )

        assert capability.state is CodexReasoningCapabilityState.KNOWN
        assert capability.efforts[-1] == "xhigh"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cached_models",
        (
            [],
            [_cached_model("gpt-other", ("low", "high"))],
            [_cached_model("gpt-5.5", (), include_metadata=False)],
        ),
        ids=("empty", "selected-model-missing", "metadata-missing"),
    )
    async def test_rpc_unavailable_without_explicit_disk_metadata_is_unknown(
        self,
        cached_models,
    ):
        app = _ConfigAwareAppServer(
            {},
            model_list_error=CodexAppServerError("method unavailable"),
        )
        capability = await resolve_codex_reasoning_capability(
            app,
            "gpt-5.5",
            lambda: cached_models,
        )

        assert capability.state is CodexReasoningCapabilityState.UNKNOWN

    @pytest.mark.asyncio
    async def test_successful_live_missing_model_never_uses_stale_disk_row(self):
        app = _ConfigAwareAppServer(
            {},
            (_app_model("gpt-other", ("low", "high")),),
        )
        capability = await resolve_codex_reasoning_capability(
            app,
            "gpt-5.5",
            lambda: [_cached_model("gpt-5.5", ("low", "high", "xhigh"))],
        )

        assert capability.state is CodexReasoningCapabilityState.UNKNOWN
        assert capability.source == "live model/list"

    @pytest.mark.asyncio
    async def test_successful_live_empty_efforts_is_authoritative_no_support(self):
        app = _ConfigAwareAppServer({}, (_app_model("gpt-5.5", ()),))
        capability = await resolve_codex_reasoning_capability(
            app,
            "gpt-5.5",
            lambda: [_cached_model("gpt-5.5", ("high", "xhigh"))],
        )

        assert capability.state is CodexReasoningCapabilityState.NO_EFFORT


class TestFullTurnSettingsFreeze:
    @pytest.mark.asyncio
    async def test_run_turn_normalizes_both_rpc_boundaries_and_collaboration(self):
        app = _ConfigAwareAppServer(
            {
                "model": "gpt-5.6-sol",
                "model_reasoning_effort": "max",
            },
            (_app_model("gpt-5.5", ("low", "medium", "high", "xhigh")),),
        )
        adapter = CodexAdapter()
        adapter._client = app

        with patch.object(
            adapter,
            "_read_codex_models_cache",
            return_value=[_cached_model("gpt-5.5", ("low", "high", "xhigh"))],
        ):
            response = await adapter.get_response(
                client=None,
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "You are Kestrel."},
                    {"role": "user", "content": "hi"},
                    {"role": "system", "content": "Operator context."},
                ],
                session_id="full-turn-normalized",
                keep_trailing_system=True,
            )

        assert response.content == "ok"
        thread_params = _request_params(app, "thread/start")[0]
        turn_params = _request_params(app, "turn/start")[0]
        assert thread_params["model"] == turn_params["model"] == "gpt-5.5"
        assert thread_params["config"]["model_reasoning_effort"] == "xhigh"
        assert turn_params["effort"] == "xhigh"
        assert (
            turn_params["collaborationMode"]["settings"]["reasoning_effort"] == "xhigh"
        )

    @pytest.mark.asyncio
    async def test_run_turn_freezes_stale_cache_exclusion_across_refresh(self):
        cache = {
            "models": [
                _cached_model(
                    "gpt-5.6-sol",
                    ("low", "high", "xhigh", "max", "ultra"),
                )
            ]
        }
        cache_reads = []

        def read_cache():
            cache_reads.append(tuple(row["slug"] for row in cache["models"]))
            return cache["models"]

        def refresh_cache_after_live_catalog():
            # Exact regression: the first serveability read excludes requested
            # GPT-5.5, then the live catalog refresh makes it available before
            # turn/start. The frozen selection must remain unchanged.
            cache["models"] = [
                *cache["models"],
                _cached_model("gpt-5.5", ("low", "high", "xhigh")),
            ]

        app = _ConfigAwareAppServer(
            {
                "model": "gpt-5.6-sol",
                "model_reasoning_effort": "ultra",
            },
            (
                _app_model(
                    "gpt-5.6-sol",
                    ("low", "medium", "high", "xhigh", "max", "ultra"),
                ),
                _app_model("gpt-5.5", ("low", "medium", "high", "xhigh")),
            ),
            on_model_list=refresh_cache_after_live_catalog,
        )
        adapter = CodexAdapter()
        adapter._client = app

        from kestrel_sovereign.llm import codex_adapter as codex_adapter_module

        real_await_or_cancelled = codex_adapter_module.await_or_cancelled
        lock_acquires = 0

        async def acquire_then_invalidate(awaitable, cancel_token):
            nonlocal lock_acquires
            result = await real_await_or_cancelled(awaitable, cancel_token)
            lock_acquires += 1
            if lock_acquires == 1:
                # Force the same-session poisoned-thread re-resolution path
                # after the first lock is acquired and after model/list has
                # refreshed the disk cache.
                adapter._session_threads.pop("stale-cache-refresh", None)
            return result

        with (
            patch.object(
                adapter,
                "_read_codex_models_cache",
                side_effect=read_cache,
            ),
            patch.object(
                codex_adapter_module,
                "await_or_cancelled",
                side_effect=acquire_then_invalidate,
            ),
        ):
            response = await adapter.get_response(
                client=None,
                model="gpt-5.5",
                messages=[{"role": "user", "content": "hi"}],
                session_id="stale-cache-refresh",
            )

        assert response.content == "ok"
        thread_starts = _request_params(app, "thread/start")
        assert len(thread_starts) == 2
        turn_params = _request_params(app, "turn/start")[0]
        assert all("model" not in params for params in thread_starts)
        assert "model" not in turn_params
        assert all(
            params["config"]["model_reasoning_effort"] == "ultra"
            for params in thread_starts
        )
        assert turn_params["effort"] == "ultra"
        assert cache_reads == [("gpt-5.6-sol",)]
