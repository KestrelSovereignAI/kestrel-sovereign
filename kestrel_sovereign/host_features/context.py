"""Concrete :class:`~kestrel_sdk.features.host_base.HostContext` for the host.

A host feature receives a ``HostContext`` at ``on_host_start`` — the
fleet-scoped analogue of the ``agent`` a per-agent ``Feature`` binds to. The
SDK defines the ``HostContext`` *Protocol* (``db`` / ``backplane`` / ``config``)
and stays dependency-free; sovereign provides the concrete implementation here
and additionally hands features an **entities session factory bound to a host
backend under a fleet tenant scope** (issue #2293 acceptance criterion).

Fleet tenancy: per-agent stores are tenant-scoped to the owning agent. A host
feature's store is fleet-wide, so its sessions run under a single reserved
fleet tenant id (:data:`FLEET_TENANT_ID`). When ``kestrel-feature-entities`` is
installed (Phase 2 layers the entities ORM on top), the factory enters
``TenantContext.use(FLEET_TENANT_ID)`` around each session so entity queries are
transparently fleet-scoped; when it is absent the factory still yields plain
sessions against the host backend (SQLite default), so Phase 1 works standalone.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Reserved tenant id for fleet/host-scoped stores. Distinct from any agent's
#: tenant id (agents scope by DID / agent id), so a host feature's entities
#: never collide with per-agent rows.
FLEET_TENANT_ID = "__fleet__"


class FleetSessionFactory:
    """Wrap a session factory so every session runs under the fleet tenant.

    Duck-types the ``read_session()`` / ``write_session()`` / ``engine`` /
    ``close()`` shape sovereign's SQLA factory exposes (see
    :class:`kestrel_sovereign.storage.sqla.session.SovereignSqlaSessionFactory`).
    Each session context enters ``TenantContext.use(FLEET_TENANT_ID)`` when the
    entities ORM is available, so host-feature entity queries are fleet-scoped
    without the feature having to manage tenancy itself.
    """

    def __init__(self, inner: Any, tenant_id: str = FLEET_TENANT_ID) -> None:
        self._inner = inner
        self._tenant_id = tenant_id

    @property
    def engine(self) -> Any:
        return self._inner.engine

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @staticmethod
    def _tenant_scope(tenant_id: str):
        """Return a context manager applying the fleet tenant, or a no-op.

        Imported lazily so the SDK/host stay decoupled from
        ``kestrel-feature-entities`` — absent it, tenancy is a no-op (the host
        backend is single-tenant by construction).
        """
        try:
            from kestrel_feature_entities import TenantContext  # type: ignore

            return TenantContext.use(tenant_id)
        except Exception:  # noqa: BLE001 - entities not installed / older shape

            @asynccontextmanager
            async def _noop():
                yield

            return _noop()

    @asynccontextmanager
    async def read_session(self):
        scope = self._tenant_scope(self._tenant_id)
        _enter = getattr(scope, "__enter__", None)
        if _enter is not None:  # sync CM (entities TenantContext.use)
            scope.__enter__()
            try:
                async with self._inner.read_session() as session:
                    yield session
            finally:
                scope.__exit__(None, None, None)
        else:  # async no-op CM
            async with scope:
                async with self._inner.read_session() as session:
                    yield session

    @asynccontextmanager
    async def write_session(self):
        scope = self._tenant_scope(self._tenant_id)
        _enter = getattr(scope, "__enter__", None)
        if _enter is not None:
            scope.__enter__()
            try:
                async with self._inner.write_session() as session:
                    yield session
            finally:
                scope.__exit__(None, None, None)
        else:
            async with scope:
                async with self._inner.write_session() as session:
                    yield session

    async def close(self) -> None:
        closer = getattr(self._inner, "close", None)
        if callable(closer):
            await closer()


class SovereignHostContext:
    """Concrete ``HostContext`` implementation handed to host features.

    Satisfies the SDK ``HostContext`` Protocol (``db`` / ``backplane`` /
    ``config``) and additionally exposes :attr:`session_factory` — the fleet
    tenant-scoped entities session factory bound to the host backend — and
    :attr:`fleet_tenant_id`.
    """

    def __init__(
        self,
        *,
        db: Any = None,
        backplane: Any = None,
        config: Any = None,
        session_factory: Optional[FleetSessionFactory] = None,
        hold_store: Any = None,
        hold_db: Any = None,
        hold_evidence_db: Any = None,
        hold_boot_state: tuple[Any, ...] = (),
        backend_error: str = "",
    ) -> None:
        self._db = db
        self._backplane = backplane
        self._config = config if config is not None else {}
        self._session_factory = session_factory
        self._hold_store = hold_store
        self._hold_db = hold_db
        self._hold_evidence_db = hold_evidence_db
        self._hold_boot_state = tuple(hold_boot_state)
        self._backend_error = str(backend_error or "")

    @property
    def db(self) -> Any:
        return self._db

    @property
    def backplane(self) -> Any:
        return self._backplane

    @property
    def config(self) -> Any:
        return self._config

    @property
    def session_factory(self) -> Optional[FleetSessionFactory]:
        """Fleet tenant-scoped session factory on the host backend."""
        return self._session_factory

    @property
    def hold_store(self) -> Any:
        """Durable host/agent Hold latches on the host control backend."""

        return self._hold_store

    @property
    def hold_db(self) -> Any:
        """Backend owned solely by Hold, or :attr:`db` when they coincide."""

        return self._hold_db

    @property
    def hold_evidence_db(self) -> Any:
        """Independent PostgreSQL rollback witness backend, when configured."""

        return self._hold_evidence_db

    @property
    def hold_boot_state(self) -> tuple[Any, ...]:
        """Validated active latches observed before work admission opens."""

        return self._hold_boot_state

    @property
    def backend_error(self) -> str:
        """Why :attr:`db` is None, when it is, and empty otherwise.

        The host is built to start without a store, so a backend that cannot
        open is a capability gap rather than an identity one and must not abort
        boot. But a capability gap has to be NAMED. #3058: the open failed on a
        schema migration, this constructor was handed ``db=None``, the
        Workflows host feature's start hook refused for want of a database, the
        feature was dropped whole, Talon's router refused without it, and every
        surface downstream reported an empty list on a host answering ok.
        Carrying the reason turns the diagnosis from "read the boot log" into
        one field beside the symptom.
        """
        return self._backend_error

    @property
    def fleet_tenant_id(self) -> str:
        return FLEET_TENANT_ID


async def _close_partial_host_resources(
    session_factory: Optional[FleetSessionFactory],
    hold_evidence_db: Any,
    hold_db: Any,
    db: Any,
) -> None:
    """Close every resource acquired before host bootstrap completed."""

    cancelled = False
    if session_factory is not None:
        try:
            await session_factory.close()
        except asyncio.CancelledError:
            cancelled = True
        except Exception as close_exc:  # noqa: BLE001 - finish later resources
            logger.warning("Could not close partial host session factory: %s", close_exc)
    if (
        hold_evidence_db is not None
        and hold_evidence_db is not hold_db
        and hold_evidence_db is not db
        and hasattr(hold_evidence_db, "close")
    ):
        try:
            await hold_evidence_db.close()
        except asyncio.CancelledError:
            cancelled = True
        except Exception as close_exc:  # noqa: BLE001 - finish later resources
            logger.warning(
                "Could not close partial Hold evidence backend: %s", close_exc
            )
    if hold_db is not None and hold_db is not db and hasattr(hold_db, "close"):
        try:
            await hold_db.close()
        except asyncio.CancelledError:
            cancelled = True
        except Exception as close_exc:  # noqa: BLE001 - finish later resources
            logger.warning("Could not close partial Hold backend: %s", close_exc)
    if db is not None and hasattr(db, "close"):
        try:
            await db.close()
        except asyncio.CancelledError:
            cancelled = True
        except Exception as close_exc:  # noqa: BLE001 - finish later resources
            logger.warning("Could not close partial host backend: %s", close_exc)
    if cancelled:
        raise asyncio.CancelledError()


async def _finish_partial_host_cleanup(
    session_factory: Optional[FleetSessionFactory],
    hold_evidence_db: Any,
    hold_db: Any,
    db: Any,
) -> None:
    """Own partial bootstrap cleanup through repeated caller cancellation."""

    cleanup = asyncio.create_task(
        _close_partial_host_resources(
            session_factory,
            hold_evidence_db,
            hold_db,
            db,
        ),
        name="partial-host-bootstrap-cleanup",
    )
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # A supervisor may cancel shutdown more than once. The resource
            # owner is independent, so every acquired backend still closes.
            cancelled = True
            continue
    await cleanup
    if cancelled:
        raise asyncio.CancelledError()


async def close_host_context_resources(context: Any) -> None:
    """Cancellation-safely close all databases owned by one host context."""

    if context is None:
        return
    await _finish_partial_host_cleanup(
        getattr(context, "session_factory", None),
        getattr(context, "hold_evidence_db", None),
        getattr(context, "hold_db", None),
        getattr(context, "db", None),
    )


async def build_host_context(
    *,
    config: Any = None,
    db_path: Optional[str] = None,
) -> SovereignHostContext:
    """Build the host context: open a host backend + fleet session factory.

    The established host-feature backend remains the dedicated SQLite file so
    an upgrade cannot make existing workflow/feature rows disappear. Hold is
    the cross-worker control plane exception: PostgreSQL deployments give it a
    separate ``KESTREL_DATABASE_URL`` backend plus an independently restored
    ``KESTREL_HOLD_EVIDENCE_DATABASE_URL``. ``db_path`` overrides the host
    SQLite location (``$KESTREL_HOST_DB_PATH`` or the private host-data root).
    Failure to secure or open either backend degrades gracefully to a context
    with no store; production Hold enforcement then fails closed at boot.
    """
    db = None
    session_factory: Optional[FleetSessionFactory] = None
    hold_store = None
    hold_db = None
    hold_evidence_db = None
    hold_boot_state: tuple[Any, ...] = ()
    backend_error = ""
    try:
        from kestrel_sovereign.host_features.storage import (
            prepare_host_database,
            validate_sqlite_family_private,
        )
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.sqla.session import make_session_factory
        from kestrel_sovereign.hold import HoldStore
        from kestrel_sovereign.hold.state import (
            hold_history_anchor_path,
            hold_initialization_witness_path,
        )

        resolved = prepare_host_database(db_path)
        db = await AsyncDatabase.sqlite(str(resolved))
        validate_sqlite_family_private(resolved)
        inner = make_session_factory(db)
        session_factory = FleetSessionFactory(inner)

        backend = os.environ.get("KESTREL_DB_BACKEND", "sqlite").lower()
        dsn = os.environ.get("KESTREL_DATABASE_URL")
        if backend == "postgres" and dsn:
            evidence_dsn = os.environ.get("KESTREL_HOLD_EVIDENCE_DATABASE_URL")
            if not evidence_dsn:
                raise RuntimeError(
                    "KESTREL_HOLD_EVIDENCE_DATABASE_URL is required for "
                    "PostgreSQL Hold rollback evidence"
                )
            if evidence_dsn == dsn:
                raise RuntimeError(
                    "KESTREL_HOLD_EVIDENCE_DATABASE_URL must identify an "
                    "independent rollback domain"
                )
            # Hold operations are serialized by their independent evidence
            # protocol, so wider pools add connection demand without adding
            # useful concurrency. Keep both pools load-bearingly small: the
            # ordinary runtime already reserves its own PostgreSQL budget.
            hold_db = await AsyncDatabase.postgres(
                dsn,
                min_pool_size=1,
                max_pool_size=1,
            )
            hold_evidence_db = await AsyncDatabase.postgres(
                evidence_dsn,
                min_pool_size=1,
                max_pool_size=1,
            )
            hold_location = "configured PostgreSQL database"
            initialization_witness_path = None
            history_anchor_path = None
        else:
            hold_db = db
            hold_location = str(resolved)
            initialization_witness_path = hold_initialization_witness_path(
                resolved
            )
            history_anchor_path = hold_history_anchor_path(resolved)
        hold_store = HoldStore(
            hold_db,
            initialization_witness_path=initialization_witness_path,
            history_anchor_path=history_anchor_path,
            evidence_db=hold_evidence_db,
        )
        await hold_store.ensure_schema()
        hold_boot_state = await hold_store.read_boot_state()
        logger.info(
            "Host backend opened at %s (fleet tenant=%s); Hold backend=%s",
            resolved,
            FLEET_TENANT_ID,
            hold_location,
        )
    except asyncio.CancelledError as cancellation:
        await _finish_partial_host_cleanup(
            session_factory,
            hold_evidence_db,
            hold_db,
            db,
        )
        raise cancellation
    except Exception as exc:  # noqa: BLE001 - host must start even without a store
        # Cleanup owns its task independently so cancellation arriving after
        # the opening failure closes every acquired backend and then
        # propagates instead of returning a degraded context during shutdown.
        await _finish_partial_host_cleanup(
            session_factory,
            hold_evidence_db,
            hold_db,
            db,
        )
        session_factory = None
        hold_store = None
        hold_evidence_db = None
        hold_db = None
        db = None
        backend_error = f"{type(exc).__name__}: {exc}"
        # ERROR, not warning: everything that depends on the host store is
        # about to be dropped, and each of those drops reports an empty
        # result rather than a failure (#3058).
        logger.error("Could not open host backend/session factory: %s", exc)

    return SovereignHostContext(
        db=db,
        backplane=None,
        config=config,
        session_factory=session_factory,
        hold_store=hold_store,
        hold_db=hold_db,
        hold_evidence_db=hold_evidence_db,
        hold_boot_state=hold_boot_state,
        backend_error=backend_error,
    )


__all__ = [
    "FLEET_TENANT_ID",
    "FleetSessionFactory",
    "SovereignHostContext",
    "build_host_context",
    "close_host_context_resources",
]
