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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from kestrel_sovereign.host_features.storage import HostDatabaseLaunchContext

from kestrel_sovereign._async_ownership import OwnedTaskAdmission, await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    execution_cleanup_debts,
    execution_native_source_error,
    execution_terminal_error,
    is_execution_control_error,
)
from kestrel_sovereign.lifecycle_checks import (
    is_isolated_nonproduction_kite_environment,
)

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
        self._native_close_owners: dict[object, _HostCloseOwner] = {}

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


@dataclass
class _HostCloseOwner:
    """One original resource and its close operation, never a proxy ACK."""

    resource: Any
    task: asyncio.Task | None = None
    error: BaseException | None = None
    closed: bool = False


def _host_close_owners(context: Any) -> dict[object, _HostCloseOwner]:
    """One context-owned inventory for original resource retirement failures."""
    owners = getattr(context, "_native_close_owners", None)
    if owners is None:
        owners = context._native_close_owners = {}
    return owners


def _retain_host_cleanup_failure(
    context: Any, resource: Any, error: BaseException, *, operation: str,
) -> None:
    # A feature may also be a database resource. Operation-qualified keys must
    # not confuse its failed stop hook with the independent resource.close.
    record = _host_close_owners(context).setdefault(
        (operation, id(resource)), _HostCloseOwner(resource)
    )
    record.error = execution_terminal_error(
        record.error, execution_native_source_error(error) or error,
        retain_originals=True, isolate_control_originals=True,
    )


def _retained_host_cleanup_error(context: Any) -> BaseException | None:
    return execution_terminal_error(
        *(record.error for record in _host_close_owners(context).values()),
        retain_originals=True, isolate_control_originals=True,
    )


async def _close_partial_host_resources(
    session_factory: Optional[FleetSessionFactory],
    hold_evidence_db: Any,
    hold_db: Any,
    db: Any,
    *,
    owners: dict[object, _HostCloseOwner],
    prior_error: BaseException | None = None,
) -> None:
    """Retire each ORIGINAL resource independently, retaining native failures.

    The caller already owns this cleanup. No all-resource collector Task is
    needed. Register each actual resource BEFORE close-Task admission, join
    every accepted close, and keep failed/unaccepted owners for reconciliation.
    Session factories still retire before their borrowed databases.
    """
    resources = []
    for resource in (session_factory, hold_evidence_db, hold_db, db):
        if resource is not None and callable(getattr(resource, "close", None)):
            if not any(resource is previous for previous in resources):
                resources.append(resource)
    errors = []
    for resource in resources:
        record = owners.setdefault(id(resource), _HostCloseOwner(resource))
        if record.closed:
            continue
        if record.task is None and record.error is None:
            admission = OwnedTaskAdmission()
            try:
                record.task = create_owned_task(
                    resource.close(),
                    retirement_resource=resource,
                    retain_originals=True,
                    admission=admission,
                    name="host-original-resource-close",
                )
            except BaseException as error:
                # Canonical admission closes refused coroutines and attaches
                # the ACTUAL resource's unstarted retirement debt.
                record.error = error
                record.task = admission.task
                debt = RuntimeError("original host closer admission failed")
                debt.__cause__ = execution_native_source_error(error) or error
                debt._kestrel_native_retirement_debts = ((resource, record.task),)
                record.error = execution_terminal_error(
                    error, debt, retain_originals=True, isolate_control_originals=True,
                )
        if record.task is not None:
            outcome = await await_owned_task(record.task)
            from kestrel_sovereign._async_ownership import owned_task_error
            physical = owned_task_error(record.task)
            record.error = execution_terminal_error(
                record.error, execution_native_source_error(physical) or physical,
                retain_originals=True,
                isolate_control_originals=True,
            )
            # Join publication/removal failures belong to this finite
            # observation, not the physical closer's sticky disposition.
            errors.extend((outcome.error, outcome.cancellation))
            if record.error is None:
                record.closed = True
                record.task = None
        if record.error is not None:
            if not execution_cleanup_debts(record.error):
                # An unsuccessful close is not physical retirement. Retain its
                # ORIGINAL resource/operation rather than a successful holder.
                debt = RuntimeError("original host close retirement failed")
                debt.__cause__ = execution_native_source_error(record.error) or record.error
                debt._kestrel_native_retirement_debts = ((resource, record.task),)
                record.error = execution_terminal_error(
                    record.error, debt, retain_originals=True,
                    isolate_control_originals=True,
                )
    # Feature stop and declarative rollback share this SAME inventory. Closing
    # their independent databases cannot acknowledge an earlier failed hook.
    errors.extend(record.error for record in owners.values())
    if any(error is not None for error in errors):
        raise execution_terminal_error(
            prior_error, *errors, retain_originals=True, prefer_cancellation=True,
            isolate_control_originals=True,
        )


async def _finish_partial_host_cleanup(
    session_factory: Optional[FleetSessionFactory],
    hold_evidence_db: Any,
    hold_db: Any,
    db: Any,
    *,
    owners: dict[object, _HostCloseOwner] | None = None,
    prior_error: BaseException | None = None,
) -> None:
    """The original caller joins each resource, including repeated Stop."""
    await _close_partial_host_resources(
        session_factory,
        hold_evidence_db,
        hold_db,
        db,
        owners=owners if owners is not None else {},
        prior_error=prior_error,
    )


async def close_host_context_resources(context: Any) -> None:
    """Cancellation-safely close all databases owned by one host context."""

    if context is None:
        return
    owners = _host_close_owners(context)
    await _finish_partial_host_cleanup(
        getattr(context, "session_factory", None),
        getattr(context, "hold_evidence_db", None),
        getattr(context, "hold_db", None),
        getattr(context, "db", None),
        owners=owners,
    )


async def build_host_context(
    *,
    config: Any = None,
    db_path: Optional[str] = None,
    host_database_launch_context: Optional[HostDatabaseLaunchContext] = None,
) -> SovereignHostContext:
    """Build the host context: open a host backend + fleet session factory.

    The established host-feature backend remains the dedicated SQLite file so
    an upgrade cannot make existing workflow/feature rows disappear. Hold is
    the cross-worker control plane exception: PostgreSQL deployments give it a
    separate ``KESTREL_DATABASE_URL`` backend plus an independently restored
    ``KESTREL_HOLD_EVIDENCE_DATABASE_URL``. An explicitly isolated single-host
    workload may select the already-owned SQLite control plane with
    ``KESTREL_HOLD_BACKEND=sqlite``. ``db_path`` overrides the host SQLite
    location (``$KESTREL_HOST_DB_PATH``, otherwise
    ``$KESTREL_DB_PATH/host-data``, otherwise the private host-data root).
    Ordinary capability failures can return a degraded context with no store;
    production Hold enforcement then fails closed at boot. Original execution
    authority, commit-outcome and native retirement controls instead propagate
    after the original acquired resources have had their cleanup attempts.
    """
    db = None
    session_factory: Optional[FleetSessionFactory] = None
    hold_store = None
    hold_db = None
    hold_evidence_db = None
    postgres_pair_id = None
    hold_boot_state: tuple[Any, ...] = ()
    backend_error = ""
    try:
        from kestrel_sovereign.hold import HoldStore
        from kestrel_sovereign.hold.state import (
            claim_hold_backend_custody,
            commit_postgres_hold_pair_custody,
            configured_postgres_hold_pair_id,
            hold_history_anchor_path,
            hold_initialization_witness_path,
            initialize_postgres_hold_databases,
            validate_hold_backend_custody,
        )
        from kestrel_sovereign.host_features.storage import (
            prepare_host_database,
            resolve_host_database_launch_context,
            validate_sqlite_family_private,
        )
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.storage.sqla.session import make_session_factory

        if db_path is not None and host_database_launch_context is not None:
            raise ValueError(
                "db_path and host_database_launch_context are mutually exclusive"
            )
        launch_context = host_database_launch_context
        if launch_context is None:
            launch_context = resolve_host_database_launch_context(
                env=os.environ,
                db_path=db_path,
            )
        launch_env = launch_context.backend_env()

        # Configuration validation is a read-only preflight. In particular it
        # must precede ``prepare_host_database``: that function may securely
        # create the SQLite file or migrate a legacy store, and a rejected Hold
        # configuration has no authority to mutate either one.
        backend = launch_env.get("KESTREL_DB_BACKEND", "sqlite").lower()
        dsn = launch_env.get("KESTREL_DATABASE_URL")
        configured_hold_backend = launch_env.get("KESTREL_HOLD_BACKEND")
        evidence_dsn = None
        external_pair_id = None
        if configured_hold_backend is None:
            hold_backend = "postgres" if backend == "postgres" and dsn else "sqlite"
        else:
            hold_backend = configured_hold_backend.lower()
            if hold_backend not in {"postgres", "sqlite"}:
                raise RuntimeError(
                    "KESTREL_HOLD_BACKEND must be 'postgres' or 'sqlite'"
                )
            if (
                hold_backend == "sqlite"
                and backend == "postgres"
                and not is_isolated_nonproduction_kite_environment(launch_env)
            ):
                raise RuntimeError(
                    "PostgreSQL runtimes may select SQLite Hold only inside "
                    "isolated Kite release evidence from an isolated "
                    "non-production Kite demo"
                )
        if hold_backend == "postgres":
            if not dsn:
                raise RuntimeError(
                    "KESTREL_DATABASE_URL is required for PostgreSQL Hold state"
                )
            evidence_dsn = launch_env.get("KESTREL_HOLD_EVIDENCE_DATABASE_URL")
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
            external_pair_id = configured_postgres_hold_pair_id(
                launch_env,
                required=(
                    launch_env.get("KESTREL_DEPLOYMENT_PERSISTENCE", "").strip().lower()
                    == "durable_sovereign"
                ),
            )

        # Surviving custody evidence is authoritative even when the selected
        # SQLite file is absent or old.  Resolve and validate it before
        # preparation can create, harden, migrate, or initialize host storage.
        preflight_path = launch_context.database_path
        validate_hold_backend_custody(preflight_path, hold_backend)

        resolved = prepare_host_database(launch_context=launch_context)
        db = await AsyncDatabase.sqlite(str(resolved))
        validate_sqlite_family_private(resolved)
        inner = make_session_factory(db)
        session_factory = FleetSessionFactory(inner)

        if hold_backend == "postgres":
            # Repeat beneath the preparation boundary so a concurrent custody
            # publication cannot race the read-only preflight.
            validate_hold_backend_custody(resolved, hold_backend)
            # Hold operations are serialized by their independent evidence
            # protocol, so wider pools add connection demand without adding
            # useful concurrency. The paired initializer keeps both pools
            # load-bearingly small and, critically, initializes schema on the
            # same connected backends whose cluster identity and custody roles
            # it inspected. Reopening by DSN here would create a failover/
            # load-balancer window between validation and the first write.
            initializer_kwargs = {"control_db_path": resolved}
            if external_pair_id is not None:
                initializer_kwargs["expected_external_pair_id"] = external_pair_id
            (
                hold_db,
                hold_evidence_db,
                postgres_pair_id,
            ) = await initialize_postgres_hold_databases(
                dsn,
                evidence_dsn,
                **initializer_kwargs,
            )
            hold_location = "configured PostgreSQL database"
            initialization_witness_path = None
            history_anchor_path = None
        else:
            claim_hold_backend_custody(resolved, hold_backend)
            hold_db = db
            hold_location = str(resolved)
            initialization_witness_path = hold_initialization_witness_path(resolved)
            history_anchor_path = hold_history_anchor_path(resolved)
        hold_store = HoldStore(
            hold_db,
            initialization_witness_path=initialization_witness_path,
            history_anchor_path=history_anchor_path,
            evidence_db=hold_evidence_db,
            expected_postgres_pair_id=(
                postgres_pair_id if hold_backend == "postgres" else None
            ),
        )
        await hold_store.ensure_schema()
        if hold_backend == "postgres":
            commit_postgres_hold_pair_custody(resolved, postgres_pair_id)
        hold_boot_state = await hold_store.read_boot_state()
        logger.info(
            "Host backend opened at %s (fleet tenant=%s); Hold backend=%s",
            resolved,
            FLEET_TENANT_ID,
            hold_location,
        )
    except BaseException as exc:
        # Cleanup owns each original resource independently so cancellation
        # arriving after the opening failure closes every acquired backend and
        # propagates instead of returning a degraded context during shutdown.
        await _finish_partial_host_cleanup(
            session_factory,
            hold_evidence_db,
            hold_db,
            db,
            prior_error=exc,
        )
        if not isinstance(exc, Exception) or is_execution_control_error(exc):
            raise
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


async def close_host_context(ctx: Any) -> None:
    """Compatibility name for cancellation-safe host resource cleanup."""

    await close_host_context_resources(ctx)


__all__ = [
    "FLEET_TENANT_ID",
    "FleetSessionFactory",
    "SovereignHostContext",
    "build_host_context",
    "close_host_context_resources",
    "close_host_context",
]
