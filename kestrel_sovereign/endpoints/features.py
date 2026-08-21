"""Feature Store API — catalog, install, enable, disable, configure features."""

import asyncio
import inspect
import logging
import subprocess
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from kestrel_sovereign.endpoints.agent_helpers import get_agent
from kestrel_sovereign.feature_registry import (
    FeaturePackageInfo,
    FeatureStatus,
    get_all_skills,
    get_package_for_feature,
    get_registry,
    get_skills_for_package,
)
from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES
from kestrel_sovereign.ui_capabilities import (
    active_feature_class_names,
    compute_feature_capabilities,
)
from kestrel_sovereign.ui_contributions import compute_ui_manifest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["features"])

# Every installer subprocess started from an HTTP request is bounded. Nothing on
# the other end of the socket can interrupt one, so an unbounded call is not a
# slow request, it is a wedged one. This bounds the feature install AND the core
# repair that may follow it (issue #2949) — the repair runs a second installer,
# and the resolver or network problem that hung the first hangs it identically.
#
# The repair gets its OWN budget rather than the remainder of the install's: a
# timed-out install is precisely when core is most likely to have been left
# swapped mid-write, so starving the restore there would trade a hung request
# for a host running a core nobody declared. The bound is per installer
# subprocess, so the worst case is a small multiple of it — two here, and one
# more on a host with no uv, where a scoped reinstall takes two passes
# (`cli_features._install_commands`). Every path still terminates, which is the
# property a caller with nobody watching needs; the multiple is the price.
INSTALL_TIMEOUT_SECONDS = 300

def _requested_extras(package_spec: str) -> Tuple[str, ...]:
    """The extras named in a spec like ``pkg[voice,web]>=1.2``."""
    if "[" not in package_spec or "]" not in package_spec:
        return ()
    inside = package_spec.split("[", 1)[1].split("]", 1)[0]
    return tuple(part.strip() for part in inside.split(",") if part.strip())


def _requirement_applies(req, extras: Tuple[str, ...]) -> bool:
    """Whether *req* is active for this interpreter and these extras.

    An unmarked requirement always applies. A marked one applies if it
    evaluates true in the base environment or under any extra the caller
    requested — `packaging` evaluates ``extra == "x"`` to False when no extra is
    supplied, so the base environment alone would silently drop every
    extra-gated requirement.
    """
    if req.marker is None:
        return True
    for env in ({}, *({"extra": extra} for extra in extras)):
        try:
            if req.marker.evaluate(env):
                return True
        except Exception:  # noqa: BLE001
            # An undefined marker name tells us nothing either way; a later
            # environment may still resolve it.
            continue
    return False


def _core_requirement_unsatisfied(package_spec: str) -> Optional[str]:
    """Describe *package_spec*'s unmet requirement on core, or None.

    Asked after a repair: the restored core may be exactly the version the
    just-installed package could not accept. Reads the installed metadata rather
    than the install command, because the requirement that matters is the one
    the artifact on disk actually declares.

    Only requirements whose environment marker is ACTIVE are considered. A
    conditional core dependency (``kestrel-sovereign>=0.60; python_version <
    "3.10"``) is not a requirement of THIS interpreter, and reporting it as
    unmet turned a healthy install into a 500. Markers gated on an extra are
    evaluated against the extras the caller actually asked for, since that is
    what decides whether such a requirement applies at all.

    Any lookup failure returns None — a diagnostic that cannot read the
    metadata must not manufacture a failure for a package that may be fine.
    This does not weaken the fail-closed rule that governs the install guard:
    core's conformance to its declared source is verified independently and has
    already passed by the time this is asked. This only decides whether to
    upgrade an otherwise-successful response to an error, so "cannot tell"
    must not become "cannot load".
    """
    import importlib.metadata as md

    from kestrel_sovereign.feature_reconcile import (
        CORE_DISTRIBUTION,
        canonical_package,
        version_satisfies,
    )

    try:
        from packaging.requirements import Requirement

        name = canonical_package(package_spec.split("[")[0].split("=")[0].strip())
        extras = _requested_extras(package_spec)
        requires = md.metadata(name).get_all("Requires-Dist") or []
        core_version = md.version(CORE_DISTRIBUTION)
    except Exception:  # noqa: BLE001
        return None

    for raw in requires:
        try:
            req = Requirement(raw)
        except Exception:  # noqa: BLE001
            continue
        if canonical_package(req.name) != CORE_DISTRIBUTION:
            continue
        if not _requirement_applies(req, extras):
            continue
        spec = str(req.specifier)
        if spec and not version_satisfies(core_version, spec):
            return (
                f"{name} requiring {CORE_DISTRIBUTION}{spec} against the "
                f"restored {core_version}"
            )
    return None


#: Serializes the whole snapshot -> install -> resolve sequence.
#
# Two overlapping install requests otherwise run it concurrently in worker
# threads against ONE environment, and both halves break:
#
#   * the installs themselves — on a host without uv the fallback is a
#     multi-pass pip sequence, and concurrent pip writes to one environment are
#     unsupported and can leave package metadata corrupt;
#   * the guard — each request snapshots core before its install and compares
#     after, so B's install lands inside A's window and A reports it as drift,
#     "repairs" a core nobody moved, and B then sees THAT as drift. Two correct
#     installs manufacture two spurious CORE_UNSAFE verdicts between them.
#
# In-process only, and deliberately named as such: agents share one host
# process, so this covers agent-vs-agent, which is the reachable case on a
# multi-agent host where features can self-install. It does NOT serialize
# against a concurrent `kestrel feature install` in a separate process — that
# needs a filesystem lock, and is tracked rather than pretended to be handled.
_INSTALL_LOCK = asyncio.Lock()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ConfigUpdateRequest(BaseModel):
    """Partial config update for a feature."""

    config: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feature_package_to_dict(info: FeaturePackageInfo) -> Dict[str, Any]:
    """Serialize a FeaturePackageInfo to a JSON-safe dict."""
    d = asdict(info)
    d["status"] = info.status.value
    d["boundary"] = info.boundary.value
    d["installable"] = info.installable
    d["skills"] = [asdict(s) for s in info.skills]
    return d


def _get_enabled_class_names(agent) -> set:
    """Return the set of Feature class names currently enabled on *agent*."""
    return active_feature_class_names(agent)


def _registry_info(agent, name: str) -> Optional[FeaturePackageInfo]:
    """Resolve a catalog stable ID or Feature class name to package metadata."""
    registry = get_registry(enabled_class_names=_get_enabled_class_names(agent))
    for stable_id, info in registry.items():
        if name in {stable_id, info.name} or name in info.features:
            return info
    return None


def _get_loaded_features_or_404(agent, name: str) -> List[tuple[str, Any]]:
    """Resolve a class name or package stable ID to loaded feature instances."""
    features = getattr(agent, "features", {}) or {}
    if name in features:
        return [(name, features[name])]

    info = _registry_info(agent, name)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Feature '{name}' not found in registry",
        )
    loaded = [
        (class_name, features[class_name])
        for class_name in info.features if class_name in features
    ]
    if not loaded:
        raise HTTPException(
            status_code=404,
            detail=f"Feature '{name}' not loaded on this agent",
        )
    return loaded


def _get_feature_or_404(agent, name: str):
    """Look up a loaded Feature by class name, raising 404 if not found."""
    features = getattr(agent, "features", {})
    feature = features.get(name)
    if feature is None:
        raise HTTPException(status_code=404, detail=f"Feature '{name}' not loaded on this agent")
    return feature


def _tool_to_dict(tool) -> Dict[str, Any]:
    """Serialize an AgentTool to a JSON-safe dict."""
    schema = tool.schema
    return {
        "name": schema.name,
        "description": schema.description,
        "category": schema.category.value if hasattr(schema.category, "value") else str(schema.category),
        "parameters": schema.parameters,
        "command_prefix": getattr(schema, "command_prefix", None),
    }


# ---------------------------------------------------------------------------
# Feature catalog endpoints
# ---------------------------------------------------------------------------


@router.get("/api/features")
async def list_features(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by status: available, installed, enabled, disabled"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
) -> Dict[str, Any]:
    """
    Full feature catalog with status per feature.

    Returns all known features from the registry with their runtime status
    (available / installed / enabled / disabled).
    """
    agent = get_agent(request)
    enabled = _get_enabled_class_names(agent)
    registry = get_registry(enabled_class_names=enabled)

    results = []
    for info in registry.values():
        if status and info.status.value != status:
            continue
        if tag and tag not in info.tags:
            continue
        results.append(_feature_package_to_dict(info))

    return {"features": results, "count": len(results)}


@router.get("/api/features/installed")
async def list_installed_features(request: Request) -> Dict[str, Any]:
    """
    Only installed/enabled features with their tools.

    Returns features that are currently loaded on the agent, along with the
    tools each feature exposes.
    """
    agent = get_agent(request)
    features = getattr(agent, "features", {})

    results = []
    for name, feature in features.items():
        tools = feature.get_tools()
        pkg = get_package_for_feature(name)
        entry: Dict[str, Any] = {
            "name": name,
            "tool_name": feature.tool_name,
            "description": feature.tool_description,
            "tools": [_tool_to_dict(t) for t in tools],
        }
        if pkg:
            entry["package"] = pkg.package
            entry["tags"] = pkg.tags
            entry["icon"] = pkg.icon
            entry["core"] = pkg.core
            entry["boundary"] = pkg.boundary.value
        results.append(entry)

    return {"features": results, "count": len(results)}


# ---------------------------------------------------------------------------
# UI capability derivation (#2041)
# ---------------------------------------------------------------------------


@router.get("/api/ui/capabilities")
async def get_ui_capabilities(request: Request) -> Dict[str, Any]:
    """Feature-backed UI capability set for the current agent.

    Each key is a feature's registry name; the value is whether that feature is
    enabled on this agent. The frontend merges this over its core/static
    defaults — see ``mergeCapabilities`` in ``api_client.mjs``. Re-fetched after
    a runtime enable/disable so the UI re-gates without a page reload.
    """
    agent = get_agent(request)
    return {"capabilities": compute_feature_capabilities(agent)}


@router.get("/api/ui/contributions")
async def get_ui_contributions(request: Request) -> Dict[str, Any]:
    """Merged UI-asset manifest for the current agent's ENABLED features.

    Each entry is ``{feature, capability, modules: [...], css: [...]}`` where
    ``modules``/``css`` are same-origin asset URLs (out-of-tree features are
    served under ``/features/{name}/static/``). The frontend boot loader
    dynamically ``import()``s each module in declared order so a pip-installed
    feature can mount slot contributions with no edits to core ``static/`` or
    ``app.js``. Disabled / uninstalled features contribute nothing, and remote /
    cross-origin module URLs are rejected when the manifest is built.
    """
    agent = get_agent(request)
    return {"contributions": compute_ui_manifest(agent)}


# ---------------------------------------------------------------------------
# Single-feature detail & lifecycle
# ---------------------------------------------------------------------------


@router.get("/api/features/{name}")
async def get_feature_detail(request: Request, name: str) -> Dict[str, Any]:
    """
    Detail view for a feature.

    Returns description, tools provided, hooks registered, config schema,
    package info, and install instructions.
    """
    agent = get_agent(request)
    features = getattr(agent, "features", {})

    # Try loaded feature first
    feature = features.get(name)
    if feature is not None:
        tools = feature.get_tools()
        hooks = feature.get_hooks()
        pkg = get_package_for_feature(name)

        detail: Dict[str, Any] = {
            "name": name,
            "tool_name": feature.tool_name,
            "description": feature.tool_description,
            "status": (
                "enabled" if getattr(feature, "enabled", True) else "disabled"
            ),
            "tools": [_tool_to_dict(t) for t in tools],
            "hooks": [{"name": h.name, "events": [e.value for e in h.events]} for h in hooks] if hooks else [],
            "config_schema": feature.config_schema,
        }
        if pkg:
            detail["package"] = pkg.package
            detail["git"] = pkg.git
            detail["tags"] = pkg.tags
            detail["icon"] = pkg.icon
            detail["core"] = pkg.core
            detail["boundary"] = pkg.boundary.value
            detail["installable"] = pkg.installable
            detail["skills"] = [asdict(s) for s in pkg.skills]
            detail["install_instructions"] = f"pip install {pkg.package}" if not pkg.core else None
        return detail

    # Not loaded — look up in registry
    info = _registry_info(agent, name)
    if info is not None:
        d = _feature_package_to_dict(info)
        d["install_instructions"] = f"pip install {info.package}" if not info.core else None
        return d

    raise HTTPException(status_code=404, detail=f"Feature '{name}' not found in registry or loaded features")


@router.post("/api/features/{name}/install")
async def install_feature(request: Request, name: str) -> Dict[str, Any]:
    """
    Install a feature package via pip.

    Requires a sovereign agent — governed agents cannot install packages.
    """
    agent = get_agent(request)

    # Look up package info from registry
    pkg_info = _registry_info(agent, name)

    if pkg_info is None:
        raise HTTPException(status_code=404, detail=f"Feature '{name}' not found in registry")

    if pkg_info.core:
        raise HTTPException(status_code=400, detail=f"Feature '{name}' is a core feature and is already installed")

    if pkg_info.status != FeatureStatus.AVAILABLE:
        raise HTTPException(status_code=400, detail=f"Feature '{name}' is already installed")

    # Install via pip in a subprocess, through the SAME core guard the CLI uses:
    # this package depends on kestrel-sovereign, so an unguarded install can
    # resolve core from the index and replace the running (often editable) core
    # with a wheel copy. Installing from the console is not a safer path than
    # installing from the CLI (issue #2949).
    from kestrel_sovereign.cli_features import CoreInstallGuard

    package_spec = pkg_info.package

    # The snapshot, the install and the resolve are ONE transaction over a
    # shared environment (see _INSTALL_LOCK). Holding the lock across all three
    # is the point: snapshotting outside it would let another install land
    # between the snapshot and the compare, which is the drift this guard would
    # then report against an environment nobody actually broke.
    #
    # Run as a shielded task rather than inline, because cancelling an
    # `asyncio.to_thread` await does NOT stop the worker thread or the installer
    # subprocess it is running. Inline, a cancelled request unwinds `async with`
    # and releases the lock immediately while that abandoned installer keeps
    # writing — so the next request snapshots a venv still being mutated, which
    # is exactly the concurrent-write and false-drift pair the lock exists to
    # prevent. Shielding keeps the task alive to its own terminal state, and the
    # lock is released by the task itself, so the guarantee survives a client
    # that hangs up mid-install.
    # The snapshot, the install and the resolve are ONE transaction over a
    # shared environment (see _INSTALL_LOCK): snapshotting outside it would let
    # another install land between the snapshot and the compare, and the guard
    # would report that as drift against an environment nobody broke.
    #
    # The lock is acquired OUTSIDE the shield, deliberately. Shielding the wait
    # as well meant a request cancelled while QUEUED behind another install
    # survived its own cancellation, kept waiting, and then installed a package
    # nobody was asking for any more. Cancellation while waiting MUST prevent
    # the mutation.
    #
    # Once the installer is running, cancellation cannot stop it — cancelling an
    # `asyncio.to_thread` await does not stop the worker thread or its
    # subprocess — so that half is shielded and the task releases the lock
    # itself. Releasing from the awaiting request would free the venv to the
    # next install while an abandoned installer was still writing.
    await _INSTALL_LOCK.acquire()

    async def _guarded_install():
        try:
            guard = await asyncio.to_thread(CoreInstallGuard.snapshot)

            install_error: Optional[Tuple[int, str]] = None
            try:
                result = await asyncio.to_thread(
                    guard.run, [package_spec], timeout=INSTALL_TIMEOUT_SECONDS,
                )
                if result.returncode != 0:
                    logger.error(
                        f"pip install failed for {package_spec}: {result.stderr}"
                    )
                    install_error = (500, f"Installation failed: {result.stderr[:500]}")
            except subprocess.TimeoutExpired:
                install_error = (504, "Installation timed out")

            # Detection half, on EVERY path — a failed or timed-out install is
            # not a reason to skip it. pip installs dependencies before the
            # requested package can fail, and a timeout kills the process
            # mid-write, so both can leave core swapped for an index wheel.
            # Returning the install error without looking would leave that swap
            # in place, unnamed.
            #
            # Bounded, because this repairs by running another installer: an
            # unbounded one would hang the request the install timeout exists to
            # prevent. A repair that hits the bound comes back as an unrepaired
            # outcome carrying the manual restore command.
            outcome = await asyncio.to_thread(
                guard.resolve, timeout=INSTALL_TIMEOUT_SECONDS,
            )

            # Inside the lock, deliberately. This reads LIVE environment state
            # (core's installed version), and a queued install is waiting to
            # mutate exactly that. Read outside, it could observe a transient
            # core that the next request puts up and takes back down, and answer
            # about an environment that never outlived the read. Every reading
            # of the environment this response describes belongs to the critical
            # section that owns it — the earlier fix moved snapshot/install/
            # resolve in and left this one, added later, outside.
            unsatisfied: Optional[str] = None
            if (
                install_error is None
                and outcome.drift is not None
                and outcome.conforming
            ):
                unsatisfied = await asyncio.to_thread(
                    _core_requirement_unsatisfied, package_spec,
                )
            return install_error, outcome, unsatisfied
        finally:
            _INSTALL_LOCK.release()

    install_error, outcome, unsatisfied = await asyncio.shield(
        asyncio.create_task(_guarded_install())
    )

    if outcome.drift is not None:
        logger.error("core install changed during %s install:\n%s", package_spec, outcome.describe())

    if install_error is not None:
        status_code, detail = install_error
        raise HTTPException(
            status_code=status_code,
            detail=detail if outcome.drift is None else f"{detail}\n{outcome.describe()}",
        )

    if not outcome.conforming:
        # The package installed, but the host is now running a core nobody
        # declared and the restore failed. That is not a 2xx: the operator has
        # to run the named command before this host is trustworthy again.
        raise HTTPException(
            status_code=500,
            detail=f"Package '{package_spec}' installed, but {outcome.describe()}",
        )

    if outcome.drift is None:
        return {
            "status": "installed",
            "package": package_spec,
            "message": f"Package '{package_spec}' installed. Restart the agent to load the feature.",
        }

    # Core drifted and was restored — but the package that MOVED core is very
    # often the reason it moved: a feature requiring core >=0.53 pulls 0.53 in,
    # the repair puts 0.52 back, and now the freshly installed feature has an
    # unsatisfied dependency. Telling the UI "installed, restart the agent"
    # there is a completed-install report over an environment that cannot load
    # it, which is the shape of failure this whole guard exists to stop.
    if unsatisfied:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Package '{package_spec}' installed and then moved core; core "
                f"was restored, which leaves {unsatisfied}. The package is "
                "present but cannot load. Move core to a version it accepts, or "
                "uninstall the package."
            ),
        )

    return {
        "status": "installed_with_core_drift",
        "package": package_spec,
        "core_drift": outcome.drift,
        "core_restored": True,
        "message": (
            f"Package '{package_spec}' installed, but {outcome.headline} — "
            "restored afterwards. Restart the agent."
        ),
    }


@router.post("/api/features/{name}/enable")
async def enable_feature(request: Request, name: str) -> Dict[str, Any]:
    """
    Enable a loaded feature.

    Runs the agent's canonical runtime *activation*
    (``KestrelAgent._activate_feature_runtime``) per member — the exact inverse
    of the disable teardown, so signal sources, wait providers, hooks, the A2A
    agent, and dynamic tools are all re-registered on the SAME loaded instance
    (kestrel-sovereign#2522). The endpoint owns only the group-transactional
    orchestration: enabling a package is all-or-nothing, so if any member fails
    to activate the already-activated members are soft-disabled again before the
    error surfaces. Per-member work is NOT duplicated here — it lives in the one
    canonical activation used by boot and the disable/enable rails alike.
    """
    agent = get_agent(request)
    loaded = _get_loaded_features_or_404(agent, name)

    to_activate = tuple(
        feature
        for _class_name, feature in loaded
        if not bool(getattr(feature, "enabled", True))
    )
    prepared = agent._prepare_feature_contribution_transition(to_activate)
    prepared_by_feature = {id(item.feature): item for item in prepared}
    activated: List[tuple[str, Any]] = []
    try:
        for class_name, feature in loaded:
            if bool(getattr(feature, "enabled", True)):
                # Already enabled — nothing to re-activate.
                continue
            await agent._activate_feature_runtime(
                feature,
                prepared_contributions=prepared_by_feature[id(feature)],
            )
            activated.append((class_name, feature))
    except Exception:
        # Group transaction: roll the already-activated members back to the
        # disabled state (soft-toggle) so a partial package-enable never leaves
        # a mix of live and dead members.
        for class_name, feature in reversed(activated):
            try:
                await agent._unregister_feature_runtime(feature, unload=False)
            except Exception:
                logger.exception(
                    "Enable rollback (re-disable) failed for feature '%s'",
                    class_name,
                )
            else:
                logger.info("Rolled back enable of feature '%s'", class_name)
        raise

    return {
        "name": name,
        "features": [class_name for class_name, _ in loaded],
        "status": "enabled",
        "capabilities": compute_feature_capabilities(agent),
    }


@router.post("/api/features/{name}/disable")
async def disable_feature(request: Request, name: str) -> Dict[str, Any]:
    """
    Disable a loaded feature.

    Runs the agent's canonical runtime *teardown*
    (``KestrelAgent._unregister_feature_runtime`` with ``unload=False``) per
    member, so a disabled feature has its signal sources, wait providers, hooks,
    A2A agent, and dynamic tools all detached — not just ``on_disable()`` +
    hooks (the old light path left ``task:`` / ``talon:`` wait providers and
    dispatcher sources live, kestrel-sovereign#2522). ``unload=False`` keeps the
    SAME instance loaded so ``/enable`` can re-activate it. The endpoint owns
    only the group-transactional orchestration: disabling a package is
    all-or-nothing, so if any member fails to tear down the members touched so
    far are re-activated before the error surfaces. Per-member cleanup is NOT
    duplicated here — it is the one canonical teardown boot rollback and runtime
    disable also use.
    """
    agent = get_agent(request)
    loaded = _get_loaded_features_or_404(agent, name)
    mandatory = sorted(
        class_name
        for class_name, _feature in loaded
        if class_name in MANDATORY_FEATURES
    )
    if mandatory:
        raise HTTPException(
            status_code=409,
            detail=(
                "Mandatory sovereignty features cannot be disabled: "
                + ", ".join(mandatory)
            ),
        )

    # ``attempted`` records every member we STARTED to tear down (before the
    # call, so a member whose teardown raises is still rolled back). The
    # canonical teardown runs its independent cleanup unconditionally even on a
    # failing ``on_disable`` (#2522 P2), so a member that raised is left in a
    # torn-down state; rolling the whole group back re-activates it too.
    attempted: List[tuple[str, Any]] = []
    try:
        for class_name, feature in loaded:
            if not bool(getattr(feature, "enabled", True)):
                # Already disabled — nothing to tear down.
                continue
            attempted.append((class_name, feature))
            await agent._unregister_feature_runtime(feature, unload=False)
    except Exception:
        for class_name, feature in reversed(attempted):
            try:
                await agent._activate_feature_runtime(feature)
            except Exception:
                logger.exception(
                    "Disable rollback (re-enable) failed for feature '%s'",
                    class_name,
                )
            else:
                logger.info("Rolled back disable of feature '%s'", class_name)
        raise

    return {
        "name": name,
        "features": [class_name for class_name, _ in loaded],
        "status": "disabled",
        "capabilities": compute_feature_capabilities(agent),
    }


@router.post("/api/features/{name}/remove")
async def remove_feature(request: Request, name: str) -> Dict[str, Any]:
    """
    Uninstall a feature package.

    Runs the agent's canonical runtime *teardown*
    (``KestrelAgent._unregister_feature_runtime`` with ``unload=True``) per
    loaded member so removal drains EVERYTHING a live feature acquired — signal
    sources, ``task:`` / ``talon:`` wait providers, hooks, the A2A agent, owned
    background tasks / sleep hooks, AND promoted direct tools — then calls
    ``on_remove()`` for stored-data cleanup, then pip-uninstalls the package. The
    old path detached only hooks and set ``enabled=False``, leaving every other
    registration live; the promoted **direct tools** in particular stayed
    executable because tool resolution gates on the feature's ``enabled`` flag,
    not on membership of a still-registered tool map, so a "removed" feature's
    ``@tool`` methods remained callable until restart (kestrel-sovereign#2522
    P1). Requires a sovereign agent — governed agents cannot remove packages.
    """
    agent = get_agent(request)

    pkg_info = get_package_for_feature(name) or _registry_info(agent, name)
    if pkg_info is None:
        raise HTTPException(status_code=404, detail=f"Feature '{name}' not found in registry")

    if pkg_info.core:
        raise HTTPException(status_code=400, detail="Cannot remove a core feature")

    # Mandatory sovereignty features must never be torn down / uninstalled: the
    # canonical teardown below (``unload=True``) would drop their signal sources,
    # wait providers, and A2A wiring, crippling the agent. Mirrors the /disable
    # guard (kestrel-sovereign#2522). Today every mandatory feature is already
    # blocked earlier (core members → 400, ``WaitFeature`` is absent from the
    # removable registry → 404), so this is defense in depth against a future
    # removable package that declares one — the guard fails closed rather than
    # relying on that coincidence.
    mandatory = sorted(
        class_name
        for class_name in pkg_info.features
        if class_name in MANDATORY_FEATURES
    )
    if mandatory:
        raise HTTPException(
            status_code=409,
            detail=(
                "Mandatory sovereignty features cannot be removed: "
                + ", ".join(mandatory)
            ),
        )

    # Check if feature is loaded — drain its full runtime, then on_remove.
    features = getattr(agent, "features", {}) or {}
    loaded = [
        (class_name, features[class_name])
        for class_name in pkg_info.features
        if class_name in features
    ]
    for class_name, feature in loaded:
        # Full canonical runtime teardown BEFORE uninstall — the SAME inverse
        # boot rollback and /disable use, not a hooks-only subset. ``unload=True``
        # drops the instance from ``agent.features`` once every registration is
        # drained (kestrel-sovereign#2522 P1).
        await agent._unregister_feature_runtime(feature, unload=True)
        # ``on_remove`` (stored-data cleanup) runs AFTER the runtime is fully
        # quiesced, but on the SAME still-referenced instance whose ``agent`` /
        # storage / config the teardown never touched — so unloading loses it no
        # feature state, and no still-live background task can race the deletion.
        await feature.on_remove()
        feature.enabled = False

    package_spec = pkg_info.package
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "pip", "uninstall", "-y", package_spec],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"pip uninstall failed for {package_spec}: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Removal failed: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Removal timed out")

    return {
        "status": "removed",
        "package": package_spec,
        "features": [class_name for class_name, _ in loaded],
        "message": f"Package '{package_spec}' uninstalled. Restart the agent to fully unload.",
    }


# ---------------------------------------------------------------------------
# Feature configuration endpoints
# ---------------------------------------------------------------------------


def _secret_field_names(schema: Optional[Dict[str, Any]]) -> set:
    """Return property names that hold secrets and must never be returned.

    A field is treated as a secret when the schema marks it ``writeOnly: true``
    or ``format: "password"`` (standard JSON Schema keywords — see
    ``docs/architecture/features/CONFIG_SCHEMA_UI_HINTS.md``).
    """
    if not schema:
        return set()
    properties = schema.get("properties", {})
    secrets = set()
    for key, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        if prop.get("writeOnly") is True or prop.get("format") == "password":
            secrets.add(key)
    return secrets


@router.get("/api/features/{name}/config")
async def get_feature_config(request: Request, name: str) -> Dict[str, Any]:
    """Current configuration for a loaded feature.

    Secret fields (``writeOnly``/``format: password``) are never returned in
    plaintext. They are stripped from ``config`` and surfaced only as a boolean
    in ``secrets_set`` so the UI can show whether a value is already stored
    (write-only semantics).
    """
    agent = get_agent(request)
    feature = _get_feature_or_404(agent, name)

    config = await feature.get_config()
    schema = feature.config_schema

    secrets_set: Dict[str, bool] = {}
    secret_fields = _secret_field_names(schema)
    if secret_fields and isinstance(config, dict):
        config = dict(config)
        for key in secret_fields:
            secrets_set[key] = bool(config.get(key))
            config.pop(key, None)

    return {
        "name": name,
        "config": config,
        "config_schema": schema,
        "secrets_set": secrets_set,
    }


@router.patch("/api/features/{name}/config")
async def update_feature_config(
    request: Request,
    name: str,
    body: ConfigUpdateRequest,
) -> Dict[str, Any]:
    """
    Update feature configuration.

    Validates against the feature's config_schema if available.

    Write-only secret fields omitted from the request body are preserved: the
    stored value is re-injected before validation/save, so the frontend can
    leave an unchanged secret out of the PATCH without clearing it.
    """
    agent = get_agent(request)
    feature = _get_feature_or_404(agent, name)

    schema = feature.config_schema
    incoming = dict(body.config)

    secret_fields = _secret_field_names(schema)
    atomic_secret_update = getattr(feature, "set_config_with_secret_preservation", None)
    has_atomic_secret_update = inspect.iscoroutinefunction(atomic_secret_update)
    if secret_fields and has_atomic_secret_update:
        # Isolated hosted features preserve omitted write-only fields from the
        # same durable snapshot used by their transition CAS.  Reading here and
        # reinjecting later would let a stale replica overwrite a concurrent
        # credential rotation.
        await atomic_secret_update(
            incoming,
            secret_fields,
            lambda effective: _validate_config(effective, schema)
            if schema is not None
            else None,
        )
    else:
        if secret_fields:
            current = await feature.get_config()
            if isinstance(current, dict):
                for key in secret_fields:
                    if key not in incoming and key in current:
                        incoming[key] = current[key]

        if schema is not None:
            _validate_config(incoming, schema)

        await feature.set_config(incoming)

    updated = await feature.get_config()

    # Never echo secret values back to the client.
    if secret_fields and isinstance(updated, dict):
        updated = {k: v for k, v in updated.items() if k not in secret_fields}

    return {
        "name": name,
        "config": updated,
        "message": "Configuration updated",
    }


def _validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """
    JSON Schema validation for feature config.

    Checks required fields, type constraints, minimum/maximum bounds, and
    enum restrictions from the schema.
    Raises HTTPException(422) on validation failure.
    """
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for field_name in required:
        if field_name not in config:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required config field: '{field_name}'",
            )

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for key, value in config.items():
        if key not in properties:
            continue

        prop_schema = properties[key]

        # Type check
        expected_type_name = prop_schema.get("type")
        if expected_type_name and expected_type_name in type_map:
            expected = type_map[expected_type_name]
            if not isinstance(value, expected):
                raise HTTPException(
                    status_code=422,
                    detail=f"Config field '{key}' must be {expected_type_name}, got {type(value).__name__}",
                )

        # Minimum / maximum bounds (for integer and number types)
        if isinstance(value, (int, float)):
            minimum = prop_schema.get("minimum")
            if minimum is not None and value < minimum:
                raise HTTPException(
                    status_code=422,
                    detail=f"Config field '{key}' must be >= {minimum}, got {value}",
                )
            maximum = prop_schema.get("maximum")
            if maximum is not None and value > maximum:
                raise HTTPException(
                    status_code=422,
                    detail=f"Config field '{key}' must be <= {maximum}, got {value}",
                )

        # Enum restriction
        enum_values = prop_schema.get("enum")
        if enum_values is not None and value not in enum_values:
            raise HTTPException(
                status_code=422,
                detail=f"Config field '{key}' must be one of {enum_values}, got {value!r}",
            )


# ---------------------------------------------------------------------------
# Skill discovery endpoints
# ---------------------------------------------------------------------------


@router.get("/api/features/{name}/skills")
async def get_feature_skills(request: Request, name: str) -> Dict[str, Any]:
    """
    Skills provided by a specific feature.

    Returns tools from the loaded feature instance if available, otherwise
    falls back to static skill declarations from the registry.
    """
    agent = get_agent(request)
    features = getattr(agent, "features", {})

    # If feature is loaded, return live tools
    feature = features.get(name)
    if feature is not None:
        tools = feature.get_tools()
        return {
            "feature": name,
            "skills": [_tool_to_dict(t) for t in tools],
            "count": len(tools),
            "source": "live",
        }

    # Fall back to registry declarations
    # Try by class name first, then by package short name
    pkg = get_package_for_feature(name)
    if pkg is not None:
        skills = [asdict(s) for s in pkg.skills]
        return {
            "feature": name,
            "skills": skills,
            "count": len(skills),
            "source": "registry",
        }

    # Try as package short name
    skills = get_skills_for_package(name)
    if skills:
        return {
            "feature": name,
            "skills": [asdict(s) for s in skills],
            "count": len(skills),
            "source": "registry",
        }

    raise HTTPException(status_code=404, detail=f"Feature '{name}' not found")


@router.get("/api/skills")
async def list_all_skills(
    request: Request,
    tag: Optional[str] = Query(None, description="Filter by skill tag"),
    category: Optional[str] = Query(None, description="Filter by skill category"),
) -> Dict[str, Any]:
    """
    All skills across all features, searchable by tag/category.

    Merges live tools from loaded features with static registry declarations
    for unloaded features.
    """
    agent = get_agent(request)
    features = getattr(agent, "features", {})
    seen_skill_names: set = set()
    results: List[Dict[str, Any]] = []

    # Live skills from loaded features
    for feature_name, feature in features.items():
        if not getattr(feature, "enabled", True):
            continue
        for tool in feature.get_tools():
            skill = _tool_to_dict(tool)
            skill["feature"] = feature_name
            skill["source"] = "live"

            if tag and tag not in (skill.get("category", ""),):
                # Check tool parameters or skip — tags aren't on live tools
                pass
            if category and skill.get("category", "") != category:
                continue

            results.append(skill)
            seen_skill_names.add(skill["name"])

    # Static registry skills for unloaded features
    enabled = _get_enabled_class_names(agent)
    registry = get_registry(enabled_class_names=enabled)

    for info in registry.values():
        # Skip features already represented by live tools
        if set(info.features) & enabled:
            continue
        for skill_info in info.skills:
            if skill_info.name in seen_skill_names:
                continue
            if tag and tag not in skill_info.tags:
                continue
            if category and skill_info.category != category:
                continue
            skill_dict = asdict(skill_info)
            skill_dict["feature"] = info.name
            skill_dict["source"] = "registry"
            results.append(skill_dict)
            seen_skill_names.add(skill_info.name)

    return {"skills": results, "count": len(results)}


@router.get("/api/skills/{skill_id}/schema")
async def get_skill_schema(request: Request, skill_id: str) -> Dict[str, Any]:
    """
    OpenAI function-calling schema for a specific skill.

    Returns the skill in the format expected by LLM function-calling APIs.
    """
    agent = get_agent(request)
    features = getattr(agent, "features", {})

    # Search loaded features for this skill
    for feature_name, feature in features.items():
        for tool in feature.get_tools():
            if tool.schema.name == skill_id:
                schema = tool.schema
                return {
                    "type": "function",
                    "function": {
                        "name": schema.name,
                        "description": schema.description,
                        "parameters": schema.parameters,
                    },
                    "feature": feature_name,
                }

    # Check registry for static declaration
    all_skills = get_all_skills()
    for skill in all_skills:
        if skill.name == skill_id:
            return {
                "type": "function",
                "function": {
                    "name": skill.name,
                    "description": skill.description,
                    "parameters": {},
                },
                "source": "registry",
            }

    raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
