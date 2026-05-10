"""Setup wizard step: configure PayerPolicy and host master credentials.

Phase 4 of the PayerPolicy foundation work.

Responsibilities:
- Reads existing ``[payments]`` table from ``kestrel.toml`` if present.
- For each resource class (llm, storage, compute, tools, comms), offers
  the operator the supported `(vendor, kind)` combinations from the
  SDK ``SUPPORTED_PAYER_COMBINATIONS`` matrix. Combinations the matrix
  marks NOT_IMPLEMENTED / OUT_OF_SCOPE / NOT_APPLICABLE are not
  offered, so the operator can never pick a path the resolver cannot
  honor.
- For ``HOST_MASTER_PROVISIONED``, collects the master credential and
  stores it via ``HostKeyStorage`` (encrypted at rest under
  ``KESTREL_DATA_KEY``). Card details are NEVER prompted — fiat onramp
  goes through Stripe (already wired in
  ``kestrel_sovereign/features/wallet/onramp/``), not through this step.
- Writes the resolved ``PayerPolicy`` to ``kestrel.toml`` under
  ``[payments]``.

Idempotent: re-running the step shows current state and offers to
change. Never overwrites an existing master credential without explicit
confirmation.

The step is wired in :data:`kestrel_sovereign.setup.steps.ORDERED` after
``keys`` (so ``KESTREL_DATA_KEY`` is in place before HostKeyStorage
encrypts) and before ``llm`` (so per-agent OpenRouter provisioning can
read its master from HostKeyStorage at first agent init).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Mapping

from kestrel_sdk.payer_policy import (
    PayerKind,
    PayerPolicy,
    PayerSpec,
    ResourceClass,
    SupportStatus,
    is_offerable,
    status_for,
    supported_kinds_for,
)

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.toml_file import read_toml, write_toml


# Vendors offered by the wizard per resource class. For each we filter
# supported_kinds_for() through the matrix to get the actual
# offerable list. Adding a new vendor here is the deliberate plan-side
# choice that lets it appear in the wizard.
_VENDORS_PER_RESOURCE: dict[ResourceClass, tuple[str, ...]] = {
    ResourceClass.LLM: ("openrouter", "local"),
    ResourceClass.STORAGE: ("lighthouse", "local-disk"),
    ResourceClass.COMPUTE: ("*",),
    ResourceClass.TOOLS: ("*",),
    ResourceClass.COMMS: ("*",),
}


_KIND_DESCRIPTIONS: dict[PayerKind, str] = {
    PayerKind.HOST_ENV: "host env var (today's behavior)",
    PayerKind.HOST_MASTER_PROVISIONED: "host master account, child credential per agent",
    PayerKind.USER_MASTER_PROVISIONED: "user's master account",
    PayerKind.SPONSOR: "sponsor's master account",
    PayerKind.SELF_WALLET: "agent's own wallet (e.g. x402)",
    PayerKind.NONE: "do not use this resource for any agent",
}


def run(ctx: SetupContext) -> None:
    """Configure PayerPolicy in kestrel.toml [payments]."""
    toml_data = read_toml(ctx.kestrel_toml_path)
    existing_section = toml_data.get("payments") if toml_data else None

    if ctx.flow is Flow.CHECK:
        _check_only(ctx, existing_section)
        return

    # Load existing policy or default. host_env_default is always valid
    # (matrix says READY for the wildcard kinds), so first-time runs
    # don't fail.
    if existing_section:
        try:
            policy = PayerPolicy.from_toml_section(existing_section)
        except Exception as e:
            ctx.prompter.info(
                f"[payments] section in kestrel.toml is malformed: {e}. "
                "Starting from default."
            )
            policy = PayerPolicy.host_env_default()
    else:
        policy = PayerPolicy.host_env_default()

    if ctx.flow is Flow.QUICKSTART:
        # Quickstart: accept existing or default; no prompts. Just
        # validate and persist if absent.
        if existing_section is None:
            _persist_policy(ctx, policy)
            ctx.record("Wrote default PayerPolicy to kestrel.toml [payments]")
        return

    # INTERACTIVE flow: walk each resource class.
    new_policy = _walk_interactive(ctx, policy)
    if new_policy is None:
        return  # user cancelled or no changes

    if new_policy == policy and existing_section is not None:
        ctx.prompter.info("PayerPolicy unchanged.")
        return

    _persist_policy(ctx, new_policy)
    ctx.record("Updated kestrel.toml [payments] PayerPolicy")


# =============================================================================
# CHECK flow
# =============================================================================


def _check_only(ctx: SetupContext, existing: Mapping[str, Any] | None) -> None:
    if existing is None:
        ctx.prompter.info(
            "kestrel.toml has no [payments] section; agents use "
            "host_env_default (today's behavior unchanged)."
        )
        return
    try:
        policy = PayerPolicy.from_toml_section(existing)
    except Exception as e:
        ctx.block(f"kestrel.toml [payments] section is malformed: {e}")
        return
    try:
        policy.validate_against_matrix()
    except Exception as e:
        ctx.block(f"kestrel.toml [payments] policy is invalid: {e}")
        return
    ctx.prompter.info("kestrel.toml [payments] PayerPolicy validates.")


# =============================================================================
# INTERACTIVE flow
# =============================================================================


def _walk_interactive(
    ctx: SetupContext, current: PayerPolicy
) -> PayerPolicy | None:
    """Walk each ResourceClass slot and let the operator pick (vendor, kind)."""
    ctx.prompter.info(
        "PayerPolicy: who pays for which metered resource per agent.\n"
        "For each resource class, pick the vendor and how it's paid for. "
        "Combinations marked READY in the support matrix are the only "
        "ones offered."
    )

    new_specs: dict[str, PayerSpec] = {}
    for resource_class, current_spec in current._iter_specs():
        new_spec = _pick_spec_for(ctx, resource_class, current_spec)
        if new_spec is None:
            return None
        new_specs[resource_class.value] = new_spec

    return PayerPolicy(
        llm=new_specs[ResourceClass.LLM.value],
        storage=new_specs[ResourceClass.STORAGE.value],
        compute=new_specs[ResourceClass.COMPUTE.value],
        tools=new_specs[ResourceClass.TOOLS.value],
        comms=new_specs[ResourceClass.COMMS.value],
    )


def _pick_spec_for(
    ctx: SetupContext,
    resource_class: ResourceClass,
    current: PayerSpec,
) -> PayerSpec | None:
    """Prompt for vendor + kind for a single resource class."""
    vendors = _VENDORS_PER_RESOURCE.get(resource_class, ("*",))
    if len(vendors) == 1:
        vendor = vendors[0]
    else:
        vendor_default = current.vendor if current.vendor in vendors else vendors[0]
        vendor = ctx.prompter.select(
            f"{resource_class.value}: pick a vendor",
            choices=list(vendors),
            default=vendor_default,
        )

    # Filter kinds to only those marked READY in the matrix for this
    # (resource, vendor) pair. This is the load-bearing wizard
    # invariant: we never offer a kind the resolver cannot honor.
    offerable = supported_kinds_for(resource_class, vendor)
    if not offerable:
        ctx.prompter.info(
            f"{resource_class.value}/{vendor}: no payer kinds are "
            "currently READY in the support matrix; skipping."
        )
        # Use NONE as a safe placeholder (always offerable).
        return PayerSpec(vendor=vendor, kind=PayerKind.NONE)

    kind_choices = [
        f"{k.value} — {_KIND_DESCRIPTIONS.get(k, '')}".rstrip(" —")
        for k in offerable
    ]
    default_label = None
    if current.vendor == vendor and current.kind in offerable:
        default_label = next(
            label for label, k in zip(kind_choices, offerable) if k is current.kind
        )

    chosen_label = ctx.prompter.select(
        f"{resource_class.value}/{vendor}: how is it paid for?",
        choices=kind_choices,
        default=default_label,
    )
    chosen_kind = offerable[kind_choices.index(chosen_label)]

    # USER_MASTER_PROVISIONED / SPONSOR require master_did.
    master_did: str | None = None
    if chosen_kind in (PayerKind.USER_MASTER_PROVISIONED, PayerKind.SPONSOR):
        master_did = ctx.prompter.text(
            f"{resource_class.value}/{vendor}/{chosen_kind.value}: "
            "DID of the principal funding the agent (e.g. did:pkh:eip155:1:0x...)",
            default=current.master_did or "",
        )
        if not master_did:
            ctx.prompter.info(
                f"No master_did provided; falling back to {current.kind.value}."
            )
            return current

    # HOST_MASTER_PROVISIONED for OpenRouter → prompt for the master
    # API key and store via HostKeyStorage.
    if (
        chosen_kind is PayerKind.HOST_MASTER_PROVISIONED
        and resource_class is ResourceClass.LLM
        and vendor == "openrouter"
    ):
        _maybe_capture_openrouter_master(ctx)

    return PayerSpec(
        vendor=vendor,
        kind=chosen_kind,
        master_did=master_did,
    )


# =============================================================================
# HostKeyStorage capture
# =============================================================================


def _maybe_capture_openrouter_master(ctx: SetupContext) -> None:
    """If no host master is stored for OpenRouter, prompt and persist.

    The master is stored via ``HostKeyStorage`` (encrypted under
    ``KESTREL_DATA_KEY``), NOT in ``.env`` or ``kestrel.toml`` plaintext.
    Card details are NEVER prompted; fiat → crypto goes through
    ``kestrel_sovereign/features/wallet/onramp/`` (Stripe), and the
    master OpenRouter API key gets paid for at openrouter.ai with the
    operator's existing card-on-file there.

    Idempotent: existing master is left untouched unless the operator
    explicitly confirms a rotation.
    """
    # Late imports because this step doesn't always exercise the DB path.
    try:
        from kestrel_sovereign.security.host_key_storage import HostKeyStorage
        from kestrel_sovereign.storage.async_database import AsyncDatabase
    except ImportError as e:
        ctx.prompter.info(
            f"HostKeyStorage not importable ({e}); skipping master capture. "
            "Set OPENROUTER_MANAGEMENT_API_KEY env var as fallback."
        )
        return

    db_path = ctx.project_dir / "agent_data" / "host.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async def _capture():
        db = await AsyncDatabase.sqlite(str(db_path))
        try:
            host_storage = HostKeyStorage(db)
            already_set = await host_storage.has_key("openrouter")

            if already_set:
                rotate = ctx.prompter.confirm(
                    "OpenRouter host master key is already configured. Rotate it?",
                    default=False,
                )
                if not rotate:
                    return False

            new_key = ctx.prompter.text(
                "OpenRouter master API key (sk-or-v1-..., from "
                "https://openrouter.ai/settings/keys with key-management permission)",
                default="",
            ).strip()
            if not new_key:
                ctx.prompter.info(
                    "No master key entered. Per-agent OpenRouter provisioning "
                    "will fail until one is configured."
                )
                return False

            await host_storage.store_key("openrouter", new_key)
            return True
        finally:
            await db.close()

    persisted = asyncio.run(_capture())
    if persisted:
        ctx.record(
            f"Stored OpenRouter host master key in HostKeyStorage "
            f"({db_path.relative_to(ctx.project_dir)}, encrypted under "
            "KESTREL_DATA_KEY)."
        )


# =============================================================================
# Persistence
# =============================================================================


def _persist_policy(ctx: SetupContext, policy: PayerPolicy) -> None:
    """Write the policy to kestrel.toml [payments]."""
    section = policy.to_toml_section()
    write_toml(ctx.kestrel_toml_path, {"payments": section})
