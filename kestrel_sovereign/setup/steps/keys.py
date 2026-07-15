"""Generate encryption + API keys into ``.env``.

Two keys matter for v1:

  ``KESTREL_DATA_KEY``
      Fernet key used to encrypt agent data at rest. **Never** regenerated
      if already present — doing so would brick existing encrypted DBs.
      The wizard refuses to overwrite a non-empty value.

  ``KESTREL_API_KEY``
      Optional API key for HTTP auth. Auto-generated if absent and the
      user opts in; otherwise left empty (server auto-generates per-run).

Custody invariant (issue #2468)
-------------------------------
For every setup target there must be exactly **one** effective
``KESTREL_DATA_KEY``: the key used to encrypt new identity artifacts during
this process **is** the key persisted to the target ``.env`` for the next
boot. Anything else is a split brain — inception encrypts with key *A* while
the target ``.env`` persists key *B*, so the agent cannot decrypt its own
identity after the first process exits (a loss-of-custody defect).

Key authority is resolved **deliberately** from the target context, never
from an import-time current-directory ``load_dotenv()`` side effect (that
mutation was removed from ``inception_service``). Precedence:

1. **Existing target key** (present in the target ``.env``): authoritative and
   never regenerated. It is propagated into ``os.environ`` so inception
   encrypts with exactly the persisted key. If a *different* key is already
   exported in the process environment, that is an unresolvable conflict —
   we block *before* inception rather than silently encrypt with one key and
   persist another.
2. **True exported key** (``KESTREL_DATA_KEY`` in ``os.environ`` but not in the
   target ``.env``): adopted as this home's key — persisted to the target
   ``.env`` so the next boot loads the same value the process is using now.
3. **Neither**: generate a fresh key, persist it, and export it into the
   process so later wizard steps (notably ``agent`` → inception) encrypt with
   it.

Invalid key material (malformed master key) blocks before inception rather
than producing an identity nobody can later re-derive.
"""

from __future__ import annotations

import os
import secrets

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.env_file import read_env, write_env

#: Env var name; kept as a constant so callers/tests reference one spelling.
DATA_KEY_ENV = "KESTREL_DATA_KEY"
API_KEY_ENV = "KESTREL_API_KEY"


def resolve_data_key_authority(
    target_key: str | None,
    exported_key: str | None,
    *,
    env_name: str = ".env",
) -> tuple[str | None, str | None]:
    """Resolve the single effective ``KESTREL_DATA_KEY`` for a setup target.

    Pure custody decision for #2468, shared by the wizard ``keys`` step **and**
    the CLI ``create`` / ``setup`` pre-inception guard, so every path that can
    trigger inception enforces identical precedence. Returns
    ``(effective_key, conflict)`` where exactly one shape applies:

    * ``(key, None)``  — ``key`` is authoritative: encrypt with it *and* persist
      it. Precedence is (1) an existing target ``.env`` key, else (2) a genuinely
      exported key.
    * ``(None, msg)``  — unresolvable; ``msg`` explains the block for the
      operator. Never contains key material.
    * ``(None, None)`` — neither a target nor an exported key exists; the caller
      must generate a fresh key (only the keys step does this).
    """
    target_key = (target_key or "").strip()
    exported_key = (exported_key or "").strip()

    if target_key:
        # (1) Existing target home — its persisted key is authoritative and must
        # never be regenerated.
        if exported_key and exported_key != target_key:
            # A different key is exported in the process. Encrypting with the
            # exported key while the target ``.env`` persists another value is
            # exactly the split-brain custody defect. Never echo the material.
            return None, (
                f"{DATA_KEY_ENV} conflict: the value exported in this shell "
                f"differs from the one persisted in {env_name}. "
                "Refusing to encrypt the identity with one key while "
                "persisting another. Unset the exported value (the target "
                f"{env_name} is authoritative) or reconcile them, "
                "then re-run."
            )
        if not _is_valid_data_key(target_key):
            return None, (
                f"{DATA_KEY_ENV} in {env_name} is not a valid master "
                "key (expected a 32-byte url-safe base64 Fernet key). Fix or "
                "remove it before an agent can be created."
            )
        return target_key, None

    if exported_key:
        # (2) True exported-key semantics: no persisted target key yet, but the
        # operator has one exported. Adopt it as this home's key.
        if not _is_valid_data_key(exported_key):
            return None, (
                f"{DATA_KEY_ENV} exported in this shell is not a valid master "
                "key (expected a 32-byte url-safe base64 Fernet key). Fix it "
                "before an agent can be created."
            )
        return exported_key, None

    # (3) Nothing persisted, nothing exported.
    return None, None


def run(ctx: SetupContext) -> None:
    """Ensure data + API keys exist in ``.env`` with a single effective key."""
    env = read_env(ctx.env_path)
    target_key = (env.get(DATA_KEY_ENV) or "").strip()
    exported_key = (os.environ.get(DATA_KEY_ENV) or "").strip()

    # CHECK is read-only and reports solely on the *persisted* target state:
    # it never prompts, never writes, and never mutates ``os.environ``. A
    # persisted key is all the next boot has to unseal encrypted memory with.
    if ctx.flow is Flow.CHECK:
        if not target_key:
            ctx.block(f"{DATA_KEY_ENV} missing — encrypted memory cannot be unsealed")
        return

    updates: dict[str, str] = {}
    # The one key this process will both encrypt with and persist. ``None``
    # means "unresolved" — the data-key branch blocked and inception must not
    # run (the ``agent`` step guards on the blocker).
    effective_key: str | None

    effective_key, conflict = resolve_data_key_authority(
        target_key, exported_key, env_name=ctx.env_path.name
    )
    if conflict:
        # Fail before inception rather than silently pick a winner. The target
        # key (if any) is left untouched.
        ctx.block(conflict)
        return

    if effective_key is not None and target_key:
        # (1) Existing target key adopted as-is; never regenerated.
        if ctx.flow is Flow.INTERACTIVE:
            ctx.prompter.info(f"{DATA_KEY_ENV} already set; leaving untouched.")
    elif effective_key is not None:
        # (2) Exported key adopted — persist it so the next boot loads the same
        # value the process encrypts with.
        updates[DATA_KEY_ENV] = effective_key
        ctx.record(
            f"Adopted exported {DATA_KEY_ENV} into {ctx.env_path.name} so the "
            "next boot uses the same key"
        )
    else:
        # (3) Nothing persisted, nothing exported — generate a fresh key.
        generated = _generate_fernet_key()
        effective_key = generated
        updates[DATA_KEY_ENV] = generated
        ctx.record(f"Generated {DATA_KEY_ENV} (Fernet, 32-byte url-safe base64)")

    api_key_present = bool(env.get(API_KEY_ENV))
    if not api_key_present:
        if ctx.flow is Flow.QUICKSTART:
            updates[API_KEY_ENV] = _generate_api_key()
            ctx.record(f"Generated {API_KEY_ENV} (32-byte url-safe token)")
        else:
            wants_key = ctx.prompter.confirm(
                "Generate a stable KESTREL_API_KEY now? "
                "(otherwise the server picks one per run)",
                default=True,
            )
            if wants_key:
                updates[API_KEY_ENV] = _generate_api_key()
                ctx.record(f"Generated {API_KEY_ENV} (32-byte url-safe token)")

    if updates:
        result = write_env(ctx.env_path, updates)
        if result.backup_path is not None:
            ctx.record(f"Backed up existing .env to {result.backup_path.name}")

    # Propagate the *effective* data key to the live process, overriding any
    # stale value, so later wizard steps (``agent`` → inception) encrypt with
    # exactly the key persisted to the target ``.env``. This is the load-bearing
    # fix for #2468: encrypt-key and persist-key must be identical.
    if effective_key:
        os.environ[DATA_KEY_ENV] = effective_key

    # Propagate a freshly-generated API key too, but never overwrite a value
    # the operator already had exported (API-key clobbering is surprising and,
    # unlike the data key, carries no custody guarantee to uphold).
    api_value = updates.get(API_KEY_ENV)
    if api_value and not os.environ.get(API_KEY_ENV):
        os.environ[API_KEY_ENV] = api_value


def _is_valid_data_key(value: str) -> bool:
    """Return True iff ``value`` is usable as a ``KESTREL_DATA_KEY``.

    ``KESTREL_DATA_KEY`` may be a Fernet-shaped key **or** a passphrase (both
    are documented and accepted by ``SecureKeyStorage``, which derives the
    encryption key via PBKDF2). So the guard is deliberately permissive about
    shape, but rejects material that is empty or that embeds whitespace /
    control characters — the latter both signals corruption and would break
    the single-line ``.env`` key format. Never logs or echoes the value.
    """
    v = (value or "").strip()
    if not v:
        return False
    return not any(ch.isspace() or ord(ch) < 0x20 for ch in v)


def _generate_fernet_key() -> str:
    """Return a fresh AEADCipher master key (44-char url-safe base64).

    Same shape as the historical Fernet key it replaces — backwards-compatible
    with KESTREL_DATA_KEY consumers, but generated through ``AEADCipher`` so
    this module no longer pulls in ``cryptography.fernet`` directly. Imported
    lazily so this module stays cheap at CLI parse time.
    """
    from kestrel_sdk.security.aead import AEADCipher

    return AEADCipher.generate_key().decode("ascii")


def _generate_api_key() -> str:
    """Return a 32-byte url-safe API key."""
    return secrets.token_urlsafe(32)
