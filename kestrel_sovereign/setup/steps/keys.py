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
from typing import Mapping

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.env_file import read_env, write_env

#: Env var name; kept as a constant so callers/tests reference one spelling.
DATA_KEY_ENV = "KESTREL_DATA_KEY"
API_KEY_ENV = "KESTREL_API_KEY"


def read_persisted_data_key(env_path) -> str | None:
    """Return the ``KESTREL_DATA_KEY`` persisted in ``env_path`` **exactly** as
    the runtime will see it.

    Custody correctness (#2468) requires that the value setup reads here is the
    identical byte string ``load_dotenv`` will place in ``os.environ`` at the
    next boot — otherwise setup could encrypt with one representation while the
    server decrypts with another. So this uses python-dotenv's own parser
    (``dotenv_values``), the *same* canonical parser the runtime uses, rather
    than the wizard's simplified ``read_env``. It never strips or normalizes the
    value: whitespace, quoting, and interpolation resolve exactly once, the way
    boot resolves them. Returns ``None`` when the file or key is absent.
    """
    from pathlib import Path

    env_path = Path(env_path)
    if not env_path.exists():
        return None
    from dotenv import dotenv_values

    return dotenv_values(str(env_path)).get(DATA_KEY_ENV)


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

    Key material is compared **verbatim** — never stripped or normalized. The
    master key feeds ``SecureKeyStorage``'s PBKDF2 as raw UTF-8 bytes, so any
    mutation (even trimming whitespace) would derive a different key than the
    next boot. A passphrase with internal spaces is a legitimate key and must
    round-trip untouched.
    """
    target = target_key if target_key not in (None, "") else None
    exported = exported_key if exported_key not in (None, "") else None

    if target is not None:
        # (1) Existing target home — its persisted key is authoritative and must
        # never be regenerated.
        if exported is not None and exported != target:
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
        if not _is_valid_data_key(target):
            return None, (
                f"{DATA_KEY_ENV} in {env_name} is not a valid master "
                "key (must be a single-line, non-empty value with no control "
                "characters). Fix or remove it before an agent can be created."
            )
        return target, None

    if exported is not None:
        # (2) True exported-key semantics: no persisted target key yet, but the
        # operator has one exported. Adopt it as this home's key.
        if not _is_valid_data_key(exported):
            return None, (
                f"{DATA_KEY_ENV} exported in this shell is not a valid master "
                "key (must be a single-line, non-empty value with no control "
                "characters). Fix it before an agent can be created."
            )
        return exported, None

    # (3) Nothing persisted, nothing exported.
    return None, None


def ensure_effective_data_key(
    env_path,
    *,
    generate_if_missing: bool,
    extra_updates: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None, str, object]:
    """Resolve, persist (round-trip-verified) and export the one effective
    ``KESTREL_DATA_KEY`` for a target home.

    This is the canonical custody primitive for #2468. Every path that can reach
    inception — the wizard ``keys`` step, the wizard ``agent`` step at the actual
    inception boundary, and the CLI ``create`` / ``setup agent`` guards — funnels
    through it so encrypt-key and persist-key are provably identical.

    ``extra_updates`` is an optional mapping of *other* ``.env`` keys to persist
    in the **same** ``write_env`` call as the data key. Batching them into one
    write matters for cleanliness: on a fresh home a separate data-key write
    would create the file and a follow-up write (e.g. the API key) would then
    back it up, leaving a spurious ``.env.backup-*``. The data key still governs
    the round-trip check; extra keys ride along.

    Returns ``(effective, conflict, action, write_result)``:

    * ``(key, None, action, result)`` — ``key`` is authoritative and now both
      persisted in ``env_path`` and exported in ``os.environ``. ``action`` is one
      of ``"existing"`` (already persisted), ``"adopted"`` (an exported key
      persisted for this home) or ``"generated"``. ``result`` is the
      ``EnvWriteResult`` when a write occurred (so callers can report a backup),
      else ``None``.
    * ``(None, msg, "conflict", None)`` — unresolvable custody state
      (exported⇄persisted conflict, invalid material, or a value that could not
      be persisted without being altered). ``msg`` never contains key material.
    * ``(None, None, "none", None)`` — nothing persisted, nothing exported, and
      ``generate_if_missing`` is False. The caller decides what to do.
    """
    from pathlib import Path

    env_path = Path(env_path)
    persisted = read_persisted_data_key(env_path)
    exported = os.environ.get(DATA_KEY_ENV)
    extra = dict(extra_updates or {})

    effective, conflict = resolve_data_key_authority(
        persisted, exported, env_name=env_path.name
    )
    if conflict:
        return None, conflict, "conflict", None

    if effective is not None:
        action = "existing" if persisted not in (None, "") else "adopted"
    else:
        if not generate_if_missing:
            return None, None, "none", None
        effective = _generate_fernet_key()
        action = "generated"

    # Persist the data key (unless the target ``.env`` already holds exactly this
    # value) together with any ``extra_updates`` in a single write, then confirm
    # the data key's round-trip: re-read through the runtime parser and require an
    # exact match. If persisting altered the value (interpolation, quoting), fail
    # closed rather than encrypt an identity the next boot could not decrypt.
    needs_key_write = persisted != effective
    result = None
    if needs_key_write or extra:
        payload = dict(extra)
        if needs_key_write:
            payload[DATA_KEY_ENV] = effective
        result = write_env(env_path, payload)
        if needs_key_write and read_persisted_data_key(env_path) != effective:
            return None, (
                f"{DATA_KEY_ENV} could not be persisted to {env_path.name} "
                "without altering its value (round-trip check failed). Refusing "
                "to encrypt an identity the next boot could not decrypt — remove "
                "shell-special characters from the key, or provide a "
                "Fernet-shaped key."
            ), "conflict", None

    # Make the effective key authoritative in the live process, overriding any
    # stale value, so inception encrypts with exactly the persisted key.
    os.environ[DATA_KEY_ENV] = effective
    return effective, None, action, result


def run(ctx: SetupContext) -> None:
    """Ensure data + API keys exist in ``.env`` with a single effective key."""
    env = read_env(ctx.env_path)

    # CHECK is read-only: it never prompts, writes, or mutates ``os.environ``.
    # It validates the *persisted* key through the same canonical parser and
    # validation setup uses (#2468) — not mere presence — so a corrupt or
    # non-round-trippable key is reported here, not discovered at inception.
    if ctx.flow is Flow.CHECK:
        persisted = read_persisted_data_key(ctx.env_path)
        if persisted in (None, ""):
            ctx.block(f"{DATA_KEY_ENV} missing — encrypted memory cannot be unsealed")
        elif not _is_valid_data_key(persisted):
            ctx.block(
                f"{DATA_KEY_ENV} in {ctx.env_path.name} is not a valid master "
                "key (must be a single-line, non-empty value with no control "
                "characters)."
            )
        return

    updates: dict[str, str] = {}

    # Decide the API-key update *before* touching the data key so both land in a
    # single ``write_env`` inside ``ensure_effective_data_key`` — a split write
    # would create the ``.env`` for the data key and then back it up for the API
    # key, leaving a spurious ``.env.backup-*`` on a fresh home (#2468).
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

    effective_key, conflict, action, result = ensure_effective_data_key(
        ctx.env_path, generate_if_missing=True, extra_updates=updates
    )
    if conflict:
        # Fail before inception rather than silently pick a winner, and HALT the
        # wizard so no later, key-dependent step (payments encrypting operator
        # credentials, etc.) runs under an unresolved custody conflict (#2468).
        ctx.block(conflict)
        ctx.halt(conflict)
        return

    if action == "existing":
        if ctx.flow is Flow.INTERACTIVE:
            ctx.prompter.info(f"{DATA_KEY_ENV} already set; leaving untouched.")
    elif action == "adopted":
        ctx.record(
            f"Adopted exported {DATA_KEY_ENV} into {ctx.env_path.name} so the "
            "next boot uses the same key"
        )
    elif action == "generated":
        ctx.record(f"Generated {DATA_KEY_ENV} (Fernet, 32-byte url-safe base64)")

    if result is not None and result.backup_path is not None:
        ctx.record(f"Backed up existing .env to {result.backup_path.name}")

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
    encryption key via PBKDF2 over the *exact* UTF-8 bytes). So the guard is
    deliberately permissive about shape — an internal space is a legitimate part
    of a passphrase and must **not** be stripped or rejected — but rejects
    material that is empty or that embeds control characters (newline, tab,
    ``\\r``, ``DEL``). Those cannot survive a single-line ``.env`` round-trip and
    signal corruption. Never logs or echoes the value.
    """
    if value in (None, ""):
        return False
    return not any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)


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
