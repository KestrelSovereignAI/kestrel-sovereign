"""Generate encryption + API keys into ``.env``.

Two keys matter for v1:

  ``KESTREL_DATA_KEY``
      Fernet key used to encrypt agent data at rest. **Never** regenerated
      if already present — doing so would brick existing encrypted DBs.
      The wizard refuses to overwrite a non-empty value.

  ``KESTREL_API_KEY``
      Optional API key for HTTP auth. Auto-generated if absent and the
      user opts in; otherwise left empty (server auto-generates per-run).
"""

from __future__ import annotations

import secrets

from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.env_file import read_env, write_env


def run(ctx: SetupContext) -> None:
    """Ensure data + API keys exist in ``.env``."""
    env = read_env(ctx.env_path)
    updates: dict[str, str] = {}

    data_key_present = bool(env.get("KESTREL_DATA_KEY"))
    if not data_key_present:
        if ctx.flow is Flow.CHECK:
            ctx.block("KESTREL_DATA_KEY missing — encrypted memory cannot be unsealed")
            return
        updates["KESTREL_DATA_KEY"] = _generate_fernet_key()
        ctx.record("Generated KESTREL_DATA_KEY (Fernet, 32-byte url-safe base64)")
    elif ctx.flow is Flow.INTERACTIVE:
        ctx.prompter.info("KESTREL_DATA_KEY already set; leaving untouched.")

    api_key_present = bool(env.get("KESTREL_API_KEY"))
    if not api_key_present:
        if ctx.flow is Flow.CHECK:
            # API key absence isn't a blocker — server auto-generates one.
            pass
        elif ctx.flow is Flow.QUICKSTART:
            updates["KESTREL_API_KEY"] = _generate_api_key()
            ctx.record("Generated KESTREL_API_KEY (32-byte url-safe token)")
        else:
            wants_key = ctx.prompter.confirm(
                "Generate a stable KESTREL_API_KEY now? "
                "(otherwise the server picks one per run)",
                default=True,
            )
            if wants_key:
                updates["KESTREL_API_KEY"] = _generate_api_key()
                ctx.record("Generated KESTREL_API_KEY (32-byte url-safe token)")

    if not updates:
        return
    if ctx.flow is Flow.CHECK:
        return

    result = write_env(ctx.env_path, updates)
    if result.backup_path is not None:
        ctx.record(f"Backed up existing .env to {result.backup_path.name}")


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
