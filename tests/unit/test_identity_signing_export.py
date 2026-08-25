"""Protected filesystem behavior for signing.sign_and_export (#2505)."""

from __future__ import annotations

import os
import stat
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.identity import signing
from kestrel_sovereign.identity.protected_export import IdentityExportSecurityError


def _signed_package(monkeypatch, payload: str = '{"signed": true}'):
    package = MagicMock()
    signed = MagicMock()
    signed.to_json.return_value = payload
    monkeypatch.setattr(signing, "sign_package", lambda *_args, **_kwargs: signed)
    return package


@pytest.mark.asyncio
async def test_sign_and_export_creates_private_file_under_permissive_umask(
    tmp_path,
    monkeypatch,
):
    package = _signed_package(monkeypatch)
    export_root = tmp_path / "exports"
    output = export_root / "identity_signed.json"
    monkeypatch.setenv("KESTREL_DATA_DIR", str(export_root))

    previous_umask = os.umask(0)
    try:
        returned_json = await signing.sign_and_export(package, output_path=output)
    finally:
        os.umask(previous_umask)

    assert returned_json == '{"signed": true}'
    assert output.read_text(encoding="utf-8") == returned_json
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_sign_and_export_refuses_implicit_clobber(tmp_path, monkeypatch):
    package = _signed_package(monkeypatch)
    output = tmp_path / "exports" / "identity_signed.json"
    output.parent.mkdir(mode=0o700)
    output.write_text("original", encoding="utf-8")
    output.chmod(0o600)
    monkeypatch.setenv("KESTREL_DATA_DIR", str(output.parent))

    with pytest.raises(FileExistsError):
        await signing.sign_and_export(package, output_path=output)

    assert output.read_text(encoding="utf-8") == "original"
    assert list(output.parent.glob(".identity-export-*")) == []


@pytest.mark.asyncio
async def test_sign_and_export_explicit_replacement_is_root_scoped(
    tmp_path,
    monkeypatch,
):
    package = _signed_package(monkeypatch, payload="replacement")
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o755)
    output = export_root / "identity_signed.json"
    output.write_text("legacy", encoding="utf-8")
    output.chmod(0o644)
    monkeypatch.setenv("KESTREL_DATA_DIR", str(export_root))

    returned_json = await signing.sign_and_export(
        package,
        output_path=output,
        replace_existing=True,
    )

    assert returned_json == "replacement"
    assert output.read_text(encoding="utf-8") == "replacement"
    assert stat.S_IMODE(export_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_sign_and_export_replacement_outside_configured_roots_is_refused(
    tmp_path,
    monkeypatch,
):
    package = _signed_package(monkeypatch, payload="replacement")
    configured = tmp_path / "configured"
    configured.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    # chmod, not mkdir(mode=...): mkdir's mode is masked by the process umask,
    # so under a 0o077 umask this deliberately-loose directory was created
    # 0o700 and the closing assertion failed against the test's own setup.
    outside.chmod(0o755)
    output = outside / "identity_signed.json"
    output.write_text("original", encoding="utf-8")
    output.chmod(0o644)
    monkeypatch.setenv("KESTREL_DATA_DIR", str(configured))

    with pytest.raises(IdentityExportSecurityError, match="outside configured"):
        await signing.sign_and_export(
            package,
            output_path=output,
            replace_existing=True,
        )

    assert output.read_text(encoding="utf-8") == "original"
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    assert stat.S_IMODE(outside.stat().st_mode) == 0o755
