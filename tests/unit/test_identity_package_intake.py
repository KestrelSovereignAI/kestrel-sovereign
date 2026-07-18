"""Security regressions for the shared identity-package intake boundary."""

from __future__ import annotations

import asyncio
import os
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest

from kestrel_sovereign.identity import package_intake
from kestrel_sovereign.identity.package_intake import (
    IdentityPackageIntakeError,
    load_identity_package_source,
)


@pytest.mark.asyncio
async def test_valid_plaintext_package_loads(tmp_path):
    package = tmp_path / "identity_plaintext.json"
    package.write_text('{"package_version": "1"}', encoding="utf-8")

    loaded = await load_identity_package_source(str(package))

    assert loaded == '{"package_version": "1"}'


@pytest.mark.asyncio
async def test_valid_sealed_capsule_text_loads_for_downstream_unsealing(tmp_path):
    package = tmp_path / "identity_sealed.json"
    capsule = '{"format": "kestrel-sealed-identity-v1", "ciphertext": "opaque"}'
    package.write_text(capsule, encoding="utf-8")

    assert await load_identity_package_source(str(package)) == capsule


@pytest.mark.asyncio
async def test_symlink_is_rejected_without_reading_target(tmp_path):
    marker = "target-package-content-must-not-leak"
    target = tmp_path / "outside.json"
    target.write_text(marker, encoding="utf-8")
    source = tmp_path / "identity_link.json"
    source.symlink_to(target)

    with pytest.raises(IdentityPackageIntakeError) as caught:
        await load_identity_package_source(str(source))

    assert "regular file" in str(caught.value)
    assert marker not in str(caught.value)


@pytest.mark.asyncio
async def test_directory_is_rejected(tmp_path):
    with pytest.raises(IdentityPackageIntakeError, match="regular file"):
        await load_identity_package_source(str(tmp_path))


@pytest.mark.asyncio
async def test_fifo_is_rejected_without_blocking(tmp_path):
    source = tmp_path / "identity_pipe.json"
    os.mkfifo(source)

    with pytest.raises(IdentityPackageIntakeError, match="regular file"):
        await asyncio.wait_for(load_identity_package_source(str(source)), timeout=1)


@pytest.mark.asyncio
async def test_unix_socket_is_rejected():
    with tempfile.TemporaryDirectory(prefix="ks-intake-", dir="/tmp") as root:
        source = Path(root) / "identity.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(source))
            with pytest.raises(IdentityPackageIntakeError, match="regular file"):
                await load_identity_package_source(str(source))
        finally:
            server.close()


@pytest.mark.asyncio
async def test_device_is_rejected_before_read():
    device = Path("/dev/null")
    if not device.exists():
        pytest.skip("no test device on this platform")

    with pytest.raises(IdentityPackageIntakeError, match="regular file"):
        await load_identity_package_source(str(device))


@pytest.mark.asyncio
async def test_group_or_other_writable_file_is_rejected(tmp_path):
    source = tmp_path / "identity_writable.json"
    source.write_text("{}", encoding="utf-8")
    source.chmod(0o666)

    with pytest.raises(IdentityPackageIntakeError, match="writable"):
        await load_identity_package_source(str(source))


@pytest.mark.asyncio
async def test_oversized_local_file_is_bounded_without_content_in_error(
    tmp_path,
    monkeypatch,
):
    marker = b"oversized-secret-package-content"
    source = tmp_path / "identity_oversized.json"
    source.write_bytes(marker)
    monkeypatch.setattr(package_intake, "MAX_IDENTITY_PACKAGE_BYTES", 8)

    with pytest.raises(IdentityPackageIntakeError) as caught:
        await load_identity_package_source(str(source))

    assert "8-byte limit" in str(caught.value)
    assert marker.decode() not in str(caught.value)


@pytest.mark.asyncio
async def test_path_swap_between_lstat_and_open_cannot_follow_link(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "identity_raced.json"
    source.write_text("{}", encoding="utf-8")
    marker = "raced-target-package-content"
    target = tmp_path / "outside.json"
    target.write_text(marker, encoding="utf-8")
    real_open = package_intake.os.open
    swapped = False

    def swap_then_open(path, flags):
        nonlocal swapped
        if Path(path) == source and not swapped:
            swapped = True
            source.unlink()
            source.symlink_to(target)
        return real_open(path, flags)

    monkeypatch.setattr(package_intake.os, "open", swap_then_open)

    with pytest.raises(IdentityPackageIntakeError) as caught:
        await load_identity_package_source(str(source))

    assert swapped
    assert marker not in str(caught.value)


@pytest.mark.asyncio
async def test_regular_file_swap_between_lstat_and_open_is_rejected(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "identity_raced.json"
    source.write_text("{}", encoding="utf-8")
    marker = "replacement-regular-file-content"
    replacement = tmp_path / "replacement.json"
    replacement.write_text(marker, encoding="utf-8")
    real_open = package_intake.os.open
    swapped = False

    def swap_then_open(path, flags):
        nonlocal swapped
        if Path(path) == source and not swapped:
            swapped = True
            os.replace(replacement, source)
        return real_open(path, flags)

    monkeypatch.setattr(package_intake.os, "open", swap_then_open)

    with pytest.raises(IdentityPackageIntakeError) as caught:
        await load_identity_package_source(str(source))

    assert swapped
    assert "changed" in str(caught.value)
    assert marker not in str(caught.value)


@pytest.mark.asyncio
async def test_slow_local_reader_does_not_block_event_loop(tmp_path, monkeypatch):
    source = tmp_path / "identity_slow.json"
    source.write_text("{}", encoding="utf-8")
    reader_started = threading.Event()

    def slow_reader(path):
        reader_started.set()
        time.sleep(0.15)
        return "{}"

    monkeypatch.setattr(package_intake, "_read_local_identity_package", slow_reader)
    load_task = asyncio.create_task(load_identity_package_source(str(source)))
    while not reader_started.is_set():
        await asyncio.sleep(0)

    await asyncio.sleep(0.02)

    assert not load_task.done()
    assert await load_task == "{}"


@pytest.mark.asyncio
async def test_cid_loader_forwards_bound_and_key_hash(monkeypatch):
    captured = {}

    class FakeAdapter:
        def retrieve_content(self, content_hash, **kwargs):
            captured["content_hash"] = content_hash
            captured.update(kwargs)
            return b"{}"

    import kestrel_sovereign.filecoin_adapter as adapter_module

    monkeypatch.setattr(adapter_module, "FilecoinAdapter", FakeAdapter)

    loaded = await load_identity_package_source("QmIdentityCid", key_hash="key-hash")

    assert loaded == "{}"
    assert captured == {
        "content_hash": "QmIdentityCid",
        "ipfs_cid": "QmIdentityCid",
        "key_hash": "key-hash",
        "max_output_bytes": package_intake.MAX_IDENTITY_PACKAGE_BYTES,
    }


@pytest.mark.asyncio
async def test_oversized_cid_result_is_rejected_even_if_adapter_ignores_bound(
    monkeypatch,
):
    marker = b"remote-secret-content"

    class NoncompliantAdapter:
        def retrieve_content(self, *args, **kwargs):
            return marker

    import kestrel_sovereign.filecoin_adapter as adapter_module

    monkeypatch.setattr(package_intake, "MAX_IDENTITY_PACKAGE_BYTES", 4)
    monkeypatch.setattr(adapter_module, "FilecoinAdapter", NoncompliantAdapter)

    with pytest.raises(IdentityPackageIntakeError) as caught:
        await load_identity_package_source("bafyIdentityCid")

    assert "4-byte limit" in str(caught.value)
    assert marker.decode() not in str(caught.value)


@pytest.mark.asyncio
async def test_cid_backend_error_does_not_echo_backend_secret(monkeypatch):
    marker = "backend-secret-package-content"

    class FailingAdapter:
        def retrieve_content(self, *args, **kwargs):
            raise RuntimeError(marker)

    import kestrel_sovereign.filecoin_adapter as adapter_module

    monkeypatch.setattr(adapter_module, "FilecoinAdapter", FailingAdapter)

    with pytest.raises(IdentityPackageIntakeError) as caught:
        await load_identity_package_source("QmIdentityCid")

    assert "RuntimeError" in str(caught.value)
    assert marker not in str(caught.value)
