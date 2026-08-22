"""Custody tests for protected local identity-export publication (#2505)."""

from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kestrel_sovereign.identity import protected_export as protected


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _temporary_entries(root: Path) -> list[Path]:
    return list(root.glob(".identity-export-*")) if root.is_dir() else []


def test_new_export_is_private_atomic_and_umask_independent(tmp_path):
    export_root = tmp_path / "exports"
    output = export_root / "identity_private.json"

    previous_umask = os.umask(0)
    try:
        returned = protected.write_protected_identity_export(output, '{"secret": true}')
    finally:
        os.umask(previous_umask)

    assert returned == output.absolute()
    assert _mode(export_root) == 0o700
    assert _mode(output) == 0o600
    assert output.read_text(encoding="utf-8") == '{"secret": true}'
    assert _temporary_entries(export_root) == []


def test_existing_export_is_not_clobbered_by_default(tmp_path):
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    output = export_root / "identity_existing.json"
    output.write_text("original", encoding="utf-8")
    output.chmod(0o600)

    with pytest.raises(FileExistsError):
        protected.write_protected_identity_export(output, "replacement")

    assert output.read_text(encoding="utf-8") == "original"
    assert _temporary_entries(export_root) == []


def test_existing_symlink_is_not_followed_or_clobbered(tmp_path):
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    output = export_root / "identity_link.json"
    output.symlink_to(outside)

    with pytest.raises(FileExistsError):
        protected.write_protected_identity_export(output, "replacement")

    assert output.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside"
    assert _temporary_entries(export_root) == []


def test_symlink_or_non_directory_export_root_is_refused(tmp_path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(OSError):
        protected.write_protected_identity_export(
            linked_root / "identity_linked.json",
            "payload",
        )
    assert list(real_root.iterdir()) == []

    file_root = tmp_path / "not-a-directory"
    file_root.write_text("sentinel", encoding="utf-8")
    with pytest.raises(OSError):
        protected.write_protected_identity_export(
            file_root / "identity_invalid.json",
            "payload",
        )
    assert file_root.read_text(encoding="utf-8") == "sentinel"


def test_write_failure_removes_private_stage_and_final(tmp_path, monkeypatch):
    export_root = tmp_path / "exports"
    output = export_root / "identity_failed.json"

    def fail_after_partial_write(descriptor: int, _payload: bytes) -> None:
        os.write(descriptor, b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(protected, "_write_payload", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated write failure"):
        protected.write_protected_identity_export(output, "complete")

    assert not output.exists()
    assert _temporary_entries(export_root) == []


def test_publish_failure_removes_private_stage_and_final(tmp_path, monkeypatch):
    export_root = tmp_path / "exports"
    output = export_root / "identity_failed.json"

    def fail_publish(
        _directory_fd: int,
        _temp_name: str,
        _final_name: str,
        _directory_path: Path,
        _expected: os.stat_result,
    ) -> None:
        raise OSError("simulated publish failure")

    monkeypatch.setattr(protected, "_publish_new_export", fail_publish)
    with pytest.raises(OSError, match="simulated publish failure"):
        protected.write_protected_identity_export(output, "complete")

    assert not output.exists()
    assert _temporary_entries(export_root) == []


def test_post_publish_validation_failure_removes_final(tmp_path, monkeypatch):
    export_root = tmp_path / "exports"
    output = export_root / "identity_failed.json"

    def fail_validation(
        _directory_fd: int,
        _directory: Path,
        _name: str,
        _expected: os.stat_result,
    ) -> None:
        raise OSError("simulated final validation failure")

    monkeypatch.setattr(protected, "_validate_published_entry", fail_validation)
    with pytest.raises(OSError, match="simulated final validation failure"):
        protected.write_protected_identity_export(output, "complete")

    assert not output.exists()
    assert _temporary_entries(export_root) == []


def test_explicit_existing_replacement_is_private_and_atomic(tmp_path):
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o755)
    output = export_root / "identity_existing.json"
    output.write_text("legacy", encoding="utf-8")
    output.chmod(0o644)

    protected.write_protected_identity_export(
        output,
        "replacement",
        replace_existing=True,
        allowed_replacement_roots=(export_root,),
    )

    assert output.read_text(encoding="utf-8") == "replacement"
    assert _mode(export_root) == 0o700
    assert _mode(output) == 0o600
    assert _temporary_entries(export_root) == []


def test_explicit_replacement_requires_generated_name_and_configured_root(tmp_path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir(mode=0o700)
    arbitrary = allowed_root / "other.json"
    arbitrary.write_text("original", encoding="utf-8")
    arbitrary.chmod(0o600)

    with pytest.raises(
        protected.IdentityExportSecurityError,
        match=r"identity_\*\.json",
    ):
        protected.write_protected_identity_export(
            arbitrary,
            "replacement",
            replace_existing=True,
            allowed_replacement_roots=(allowed_root,),
        )
    assert arbitrary.read_text(encoding="utf-8") == "original"

    outside_root = tmp_path / "outside"
    outside_root.mkdir(mode=0o700)
    outside = outside_root / "identity_existing.json"
    outside.write_text("original", encoding="utf-8")
    outside.chmod(0o644)
    with pytest.raises(
        protected.IdentityExportSecurityError,
        match="outside configured data roots",
    ):
        protected.write_protected_identity_export(
            outside,
            "replacement",
            replace_existing=True,
            allowed_replacement_roots=(allowed_root,),
        )
    assert outside.read_text(encoding="utf-8") == "original"
    assert _mode(outside) == 0o644


def test_explicit_replacement_refuses_symlink(tmp_path):
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    output = export_root / "identity_existing.json"
    output.symlink_to(outside)

    with pytest.raises(protected.IdentityExportSecurityError):
        protected.write_protected_identity_export(
            output,
            "replacement",
            replace_existing=True,
            allowed_replacement_roots=(export_root,),
        )

    assert outside.read_text(encoding="utf-8") == "outside"
    assert output.is_symlink()
    assert _temporary_entries(export_root) == []


def test_replace_failure_preserves_existing_and_cleans_stage(tmp_path, monkeypatch):
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    output = export_root / "identity_existing.json"
    output.write_text("original", encoding="utf-8")
    output.chmod(0o644)
    real_replace = protected.os.replace

    def fail_initial_replace(src, dst, **kwargs):
        if str(src).endswith(".tmp"):
            raise OSError("simulated replace failure")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(protected.os, "replace", fail_initial_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        protected.write_protected_identity_export(
            output,
            "replacement",
            replace_existing=True,
            allowed_replacement_roots=(export_root,),
        )

    assert output.read_text(encoding="utf-8") == "original"
    assert _mode(output) == 0o600
    assert _temporary_entries(export_root) == []


def test_replace_validation_failure_rolls_back_existing(tmp_path, monkeypatch):
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    output = export_root / "identity_existing.json"
    output.write_text("original", encoding="utf-8")
    output.chmod(0o644)

    def fail_validation(
        _directory_fd: int,
        _directory: Path,
        _name: str,
        _expected: os.stat_result,
    ) -> None:
        raise OSError("simulated final validation failure")

    monkeypatch.setattr(protected, "_validate_published_entry", fail_validation)
    with pytest.raises(OSError, match="simulated final validation failure"):
        protected.write_protected_identity_export(
            output,
            "replacement",
            replace_existing=True,
            allowed_replacement_roots=(export_root,),
        )

    assert output.read_text(encoding="utf-8") == "original"
    assert _mode(output) == 0o600
    assert _temporary_entries(export_root) == []


def test_final_validation_rejects_a_different_regular_inode(tmp_path):
    """A same-mode raced-in file is not mistaken for the staged payload."""

    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    expected_path = export_root / "expected.json"
    expected_path.write_text("expected", encoding="utf-8")
    expected_path.chmod(0o600)
    final_path = export_root / "identity_raced.json"
    final_path.write_text("raced", encoding="utf-8")
    final_path.chmod(0o600)
    directory_fd = os.open(export_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(
            protected.IdentityExportSecurityError,
            match="changed during publication",
        ):
            protected._validate_published_entry(
                directory_fd,
                export_root,
                final_path.name,
                expected_path.stat(),
            )
    finally:
        os.close(directory_fd)


def test_concurrent_new_exports_remain_distinct_and_private(tmp_path):
    export_root = tmp_path / "exports"
    outputs = [export_root / f"identity_concurrent_{index}.json" for index in range(32)]

    with ThreadPoolExecutor(max_workers=12) as pool:
        returned = list(
            pool.map(
                lambda item: protected.write_protected_identity_export(
                    item[1],
                    f"payload-{item[0]}",
                ),
                enumerate(outputs),
            )
        )

    assert len(set(returned)) == len(outputs)
    assert _mode(export_root) == 0o700
    assert _temporary_entries(export_root) == []
    for index, output in enumerate(outputs):
        assert output.read_text(encoding="utf-8") == f"payload-{index}"
        assert _mode(output) == 0o600


def test_legacy_hardening_is_metadata_only_and_refuses_unsafe_entries(tmp_path):
    export_root = tmp_path / "exports"
    export_root.mkdir()
    # chmod, not mkdir(mode=...): mkdir's mode is masked by the process umask,
    # so under a 0o077 umask this deliberately-legacy root was created 0o700
    # and the "export root is not mode 0700" finding could never be produced.
    export_root.chmod(0o755)
    legacy = export_root / "identity_legacy.json"
    legacy.write_text("legacy-secret", encoding="utf-8")
    legacy.chmod(0o644)
    private = export_root / "identity_private.json"
    private.write_text("private-secret", encoding="utf-8")
    private.chmod(0o600)
    outside = tmp_path / "outside.json"
    outside.write_text("outside-secret", encoding="utf-8")
    linked = export_root / "identity_link.json"
    linked.symlink_to(outside)
    unrelated = export_root / "notes.json"
    unrelated.write_text("unrelated", encoding="utf-8")
    unrelated.chmod(0o644)

    findings = protected.audit_legacy_identity_exports((export_root,))
    assert {finding.reason for finding in findings} >= {
        "export root is not mode 0700",
        "entry is not mode 0600",
        "entry is a symbolic link",
    }

    result = protected.harden_legacy_identity_exports((export_root,))
    assert result == protected.LegacyIdentityExportHardeningResult(
        hardened=1,
        already_private=1,
        refused=1,
        missing_roots=0,
    )
    assert _mode(export_root) == 0o700
    assert _mode(legacy) == 0o600
    assert _mode(private) == 0o600
    assert _mode(unrelated) == 0o644
    assert outside.read_text(encoding="utf-8") == "outside-secret"
    assert linked.is_symlink()


def test_legacy_hardening_refuses_foreign_owned_regular_entry(tmp_path, monkeypatch):
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    foreign = export_root / "identity_foreign.json"
    foreign.write_text("foreign", encoding="utf-8")
    foreign.chmod(0o644)
    foreign_inode = foreign.stat().st_ino
    real_owner_check = protected._owned_by_operator

    monkeypatch.setattr(
        protected,
        "_owned_by_operator",
        lambda metadata: (
            False if metadata.st_ino == foreign_inode else real_owner_check(metadata)
        ),
    )
    result = protected.harden_legacy_identity_exports((export_root,))

    assert result.refused == 1
    assert result.hardened == 0
    assert _mode(foreign) == 0o644
