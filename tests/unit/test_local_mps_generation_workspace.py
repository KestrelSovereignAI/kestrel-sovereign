"""Adversarial filesystem contracts for Local MPS generation workspaces."""

import json
import os
import shutil
import sys

import pytest

from kestrel_sovereign.features.training.adapters import (
    _local_mps_generation_lifecycle as generation_lifecycle,
)
from kestrel_sovereign.features.training.adapters import (
    _local_mps_generation_workspace as generation_workspace,
)


def test_workspace_creation_rejects_inode_swapped_before_open(tmp_path, monkeypatch):
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    moved_workspace = tmp_path / "created-workspace-moved"
    replacement_workspace = None
    real_open = generation_workspace.os.open

    def swap_before_workspace_open(path, flags, *args, dir_fd=None):
        nonlocal replacement_workspace
        if (
            isinstance(path, str)
            and path.startswith(generation_lifecycle.GENERATION_WORKSPACE_PREFIX)
            and dir_fd is not None
        ):
            workspace = working_dir / path
            workspace.rename(moved_workspace)
            workspace.mkdir()
            (workspace / "replacement-must-survive").write_text(
                "replacement",
                encoding="utf-8",
            )
            replacement_workspace = workspace
        return real_open(path, flags, *args, dir_fd=dir_fd)

    monkeypatch.setattr(generation_workspace.os, "open", swap_before_workspace_open)

    with pytest.raises(
        generation_lifecycle.GenerationWorkspaceIdentityError,
        match="changed identity while it was opened",
    ):
        generation_lifecycle._create_generation_workspace(working_dir)

    assert replacement_workspace is not None
    assert (replacement_workspace / "replacement-must-survive").read_text(
        encoding="utf-8"
    ) == "replacement"
    assert moved_workspace.is_dir()
    shutil.rmtree(replacement_workspace)
    shutil.rmtree(moved_workspace)


def test_workspace_cleanup_refuses_replacement_leaf_inode(tmp_path):
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    lease = generation_lifecycle._create_generation_workspace(working_dir)
    original_workspace = lease.path
    moved_workspace = working_dir / "original-workspace"
    original_workspace.rename(moved_workspace)
    original_workspace.mkdir()
    replacement_sentinel = original_workspace / "replacement-must-survive"
    replacement_sentinel.write_text("replacement", encoding="utf-8")

    with pytest.raises(
        generation_lifecycle.GenerationWorkspaceIdentityError,
        match="workspace path was replaced",
    ):
        generation_lifecycle._cleanup_generation_workspace(lease)

    assert replacement_sentinel.read_text(encoding="utf-8") == "replacement"
    assert moved_workspace.is_dir()
    shutil.rmtree(original_workspace)
    shutil.rmtree(moved_workspace)


def test_workspace_cleanup_refuses_leaf_swapped_after_initial_validation(
    tmp_path,
    monkeypatch,
):
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    lease = generation_lifecycle._create_generation_workspace(working_dir)
    generation_lifecycle._create_generation_artifact(
        lease,
        "generated.png",
        b"private-output",
    )
    original_workspace = lease.path
    moved_workspace = working_dir / "original-workspace"
    replacement_sentinel = original_workspace / "replacement-must-survive"
    real_clear = generation_workspace._clear_directory_contents

    def swap_then_clear(directory_fd):
        original_workspace.rename(moved_workspace)
        original_workspace.mkdir()
        replacement_sentinel.write_text("replacement", encoding="utf-8")
        real_clear(directory_fd)

    monkeypatch.setattr(
        generation_workspace,
        "_clear_directory_contents",
        swap_then_clear,
    )

    with pytest.raises(
        generation_lifecycle.GenerationWorkspaceIdentityError,
        match="workspace path was replaced",
    ):
        generation_lifecycle._cleanup_generation_workspace(lease)

    assert replacement_sentinel.read_text(encoding="utf-8") == "replacement"
    assert moved_workspace.is_dir()
    shutil.rmtree(original_workspace)
    shutil.rmtree(moved_workspace)


def test_workspace_cleanup_refuses_replacement_root_inode(tmp_path):
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    lease = generation_lifecycle._create_generation_workspace(working_dir)
    moved_root = tmp_path / "original-working-root"
    working_dir.rename(moved_root)
    working_dir.mkdir()
    replacement_sentinel = working_dir / "replacement-root-must-survive"
    replacement_sentinel.write_text("replacement", encoding="utf-8")

    with pytest.raises(
        generation_lifecycle.GenerationWorkspaceIdentityError,
        match="working root was replaced",
    ):
        generation_lifecycle._cleanup_generation_workspace(lease)

    assert replacement_sentinel.read_text(encoding="utf-8") == "replacement"
    assert (moved_root / lease.workspace_name).is_dir()
    shutil.rmtree(working_dir)
    shutil.rmtree(moved_root)


def test_private_lora_creation_refuses_swapped_workspace_symlink(tmp_path):
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_sentinel = outside / "outside-must-survive"
    outside_sentinel.write_text("outside", encoding="utf-8")
    lease = generation_lifecycle._create_generation_workspace(working_dir)
    original_workspace = lease.path
    moved_workspace = working_dir / "original-workspace"
    original_workspace.rename(moved_workspace)
    original_workspace.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        generation_lifecycle.GenerationWorkspaceIdentityError,
        match="workspace path was replaced",
    ):
        generation_lifecycle._create_generation_artifact(
            lease,
            "lora.safetensors",
            b"private-lora",
        )

    assert outside_sentinel.read_text(encoding="utf-8") == "outside"
    assert not (outside / "lora.safetensors").exists()
    with pytest.raises(generation_lifecycle.GenerationWorkspaceIdentityError):
        generation_lifecycle._cleanup_generation_workspace(lease)
    original_workspace.unlink()
    shutil.rmtree(moved_workspace)
    shutil.rmtree(outside)


@pytest.mark.skipif(os.name != "posix", reason="POSIX inherited-fd contract")
@pytest.mark.asyncio
async def test_subprocess_output_fd_cannot_follow_swapped_workspace_leaf(tmp_path):
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    lease = generation_lifecycle._create_generation_workspace(working_dir)
    output = generation_lifecycle._create_generation_artifact(
        lease,
        "generated.png",
        None,
    )
    gate = tmp_path / "allow-output-write"
    script = r"""
import json
import os
import sys
import time
from pathlib import Path

payload = json.loads(sys.argv[1])
while not Path(payload["gate"]).exists():
    time.sleep(0.01)
os.ftruncate(payload["output_fd"], 0)
os.pwrite(payload["output_fd"], b"fd-bound-output", 0)
"""
    process_lease = await generation_lifecycle.start_generation_process(
        sys.executable,
        script,
        json.dumps({"gate": str(gate), "output_fd": output.fd}),
        inherited_fds=[output.fd],
        workspace=lease,
    )

    original_workspace = lease.path
    moved_workspace = working_dir / "original-workspace"
    original_workspace.rename(moved_workspace)
    original_workspace.symlink_to(outside, target_is_directory=True)
    gate.touch()
    await process_lease.process.communicate()

    output_size = os.fstat(output.fd).st_size
    assert os.pread(output.fd, output_size, 0) == b"fd-bound-output"
    assert not (outside / "generated.png").exists()
    with pytest.raises(
        generation_lifecycle.GenerationWorkspaceIdentityError,
        match="workspace path was replaced",
    ):
        await generation_lifecycle.finalize_generation_resources(
            process=process_lease,
            process_communicated=True,
            workspace=lease,
        )
    original_workspace.unlink()
    shutil.rmtree(moved_workspace)
    shutil.rmtree(outside)
