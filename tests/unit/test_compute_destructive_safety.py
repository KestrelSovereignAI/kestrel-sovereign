"""Focused regressions for compute shell rewriting and trash containment."""

import subprocess
import sys
from pathlib import Path

import pytest

from kestrel_sovereign.features.compute.destructive_policy import (
    DestructiveOperationPolicy,
    ShellRewriteError,
)
from kestrel_sovereign.features.compute.executors.local_executor import LocalExecutor
from kestrel_sovereign.features.compute.models import ComputeScript
from kestrel_sovereign.features.compute.trash_manager import TrashManager
from kestrel_sovereign.features.compute import trash_manager as trash_manager_module


@pytest.fixture
def trash_dir(tmp_path: Path) -> Path:
    path = tmp_path / "trash"
    path.mkdir()
    return path


@pytest.mark.parametrize(
    ("script", "preserved_boundary"),
    [
        ("true; rm /data/late", "true; ("),
        ("true && rm /data/late", "true && ("),
        ("false || rm /data/late", "false || ("),
        ("printf x | rm /data/late", "printf x | ("),
        ("rm /data/late & wait", ") & wait"),
        ("printf before\nrm /data/late", "printf before\n("),
    ],
)
def test_each_compound_rm_segment_is_rewritten(
    trash_dir: Path,
    script: str,
    preserved_boundary: str,
) -> None:
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)

    rewritten = policy.rewrite_bash_script(script)

    assert rewritten.count("command -p mv --") == 1
    assert "builtin command" not in rewritten
    assert preserved_boundary in rewritten


def test_multiple_quoted_rm_commands_execute_as_independent_safe_moves(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    first = tmp_path / "first file.txt"
    second = tmp_path / "second file.txt"
    first.write_text("first")
    second.write_text("second")
    script = f'rm -f "{first}"; printf between; rm -- "{second}"'
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)

    rewritten = policy.rewrite_bash_script(script)
    completed = subprocess.run(
        ["sh", "-c", rewritten],
        check=False,
        capture_output=True,
        text=True,
    )

    assert rewritten.count("command -p mv --") == 2
    assert "; printf between; " in rewritten
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "between"
    assert not first.exists()
    assert not second.exists()
    assert sorted(path.name for path in trash_dir.glob("*/*")) == [
        "first file.txt",
        "second file.txt",
    ]


def test_one_rm_with_duplicate_basenames_preserves_both_files(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    first = tmp_path / "first" / "same.txt"
    second = tmp_path / "second" / "same.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first")
    second.write_text("second")
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)

    rewritten = policy.rewrite_bash_script(f'rm "{first}" "{second}"')
    completed = subprocess.run(
        ["sh", "-c", rewritten],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    trashed = sorted(trash_dir.glob("*/same.txt"))
    assert len(trashed) == 2
    assert {path.read_text() for path in trashed} == {"first", "second"}


def test_generated_commands_bypass_utility_function_overrides(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    target = tmp_path / "override.txt"
    target.write_text("keep me")
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)
    script = f'mkdir() {{ return 97; }}; mv() {{ return 98; }}; rm "{target}"'

    rewritten = policy.rewrite_bash_script(script)
    completed = subprocess.run(
        ["sh", "-c", rewritten],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not target.exists()
    assert next(trash_dir.glob("*/override.txt")).read_text() == "keep me"


def test_command_override_is_rejected_fail_closed(trash_dir: Path) -> None:
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)

    with pytest.raises(ShellRewriteError, match="POSIX command builtin"):
        policy.rewrite_bash_script("command() { return 96; }; rm /data/override")


def test_static_non_rm_dispatcher_remains_compatible(trash_dir: Path) -> None:
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)

    rewritten = policy.rewrite_bash_script("env printf compatible")

    assert rewritten == "env printf compatible"


def test_non_dispatching_command_may_print_rm_word(trash_dir: Path) -> None:
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)

    rewritten = policy.rewrite_bash_script("printf '%s\\n' rm")

    assert rewritten == "printf '%s\\n' rm"


def test_comments_and_quoted_rm_text_are_not_commands(
    trash_dir: Path,
) -> None:
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)
    script = 'printf "rm /quoted" # rm /comment\nrm /data/real # retain me'

    rewritten = policy.rewrite_bash_script(script)

    assert rewritten.count("command -p mv --") == 1
    assert 'printf "rm /quoted" # rm /comment' in rewritten
    assert rewritten.endswith("# retain me")


def test_absolute_rm_executable_is_rewritten(trash_dir: Path) -> None:
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)

    rewritten = policy.rewrite_bash_script("/bin/rm /data/absolute")

    assert "command -p mv --" in rewritten
    assert "/bin/rm" not in rewritten


def test_safe_workdir_rm_segments_keep_guarded_real_deletion(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_text("first")
    second.write_text("second")
    script = "rm -fv first; rm -f second"

    rewritten = policy.rewrite_bash_script(script, str(tmp_path))

    completed = subprocess.run(
        ["sh", "-c", rewritten],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert "command -p rm -fv --" in rewritten
    assert "command -p rm -f --" in rewritten
    assert completed.returncode == 0, completed.stderr
    assert "first" in completed.stdout
    assert not first.exists()
    assert not second.exists()
    assert list(trash_dir.iterdir()) == []


def test_guarded_rm_still_deletes_valid_operand_when_another_is_missing(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    existing = tmp_path / "existing.txt"
    missing = tmp_path / "missing.txt"
    existing.write_text("existing")
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)

    rewritten = policy.rewrite_bash_script(
        f'rm "{missing}" "{existing}"',
        str(tmp_path),
    )
    completed = subprocess.run(
        ["sh", "-c", rewritten],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not existing.exists()
    assert list(trash_dir.iterdir()) == []


@pytest.mark.parametrize("option", ["-i", "-I", "-v", "--interactive=never"])
def test_semantic_rm_flags_fail_closed_for_trash_moves(
    trash_dir: Path,
    option: str,
) -> None:
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)

    with pytest.raises(ShellRewriteError, match="cannot be preserved"):
        policy.rewrite_bash_script(f"rm {option} /data/important")


def test_one_rewritten_script_can_run_twice_without_trash_id_collision(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    target = tmp_path / "repeat.txt"
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)
    rewritten = policy.rewrite_bash_script(f'rm -f "{target}"')

    for content in ("first", "second"):
        target.write_text(content)
        completed = subprocess.run(
            ["sh", "-c", rewritten],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    trashed = sorted(trash_dir.glob("*/repeat.txt"))
    assert len(trashed) == 2
    assert {path.read_text() for path in trashed} == {"first", "second"}


def test_direct_delete_revalidates_root_identity_at_execution(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    owned_root = tmp_path / "owned-root"
    owned_root.mkdir()
    target = owned_root / "target.txt"
    target.write_text("original")
    policy = DestructiveOperationPolicy(
        trash_dir=trash_dir,
        deletable_prefixes=[str(owned_root)],
    )
    rewritten = policy.rewrite_bash_script(f'rm -f "{target}"')

    moved_root = tmp_path / "moved-owned-root"
    owned_root.rename(moved_root)
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    outside = outside_root / target.name
    outside.write_text("outside")
    owned_root.symlink_to(outside_root, target_is_directory=True)

    completed = subprocess.run(
        ["sh", "-c", rewritten],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "runtime ownership changed" in completed.stderr
    assert outside.read_text() == "outside"
    assert (moved_root / target.name).read_text() == "original"


def test_trash_move_rechecks_cross_agent_ownership_at_execution(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    lexical_parent = tmp_path / "lexical"
    lexical_parent.mkdir()
    target = lexical_parent / "target.txt"
    target.write_text("ordinary")
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)
    rewritten = policy.rewrite_bash_script(f'rm "{target}"')

    lexical_parent.rename(tmp_path / "original-parent")
    other_agent = tmp_path / "agent_data" / "other"
    other_agent.mkdir(parents=True)
    protected = other_agent / target.name
    protected.write_text("protected")
    lexical_parent.symlink_to(other_agent, target_is_directory=True)

    completed = subprocess.run(
        ["sh", "-c", rewritten],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "another agent's runtime data" in completed.stderr
    assert protected.read_text() == "protected"
    assert list(trash_dir.iterdir()) == []


def test_python_runtime_allocates_unique_trash_subdirs(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    target = tmp_path / "python-repeat.txt"
    execution_root = tmp_path / "execution-root"
    execution_root.mkdir()
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)
    content = (
        "import os\n"
        f"target = {str(target)!r}\n"
        "for value in ('first', 'second'):\n"
        "    with open(target, 'w', encoding='utf-8') as handle:\n"
        "        handle.write(value)\n"
        "    os.remove(target)\n"
    )
    script_path = tmp_path / "python-delete.py"
    script_path.write_text(
        policy.rewrite_python_script(content, workdir=str(execution_root))
    )

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    trashed = sorted(trash_dir.glob("*/python-repeat.txt"))
    assert len(trashed) == 2
    assert {path.read_text() for path in trashed} == {"first", "second"}


def test_python_runtime_preserves_legal_module_prologue(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)
    content = (
        "#!/usr/bin/env python3\n"
        "# coding: utf-8\n"
        '"""kept module docstring"""\n'
        "from __future__ import annotations\n"
        "print(__doc__)\n"
    )
    rewritten = policy.rewrite_python_script(content)
    script_path = tmp_path / "prologue.py"
    script_path.write_text(rewritten)

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert rewritten.startswith(content.split("print", 1)[0])
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "kept module docstring"


def test_workdir_prefix_sibling_is_not_deletable(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)
    workdir = tmp_path / "work"
    sibling = tmp_path / "work-escape" / "file.txt"

    assert not policy.is_deletable_path(str(sibling), str(workdir))


@pytest.mark.parametrize(
    "script",
    [
        'echo "$(rm /data/nested)"',
        "sudo rm /data/wrapped",
        "printf '%s\\n' /data/file | xargs rm",
        'rm "$TARGET"',
        "rm ~/important",
        "rm /data/file 2>/dev/null",
        'rm "unterminated',
        "cat <<'EOF'\nrm /data/heredoc\nEOF",
    ],
)
def test_unclassified_rm_forms_fail_closed(
    trash_dir: Path,
    script: str,
) -> None:
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)

    with pytest.raises(ShellRewriteError):
        policy.rewrite_bash_script(script)


@pytest.mark.asyncio
async def test_local_executor_uses_the_same_compound_rewrite(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    first = tmp_path / "runtime first.txt"
    second = tmp_path / "runtime second.txt"
    first.write_text("first")
    second.write_text("second")
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)
    executor = LocalExecutor(require_env_flag=False)
    executor._policy = policy
    script = ComputeScript(
        id="compound-runtime",
        name="compound-runtime",
        language="bash",
        content=f'rm "{first}" && rm "{second}"',
        purpose="exercise the executor rewrite boundary",
    )

    record = await executor.execute(script)

    assert record.exit_code == 0, record.stderr
    assert not first.exists()
    assert not second.exists()
    assert sorted(path.name for path in trash_dir.glob("*/*")) == [
        "runtime first.txt",
        "runtime second.txt",
    ]


def test_restore_accepts_a_real_item_inside_trash(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    manager = TrashManager(trash_dir)
    item = trash_dir / "20260715_120000_000001" / "item.txt"
    item.parent.mkdir()
    item.write_text("trashed")
    destination = tmp_path / "restored.txt"

    restored = manager.restore(item, str(destination))

    assert restored == destination
    assert destination.read_text() == "trashed"
    assert not item.exists()


@pytest.mark.parametrize("source_kind", ["direct", "traversal"])
def test_restore_rejects_an_outside_host_source(
    tmp_path: Path,
    trash_dir: Path,
    source_kind: str,
) -> None:
    manager = TrashManager(trash_dir)
    outside = tmp_path / "outside.txt"
    outside.write_text("host data")
    source = outside if source_kind == "direct" else trash_dir / ".." / outside.name

    with pytest.raises(PermissionError, match="outside trash directory"):
        manager.restore(source, str(tmp_path / "destination.txt"))

    assert outside.read_text() == "host data"


def test_restore_rejects_symlink_that_resolves_outside_trash(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    manager = TrashManager(trash_dir)
    outside = tmp_path / "outside.txt"
    outside.write_text("host data")
    link = trash_dir / "20260715_120000_000001" / "link.txt"
    link.parent.mkdir()
    link.symlink_to(outside)

    with pytest.raises(PermissionError, match="outside trash directory"):
        manager.restore(link, str(tmp_path / "destination.txt"))

    assert outside.read_text() == "host data"
    assert link.is_symlink()


def test_restore_accepts_symlink_whose_target_stays_in_trash(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    manager = TrashManager(trash_dir)
    operation = trash_dir / "20260715_120000_000001"
    operation.mkdir()
    target = operation / "target.txt"
    target.write_text("trashed")
    link = operation / "link.txt"
    link.symlink_to(target.name)
    destination = tmp_path / "restored-link.txt"

    manager.restore(link, str(destination))

    assert destination.is_symlink()
    assert destination.readlink() == Path("target.txt")
    assert target.read_text() == "trashed"


def test_restore_rejects_parent_symlink_escape(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    manager = TrashManager(trash_dir)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "host.txt"
    outside.write_text("host data")
    timestamp_dir = trash_dir / "20260715_120000_000001"
    timestamp_dir.mkdir()
    escape = timestamp_dir / "escape"
    escape.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(PermissionError, match="outside trash directory"):
        manager.restore(escape / outside.name, str(tmp_path / "destination.txt"))

    assert outside.read_text() == "host data"


def test_restore_rejects_the_trash_root_itself(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    manager = TrashManager(trash_dir)

    with pytest.raises(PermissionError, match="outside trash directory"):
        manager.restore(trash_dir, str(tmp_path / "destination"))

    assert trash_dir.exists()


def test_restore_never_replaces_destination_created_during_operation(
    tmp_path: Path,
    trash_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrashManager(trash_dir)
    item = trash_dir / "20260715_120000_000001" / "item.txt"
    item.parent.mkdir()
    item.write_text("trashed")
    destination = tmp_path / "destination.txt"
    original_rename = trash_manager_module._rename_noreplace

    def create_destination_then_rename(*args, **kwargs):
        destination.write_text("concurrent")
        return original_rename(*args, **kwargs)

    monkeypatch.setattr(
        trash_manager_module,
        "_rename_noreplace",
        create_destination_then_rename,
    )

    with pytest.raises(FileExistsError, match="Destination already exists"):
        manager.restore(item, str(destination))

    assert destination.read_text() == "concurrent"
    assert item.read_text() == "trashed"


def test_restore_rejects_source_item_replaced_during_operation(
    tmp_path: Path,
    trash_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrashManager(trash_dir)
    item = trash_dir / "20260715_120000_000001" / "item.txt"
    item.parent.mkdir()
    item.write_text("original")
    destination = tmp_path / "destination.txt"
    original_rename = trash_manager_module._rename_noreplace
    displaced = item.with_name("original-item.txt")

    def replace_source_then_rename(*args, **kwargs):
        item.rename(displaced)
        item.write_text("replacement")
        return original_rename(*args, **kwargs)

    monkeypatch.setattr(
        trash_manager_module,
        "_rename_noreplace",
        replace_source_then_rename,
    )

    with pytest.raises(PermissionError, match="Trash item changed"):
        manager.restore(item, str(destination))

    assert item.read_text() == "replacement"
    assert displaced.read_text() == "original"
    assert not destination.exists()


def test_restore_does_not_create_parent_under_another_agents_data(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    own_data = tmp_path / "agent_data" / "own"
    own_data.mkdir(parents=True)
    manager = TrashManager(trash_dir, current_agent_data_path=own_data)
    item = trash_dir / "20260715_120000_000001" / "item.txt"
    item.parent.mkdir()
    item.write_text("trashed")
    other_parent = tmp_path / "agent_data" / "other" / "new-parent"
    destination = other_parent / "destination.txt"

    with pytest.raises(PermissionError, match="another agent's data"):
        manager.restore(item, str(destination))

    assert not other_parent.exists()
    assert item.read_text() == "trashed"


def test_restore_uses_resolved_destination_parent_if_symlink_changes(
    tmp_path: Path,
    trash_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrashManager(trash_dir)
    item = trash_dir / "20260715_120000_000001" / "item.txt"
    item.parent.mkdir()
    item.write_text("trashed")
    first_parent = tmp_path / "first"
    second_parent = tmp_path / "second"
    first_parent.mkdir()
    second_parent.mkdir()
    destination_parent = tmp_path / "destination-parent"
    destination_parent.symlink_to(first_parent, target_is_directory=True)
    destination = destination_parent / "restored.txt"
    original_open_chain = trash_manager_module._open_directory_chain
    swapped = False

    def swap_symlink_then_open(path: Path, *, create: bool = False) -> int:
        nonlocal swapped
        if path == first_parent.resolve() and not swapped:
            destination_parent.unlink()
            destination_parent.symlink_to(second_parent, target_is_directory=True)
            swapped = True
        return original_open_chain(path, create=create)

    monkeypatch.setattr(
        trash_manager_module,
        "_open_directory_chain",
        swap_symlink_then_open,
    )

    manager.restore(item, str(destination))

    assert (first_parent / "restored.txt").read_text() == "trashed"
    assert not (second_parent / "restored.txt").exists()


def test_restore_rejects_trash_root_replaced_during_startup(
    tmp_path: Path,
    trash_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrashManager(trash_dir)
    item = trash_dir / "20260715_120000_000001" / "item.txt"
    item.parent.mkdir()
    item.write_text("trashed")
    destination = tmp_path / "destination.txt"
    original_open_chain = trash_manager_module._open_directory_chain
    original_trash = tmp_path / "original-trash"
    replaced = False

    def replace_root_then_open(path: Path, *, create: bool = False) -> int:
        nonlocal replaced
        if path == trash_dir.resolve() and not replaced:
            trash_dir.rename(original_trash)
            trash_dir.mkdir()
            replaced = True
        return original_open_chain(path, create=create)

    monkeypatch.setattr(
        trash_manager_module,
        "_open_directory_chain",
        replace_root_then_open,
    )

    with pytest.raises(PermissionError, match="Trash directory changed"):
        manager.restore(item, str(destination))

    assert (original_trash / item.parent.name / item.name).read_text() == "trashed"
    assert not destination.exists()

def test_invalid_shell_fails_closed_even_without_rm(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    """A parse error must never skip command/path policy checks.

    Shells execute commands preceding a later syntax error, so returning the
    original script would let a protected-path mv run before the shell
    reports the error.
    """
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)
    script = f"mv {tmp_path}/agent_data/other /tmp/x\nif"

    with pytest.raises(ShellRewriteError, match="syntactically invalid"):
        policy.rewrite_bash_script(script)


def test_python_runtime_rename_moves_the_symlink_not_its_target(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    """os.rename on a symlink must rename the link itself, never the target."""
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)
    target = tmp_path / "target.txt"
    target.write_text("target data")
    link = tmp_path / "link"
    link.symlink_to(target)
    renamed = tmp_path / "renamed-link"
    content = (
        "import os\n"
        f"os.rename({str(link)!r}, {str(renamed)!r})\n"
    )
    script_path = tmp_path / "rename-link.py"
    script_path.write_text(policy.rewrite_python_script(content))

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert target.read_text() == "target data"
    assert not link.exists() and not link.is_symlink()
    assert renamed.is_symlink()
    assert renamed.read_text() == "target data"


def test_caller_working_dir_is_not_a_direct_delete_root(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    """A caller-supplied cwd resolves relative operands but never
    authorizes real deletion; only the executor-owned workspace does."""
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)
    executor_workdir = tmp_path / "executor-owned"
    executor_workdir.mkdir()
    host_working_dir = tmp_path / "host-project"
    host_working_dir.mkdir()
    precious = host_working_dir / "precious.txt"
    precious.write_text("host data")

    rewritten = policy.rewrite_bash_script(
        "rm precious.txt",
        str(executor_workdir),
        script_cwd=str(host_working_dir),
    )
    completed = subprocess.run(
        ["sh", "-c", rewritten],
        cwd=host_working_dir,
        check=False,
        capture_output=True,
        text=True,
    )

    assert "command -p mv --" in rewritten
    assert 'rm -- "$@"' not in rewritten
    assert completed.returncode == 0, completed.stderr
    assert not precious.exists()
    assert next(trash_dir.glob("*/precious.txt")).read_text() == "host data"


@pytest.mark.asyncio
async def test_local_executor_trashes_relative_delete_in_caller_working_dir(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    working_dir = tmp_path / "caller-cwd"
    working_dir.mkdir()
    precious = working_dir / "keep-safe.txt"
    precious.write_text("irreplaceable")
    policy = DestructiveOperationPolicy(trash_dir=trash_dir)
    executor = LocalExecutor(require_env_flag=False)
    executor._policy = policy
    script = ComputeScript(
        id="cwd-containment",
        name="cwd-containment",
        language="bash",
        content='rm "keep-safe.txt"',
        purpose="prove caller working dirs stay trash-contained",
    )

    record = await executor.execute(script, working_dir=str(working_dir))

    assert record.exit_code == 0, record.stderr
    assert not precious.exists()
    assert next(trash_dir.glob("*/keep-safe.txt")).read_text() == "irreplaceable"


def test_shell_rm_rejects_symlink_entry_inside_another_agents_data(
    tmp_path: Path,
    trash_dir: Path,
) -> None:
    """A symlink ENTRY parked in another agent's data dir must be protected
    even when its target resolves outside agent_data: deleting the link
    removes the entry itself from that directory (#2485 review P2)."""
    outside_target = tmp_path / "outside-target.txt"
    outside_target.write_text("outside")
    other_agent = tmp_path / "agent_data" / "other"
    other_agent.mkdir(parents=True)
    link = other_agent / "their-link"
    link.symlink_to(outside_target)

    policy = DestructiveOperationPolicy(trash_dir=trash_dir)
    rewritten = policy.rewrite_bash_script(f'rm "{link}"')

    completed = subprocess.run(
        ["sh", "-c", rewritten],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "another agent's runtime data" in completed.stderr
    assert link.is_symlink(), "the other agent's symlink entry must survive"
    assert list(trash_dir.iterdir()) == []


def test_trash_listing_skips_hidden_staging_directories(
    trash_dir: Path,
) -> None:
    """Per-execution `.staging-*` bind-mount dirs (Docker executor) must not
    surface as trash entries before promotion."""
    from kestrel_sovereign.features.compute.trash_manager import TrashManager

    staging = trash_dir / ".staging-abc123"
    staging.mkdir(parents=True)
    (staging / "rm_pending123").mkdir()
    (staging / "rm_pending123" / "inflight.txt").write_text("mid-run")

    manager = TrashManager(trash_dir=trash_dir)
    items = manager.list_items()
    assert items == [], "mid-flight staged entries must not be listed"
