"""Custody, isolation and packaging contracts for the modified official SDK."""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from scripts.vendor_a2a import check_vendor, import_vendor, relocate


def test_import_uses_the_pinned_git_tree_and_refuses_overwrite(tmp_path):
    source = tmp_path / "upstream"
    source.mkdir()
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    sdk = source / "src" / "a2a"
    sdk.mkdir(parents=True)
    (sdk / "__init__.py").write_text('"""Upstream SDK."""\n')
    (sdk / "sample.py").write_text('from a2a.types import Task\n')
    (source / "LICENSE").write_text("Apache-2.0\n")
    (source / "pyproject.toml").write_text('[project]\nname = "a2a-sdk"\n')
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Vendor Test", "-c", "user.email=vendor@example.invalid", "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "-m", "source"], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    (sdk / "sample.py").write_text("raise RuntimeError('unreviewed working tree')\n")
    destination = tmp_path / "host"
    import_vendor(destination, source, revision, revision)
    assert check_vendor(destination) == []
    assert "unreviewed" not in (destination / "kestrel_sovereign" / "_vendor" / "a2a" / "sample.py").read_text()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        import_vendor(destination, source, revision, revision)
    (sdk / "sample.py").write_text("from a2a.future_module import Task\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Vendor Test", "-c", "user.email=vendor@example.invalid", "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "-m", "unsupported module"], check=True)
    invalid = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    refused = tmp_path / "refused"
    with pytest.raises(ValueError, match="Unrelocated SDK import"):
        import_vendor(refused, source, invalid, revision)
    assert not (refused / "kestrel_sovereign" / "_vendor" / "a2a").exists()


def test_relocation_changes_modules_but_preserves_protocol_identifiers():
    original = 'from a2a.types import Task\n"a2a.utils.signing"\n"lf.a2a.v1.Task"\n"/a2a.v1.A2AService/GetTask"\n'
    assert relocate(original) == 'from kestrel_sovereign._vendor.a2a.types import Task\n"kestrel_sovereign._vendor.a2a.utils.signing"\n"lf.a2a.v1.Task"\n"/a2a.v1.A2AService/GetTask"\n'
    signals = '"a2a.task_changed" "a2a.push-config.v1" "a2a.question_answered" "a2a.inbound_authorization"'
    assert relocate(signals) == signals
    assert relocate("from a2a import types\n") == "from kestrel_sovereign._vendor.a2a import types\n"


def test_checked_in_sdk_reproduces_reviewed_source_and_license():
    assert check_vendor(Path(__file__).resolve().parents[2]) == []


@pytest.mark.parametrize("mutation", ["source", "license", "missing", "extra"])
def test_vendor_custody_rejects_changed_sources_and_provenance(tmp_path, mutation):
    import shutil

    root = Path(__file__).resolve().parents[2]
    destination = tmp_path / "kestrel_sovereign" / "_vendor" / "a2a"
    shutil.copytree(root / "kestrel_sovereign" / "_vendor" / "a2a", destination, ignore=shutil.ignore_patterns("__pycache__"))
    if mutation == "source":
        (destination / "utils" / "signing.py").write_text("raise RuntimeError('changed')\n")
    elif mutation == "license":
        (destination / "LICENSE").write_text("changed")
    elif mutation == "missing":
        (destination / "utils" / "signing.py").unlink()
    else:
        (destination / "extra.py").write_text("unexpected = True\n")
    assert check_vendor(tmp_path)


def test_sdk_imports_never_load_the_external_a2a_package():
    program = '''
import importlib.abc
import sys
class RefuseExternalA2A(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "a2a" or fullname.startswith("a2a."):
            raise AssertionError("external A2A import: " + fullname)
sys.meta_path.insert(0, RefuseExternalA2A())
import kestrel_sovereign.a2a.native_publication
import kestrel_sovereign.a2a.native_client
import kestrel_sovereign._vendor.a2a.utils.signing
assert not any(name == "a2a" or name.startswith("a2a.") for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-B", "-c", program], check=False, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_vendored_runtime_dependencies_are_declared_without_external_sdk():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    dependencies = {Requirement(raw).name.lower(): Requirement(raw) for raw in project["dependencies"]}
    assert "a2a-sdk" not in dependencies
    upstream = tomllib.loads((root / "kestrel_sovereign" / "_vendor" / "a2a" / "UPSTREAM-PYPROJECT.toml").read_text())["project"]
    required = [*upstream["dependencies"], *upstream["optional-dependencies"]["http-server"], *upstream["optional-dependencies"]["signing"]]
    for raw in required:
        requirement = Requirement(raw)
        assert requirement.name.lower() in dependencies, requirement
    lock = tomllib.loads((root / "uv.lock").read_text())
    assert all(package["name"] != "a2a-sdk" for package in lock["package"])
