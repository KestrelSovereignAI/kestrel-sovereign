"""Import and verify the reviewed A2A SDK in Kestrel's isolated namespace.

The input is an explicit Git tree, never an installed package or a moving branch.
UPSTREAM.patch reconstructs the reviewed SDK and its regression tests from the
recorded public base. Relocation changes Python module paths, not protocol names,
protobuf descriptors, JSON fields, routes, or cryptographic algorithms.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


def relocate(text: str) -> str:
    """Relocate module references without changing qualified protobuf names."""
    relocated = re.sub(
        r"(?<![\w./])a2a\.(?=(?:_base|auth|client|compat|extensions|helpers|migrations|server|types|utils|a2a_db_cli)\b)",
        "kestrel_sovereign._vendor.a2a.",
        text,
    )
    return re.sub(r"\bfrom a2a(?=\s+import\b)", "from kestrel_sovereign._vendor.a2a", relocated)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(source: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(source), *arguments])


def import_vendor(root: Path, source: Path, revision: str, base: str) -> None:
    """Materialize a pinned, licensed snapshot with reversible relocation."""
    tree = _git(source, "rev-parse", revision + "^{tree}").decode().strip()
    base_commit = _git(source, "rev-parse", base + "^{commit}").decode().strip()
    destination = root / "kestrel_sovereign" / "_vendor" / "a2a"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite vendored source: {destination}")
    paths = _git(source, "ls-tree", "-rz", "--name-only", tree, "src/a2a")
    files = {}
    contents = {}
    for raw_path in paths.decode().split("\0"):
        if not raw_path:
            continue
        relative = Path(raw_path).relative_to("src/a2a")
        original = _git(source, "show", f"{tree}:{raw_path}")
        text = original.decode("utf-8")
        if "kestrel_sovereign._vendor.a2a." in text:
            raise ValueError(f"Source already contains the vendor namespace: {raw_path}")
        relocated_text = relocate(text)
        modified = relocated_text != text
        if relative.suffix == ".py":
            for node in ast.walk(ast.parse(relocated_text)):
                if isinstance(node, ast.ImportFrom) and node.module and (node.module == "a2a" or node.module.startswith("a2a.")):
                    raise ValueError(f"Unrelocated SDK import in {raw_path}: {node.module}")
                if isinstance(node, ast.Import) and any(alias.name == "a2a" or alias.name.startswith("a2a.") for alias in node.names):
                    raise ValueError(f"Unrelocated SDK import in {raw_path}")
            relocated_text = "# Modified by Kestrel: isolated SDK namespace; see VENDORED.json.\n" + relocated_text
        relocated = relocated_text.encode("utf-8")
        contents[relative.as_posix()] = relocated
        files[relative.as_posix()] = {
            "upstream_sha256": _digest(original),
            "vendored_sha256": _digest(relocated),
            "namespace_modified": modified,
        }
    provenance = {}
    for name, original_path in [("LICENSE", "LICENSE"), ("UPSTREAM-PYPROJECT.toml", "pyproject.toml")]:
        data = _git(source, "show", f"{tree}:{original_path}")
        contents[name] = data
        provenance[name] = _digest(data)
    patch = _git(source, "diff", "--binary", base_commit, tree, "--", "src/a2a", "tests")
    contents["UPSTREAM.patch"] = patch
    provenance["UPSTREAM.patch"] = _digest(patch)
    notice = (
        b"A2A Python SDK\nCopyright Google LLC and A2A Python SDK contributors.\n"
        b"Licensed under the Apache License, Version 2.0; see LICENSE.\n\n"
        b"This copy is modified and maintained by Kestrel. It is not an official\n"
        b"A2A Python SDK release. VENDORED.json identifies the source and file digests;\n"
        b"UPSTREAM.patch records the functional changes and their regression tests.\n"
        b"Python imports are relocated into kestrel_sovereign._vendor.a2a.\n"
        b"The Agent2Agent wire protocol remains A2A 1.0.1.\n"
    )
    contents["NOTICE"] = notice
    provenance["NOTICE"] = _digest(notice)
    manifest = {
        "repository": "https://github.com/a2aproject/a2a-python",
        "protocol": "1.0.1",
        "base_release": "1.2.3",
        "base_commit": base_commit,
        "reviewed_source_tree": tree,
        "namespace": "kestrel_sovereign._vendor.a2a",
        "modifications": "Reviewed signing, HTTP authorization, client ownership and response decoding fixes in UPSTREAM.patch; Python module references relocated by scripts/vendor_a2a.py. This is a modified upstream SDK, not an official release.",
        "files": files,
        "provenance_sha256": provenance,
    }
    contents["VENDORED.json"] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".a2a-import-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "a2a"
        for name, data in contents.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        staging.rename(destination)


def check_vendor(root: Path) -> list[str]:
    """Verify complete file custody and reversible, module-only relocation."""
    destination = root / "kestrel_sovereign" / "_vendor" / "a2a"
    manifest = json.loads((destination / "VENDORED.json").read_text())
    failures = []
    expected = set(manifest["files"]) | set(manifest["provenance_sha256"]) | {"VENDORED.json"}
    actual = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual != expected:
        failures.append(f"Vendored file set differs: missing={expected - actual}, extra={actual - expected}")
    for name, record in manifest["files"].items():
        path = destination / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if _digest(data) != record["vendored_sha256"]:
            failures.append(f"Vendored source digest differs: {name}")
        text = data.decode("utf-8")
        if path.suffix == ".py":
            text = text.removeprefix("# Modified by Kestrel: isolated SDK namespace; see VENDORED.json.\n")
        original = text.replace("kestrel_sovereign._vendor.a2a", "a2a").encode("utf-8")
        if _digest(original) != record["upstream_sha256"]:
            failures.append(f"Namespace relocation does not reproduce reviewed source: {name}")
    for name, digest in manifest["provenance_sha256"].items():
        path = destination / name
        if path.is_file() and _digest(path.read_bytes()) != digest:
            failures.append(f"Vendored provenance digest differs: {name}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--base")
    arguments = parser.parse_args()
    if not arguments.check:
        if not all((arguments.source, arguments.revision, arguments.base)):
            parser.error("Import requires --source, --revision and --base")
        import_vendor(arguments.root, arguments.source, arguments.revision, arguments.base)
    failures = check_vendor(arguments.root)
    if failures:
        parser.exit(1, "\n".join(failures) + "\n")


if __name__ == "__main__":
    main()
