"""Focused contracts for the offline semantic knowledge registry."""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest

from kestrel_sovereign.knowledge.registry import (
    DuplicateNamespaceError,
    ExperimentalCapabilityError,
    ImportCycleError,
    IncompatibleSemanticVersionError,
    KnowledgeRegistryError,
    MissingPackageResourceError,
    ResourceDigestMismatchError,
    ResourceKind,
    ResourceNotFoundError,
    ResourceRequirement,
    SemanticKnowledgeRegistry,
    SemanticResource,
    SemanticVersion,
    StandardsMaturity,
    VersionRequiredError,
    load_knowledge_registry,
    refresh_manifest_digest,
)


def _resource(
    identifier: str,
    *,
    version: str = "1.0.0",
    namespace: str | None = None,
    imports: tuple[ResourceRequirement, ...] = (),
    sha256: str = "a" * 64,
    capabilities: tuple[str, ...] = (),
) -> SemanticResource:
    return SemanticResource(
        identifier=identifier,
        version=SemanticVersion.parse(version),
        namespace=namespace or f"https://example.test/semantic/{identifier}/{version}",
        package_resource=f"semantic/{identifier}-{version}.json",
        sha256=sha256,
        maturity=StandardsMaturity.STABLE,
        kind=ResourceKind.ONTOLOGY,
        uri=f"https://example.test/semantic/{identifier}/{version}",
        published_date="2026-07-26",
        description="Test-only schema resource.",
        imports=imports,
        capabilities=capabilities or (f"test:{identifier}:{version}",),
    )


def test_lookup_has_exact_version_and_digest_round_trip():
    registry = load_knowledge_registry()

    resource = registry.resolve("kestrel-vocab", "1.0.0")

    assert resource.version == SemanticVersion(1, 0, 0)
    assert resource.pin.sha256 == resource.sha256
    assert resource.pin.package_resource == resource.package_resource
    assert resource.pin.uri == "https://kestrel.ai/vocab/1.0.0"

    with pytest.raises(VersionRequiredError, match="requires an exact semantic version"):
        registry.resolve("kestrel-vocab", None)


def test_loading_and_resolution_are_offline(monkeypatch):
    def forbid_network(*_args, **_kwargs):
        raise AssertionError("semantic registry attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbid_network)
    monkeypatch.setattr(socket.socket, "connect", forbid_network)

    registry = load_knowledge_registry()
    capability = registry.resolve_capability("kestrel-assertion-shapes", "1.0.0")

    assert capability.resource.identifier == "kestrel-assertion-shapes"
    assert capability.artifact_pins[-1].identifier == "kestrel-assertion-shapes"


def test_import_cycle_is_rejected_with_the_cycle_path():
    first = _resource("first", imports=(ResourceRequirement.exact("second", "1.0.0"),))
    second = _resource("second", imports=(ResourceRequirement.exact("first", "1.0.0"),))
    registry = SemanticKnowledgeRegistry([first, second])

    with pytest.raises(ImportCycleError, match=r"first@1\.0\.0 -> second@1\.0\.0"):
        registry.resolve_import_closure(ResourceRequirement.exact("first", "1.0.0"))


def test_missing_import_is_rejected_without_a_fallback():
    root = _resource("root", imports=(ResourceRequirement.exact("missing", "1.0.0"),))
    registry = SemanticKnowledgeRegistry([root])

    with pytest.raises(ResourceNotFoundError, match="unknown semantic import 'missing'"):
        registry.resolve_import_closure(ResourceRequirement.exact("root", "1.0.0"))


def test_duplicate_namespace_is_rejected():
    first = _resource("first", namespace="https://example.test/vocab/")
    second = _resource("second", namespace="https://example.test/vocab/")

    with pytest.raises(DuplicateNamespaceError, match="duplicate semantic namespace"):
        SemanticKnowledgeRegistry([first, second])


def test_digest_mismatch_is_rejected():
    content = b"locally pinned semantic snapshot"
    resource = _resource("root", sha256="0" * 64)
    registry = SemanticKnowledgeRegistry([resource], resource_reader=lambda _path: content)

    with pytest.raises(ResourceDigestMismatchError, match="digest mismatch"):
        registry.verify_resources()


def test_missing_package_resource_is_rejected():
    digest = hashlib.sha256(b"resource").hexdigest()
    resource = _resource("root", sha256=digest)
    registry = SemanticKnowledgeRegistry(
        [resource],
        resource_reader=lambda _path: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )

    with pytest.raises(MissingPackageResourceError, match="package resource is missing"):
        registry.verify_resources()


def test_incompatible_import_version_is_rejected():
    root = _resource(
        "root",
        imports=(ResourceRequirement.parse("dependency@>=2.0.0,<3.0.0"),),
    )
    dependency = _resource("dependency", version="1.0.0")
    registry = SemanticKnowledgeRegistry([root, dependency])

    with pytest.raises(IncompatibleSemanticVersionError, match="incompatible"):
        registry.resolve_import_closure(ResourceRequirement.exact("root", "1.0.0"))


def test_conflicting_import_versions_are_rejected():
    first = _resource(
        "first",
        imports=(ResourceRequirement.exact("dependency", "1.0.0"),),
    )
    second = _resource(
        "second",
        imports=(ResourceRequirement.exact("dependency", "2.0.0"),),
    )
    dependency_v1 = _resource("dependency", version="1.0.0")
    dependency_v2 = _resource("dependency", version="2.0.0")
    registry = SemanticKnowledgeRegistry([first, second, dependency_v1, dependency_v2])

    with pytest.raises(IncompatibleSemanticVersionError, match="version conflict"):
        registry.resolve_import_closure(
            [
                ResourceRequirement.exact("first", "1.0.0"),
                ResourceRequirement.exact("second", "1.0.0"),
            ]
        )


def test_experimental_profiles_require_explicit_opt_in():
    registry = load_knowledge_registry()

    with pytest.raises(ExperimentalCapabilityError, match="experimental"):
        registry.resolve_capability("rdf12-cr-20260407", "0.1.0")

    capability = registry.resolve_capability(
        "rdf12-cr-20260407",
        "0.1.0",
        allow_experimental=True,
    )

    assert capability.resource.maturity is StandardsMaturity.EXPERIMENTAL
    assert capability.artifact_pins[-1].identifier == "rdf12-cr-20260407"

    with pytest.raises(ExperimentalCapabilityError, match="experimental"):
        registry.select_capability("rdf-profile:rdf12-cr-20260407-experimental")

    assert (
        registry.select_capability(
            "rdf-profile:rdf12-cr-20260407-experimental",
            allow_experimental=True,
        ).resource.identifier
        == "rdf12-cr-20260407"
    )


@pytest.mark.parametrize(
    "capability",
    (
        "serialization:turtle-20140225",
        "serialization:ntriples-20140225",
        "serialization:nquads-20140225",
        "serialization:trig-20140225",
    ),
)
def test_rdf11_serialization_profiles_are_pinned_to_dated_artifacts(capability):
    registry = load_knowledge_registry()

    resolved = registry.select_capability(capability)

    assert resolved.resource.package_resource.startswith("data/semantic/standards/")
    assert resolved.resource.package_resource.endswith(".html")
    assert [resource.identifier for resource in resolved.import_closure] == [
        "rdf11-concepts-20140225",
        resolved.resource.identifier,
    ]


def test_package_resources_are_available_from_an_installed_wheel(tmp_path):
    """Build and check the wheel without allowing imports from this checkout."""
    repository = Path(__file__).resolve().parents[2]
    dist_dir = tmp_path / "dist"
    venv_dir = tmp_path / "wheel-venv"
    environment = os.environ.copy()
    # The installed wheel, rather than this test's source checkout or an
    # inherited developer ``PYTHONPATH``, must satisfy the registry command.
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=repository,
        env=environment,
        check=True,
    )
    wheels = list(dist_dir.glob("kestrel_sovereign-*.whl"))
    assert len(wheels) == 1, f"expected one Kestrel wheel, found {wheels}"

    subprocess.run(
        ["uv", "venv", "--no-project", str(venv_dir)],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
    python_name = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    python = venv_dir / python_name
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            str(wheels[0]),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
    check = subprocess.run(
        [str(python), "-W", "error", "-m", "kestrel_sovereign.knowledge.registry", "check"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert check.returncode == 0, check.stderr
    assert "semantic registry OK:" in check.stdout


def test_contract_rejects_non_exact_capability_version_request():
    registry = load_knowledge_registry()

    with pytest.raises(KnowledgeRegistryError, match="exact resource version"):
        registry.resolve_capability_contract(
            [ResourceRequirement.parse("kestrel-vocab@>=1.0.0,<2.0.0")]
        )


def test_refresh_is_explicit_and_uses_a_local_snapshot(tmp_path):
    package_root = tmp_path / "kestrel_sovereign"
    semantic_root = package_root / "data" / "semantic"
    manifest = semantic_root / "registry.toml"
    snapshot = tmp_path / "kestrel-vocab.ttl"
    source_manifest = resources.files("kestrel_sovereign").joinpath(
        "data",
        "semantic",
        "registry.toml",
    )
    source_snapshot = resources.files("kestrel_sovereign").joinpath(
        "data",
        "semantic",
        "ontologies",
        "kestrel-vocab-1.0.0.ttl",
    )
    shutil.copytree(
        resources.files("kestrel_sovereign").joinpath("data", "semantic"),
        semantic_root,
    )
    assert manifest.read_bytes() == source_manifest.read_bytes()
    refreshed_bytes = source_snapshot.read_bytes() + b"\n# reviewed pin refresh\n"
    snapshot.write_bytes(refreshed_bytes)
    (package_root / "data" / "semantic" / "ontologies" / "kestrel-vocab-1.0.0.ttl").write_bytes(
        refreshed_bytes
    )

    digest = refresh_manifest_digest(
        manifest,
        identifier="kestrel-vocab",
        version="1.0.0",
        snapshot_path=snapshot,
    )

    assert digest == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert digest in manifest.read_text()
    refreshed = load_knowledge_registry(manifest)
    assert refreshed.resolve("kestrel-vocab", "1.0.0").sha256 == digest


def test_refresh_rejects_a_snapshot_that_does_not_replace_the_package_resource(tmp_path):
    package_root = tmp_path / "kestrel_sovereign"
    semantic_root = package_root / "data" / "semantic"
    manifest = semantic_root / "registry.toml"
    source_semantic = resources.files("kestrel_sovereign").joinpath("data", "semantic")
    shutil.copytree(source_semantic, semantic_root)
    snapshot = tmp_path / "unpackaged-kestrel-vocab.ttl"
    snapshot.write_bytes(b"not the registered package resource")
    before = manifest.read_bytes()

    with pytest.raises(ResourceDigestMismatchError, match="does not match"):
        refresh_manifest_digest(
            manifest,
            identifier="kestrel-vocab",
            version="1.0.0",
            snapshot_path=snapshot,
        )

    assert manifest.read_bytes() == before
