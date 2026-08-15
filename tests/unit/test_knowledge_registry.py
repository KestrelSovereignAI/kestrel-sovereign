"""Focused contracts for the offline semantic knowledge registry."""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import socket
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest

from kestrel_sovereign.knowledge.registry import (
    MANIFEST_RESOURCE,
    SEMANTIC_CHECKOUT_PATH,
    DuplicateNamespaceError,
    ExperimentalCapabilityError,
    ImportCycleError,
    IncompatibleSemanticVersionError,
    KnowledgeRegistryError,
    MalformedManifestError,
    MissingPackageResourceError,
    ResourceDigestMismatchError,
    ResourceIntegrityIssue,
    ResourceKind,
    ResourceNotFoundError,
    ResourceRequirement,
    SemanticKnowledgeRegistry,
    SemanticResource,
    SemanticVersion,
    StandardsMaturity,
    VersionRequiredError,
    audit_semantic_resources,
    classify_digest_mismatch,
    crlf_checkout_repair_commands,
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


def test_new_codec_vocabulary_release_preserves_the_existing_immutable_pin():
    registry = load_knowledge_registry()

    original = registry.resolve("kestrel-vocab", "1.0.0")
    codec_release = registry.resolve("kestrel-vocab", "1.1.0")
    selected = registry.select_capability("ontology:kestrel-vocab-1.1")

    assert original.package_resource.endswith("kestrel-vocab-1.0.0.ttl")
    assert original.sha256 == "db708b6790e5212bcbfd5040a1d7883da1161b05e73c809ee8d924c31b2a8044"
    assert codec_release.package_resource.endswith("kestrel-vocab-1.1.0.ttl")
    assert codec_release.uri == "https://kestrel.ai/vocab/1.1.0"
    assert selected.resource == codec_release


def test_loading_and_resolution_are_offline(monkeypatch):
    def forbid_network(*_args, **_kwargs):
        raise AssertionError("semantic registry attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbid_network)
    monkeypatch.setattr(socket.socket, "connect", forbid_network)

    registry = load_knowledge_registry()
    capability = registry.resolve_capability("kestrel-assertion-shapes", "1.0.0")

    assert capability.resource.identifier == "kestrel-assertion-shapes"
    assert capability.artifact_pins[-1].identifier == "kestrel-assertion-shapes"


def test_iri_profile_closure_pins_the_normative_rfc3986_source():
    registry = load_knowledge_registry()

    capability = registry.select_capability(
        "iri-profile:iri-normalization-v1-rfc3986-200501"
    )

    assert [resource.identifier for resource in capability.import_closure] == [
        "rfc3986-200501",
        "iri-normalization-v1-rfc3986-200501",
    ]
    rfc3986_pin = capability.artifact_pins[0]
    assert rfc3986_pin.uri == "https://www.rfc-editor.org/rfc/rfc3986.txt"
    assert rfc3986_pin.published_date == "2005-01-01"
    assert rfc3986_pin.package_resource == "data/semantic/standards/rfc3986-200501.txt"
    assert rfc3986_pin.sha256 == registry.resolve("rfc3986-200501", "1.0.0").sha256


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


def test_digest_pinned_resources_force_lf_checkout_bytes():
    """Git must not rewrite immutable semantic resources on Windows."""
    repository = Path(__file__).resolve().parents[2]
    registry = load_knowledge_registry()
    manifest_paths = {
        Path("kestrel_sovereign") / resource.package_resource
        for resource in registry.resources
    }
    manifest_paths.add(Path("kestrel_sovereign") / MANIFEST_RESOURCE)

    # Assert the whole tracked tree, not only the manifest entries. Two tracked
    # files under data/semantic are byte-pinned without appearing in
    # registry.toml: tests/unit/test_knowledge_rdf_codec.py compares
    # fixtures/rdf11-direct-language.nt byte-for-byte against codec output and
    # reads its golden digests from fixtures/rdf11-projection-digests.txt. A
    # glob narrowed to the manifest's own directories would satisfy every
    # manifest entry and still corrupt those two on a CRLF checkout, so the
    # rule — and this assertion — are pinned to the directory.
    tracked = subprocess.run(
        ["git", "ls-files", "--", "kestrel_sovereign/data/semantic"],
        cwd=repository,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    tracked_paths = {Path(line) for line in tracked.stdout.splitlines() if line}
    assert manifest_paths <= tracked_paths
    assert Path(
        "kestrel_sovereign/data/semantic/fixtures/rdf11-direct-language.nt"
    ) in tracked_paths
    resource_paths = sorted(tracked_paths)

    check = subprocess.run(
        [
            "git",
            "check-attr",
            "text",
            "eol",
            "--",
            *(path.as_posix() for path in resource_paths),
        ],
        cwd=repository,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    attributes: dict[tuple[str, str], str] = {}
    for line in check.stdout.splitlines():
        path, attribute, value = line.rsplit(": ", 2)
        attributes[(path, attribute)] = value

    for path in resource_paths:
        relative = path.as_posix()
        assert attributes[(relative, "text")] == "set"
        assert attributes[(relative, "eol")] == "lf"


def test_autocrlf_checkout_preserves_registered_resource_digests(tmp_path):
    """A Windows-style checkout must retain every pinned resource byte."""
    repository = Path(__file__).resolve().parents[2]
    registry = load_knowledge_registry()
    resource_paths = [
        Path("kestrel_sovereign") / resource.package_resource
        for resource in registry.resources
    ]
    manifest_path = Path("kestrel_sovereign") / MANIFEST_RESOURCE
    checkout = tmp_path / "autocrlf-checkout"
    checkout.mkdir()

    subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "checkout-index",
            "--force",
            f"--prefix={checkout.as_posix()}/",
            "--",
            manifest_path.as_posix(),
            *(path.as_posix() for path in resource_paths),
        ],
        cwd=repository,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )

    assert (checkout / manifest_path).read_bytes() == (
        repository / manifest_path
    ).read_bytes()
    resources_by_path = {
        Path("kestrel_sovereign") / resource.package_resource: resource
        for resource in registry.resources
    }
    for path, resource in resources_by_path.items():
        digest = hashlib.sha256((checkout / path).read_bytes()).hexdigest()
        assert digest == resource.sha256


def _smudge_to_crlf(content: bytes) -> bytes:
    """The bytes ``core.autocrlf=true`` writes into a working tree."""
    return content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def test_crlf_smudged_checkout_is_named_as_a_line_ending_issue():
    """A stale autocrlf checkout must say so, and must still fail closed."""
    content = b"@prefix ex: <https://example.test/> .\nex:a ex:b ex:c .\n"
    resource = _resource("root", sha256=hashlib.sha256(content).hexdigest())
    registry = SemanticKnowledgeRegistry(
        [resource], resource_reader=lambda _path: _smudge_to_crlf(content)
    )

    with pytest.raises(ResourceDigestMismatchError) as excinfo:
        registry.verify_resources()

    message = str(excinfo.value)
    assert "digest mismatch" in message
    assert "line-ending mismatch, not a corrupted resource" in message
    assert "core.autocrlf false" in message
    # The remedy repairs the whole smudged directory, and says outright that
    # the index-only command does not — that dead end is what let the reported
    # checkout keep failing while git called it clean.
    for command in crlf_checkout_repair_commands():
        assert f"`{command}`" in message
    assert SEMANTIC_CHECKOUT_PATH in message
    assert "`git add --renormalize` will not do it" in message

    (finding,) = registry.audit_resource_digests()
    assert finding.issue is ResourceIntegrityIssue.CRLF_CHECKOUT
    assert finding.repair_commands == crlf_checkout_repair_commands()


def test_crlf_poisoned_manifest_pin_blames_the_pin_not_the_checkout():
    """A pin refreshed from a smudged checkout has the opposite remedy."""
    content = b"@prefix ex: <https://example.test/> .\nex:a ex:b ex:c .\n"
    # The registered digest is the CRLF form; the checkout below is correct LF.
    resource = _resource(
        "root", sha256=hashlib.sha256(_smudge_to_crlf(content)).hexdigest()
    )
    registry = SemanticKnowledgeRegistry([resource], resource_reader=lambda _path: content)

    with pytest.raises(ResourceDigestMismatchError) as excinfo:
        registry.verify_resources()

    message = str(excinfo.value)
    assert "line-ending mismatch in the manifest" in message
    assert "data/semantic/registry.toml" in message
    assert "do not renormalize this checkout" in message
    # Sending this developer to rewrite a correct working tree would destroy
    # it; the checkout repair must not appear, in prose or as commands.
    for command in crlf_checkout_repair_commands():
        assert command not in message

    (finding,) = registry.audit_resource_digests()
    assert finding.issue is ResourceIntegrityIssue.CRLF_MANIFEST_PIN
    # No mechanical repair: a poisoned pin needs a reviewed manifest edit.
    assert finding.repair_commands == ()


def test_mixed_endings_never_accuse_the_manifest_pin():
    """Mixed endings cannot support the sentence the pin diagnosis prints.

    Re-applying CRLF to these bytes does reach the pinned digest, but that
    only shows the pin holds their all-CRLF form — not that this checkout is
    the declared LF.  Claiming ``CRLF_MANIFEST_PIN`` would tell the operator
    their working tree is fine and to fix the pin instead, which re-pins the
    smudge and leaves the resource failing.  Undecided is the honest answer.
    """
    mixed = b"first\r\nsecond\nthird\n"
    expected = hashlib.sha256(b"first\r\nsecond\r\nthird\r\n").hexdigest()
    assert hashlib.sha256(mixed.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest() == expected

    assert (
        classify_digest_mismatch(mixed, expected)
        is ResourceIntegrityIssue.DIGEST_MISMATCH
    )


def test_mixed_endings_still_convict_a_smudged_checkout():
    """The forward direction takes mixed endings, because its remedy repairs them.

    Restoring the committed bytes fixes every ending at once, so the checkout
    accusation stays true and actionable however mixed the file is.
    """
    content = b"first\r\nsecond\nthird\n"
    expected = hashlib.sha256(b"first\nsecond\nthird\n").hexdigest()

    assert (
        classify_digest_mismatch(content, expected)
        is ResourceIntegrityIssue.CRLF_CHECKOUT
    )


def test_altered_resource_still_reports_a_plain_mismatch():
    """Tampering stays indistinguishable from tampering — no invented cause."""
    content = b"line one\nline two\n"
    resource = _resource("root", sha256=hashlib.sha256(b"line one\nline three\n").hexdigest())
    registry = SemanticKnowledgeRegistry([resource], resource_reader=lambda _path: content)

    with pytest.raises(ResourceDigestMismatchError) as excinfo:
        registry.verify_resources()

    message = str(excinfo.value)
    assert "line-ending" not in message
    assert message == (
        "root@1.0.0 digest mismatch for semantic/root-1.0.0.json: "
        f"expected {resource.sha256}, got {hashlib.sha256(content).hexdigest()}"
    )
    assert [finding.issue for finding in registry.audit_resource_digests()] == [
        ResourceIntegrityIssue.DIGEST_MISMATCH
    ]


def test_audit_diagnoses_a_whole_crlf_smudged_semantic_checkout(tmp_path):
    """The failure that bricked boot: every pin broken by one bad checkout."""
    package_root = tmp_path / "kestrel_sovereign"
    semantic_root = package_root / "data" / "semantic"
    manifest = semantic_root / "registry.toml"
    shutil.copytree(
        resources.files("kestrel_sovereign").joinpath("data", "semantic"),
        semantic_root,
    )
    assert audit_semantic_resources(manifest) == ()

    registry = load_knowledge_registry(manifest)
    # Distinct paths, because two pins share one shape-set file.
    for path in {
        package_root.joinpath(*Path(resource.package_resource).parts)
        for resource in registry.resources
    }:
        original = path.read_bytes()
        smudged = _smudge_to_crlf(original)
        assert smudged != original, f"{path} has no LF endings for autocrlf to smudge"
        path.write_bytes(smudged)

    findings = audit_semantic_resources(manifest)
    assert {finding.key for finding in findings} == {
        resource.key for resource in registry.resources
    }
    assert {finding.issue for finding in findings} == {
        ResourceIntegrityIssue.CRLF_CHECKOUT
    }

    # Diagnosing the cause must not make the boot gate accept the bytes.
    with pytest.raises(ResourceDigestMismatchError, match="line-ending mismatch"):
        load_knowledge_registry(manifest)


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )


def test_the_published_remedy_actually_repairs_a_smudged_checkout(tmp_path):
    """Run the remedy verbatim against the worst reported state.

    The remedy is a claim about the world, so this executes the exact commands
    the operator is handed and then re-reads the bytes.  The state it runs
    against is the one from the report: the index already holds LF, so
    ``git status`` is clean while the working tree is CRLF and boot keeps
    failing.  An index-only command such as ``git add --renormalize`` passes
    every assertion about *messages* and still leaves the fleet unbootable —
    only running it can tell the two apart.
    """
    checkout = tmp_path / "checkout"
    semantic_root = checkout / SEMANTIC_CHECKOUT_PATH
    manifest = semantic_root / "registry.toml"
    package_root = semantic_root.parent.parent
    shutil.copytree(
        resources.files("kestrel_sovereign").joinpath("data", "semantic"),
        semantic_root,
    )
    # The shipped declaration, not a hand-written stand-in: if the repository
    # stops declaring these paths eol=lf, this remedy stops working.
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / ".gitattributes",
        checkout / ".gitattributes",
    )

    _git("init", "--quiet", cwd=checkout)
    _git("config", "user.email", "tortoise@example.test", cwd=checkout)
    _git("config", "user.name", "Tortoise", cwd=checkout)
    _git("add", "--all", cwd=checkout)
    _git("commit", "--quiet", "--message", "seed", cwd=checkout)
    assert audit_semantic_resources(manifest) == ()

    resource_paths = {
        package_root.joinpath(*Path(resource.package_resource).parts)
        for resource in load_knowledge_registry(manifest).resources
    }
    for path in resource_paths:
        path.write_bytes(_smudge_to_crlf(path.read_bytes()))
    # Reproduce the reported dead end: the index is normalized, so git calls
    # the tree clean even though every pinned resource is still CRLF.
    _git("-c", "core.autocrlf=true", "add", "--renormalize", "--", ".", cwd=checkout)
    assert _git("status", "--porcelain", cwd=checkout).stdout == ""
    findings = audit_semantic_resources(manifest)
    assert {finding.issue for finding in findings} == {ResourceIntegrityIssue.CRLF_CHECKOUT}

    for command in findings[0].repair_commands:
        argv = shlex.split(command)
        assert argv[0] == "git", command
        _git(*argv[1:], cwd=checkout)

    for path in resource_paths:
        assert b"\r\n" not in path.read_bytes(), path
    assert audit_semantic_resources(manifest) == ()
    load_knowledge_registry(manifest)


def test_an_unparseable_manifest_stays_inside_the_registry_error_contract(tmp_path):
    """A broken manifest is a registry failure, not a decoder escaping.

    Callers hold ``KnowledgeRegistryError`` and nothing else; letting
    ``tomllib.TOMLDecodeError`` through ends the doctor and ``setup --check``
    in a traceback instead of the report they exist to produce.
    """
    manifest = tmp_path / "kestrel_sovereign" / "data" / "semantic" / "registry.toml"
    manifest.parent.mkdir(parents=True)

    manifest.write_text("version = 1\n[resource.truncated\n", encoding="utf-8")
    for load in (load_knowledge_registry, audit_semantic_resources):
        with pytest.raises(MalformedManifestError, match="not valid TOML") as excinfo:
            load(manifest)
        assert isinstance(excinfo.value, KnowledgeRegistryError)

    manifest.write_bytes(b"version = \xff\n")
    with pytest.raises(MalformedManifestError, match="not valid UTF-8"):
        audit_semantic_resources(manifest)


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
    ("identifier", "version"),
    (
        ("rdf12-cr-20260407", "0.1.0"),
        ("shacl12-core-20260602-experimental", "0.1.0"),
        ("sparql12-20260605-experimental", "0.1.0"),
    ),
)
def test_public_import_closure_requires_experimental_opt_in(identifier, version):
    registry = load_knowledge_registry()
    requirement = ResourceRequirement.exact(identifier, version)

    with pytest.raises(ExperimentalCapabilityError, match="experimental"):
        registry.resolve_import_closure(requirement)

    closure = registry.resolve_import_closure(requirement, allow_experimental=True)

    assert closure[-1].identifier == identifier
    assert any(resource.maturity is StandardsMaturity.EXPERIMENTAL for resource in closure)


@pytest.mark.parametrize(
    ("resource_path", "required_declarations", "rejected_declarations"),
    (
        (
            "vocabularies/prov-o-20130430.ttl",
            (
                "prov:Entity a owl:Class .",
                "prov:Activity a owl:Class .",
                "prov:Agent a owl:Class .",
                "prov:wasDerivedFrom a owl:ObjectProperty ;",
                "prov:wasGeneratedBy a owl:ObjectProperty ;",
                "prov:wasAttributedTo a owl:ObjectProperty ;",
            ),
            (
                "prov:Entity a prov:Entity .",
                "prov:wasDerivedFrom a prov:Entity .",
            ),
        ),
        (
            "vocabularies/owl-time-20171019.ttl",
            (
                "time:TemporalEntity a owl:Class .",
                "time:Interval a owl:Class ;",
                "time:Instant a owl:Class ;",
                "time:hasBeginning a owl:ObjectProperty ;",
                "time:hasEnd a owl:ObjectProperty ;",
            ),
            (
                "time:TemporalEntity a time:TemporalEntity .",
                "time:hasBeginning a time:TemporalEntity .",
            ),
        ),
        (
            "vocabularies/skos-20090818.ttl",
            (
                "skos:Concept a owl:Class .",
                "skos:prefLabel a owl:AnnotationProperty .",
                "skos:altLabel a owl:AnnotationProperty .",
                "skos:exactMatch a owl:ObjectProperty , owl:SymmetricProperty , owl:TransitiveProperty .",
                "skos:closeMatch a owl:ObjectProperty , owl:SymmetricProperty .",
            ),
            (
                "skos:Concept a skos:Concept .",
                "skos:prefLabel a skos:Concept .",
            ),
        ),
    ),
)
def test_selected_vocabulary_terms_preserve_their_rdf_kinds(
    resource_path, required_declarations, rejected_declarations
):
    vocabulary = (
        resources.files("kestrel_sovereign")
        .joinpath("data", "semantic", *resource_path.split("/"))
        .read_text(encoding="utf-8")
    )

    for declaration in required_declarations:
        assert declaration in vocabulary
    for declaration in rejected_declarations:
        assert declaration not in vocabulary


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

    fixture_check = subprocess.run(
        [
            str(python),
            "-W",
            "error",
            "-c",
            (
                "from importlib import resources; "
                "fixture = resources.files('kestrel_sovereign').joinpath("
                "'data', 'semantic', 'fixtures', 'rdf11-direct-language.nt'); "
                "digests = resources.files('kestrel_sovereign').joinpath("
                "'data', 'semantic', 'fixtures', 'rdf11-projection-digests.txt'); "
                "assert fixture.read_bytes().startswith(b'<urn:kestrel:assertion:sha256:'); "
                "assert b'language-derived' in digests.read_bytes()"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert fixture_check.returncode == 0, fixture_check.stderr


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


def test_refreshing_a_version_qualified_table_preserves_the_prior_immutable_release(tmp_path):
    package_root = tmp_path / "kestrel_sovereign"
    semantic_root = package_root / "data" / "semantic"
    manifest = semantic_root / "registry.toml"
    shutil.copytree(
        resources.files("kestrel_sovereign").joinpath("data", "semantic"),
        semantic_root,
    )
    before = load_knowledge_registry(manifest).resolve("kestrel-vocab", "1.0.0").sha256
    snapshot = tmp_path / "kestrel-vocab-1.1.0.ttl"
    refreshed_bytes = (
        resources.files("kestrel_sovereign")
        .joinpath("data", "semantic", "ontologies", "kestrel-vocab-1.1.0.ttl")
        .read_bytes()
        + b"\n# reviewed 1.1 pin refresh\n"
    )
    snapshot.write_bytes(refreshed_bytes)
    (semantic_root / "ontologies" / "kestrel-vocab-1.1.0.ttl").write_bytes(refreshed_bytes)

    digest = refresh_manifest_digest(
        manifest,
        identifier="kestrel-vocab",
        version="1.1.0",
        snapshot_path=snapshot,
    )

    refreshed = load_knowledge_registry(manifest)
    assert refreshed.resolve("kestrel-vocab", "1.0.0").sha256 == before
    assert refreshed.resolve("kestrel-vocab", "1.1.0").sha256 == digest


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
