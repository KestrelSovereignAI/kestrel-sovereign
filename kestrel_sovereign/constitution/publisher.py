"""Check the platform publisher's release offline before accepting edition two."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from importlib.resources import as_file, files
import json
from pathlib import Path
import re
import tarfile
from tempfile import TemporaryDirectory

from sigstore.errors import Error as SigstoreError
from sigstore.errors import VerificationError
from sigstore.models import Bundle, TrustedRoot
from sigstore.verify import Verifier, policy

from .amendment_artifact import AmendmentArtifactError
from .emancipation import EmancipationContract, apply_emancipation


_REPOSITORY = "KestrelSovereignAI/kestrel-sovereign"
_REPOSITORY_ID = "1147541224"
_REPOSITORY_URL = f"https://github.com/{_REPOSITORY}"
_WORKFLOW = f"{_REPOSITORY_URL}/.github/workflows/constitution-release.yml"
_SOURCE_REF = "refs/heads/main"
_EDITION = "kestrel_sovereign/data/constitutions/v2/"
_CONSTITUTION = _EDITION + "KESTREL_CONSTITUTION.md"
_PLEDGE = _EDITION + "PLEDGE.md"


def _publisher_directory(source_sha256: str):
    return files("kestrel_sovereign").joinpath("data", "constitutions", "v2", "publisher", source_sha256)


def has_packaged_publisher(source_sha256: str | None) -> bool:
    """Identify digests for which the distribution contains edition-two evidence."""
    return (
        isinstance(source_sha256, str) and re.fullmatch(r"[a-f0-9]{64}", source_sha256) is not None
        and _publisher_directory(source_sha256).is_dir()
    )


def verify_packaged_publisher(source_sha256: str, *, declared_commit: str | None = None) -> dict[str, str]:
    """Verify the packaged release and return its reproducible publisher evidence."""
    evidence, _ = _verified_release(source_sha256, declared_commit=declared_commit)
    return evidence


def _verified_release(source_sha256: str, *, declared_commit: str | None) -> tuple[dict[str, str], bytes]:
    """Authenticate the archive and retain the exact constitutional source it contains.

    The distribution supplies the independently obtained Sigstore trusted roots
    and publisher coordinates. Never take trusted roots from an agent, the
    operator-selected source descriptor, or the supplied adoption artifact.
    """
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", source_sha256):
        raise AmendmentArtifactError("Publisher verification requires the exact raw source SHA-256")
    by_source = _publisher_directory(source_sha256)
    try:
        if not by_source.is_dir():
            raise AmendmentArtifactError("Constitution differs from every installed publisher release")
        commit = declared_commit
        if commit is None:
            commit = json.loads(by_source.joinpath("current.json").read_bytes())["publisher_commit"]
        if not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40}", commit):
            raise AmendmentArtifactError("Invalid publisher source commit")
        evidence = by_source.joinpath(commit)
        if not evidence.is_dir():
            raise AmendmentArtifactError("Declared commit differs from every installed release for these bytes")
        manifest = json.loads(evidence.joinpath("release.json").read_bytes())
        if manifest["publisher_commit"] != commit or manifest["source_sha256"] != source_sha256:
            raise AmendmentArtifactError("Constitution or declared commit differs from the installed publisher release")
        with ExitStack() as stack:
            archive, bundle, roots = (
                stack.enter_context(as_file(evidence.joinpath(name)))
                for name in ("constitution-v2.tar.gz", "attestation.jsonl", "trusted-root.jsonl")
            )
            archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            attestation_digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
            if (archive_digest != manifest["archive_sha256"] or
                    attestation_digest != manifest["attestation_sha256"]):
                raise AmendmentArtifactError("Publisher release evidence does not match its registered digests")
            governing_source = _verify_members(
                archive, {_CONSTITUTION: source_sha256, _PLEDGE: manifest["pledge_sha256"]}, commit,
            )
            issued_at = _verify_attestation(archive, bundle, roots, commit)
    except (OSError, tarfile.TarError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AmendmentArtifactError("Publisher release evidence is unavailable or malformed") from exc
    return {
        "publisher_commit": commit, "source_sha256": source_sha256,
        "archive_sha256": archive_digest, "attestation_sha256": attestation_digest,
        "issued_at": issued_at,
    }, governing_source


def verify_recorded_publisher(
    record, source_sha256: str, *, governing_text: str | None = None,
    contract: EmancipationContract | None = None,
) -> dict[str, str]:
    """Independently authenticate the exact publisher event recorded at birth or adoption."""
    if not isinstance(record, dict) or not isinstance(record.get("publisher_commit"), str):
        raise AmendmentArtifactError("Edition 2 requires verifiable publisher release evidence")
    expected, source = _verified_release(source_sha256, declared_commit=record["publisher_commit"])
    if record != expected:
        raise AmendmentArtifactError("Recorded publisher evidence does not match the verified release")
    if governing_text is not None and governing_text.encode("utf-8") != source:
        if (
            contract is None or not contract.enabled
            or apply_emancipation(source.decode("utf-8"), contract) != governing_text
        ):
            raise AmendmentArtifactError(
                "Constitution governing bytes differ from the authenticated publisher release and authorized terms"
            )
    return expected


def _verify_members(archive: Path, expected: dict[str, str], commit: str) -> bytes:
    with tarfile.open(archive, "r:gz") as tar:
        if tar.pax_headers.get("comment") != commit:
            raise AmendmentArtifactError("Publisher archive does not name its source commit")
        members: dict[str, bytes] = {}
        for member in tar:
            if member.isdir() and _EDITION.startswith(member.name.rstrip("/") + "/"):
                continue
            if (member.name not in expected or member.name in members or not member.isfile()
                    or member.size > 512 * 1024):
                raise AmendmentArtifactError("Publisher archive has an unexpected member")
            content = tar.extractfile(member)
            if content is None:
                raise AmendmentArtifactError("Publisher archive member is unreadable")
            data = content.read()
            if hashlib.sha256(data).hexdigest() != expected[member.name]:
                raise AmendmentArtifactError("Publisher archive member differs from its registered source")
            members[member.name] = data
        if members.keys() != expected.keys():
            raise AmendmentArtifactError("Publisher archive is incomplete")
        return members[_CONSTITUTION]


def _verify_attestation(archive: Path, bundle: Path, roots: Path, commit: str) -> str:
    workflow = f"{_WORKFLOW}@{_SOURCE_REF}"
    claims = policy.AllOf([
        policy.Identity(identity=workflow, issuer="https://token.actions.githubusercontent.com"),
        policy.OIDCBuildSignerURI(workflow),
        policy.OIDCBuildSignerDigest(commit),
        policy.OIDCSourceRepositoryURI(_REPOSITORY_URL),
        policy.OIDCSourceRepositoryDigest(commit),
        policy.OIDCSourceRepositoryRef(_SOURCE_REF),
        policy.OIDCSourceRepositoryIdentifier(_REPOSITORY_ID),
        policy.OIDCRunnerEnvironment("github-hosted"),
    ])
    failure = "The official publisher's workflow and repository identity did not verify"
    try:
        attestation = bundle.read_bytes()
        sigstore_bundle = Bundle.from_json(attestation)
        # GitHub's offline trust export contains one JSON document per trust
        # root. The verifier accepts only the packaged roots and uses no login.
        with TemporaryDirectory() as directory:
            trust_file = Path(directory) / "trusted-root.json"
            for root in roots.read_bytes().splitlines():
                trust_file.write_bytes(root)
                trusted_root = TrustedRoot.from_file(str(trust_file))
                try:
                    media_type, payload = Verifier(trusted_root=trusted_root).verify_dsse(
                        sigstore_bundle, claims,
                    )
                except VerificationError:
                    continue
                break
            else:
                raise AmendmentArtifactError(failure)
        statement = json.loads(payload)
        expected_subject = {
            "name": "constitution-v2.tar.gz",
            "digest": {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest()},
        }
        if (
            media_type != "application/vnd.in-toto+json"
            or statement["_type"] != "https://in-toto.io/Statement/v1"
            or statement["predicateType"] != "https://slsa.dev/provenance/v1"
            or statement["subject"] != [expected_subject]
        ):
            raise AmendmentArtifactError("The signed publisher archive digest did not verify")
        # The installed releases use a Rekor v1 signed inclusion promise.
        # Sigstore verifies that promise, including its integrated timestamp.
        entry, = json.loads(attestation)["verificationMaterial"]["tlogEntries"]
        if not entry["inclusionPromise"]["signedEntryTimestamp"]:
            raise AmendmentArtifactError("The authenticated publisher issuance time did not verify")
        return datetime.fromtimestamp(int(entry["integratedTime"]), tz=timezone.utc).isoformat()
    except AmendmentArtifactError:
        raise
    except (SigstoreError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise AmendmentArtifactError(failure) from exc
