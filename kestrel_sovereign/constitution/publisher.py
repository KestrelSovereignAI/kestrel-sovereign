"""Check the platform publisher's release offline before accepting edition two."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from importlib.resources import as_file, files
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

from .amendment_artifact import AmendmentArtifactError


_REPOSITORY = "KestrelSovereignAI/kestrel-sovereign"
_REPOSITORY_ID = "1147541224"
_WORKFLOW = f"{_REPOSITORY}/.github/workflows/constitution-release.yml"
_EDITION = "kestrel_sovereign/data/constitutions/v2/"
_CONSTITUTION = _EDITION + "KESTREL_CONSTITUTION.md"
_PLEDGE = _EDITION + "PLEDGE.md"


def verify_packaged_publisher(source_sha256: str, *, declared_commit: str | None = None) -> dict[str, str]:
    """Return reproducible evidence for these released bytes and no other source.

    The distribution supplies the independently obtained Sigstore trusted roots
    and publisher coordinates. Never take trusted roots from an agent, the
    operator-selected source descriptor, or the supplied adoption artifact.
    """
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", source_sha256):
        raise AmendmentArtifactError("Publisher verification requires the exact raw source SHA-256")
    by_source = files("kestrel_sovereign").joinpath(
        "data", "constitutions", "v2", "publisher", source_sha256,
    )
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
            _verify_members(archive, {_CONSTITUTION: source_sha256, _PLEDGE: manifest["pledge_sha256"]}, commit)
            issued_at = _verify_attestation(archive, bundle, roots, commit)
    except (OSError, tarfile.TarError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AmendmentArtifactError("Publisher release evidence is unavailable or malformed") from exc
    return {
        "publisher_commit": commit, "source_sha256": source_sha256,
        "archive_sha256": archive_digest, "attestation_sha256": attestation_digest,
        "issued_at": issued_at,
    }


def verify_recorded_publisher(record, source_sha256: str) -> dict[str, str]:
    """Independently authenticate the exact publisher event recorded at birth or adoption."""
    if not isinstance(record, dict) or not isinstance(record.get("publisher_commit"), str):
        raise AmendmentArtifactError("Edition 2 requires verifiable publisher release evidence")
    expected = verify_packaged_publisher(source_sha256, declared_commit=record["publisher_commit"])
    if record != expected:
        raise AmendmentArtifactError("Recorded publisher evidence does not match the verified release")
    return expected


def _verify_members(archive: Path, expected: dict[str, str], commit: str) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        if tar.pax_headers.get("comment") != commit:
            raise AmendmentArtifactError("Publisher archive does not name its source commit")
        seen: set[str] = set()
        for member in tar:
            if member.isdir() and _EDITION.startswith(member.name.rstrip("/") + "/"):
                continue
            if (member.name not in expected or member.name in seen or not member.isfile()
                    or member.size > 512 * 1024):
                raise AmendmentArtifactError("Publisher archive has an unexpected member")
            content = tar.extractfile(member)
            if content is None or hashlib.sha256(content.read()).hexdigest() != expected[member.name]:
                raise AmendmentArtifactError("Publisher archive member differs from its registered source")
            seen.add(member.name)
        if seen != expected.keys():
            raise AmendmentArtifactError("Publisher archive is incomplete")


def _verify_attestation(archive: Path, bundle: Path, roots: Path, commit: str) -> str:
    executable = shutil.which("gh")
    if executable is None:
        raise AmendmentArtifactError("Install GitHub CLI with offline attestation support to verify the publisher")
    args = [
        executable, "attestation", "verify", str(archive), "--bundle", str(bundle),
        "--custom-trusted-root", str(roots), "--repo", _REPOSITORY,
        "--signer-workflow", _WORKFLOW, "--source-ref", "refs/heads/main",
        "--source-digest", commit, "--signer-digest", commit,
        "--deny-self-hosted-runners", "--format", "json",
        "--jq", "[.[] | .verificationResult | {repository: .signature.certificate.sourceRepositoryIdentifier,"
                " timestamps: [.verifiedTimestamps[].timestamp]}]",
    ]
    # Offline verification needs neither an operator token nor a network client.
    env = {name: os.environ[name] for name in ("SystemRoot", "WINDIR") if name in os.environ}
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, env=env, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AmendmentArtifactError("Could not run the offline publisher attestation verifier") from exc
    if result.returncode:
        raise AmendmentArtifactError("The official publisher's workflow and repository identity did not verify")
    try:
        witnesses = json.loads(result.stdout)
        if not witnesses or any(witness["repository"] != _REPOSITORY_ID for witness in witnesses):
            raise ValueError("Unexpected publisher repository")
        timestamps = [
            datetime.fromisoformat(value)
            for witness in witnesses for value in witness["timestamps"]
        ]
        if not timestamps or any(stamp.utcoffset() is None for stamp in timestamps):
            raise ValueError("Missing authenticated publisher timestamp")
        return min(timestamps).astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError, KeyError) as exc:
        raise AmendmentArtifactError("The publisher identity or authenticated issuance time did not verify") from exc
