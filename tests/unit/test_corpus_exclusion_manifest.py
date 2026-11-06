"""The Sovereign-signed corpus exclusion manifest and its operator check (#3893).

The corpus path itself is exercised in
``tests/unit/storage/test_governed_semantic_artifacts.py``; these cover the
manifest's own grammar, signature forms, file selection and the
``kestrel corpus exclusions check`` command an operator confirms it with.
"""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from kestrel_sovereign import cli_corpus
from kestrel_sovereign.knowledge.corpus_exclusions import (
    CORPUS_EXCLUSIONS_PATH_ENV,
    DEFAULT_CORPUS_EXCLUSIONS_FILENAME,
    CorpusExclusionKind,
    CorpusExclusionManifestError,
    build_hybrid_signed_corpus_exclusions,
    configured_corpus_exclusions_path,
    load_corpus_exclusions,
    validate_exclusion_id,
)
from tests.utils.corpus_exclusions import ExclusionAuthority

MERIDIAN = "did:web:agents.example:meridian"
EMMA = "did:web:agents.example:emma"
EXAM_SOURCE = f"source:memory-agency-save-fact-v1:{'a' * 64}"
EXAM_SESSION = "dcd6f767-829d-4a7d-b8c7-d7a722e5de13"


@pytest.fixture
def authority(tmp_path, monkeypatch) -> ExclusionAuthority:
    return ExclusionAuthority(tmp_path / "sovereign", monkeypatch)


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        (CorpusExclusionKind.SOURCE_OCCURRENCE, EXAM_SOURCE),
        (CorpusExclusionKind.SOURCE_OCCURRENCE, f"source:strategic-memory-ledger-v1:{'0' * 64}"),
        (CorpusExclusionKind.SESSION, EXAM_SESSION),
        (CorpusExclusionKind.SESSION, EXAM_SESSION.upper()),
        (CorpusExclusionKind.SESSION, "sess_k3y_from-a-client"),
        (CorpusExclusionKind.SAVED_ITEM, "2ca40dbd-1111-4222-8333-444455556666"),
    ],
)
def test_whole_ids_are_accepted(kind, value) -> None:
    assert validate_exclusion_id(kind, value) == value


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        (CorpusExclusionKind.SOURCE_OCCURRENCE, "source:memory-agency-save-fact-v1:3a9b12"),
        (CorpusExclusionKind.SOURCE_OCCURRENCE, "a" * 64),
        (CorpusExclusionKind.SOURCE_OCCURRENCE, f"source:x:{'A' * 64}"),
        (CorpusExclusionKind.SESSION, "dcd6f767"),
        (CorpusExclusionKind.SESSION, "dcd6f767-829d-4a7d-b8c7"),
        (CorpusExclusionKind.SESSION, "12345"),
        (CorpusExclusionKind.SESSION, "exam session"),
        (CorpusExclusionKind.SESSION, ""),
        (CorpusExclusionKind.SAVED_ITEM, "2ca40dbd"),
        (CorpusExclusionKind.SAVED_ITEM, "2CA40DBD-1111-4222-8333-444455556666"),
        (CorpusExclusionKind.SAVED_ITEM, 42),
    ],
)
def test_shortened_or_malformed_ids_are_refused(kind, value) -> None:
    with pytest.raises(CorpusExclusionManifestError):
        validate_exclusion_id(kind, value)


def test_operator_tooling_cannot_sign_a_manifest_the_host_would_refuse(authority) -> None:
    for agents in (
        {MERIDIAN: {"saved_item": ["469a149e"]}},
        {MERIDIAN: {"exam": [EXAM_SOURCE]}},
        {MERIDIAN: {"session": EXAM_SESSION}},
        {MERIDIAN: {"session": [EXAM_SESSION, EXAM_SESSION]}},
        {"meridian": {}},
    ):
        with pytest.raises(CorpusExclusionManifestError):
            authority.sign(agents, sequence=1)
    for sequence in (0, -1, True, 2**63):
        with pytest.raises(CorpusExclusionManifestError, match="sequence"):
            authority.sign({MERIDIAN: {}}, sequence=sequence)


def test_a_manifest_names_each_agents_exclusions(authority) -> None:
    authority.exclude(MERIDIAN, source_occurrence=[EXAM_SOURCE], session=[EXAM_SESSION])
    authority.admit(EMMA)

    manifest = load_corpus_exclusions()
    meridian = manifest.exclusions_for(MERIDIAN)
    emma = manifest.exclusions_for(EMMA)

    assert manifest.sequence == 2
    assert meridian.source_occurrence_ids == (EXAM_SOURCE,)
    assert meridian.session_ids == (EXAM_SESSION,)
    assert (emma.source_occurrence_ids, emma.session_ids, emma.saved_item_ids) == ((), (), ())
    assert meridian.receipt == emma.receipt == manifest.receipt
    with pytest.raises(CorpusExclusionManifestError, match="no section"):
        manifest.exclusions_for("did:web:agents.example:kite")


def test_re_signing_unchanged_content_keeps_the_manifest_digest(authority) -> None:
    sections = {MERIDIAN: {"source_occurrence": [EXAM_SOURCE]}}
    first = authority.sign(sections, sequence=1)
    authority.write(first)
    digest = load_corpus_exclusions().manifest_sha256
    authority.write(authority.sign(sections, sequence=1, issued_at=first["issued_at"]))

    assert load_corpus_exclusions().manifest_sha256 == digest


def test_a_hybrid_signed_manifest_verifies(tmp_path, monkeypatch) -> None:
    from kestrel_sovereign.constitution.trust_root import SOVEREIGN_TRUST_ROOT_ENV
    from kestrel_sovereign.identity.inception_did_web import create_did_web_identity

    root = create_did_web_identity("sovereign.example", "root")
    root_path = tmp_path / "hybrid-root.did.json"
    root_path.write_text(json.dumps(root.did_document), encoding="utf-8")
    manifest_path = tmp_path / "corpus-exclusions.json"
    manifest_path.write_text(json.dumps(build_hybrid_signed_corpus_exclusions(
        signer_did=root.did,
        sequence=1,
        agents={MERIDIAN: {"session": [EXAM_SESSION]}},
        keypair=root.keypair,
    )))
    monkeypatch.setenv(SOVEREIGN_TRUST_ROOT_ENV, str(root_path))
    monkeypatch.setenv(CORPUS_EXCLUSIONS_PATH_ENV, str(manifest_path))

    manifest = load_corpus_exclusions()

    assert manifest.signer == root.did
    assert manifest.exclusions_for(MERIDIAN).session_ids == (EXAM_SESSION,)


def test_the_manifest_lives_in_host_data_unless_configured(tmp_path) -> None:
    home = tmp_path / "home"
    assert configured_corpus_exclusions_path(environ={"KESTREL_HOME": str(home)}) == (
        home / "host-data" / DEFAULT_CORPUS_EXCLUSIONS_FILENAME
    )
    explicit = tmp_path / "elsewhere.json"
    assert configured_corpus_exclusions_path(
        environ={CORPUS_EXCLUSIONS_PATH_ENV: str(explicit)}
    ) == explicit


@pytest.mark.parametrize("value", ["", "  ", "relative/corpus-exclusions.json"])
def test_a_blank_or_relative_manifest_path_is_refused(value) -> None:
    with pytest.raises(CorpusExclusionManifestError, match=CORPUS_EXCLUSIONS_PATH_ENV):
        configured_corpus_exclusions_path(environ={CORPUS_EXCLUSIONS_PATH_ENV: value})


def test_a_fifo_in_place_of_the_manifest_is_refused_without_blocking(authority) -> None:
    """An agent can replace the manifest with a FIFO; reading it must not wait (#3893)."""
    from tests.utils.fifo import call_against_fifo

    authority.admit(MERIDIAN)
    authority.manifest_path.unlink()
    error = call_against_fifo(authority.manifest_path, load_corpus_exclusions)
    assert isinstance(error, CorpusExclusionManifestError)
    assert "is not a regular file" in str(error)


def test_a_directory_in_place_of_the_manifest_is_refused(authority) -> None:
    authority.manifest_path.mkdir()
    with pytest.raises(CorpusExclusionManifestError, match="is not a regular file"):
        load_corpus_exclusions()


def _check(*agent_dids: str, as_json: bool = False) -> int:
    return cli_corpus.run(Namespace(
        corpus_command="exclusions",
        exclusions_command="check",
        agent_did=list(agent_dids),
        json=as_json,
    ))


def test_the_operator_check_lists_every_excluded_id(authority, capsys) -> None:
    authority.exclude(MERIDIAN, source_occurrence=[EXAM_SOURCE], session=[EXAM_SESSION])
    authority.admit(EMMA)

    assert _check(MERIDIAN, EMMA) == 0

    out = capsys.readouterr().out
    assert f"Agent {MERIDIAN}: usable" in out
    assert f"    {EXAM_SOURCE}" in out
    assert f"    {EXAM_SESSION}" in out
    assert f"Agent {EMMA}: usable\n  excludes nothing" in out
    assert "sequence:  2" in out
    assert out.rstrip().endswith("OK")


def test_the_operator_check_reports_json(authority, capsys) -> None:
    authority.exclude(MERIDIAN, source_occurrence=[EXAM_SOURCE])

    assert _check(MERIDIAN, as_json=True) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["agents"] == [{
        "agent_did": MERIDIAN,
        "ids": {"source_occurrence": [EXAM_SOURCE], "session": [], "saved_item": []},
        "usable": True,
    }]


def test_the_operator_check_fails_an_agent_without_a_section(authority, capsys) -> None:
    authority.admit(MERIDIAN)

    assert _check(MERIDIAN, EMMA) == 1

    out = capsys.readouterr().out
    assert f"Agent {EMMA}: REFUSED: no section" in out


def test_the_operator_check_fails_a_section_that_would_match_nothing(
    authority, capsys
) -> None:
    authority.exclude(MERIDIAN, saved_item=["469a149e-0000-4000-8000-000000000000"])

    assert _check() == 1

    assert "no producer records" in capsys.readouterr().out


def test_the_operator_check_fails_without_a_manifest(authority, capsys) -> None:
    assert _check(MERIDIAN) == 1

    assert "no manifest at" in capsys.readouterr().err
