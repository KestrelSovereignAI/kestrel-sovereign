"""Attribution of sovereignty export upload records to exactly one agent (#2874)."""

from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.storage.sync.retention import RetentionPolicy
from kestrel_sovereign.storage.sync.sovereignty_exports import (
    is_sovereignty_export_candidate,
    sovereignty_export_deletions,
    sovereignty_export_owner,
    sovereignty_export_tag,
    sovereignty_export_upload_name,
    sovereignty_export_uploads,
)

DID = "did:pkh:eip155:1:0xabc"
AT = datetime(2026, 10, 12, 9, 30, 5, tzinfo=timezone.utc)


def _upload(name, tag=None):
    return {"cid": "bafy", "fileName": name, "tag": tag}


def test_a_did_is_named_verbatim_and_attributed_back_to_itself():
    name = sovereignty_export_upload_name(DID, AT)

    assert name == f"kestrel_sovereignty__{DID}__20261012_093005.bin"
    assert sovereignty_export_owner(_upload(name)) == DID
    assert sovereignty_export_owner(_upload(name, sovereignty_export_tag(DID))) == DID


@pytest.mark.parametrize(
    "agent_id",
    ["default", "agent/with/slashes", "did:web:x__20260101_000000", "did:x:%3Aenc", "ünï"],
)
def test_every_agent_id_round_trips_to_exactly_itself(agent_id):
    name = sovereignty_export_upload_name(agent_id, AT)

    assert sovereignty_export_owner(_upload(name)) == agent_id


def test_an_agent_whose_id_extends_another_is_never_attributed_to_it():
    extended = DID + "__20260101_000000"
    uploads = [
        _upload(sovereignty_export_upload_name(DID, AT)),
        _upload(sovereignty_export_upload_name(extended, AT)),
    ]

    assert [sovereignty_export_owner(u) for u in uploads] == [DID, extended]
    assert sovereignty_export_uploads(uploads, DID) == [uploads[0]]


def test_slash_and_underscore_ids_do_not_collide():
    slash = sovereignty_export_upload_name("a/b", AT)
    underscore = sovereignty_export_upload_name("a_b", AT)

    assert slash != underscore


@pytest.mark.parametrize(
    "name",
    [
        "content.bin",
        f"kestrel_sovereignty__{DID}.bin",
        "kestrel_sovereignty____20261012_093005.bin",
        f"kestrel_sovereignty__{DID}__20261012_093005.car",
        f"kestrel_state__{DID}__20261012_093005.car",
        # A non-canonical encoding of DID's colon names no agent.
        "kestrel_sovereignty__did%3Apkh:eip155:1:0xabc__20261012_093005.bin",
        "kestrel_sovereignty__agent/b__20261012_093005.bin",
        "",
    ],
)
def test_a_record_outside_the_family_or_not_canonical_belongs_to_no_agent(name):
    assert sovereignty_export_owner(_upload(name)) is None


def test_a_tag_naming_another_agent_leaves_the_record_unattributed():
    name = sovereignty_export_upload_name(DID, AT)

    assert sovereignty_export_owner(_upload(name, "kestrel-sovereignty-did:other")) is None
    assert sovereignty_export_owner(_upload(name, "kestrel-sovereignty")) is None


def test_an_agent_id_is_required():
    with pytest.raises(ValueError):
        sovereignty_export_upload_name("", AT)
    with pytest.raises(ValueError):
        sovereignty_export_tag(None)


@pytest.mark.parametrize(
    "upload",
    [
        # Attributed to one agent.
        _upload(sovereignty_export_upload_name(DID, AT), sovereignty_export_tag(DID)),
        # In the family but attributable to no one agent.
        _upload(sovereignty_export_upload_name(DID, AT), "kestrel-sovereignty-did:other"),
        _upload(f"kestrel_sovereignty__{DID}.bin"),
        _upload("kestrel_sovereignty__did%3Apkh:eip155:1:0xabc__20261012_093005.bin"),
        # What an export carried before #2874: no agent in name or tag.
        _upload("content.bin"),
        _upload("content.bin", "kestrel-sovereignty"),
        _upload("export.bin", "kestrel-sovereignty"),
        _upload("export.bin", f"kestrel-sovereignty-{DID}"),
    ],
)
def test_every_upload_that_is_or_may_be_an_export_is_a_candidate(upload):
    assert is_sovereignty_export_candidate(upload)


@pytest.mark.parametrize(
    "upload",
    [
        _upload(f"kestrel_state__{DID}__20261012_093005.car", f"kestrel-state-{DID}"),
        _upload(f"kestrel_manifest__{DID}__20261012_093005.json"),
        _upload("kestrel_prime.db-wal"),
        _upload("0123456789abcdef.bin", "kestrel-storage"),
        _upload("", None),
    ],
)
def test_other_uploads_are_not_candidates(upload):
    assert not is_sovereignty_export_candidate(upload)


NEWEST_ONLY = RetentionPolicy.from_config(
    {
        "backup": {
            "retention": {
                "identity": {
                    "keep_all_days": 0,
                    "weekly_until_months": 0,
                    "monthly_forever": False,
                }
            }
        }
    }
)


def _export(agent_id, at, cid, *, tag="own"):
    return {
        "id": f"file-{cid}",
        "cid": cid,
        "fileName": sovereignty_export_upload_name(agent_id, at),
        "tag": sovereignty_export_tag(agent_id) if tag == "own" else tag,
        "createdAt": int(at.timestamp() * 1000),
    }


def test_deletions_thin_only_the_agents_own_exports():
    uploads = [
        _export(DID, AT - timedelta(days=3), "a-old"),
        _export(DID, AT - timedelta(days=2), "a-mid"),
        _export(DID, AT, "a-new"),
        _export("did:other", AT - timedelta(days=9), "b-old"),
        _export(DID + "__20260101_000000", AT - timedelta(days=9), "extended-old"),
        _export(DID, AT - timedelta(days=9), "conflict", tag="kestrel-sovereignty-did:other"),
        {"id": "file-legacy", "cid": "legacy", "fileName": "content.bin", "createdAt": 1},
    ]

    deleted = sovereignty_export_deletions(uploads, DID, NEWEST_ONLY, now=AT)

    assert {u["cid"] for u in deleted} == {"a-old", "a-mid"}


def test_a_kept_cid_counts_as_the_newest_but_is_never_returned():
    uploads = [
        _export(DID, AT - timedelta(days=1), "a-old"),
        _export(DID, AT, "a-just-published"),
    ]

    deleted = sovereignty_export_deletions(
        uploads, DID, NEWEST_ONLY, keep_cids={"a-just-published"}, now=AT
    )
    assert [u["cid"] for u in deleted] == ["a-old"]

    # Kept even when the listing places it before an older-looking record.
    clock_skewed = [
        _export(DID, AT, "a-old"),
        _export(DID, AT - timedelta(days=1), "a-just-published"),
    ]
    deleted = sovereignty_export_deletions(
        clock_skewed, DID, NEWEST_ONLY, keep_cids={"a-just-published"}, now=AT
    )
    assert deleted == []


def test_an_export_without_a_cid_is_never_deleted():
    no_cid = _export(DID, AT - timedelta(days=5), "x")
    del no_cid["cid"]
    uploads = [no_cid, _export(DID, AT, "a-new")]

    assert sovereignty_export_deletions(uploads, DID, NEWEST_ONLY, now=AT) == []


def test_an_export_without_a_server_time_is_dated_by_its_name():
    undated = _export(DID, AT - timedelta(days=5), "a-old")
    del undated["createdAt"]
    uploads = [undated, _export(DID, AT, "a-new")]

    deleted = sovereignty_export_deletions(uploads, DID, NEWEST_ONLY, now=AT)
    assert [u["cid"] for u in deleted] == ["a-old"]
