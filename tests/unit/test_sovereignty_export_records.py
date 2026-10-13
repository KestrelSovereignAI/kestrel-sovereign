"""Attribution of sovereignty export upload records to exactly one agent (#2874)."""

import re
from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.storage.sync.retention import RetentionPolicy
from kestrel_sovereign.storage.sync.sovereignty_exports import (
    AmbiguousPublishedExportError,
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

    assert re.fullmatch(
        rf"kestrel_sovereignty__{re.escape(DID)}__20261012_093005__[0-9a-f]{{16}}\.bin",
        name,
    )
    assert sovereignty_export_owner(_upload(name)) == DID
    assert sovereignty_export_owner(_upload(name, sovereignty_export_tag(DID))) == DID


def test_every_publication_is_named_apart_even_in_the_same_second():
    """Lighthouse answers an upload with its CID alone, so the name is how
    the publisher finds the record it made (#3602)."""
    names = {sovereignty_export_upload_name(DID, AT) for _ in range(64)}

    assert len(names) == 64
    assert {sovereignty_export_owner(_upload(name)) for name in names} == {DID}


def test_an_export_named_before_3602_is_still_attributed():
    name = f"kestrel_sovereignty__{DID}__20261012_093005.bin"

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
    # Without the nonce, which sets any two names apart.
    slash = sovereignty_export_upload_name("a/b", AT).rsplit("__", 1)[0]
    underscore = sovereignty_export_upload_name("a_b", AT).rsplit("__", 1)[0]

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
        # A nonce is 16 lowercase hex digits.
        f"kestrel_sovereignty__{DID}__20261012_093005__0123456789ABCDEF.bin",
        f"kestrel_sovereignty__{DID}__20261012_093005__0123456789abcde.bin",
        f"kestrel_sovereignty__{DID}__20261012_093005__.bin",
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


def _export(agent_id, at, cid, *, tag="own", record_id=None):
    return {
        "id": record_id or f"file-{cid}",
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


def _published(upload):
    """What the publisher knows of the record it just made."""
    return {"published_cid": upload["cid"], "published_name": upload["fileName"]}


def test_the_just_published_record_counts_as_the_newest_but_is_never_returned():
    published = _export(DID, AT, "a-just-published")
    uploads = [_export(DID, AT - timedelta(days=1), "a-old"), published]

    deleted = sovereignty_export_deletions(
        uploads, DID, NEWEST_ONLY, **_published(published), now=AT
    )
    assert [u["cid"] for u in deleted] == ["a-old"]

    # Kept even when the listing places it before an older-looking record.
    published = _export(DID, AT - timedelta(days=1), "a-just-published")
    clock_skewed = [_export(DID, AT, "a-old"), published]
    deleted = sovereignty_export_deletions(
        clock_skewed, DID, NEWEST_ONLY, **_published(published), now=AT
    )
    assert deleted == []


def test_an_older_record_of_the_just_published_cid_is_thinned():
    """An identical export uploaded again: two records hold one CID, each
    taking quota. The older is thinned by its own record id; the record just
    published is kept (#3602)."""
    older = _export(DID, AT - timedelta(days=1), "same", record_id="file-older")
    published = _export(DID, AT, "same", record_id="file-published")

    for uploads in ([older, published], [published, older]):
        deleted = sovereignty_export_deletions(
            uploads, DID, NEWEST_ONLY, **_published(published), now=AT
        )
        assert deleted == [older]

    # A duplicate the clock places after the record just published is the
    # newest, so both are kept: the record just published is never returned,
    # whatever its time.
    published = _export(DID, AT - timedelta(days=1), "same", record_id="file-published")
    later = _export(DID, AT, "same", record_id="file-later")
    deleted = sovereignty_export_deletions(
        [published, later], DID, NEWEST_ONLY, **_published(published), now=AT
    )
    assert deleted == []


@pytest.mark.parametrize(
    ("older_id", "published_id"),
    [("z-old", "a-new"), ("a-old", "z-new")],
    ids=["published-id-smaller", "published-id-larger"],
)
def test_an_older_record_of_the_same_time_is_thinned_whatever_the_ids(
    older_id, published_id
):
    """An identical export uploaded in the same second as the one just
    published: the record published is the one with its name, so it is kept
    and the other thinned, whichever id is larger and whichever is listed
    first (#3602)."""
    older = _export(DID, AT, "same", record_id=older_id)
    published = _export(DID, AT, "same", record_id=published_id)
    assert older["fileName"] != published["fileName"]

    for uploads in ([older, published], [published, older]):
        assert sovereignty_export_deletions(
            uploads, DID, NEWEST_ONLY, **_published(published), now=AT
        ) == [older]


def test_the_published_name_listed_as_two_records_thins_nothing():
    """The account lists the publication twice: one name, two ids. Either
    could be the record just made, so neither, nor anything else, is
    thinned (#3602)."""
    published = _export(DID, AT, "same", record_id="z-old")
    twin = dict(published, id="a-new")
    older = _export(DID, AT - timedelta(days=3), "older", record_id="file-older")

    for uploads in ([older, published, twin], [twin, older, published]):
        with pytest.raises(AmbiguousPublishedExportError):
            sovereignty_export_deletions(
                uploads, DID, NEWEST_ONLY, **_published(published), now=AT
            )

    # A twin with no id is a second record too.
    no_id = dict(published)
    del no_id["id"]
    with pytest.raises(AmbiguousPublishedExportError):
        sovereignty_export_deletions(
            [older, published, no_id], DID, NEWEST_ONLY, **_published(published), now=AT
        )


def test_the_published_name_holding_another_cid_thins_nothing():
    published = _export(DID, AT, "a-new")
    older = _export(DID, AT - timedelta(days=3), "a-old")

    with pytest.raises(AmbiguousPublishedExportError):
        sovereignty_export_deletions(
            [older, published],
            DID,
            NEWEST_ONLY,
            published_cid="a-other",
            published_name=published["fileName"],
            now=AT,
        )


def test_the_published_record_not_yet_listed_leaves_the_others_to_the_policy():
    """A listing that does not show the record just published yet cannot
    delete it; the agent's other exports are thinned among themselves."""
    uploads = [
        _export(DID, AT - timedelta(days=2), "a-old"),
        _export(DID, AT - timedelta(days=1), "a-mid"),
    ]

    deleted = sovereignty_export_deletions(
        uploads,
        DID,
        NEWEST_ONLY,
        published_cid="a-new",
        published_name=sovereignty_export_upload_name(DID, AT),
        now=AT,
    )

    assert [u["cid"] for u in deleted] == ["a-old"]


def test_records_sharing_a_cid_are_each_decided_on():
    uploads = [
        _export(DID, AT - timedelta(days=2), "same", record_id="file-1"),
        _export(DID, AT - timedelta(days=1), "same", record_id="file-2"),
        _export(DID, AT, "same", record_id="file-3"),
    ]

    deleted = sovereignty_export_deletions(uploads, DID, NEWEST_ONLY, now=AT)

    assert [u["id"] for u in deleted] == ["file-2", "file-1"]


def test_another_agents_record_of_the_same_cid_is_never_thinned():
    published = _export(DID, AT, "same", record_id="file-published")
    uploads = [
        _export("did:other", AT - timedelta(days=3), "same", record_id="file-other"),
        _export(
            DID,
            AT - timedelta(days=3),
            "same",
            tag="kestrel-sovereignty-did:other",
            record_id="file-conflict",
        ),
        {"id": "file-legacy", "cid": "same", "fileName": "content.bin", "createdAt": 1},
        published,
    ]

    assert (
        sovereignty_export_deletions(
            uploads, DID, NEWEST_ONLY, **_published(published), now=AT
        )
        == []
    )


def test_a_record_listed_twice_is_decided_on_once():
    newest = _export(DID, AT, "a-new")
    uploads = [_export(DID, AT - timedelta(days=1), "a-old"), newest, dict(newest)]

    deleted = sovereignty_export_deletions(uploads, DID, NEWEST_ONLY, now=AT)

    assert [u["cid"] for u in deleted] == ["a-old"]


def test_an_export_without_a_record_id_is_never_deleted():
    no_id = _export(DID, AT - timedelta(days=5), "x")
    del no_id["id"]
    uploads = [no_id, _export(DID, AT, "a-new")]

    assert sovereignty_export_deletions(uploads, DID, NEWEST_ONLY, now=AT) == []


@pytest.mark.parametrize(
    "published", [{"published_cid": "a-new"}, {"published_name": "x.bin"}]
)
def test_the_just_published_record_is_named_by_cid_and_name_together(published):
    with pytest.raises(ValueError):
        sovereignty_export_deletions([], DID, NEWEST_ONLY, **published)


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
