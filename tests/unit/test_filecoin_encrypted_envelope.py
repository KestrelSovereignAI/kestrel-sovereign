"""The self-describing encrypted envelope (#2873).

An encrypted export used to carry no copy of the content hash its key derives
from; that hash lived only in the exporter's ``backup_artifact`` record. The
envelope carries it, a master-key identifier and an owning-agent identifier in
an authenticated header, so the payload and ``KESTREL_DATA_KEY`` alone decrypt
it, for the agent that sealed it.
"""

import hashlib
import json
import zlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.fernet import Fernet

# Import the storage providers package before filecoin_adapter so this module
# is runnable in isolation (see test_filecoin_encryption_portability.py).
import kestrel_sovereign.storage.providers.base  # noqa: F401
from kestrel_sovereign import filecoin_adapter
from kestrel_sovereign.agent.backup import BackupMixin
from kestrel_sovereign.filecoin_adapter import (
    ENVELOPE_MAGIC,
    DecryptionKeyUnavailableError,
    EncryptedEnvelopeError,
    EnvelopeDecryptionError,
    EnvelopeFormatError,
    EnvelopeKeyMismatchError,
    EnvelopeOwnerMismatchError,
    EnvelopePolicy,
    FilecoinAdapter,
    LegacyDecryptionUnavailableError,
    StorageTier,
    UnencryptedContentError,
    envelope_cache_name,
    envelope_key_id,
    envelope_owner_id,
    is_cache_name_of,
)

PLAINTEXT = b"a whole agent database, in a tarball"
OWNER = "did:pkh:eip155:1:0x2873owner"
OTHER_AGENT = "did:pkh:eip155:1:0x2873other"


@pytest.fixture(autouse=True)
def no_ipfs_daemon():
    with patch(
        "kestrel_sovereign.filecoin_adapter.requests.post",
        side_effect=ConnectionError("no IPFS daemon"),
    ):
        yield


@pytest.fixture
def data_key(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("KESTREL_DATA_KEY", key)
    return key


def _adapter(tmp_path, name="host"):
    return FilecoinAdapter(cache_dir=str(tmp_path / name))


def _split(payload):
    """Header, authenticated prefix, and token of an envelope."""
    header_start = len(ENVELOPE_MAGIC) + 4
    length = int.from_bytes(payload[len(ENVELOPE_MAGIC):header_start], "big")
    header_end = header_start + length
    return json.loads(payload[header_start:header_end]), payload[:header_end], payload[header_end:]


def _sealed(tmp_path):
    adapter = _adapter(tmp_path)
    payload, key_hash = adapter._seal_envelope(
        PLAINTEXT, content_hash=hashlib.sha256(PLAINTEXT).hexdigest(), owner=OWNER
    )
    return adapter, payload, key_hash


def _rewrite_header(payload, *, separators=(",", ":"), **fields):
    header, _, token = _split(payload)
    header.update(fields)
    raw = json.dumps(header, sort_keys=True, separators=separators).encode("ascii")
    return ENVELOPE_MAGIC + len(raw).to_bytes(4, "big") + raw + token


def test_header_names_what_decryption_needs_and_no_key(tmp_path, data_key):
    adapter, payload, key_hash = _sealed(tmp_path)
    header, _, _ = _split(payload)

    master_key = adapter._get_master_key()
    assert header == {
        "format": "kestrel.encrypted-envelope",
        "version": 1,
        "cipher": "aes-256-gcm",
        "kdf": "hkdf-sha256:kestrel-filecoin-content-key-v1",
        "key_hash": hashlib.sha256(PLAINTEXT).hexdigest(),
        "key_id": envelope_key_id(master_key),
        "owner_id": envelope_owner_id(master_key, OWNER),
    }
    assert key_hash == header["key_hash"]
    content_key = adapter._derive_content_key(master_key, key_hash)
    for secret in (master_key, data_key.encode(), content_key, PLAINTEXT, OWNER.encode()):
        assert secret not in payload


def test_header_is_the_tokens_associated_data(tmp_path, data_key):
    from kestrel_sdk.security.aead import AEADCipher
    from kestrel_sdk.security.exceptions import DecryptionError

    adapter, payload, key_hash = _sealed(tmp_path)
    _, prefix, token = _split(payload)
    cipher = AEADCipher(adapter._derive_content_key(adapter._get_master_key(), key_hash))

    assert cipher.decrypt(token, aad=prefix) == PLAINTEXT
    with pytest.raises(DecryptionError):
        cipher.decrypt(token)
    with pytest.raises(DecryptionError):
        cipher.decrypt(token, aad=prefix.replace(b'"version":1', b'"version":9'))


def test_envelope_opens_on_another_host_from_the_payload_and_data_key(tmp_path, data_key):
    _, payload, _ = _sealed(tmp_path)
    fresh_host = _adapter(tmp_path, "fresh")
    assert not list(fresh_host.cache_dir.iterdir())

    assert fresh_host.open_envelope(payload, owner=OWNER) == PLAINTEXT


def test_envelope_under_another_key_names_both_key_ids(tmp_path, data_key, monkeypatch):
    _, payload, _ = _sealed(tmp_path)
    sealed_under = _split(payload)[0]["key_id"]
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    fresh_host = _adapter(tmp_path, "fresh")

    with pytest.raises(EnvelopeKeyMismatchError) as caught:
        fresh_host.open_envelope(payload, owner=OWNER)

    configured = envelope_key_id(fresh_host._get_master_key())
    assert caught.value.envelope_key_id == sealed_under
    assert caught.value.configured_key_id == configured
    assert sealed_under in str(caught.value) and configured in str(caught.value)


def test_envelope_without_a_configured_key_is_refused(tmp_path, data_key):
    adapter, payload, _ = _sealed(tmp_path)
    with patch(
        "kestrel_sovereign.security.encryption.get_master_key_bytes",
        return_value=None,
    ):
        with pytest.raises(DecryptionKeyUnavailableError):
            adapter.open_envelope(payload, owner=OWNER)


def test_envelope_whose_plaintext_misses_its_key_hash_is_refused(tmp_path, data_key):
    """Authentic but inconsistent: sealed under the key its header derives,
    yet the plaintext does not hash to that header's ``key_hash``."""
    adapter = _adapter(tmp_path)
    payload, _ = adapter._seal_envelope(PLAINTEXT, content_hash="d" * 64, owner=OWNER)

    with pytest.raises(EnvelopeDecryptionError, match="does not match"):
        adapter.open_envelope(payload, owner=OWNER)


def test_envelope_naming_another_key_hash_than_its_record_is_refused(tmp_path, data_key):
    adapter, payload, key_hash = _sealed(tmp_path)

    assert adapter.open_envelope(payload, owner=OWNER, expected_key_hash=key_hash) == PLAINTEXT
    with pytest.raises(EnvelopeDecryptionError, match="b" * 64):
        adapter.open_envelope(payload, owner=OWNER, expected_key_hash="b" * 64)


def test_envelope_sealed_for_another_agent_is_refused_before_decryption(tmp_path, data_key):
    """Every co-hosted agent holds the master key, so it would decrypt any
    of their envelopes. The owner is checked before any key is derived."""
    adapter, payload, _ = _sealed(tmp_path)
    master_key = adapter._get_master_key()

    with patch.object(adapter, "_derive_content_key") as derive:
        with pytest.raises(EnvelopeOwnerMismatchError) as caught:
            adapter.open_envelope(payload, owner=OTHER_AGENT)

    derive.assert_not_called()
    assert caught.value.envelope_owner_id == envelope_owner_id(master_key, OWNER)
    assert caught.value.owner_id == envelope_owner_id(master_key, OTHER_AGENT)
    assert caught.value.envelope_owner_id != caught.value.owner_id
    assert caught.value.envelope_owner_id in str(caught.value)
    assert caught.value.owner_id in str(caught.value)


def test_owner_id_rewritten_for_another_agent_fails_authentication(tmp_path, data_key):
    """The owner id is authenticated: a header rewritten to name the agent
    opening it passes the owner check and then fails decryption."""
    adapter, payload, _ = _sealed(tmp_path)
    rewritten = _rewrite_header(
        payload, owner_id=envelope_owner_id(adapter._get_master_key(), OTHER_AGENT)
    )

    with pytest.raises(EnvelopeDecryptionError, match="failed authentication"):
        adapter.open_envelope(rewritten, owner=OTHER_AGENT)


def test_owner_id_depends_on_the_master_key_and_the_did(tmp_path, data_key):
    master_key = _adapter(tmp_path)._get_master_key()
    owner_id = envelope_owner_id(master_key, OWNER)

    assert owner_id == envelope_owner_id(master_key, OWNER)
    assert owner_id != envelope_owner_id(master_key, OTHER_AGENT)
    assert owner_id != envelope_owner_id(b"\x00" * 32, OWNER)
    with pytest.raises(ValueError, match="agent DID"):
        envelope_owner_id(master_key, "")


def _flip_token_byte(payload):
    return payload[:-10] + (b"A" if payload[-10:-9] != b"A" else b"B") + payload[-9:]


def _oversized_header_length(payload):
    magic = len(ENVELOPE_MAGIC)
    return payload[:magic] + (2**31).to_bytes(4, "big") + payload[magic + 4:]


@pytest.mark.parametrize(
    ("tamper", "error", "message"),
    [
        (lambda p: _rewrite_header(p, key_hash="0" * 64), EnvelopeDecryptionError, "failed authentication"),
        (lambda p: _rewrite_header(p, key_id="0" * 32), EnvelopeKeyMismatchError, "key id " + "0" * 32),
        (lambda p: _rewrite_header(p, owner_id="0" * 32), EnvelopeOwnerMismatchError, "owner id " + "0" * 32),
        (lambda p: _rewrite_header(p, version=2), EnvelopeFormatError, "version 2 is not supported"),
        (lambda p: _rewrite_header(p, cipher="none"), EnvelopeFormatError, "cipher 'none'"),
        (lambda p: _rewrite_header(p, extra="field"), EnvelopeFormatError, "fields are invalid"),
        (lambda p: _rewrite_header(p, separators=(", ", ": ")), EnvelopeFormatError, "not canonical"),
        (_oversized_header_length, EnvelopeFormatError, "header length is invalid"),
        (lambda p: _split(p)[1], EnvelopeFormatError, "header length is invalid"),
        (_flip_token_byte, EnvelopeDecryptionError, "failed authentication"),
    ],
    ids=[
        "key-hash", "key-id", "owner-id", "version", "cipher", "extra-field",
        "non-canonical", "header-length", "no-token", "token",
    ],
)
def test_tampered_envelope_is_refused(tmp_path, data_key, tamper, error, message):
    adapter, payload, _ = _sealed(tmp_path)
    tampered = tamper(payload)
    assert tampered != payload

    with pytest.raises(error, match=message):
        adapter.open_envelope(tampered, owner=OWNER)


def test_header_nested_past_the_json_recursion_limit_is_a_format_error(
    tmp_path, data_key, monkeypatch
):
    """A 4 KB header of nested brackets: Python 3.11's JSON scanner raises
    RecursionError on it, not ValueError, and later versions parse it. Either
    way the envelope is malformed, so both are simulated here."""
    raw = b"[" * 2040 + b"]" * 2040
    payload = ENVELOPE_MAGIC + len(raw).to_bytes(4, "big") + raw + b"token"
    adapter = _adapter(tmp_path)

    with pytest.raises(EnvelopeFormatError):
        adapter.open_envelope(payload, owner=OWNER)

    def scanner_recursion(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded while decoding a JSON array")

    monkeypatch.setattr(
        filecoin_adapter, "json", SimpleNamespace(loads=scanner_recursion, dumps=json.dumps)
    )
    with pytest.raises(EnvelopeFormatError, match="not JSON"):
        adapter.open_envelope(payload, owner=OWNER)


def _cached(adapter, *, encrypt, envelope_owner=None):
    return adapter.store_content(
        PLAINTEXT,
        storage_tier=StorageTier.LOCAL_ONLY,
        encrypt=encrypt,
        envelope_owner=envelope_owner,
    )


def _owner_cache_name(adapter, content_hash, owner):
    return envelope_cache_name(content_hash, envelope_owner_id(adapter._get_master_key(), owner))


def test_sealed_store_writes_an_envelope_and_records_its_key_hash(tmp_path, data_key):
    adapter = _adapter(tmp_path)
    result = _cached(adapter, encrypt=True, envelope_owner=OWNER)

    name = _owner_cache_name(adapter, result.content_hash, OWNER)
    stored = zlib.decompress((adapter.cache_dir / f"{name}.cache").read_bytes())
    assert stored.startswith(ENVELOPE_MAGIC)
    assert _split(stored)[0]["owner_id"] == name.split(".", 1)[1]
    assert not (adapter.cache_dir / f"{result.content_hash}.cache").exists()
    assert result.encrypted is True
    assert result.encryption_key_hash == result.content_hash == _split(stored)[0]["key_hash"]
    # The name derives from the master key; the result records it.
    assert result.cache_name == name


@pytest.mark.parametrize("encrypt", [False, True], ids=["plaintext", "legacy-ciphertext"])
def test_store_without_an_envelope_records_its_content_hash_as_its_cache_name(
    tmp_path, data_key, encrypt
):
    adapter = _adapter(tmp_path)
    result = _cached(adapter, encrypt=encrypt)

    assert result.cache_name == result.content_hash
    assert (adapter.cache_dir / f"{result.cache_name}.cache").is_file()


def test_is_cache_name_of_accepts_only_the_names_a_store_writes(tmp_path, data_key):
    content_hash = hashlib.sha256(PLAINTEXT).hexdigest()
    envelope_name = f"{content_hash}.{'b' * 32}"

    assert is_cache_name_of(content_hash, content_hash)
    assert is_cache_name_of(envelope_name, content_hash)
    for name in (
        "c" * 64,
        f"{'c' * 64}.{'b' * 32}",
        f"{content_hash}.{'b' * 31}",
        f"{content_hash}.{'B' * 32}",
        f"{content_hash}.",
        f"{content_hash}.{'b' * 32}.cache",
        f"{content_hash}/../{'b' * 32}",
        content_hash.upper(),
        "",
        None,
        7,
    ):
        assert not is_cache_name_of(name, content_hash), name
    # A content hash is a lowercase SHA-256, never a CID.
    assert not is_cache_name_of("bafyabc", "bafyabc")


def test_two_agents_sealing_one_plaintext_each_restore_their_own(tmp_path, data_key):
    """The host cache is shared, and envelopes of one plaintext sealed for
    two agents share its content hash. Cached under that hash alone, the
    second would overwrite the first, whose restore would then be refused
    as another agent's export (#2873)."""
    adapter = _adapter(tmp_path)
    mine = _cached(adapter, encrypt=True, envelope_owner=OWNER)
    theirs = _cached(adapter, encrypt=True, envelope_owner=OTHER_AGENT)
    assert mine.content_hash == theirs.content_hash

    assert mine.cache_name != theirs.cache_name
    for owner, stored in ((OWNER, mine), (OTHER_AGENT, theirs)):
        assert adapter.retrieve_content(
            stored.content_hash,
            envelope=EnvelopePolicy.ENCRYPTED_ONLY,
            envelope_owner=owner,
            envelope_key_hash=stored.content_hash,
            cache_name=stored.cache_name,
        ) == PLAINTEXT


def test_an_envelope_fetched_from_ipfs_is_cached_under_its_owner(tmp_path, data_key):
    """A restore that reads an envelope from IPFS caches it as the store
    does, so another agent's later read of the same name is not answered by
    this agent's envelope."""
    origin = _adapter(tmp_path)
    stored = origin.store_content(
        PLAINTEXT, storage_tier=StorageTier.LOCAL_ONLY, encrypt=True, envelope_owner=OWNER
    )
    name = _owner_cache_name(origin, stored.content_hash, OWNER)
    blob = (origin.cache_dir / f"{name}.cache").read_bytes()
    fresh = _adapter(tmp_path, "fresh")

    with patch.object(fresh, "_retrieve_ipfs", return_value=blob):
        assert fresh.retrieve_content(
            stored.content_hash,
            ipfs_cid="bafy-owner",
            envelope=EnvelopePolicy.ENCRYPTED_ONLY,
            envelope_owner=OWNER,
        ) == PLAINTEXT

    assert (fresh.cache_dir / f"{name}.cache").read_bytes() == blob
    assert not (fresh.cache_dir / f"{stored.content_hash}.cache").exists()


@pytest.mark.parametrize("recorded", [True, False], ids=["receipt", "no-receipt"])
def test_an_envelope_fetched_by_cid_is_cached_under_its_record_s_name(
    tmp_path, data_key, recorded
):
    """An import requests an export by its IPFS CID. Fetched from IPFS, the
    envelope is cached under the name its header authenticates, which is the
    name ``store_content`` recorded, not under the CID: a copy under the CID
    is one a later read by the recorded name never finds (#2873)."""
    origin = _adapter(tmp_path)
    stored = _cached(origin, encrypt=True, envelope_owner=OWNER)
    blob = (origin.cache_dir / f"{stored.cache_name}.cache").read_bytes()
    host = _adapter(tmp_path, "restoring")
    cache_name = stored.cache_name if recorded else None

    with patch.object(host, "_retrieve_ipfs", return_value=blob):
        assert host.retrieve_content(
            "bafy-export",
            ipfs_cid="bafy-export",
            envelope=EnvelopePolicy.ENCRYPTED_ONLY,
            envelope_owner=OWNER,
            envelope_key_hash=stored.content_hash,
            cache_name=cache_name,
        ) == PLAINTEXT

    assert sorted(path.name for path in host.cache_dir.iterdir()) == [
        f"{stored.cache_name}.cache",
        f"{stored.cache_name}.meta",
    ]
    assert (host.cache_dir / f"{stored.cache_name}.cache").read_bytes() == blob
    # IPFS is gone; the record's name still finds the export.
    with patch.object(host, "_retrieve_ipfs", side_effect=ConnectionError("IPFS down")):
        assert host.retrieve_content(
            "bafy-export",
            ipfs_cid="bafy-export",
            envelope=EnvelopePolicy.ENCRYPTED_ONLY,
            envelope_owner=OWNER,
            envelope_key_hash=stored.content_hash,
            cache_name=stored.cache_name,
        ) == PLAINTEXT


def test_plaintext_fetched_by_cid_is_cached_under_the_content_hash_it_was_checked_against(
    tmp_path, data_key
):
    """With a recorded cache name the content is checked against the content
    hash the name begins with, and cached under it; with none, a CID lookup
    checks nothing and keeps the name it was requested by, as before."""
    origin = _adapter(tmp_path)
    plain = _cached(origin, encrypt=False)
    blob = (origin.cache_dir / f"{plain.cache_name}.cache").read_bytes()
    recorded = _adapter(tmp_path, "recorded")
    unrecorded = _adapter(tmp_path, "unrecorded")

    for host, cache_name in ((recorded, plain.cache_name), (unrecorded, None)):
        with patch.object(host, "_retrieve_ipfs", return_value=blob):
            assert host.retrieve_content(
                "bafy-plain",
                ipfs_cid="bafy-plain",
                envelope=EnvelopePolicy.OPEN,
                envelope_owner=OWNER,
                cache_name=cache_name,
            ) == PLAINTEXT

    assert (recorded.cache_dir / f"{plain.content_hash}.cache").read_bytes() == blob
    assert not (recorded.cache_dir / "bafy-plain.cache").exists()
    assert (unrecorded.cache_dir / "bafy-plain.cache").read_bytes() == blob
    assert not (unrecorded.cache_dir / f"{plain.content_hash}.cache").exists()


def test_content_fetched_under_a_recorded_cache_name_must_hash_to_it(tmp_path, data_key):
    """A recorded cache name names the content hash of its export, so IPFS
    answering a CID with another export's bytes is refused and nothing is
    cached."""
    origin = _adapter(tmp_path)
    plain = _cached(origin, encrypt=False)
    host = _adapter(tmp_path, "restoring")

    with patch.object(host, "_retrieve_ipfs", return_value=zlib.compress(b"another export")):
        with pytest.raises(ValueError, match="integrity check failed"):
            host.retrieve_content(
                "bafy-plain",
                ipfs_cid="bafy-plain",
                envelope=EnvelopePolicy.OPEN,
                envelope_owner=OWNER,
                cache_name=plain.cache_name,
            )

    assert list(host.cache_dir.iterdir()) == []


def test_retrieve_without_a_master_key_still_reads_a_plaintext_export(tmp_path, data_key):
    """Without a master key no envelope can be opened, and a plaintext
    export is read as before."""
    adapter = _adapter(tmp_path)
    plain = _cached(adapter, encrypt=False)

    with patch(
        "kestrel_sovereign.security.encryption.get_master_key_bytes",
        return_value=None,
    ):
        assert adapter.retrieve_content(
            plain.content_hash, envelope=EnvelopePolicy.OPEN, envelope_owner=OWNER
        ) == PLAINTEXT


def test_sealing_an_envelope_requires_encryption(tmp_path, data_key):
    with pytest.raises(ValueError, match="requires encrypt=True"):
        _cached(_adapter(tmp_path), encrypt=False, envelope_owner=OWNER)


@pytest.mark.parametrize("policy", [EnvelopePolicy.OPEN, EnvelopePolicy.ENCRYPTED_ONLY])
def test_retrieve_opens_an_envelope_without_a_key_hash(tmp_path, data_key, policy):
    adapter = _adapter(tmp_path)
    result = _cached(adapter, encrypt=True, envelope_owner=OWNER)

    assert adapter.retrieve_content(
        result.content_hash, envelope=policy, envelope_owner=OWNER, cache_name=result.cache_name
    ) == PLAINTEXT


def test_retrieve_binds_an_envelope_to_the_key_hash_it_is_given(tmp_path, data_key):
    adapter = _adapter(tmp_path)
    result = _cached(adapter, encrypt=True, envelope_owner=OWNER)

    with pytest.raises(EnvelopeDecryptionError):
        adapter.retrieve_content(
            result.content_hash,
            key_hash="c" * 64,
            envelope=EnvelopePolicy.OPEN,
            envelope_owner=OWNER,
            cache_name=result.cache_name,
        )


def test_retrieve_binds_an_envelope_to_the_content_hash_its_record_names(tmp_path, data_key):
    """A record with no key hash still names its export by content hash."""
    adapter = _adapter(tmp_path)
    result = _cached(adapter, encrypt=True, envelope_owner=OWNER)

    assert adapter.retrieve_content(
        result.content_hash,
        envelope=EnvelopePolicy.OPEN,
        envelope_owner=OWNER,
        envelope_key_hash=result.content_hash,
        cache_name=result.cache_name,
    ) == PLAINTEXT
    with pytest.raises(EnvelopeDecryptionError, match="c" * 64):
        adapter.retrieve_content(
            result.content_hash,
            envelope=EnvelopePolicy.OPEN,
            envelope_owner=OWNER,
            envelope_key_hash="c" * 64,
            cache_name=result.cache_name,
        )


@pytest.mark.parametrize("policy", [EnvelopePolicy.OPEN, EnvelopePolicy.ENCRYPTED_ONLY])
def test_retrieve_refuses_an_envelope_sealed_for_another_agent(tmp_path, data_key, policy):
    """The cache keys an envelope by its owner, so another agent's read of
    its content hash does not find it there. One that arrives another way,
    under its owner's name or the plain one, is refused."""
    adapter = _adapter(tmp_path)
    result = _cached(adapter, encrypt=True, envelope_owner=OWNER)
    with pytest.raises(ValueError, match="Content not found"):
        adapter.retrieve_content(
            result.content_hash, envelope=policy, envelope_owner=OTHER_AGENT
        )
    with pytest.raises(EnvelopeOwnerMismatchError):
        adapter.retrieve_content(
            result.content_hash,
            envelope=policy,
            envelope_owner=OTHER_AGENT,
            cache_name=result.cache_name,
        )
    name = _owner_cache_name(adapter, result.content_hash, OWNER)
    _cached_under(
        adapter,
        result.content_hash,
        zlib.decompress((adapter.cache_dir / f"{name}.cache").read_bytes()),
    )

    with pytest.raises(EnvelopeOwnerMismatchError):
        adapter.retrieve_content(
            result.content_hash,
            key_hash=result.encryption_key_hash,
            envelope=policy,
            envelope_owner=OTHER_AGENT,
        )


def test_recorded_cache_name_finds_an_envelope_under_another_key_which_it_names(
    tmp_path, data_key, monkeypatch
):
    """The cache name derives from the key the envelope was sealed under, so
    a reader under another key cannot derive it. Read by the name its record
    holds, the envelope is found and names the key it needs, rather than the
    export being reported missing (#2873)."""
    adapter = _adapter(tmp_path)
    result = _cached(adapter, encrypt=True, envelope_owner=OWNER)
    sealed_key_id = envelope_key_id(adapter._get_master_key())
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())

    with pytest.raises(ValueError, match="Content not found"):
        adapter.retrieve_content(
            result.content_hash, envelope=EnvelopePolicy.ENCRYPTED_ONLY, envelope_owner=OWNER
        )
    with pytest.raises(EnvelopeKeyMismatchError) as caught:
        adapter.retrieve_content(
            result.content_hash,
            envelope=EnvelopePolicy.ENCRYPTED_ONLY,
            envelope_owner=OWNER,
            cache_name=result.cache_name,
        )

    assert caught.value.envelope_key_id == sealed_key_id
    assert caught.value.configured_key_id == envelope_key_id(adapter._get_master_key())
    assert (adapter.cache_dir / f"{result.cache_name}.cache").is_file()


def test_recorded_cache_name_finds_an_envelope_with_no_key_configured(tmp_path, data_key):
    adapter = _adapter(tmp_path)
    result = _cached(adapter, encrypt=True, envelope_owner=OWNER)

    with patch(
        "kestrel_sovereign.security.encryption.get_master_key_bytes",
        return_value=None,
    ):
        with pytest.raises(DecryptionKeyUnavailableError):
            adapter.retrieve_content(
                result.content_hash,
                envelope=EnvelopePolicy.ENCRYPTED_ONLY,
                envelope_owner=OWNER,
                cache_name=result.cache_name,
            )


@pytest.mark.parametrize(
    "cache_name",
    ["../../etc/passwd", "bafyabc", f"{'a' * 64}.{'b' * 31}", ("a" * 64).upper(), "", 7],
)
def test_retrieve_reads_no_cache_entry_under_a_name_no_store_writes(
    tmp_path, data_key, cache_name
):
    adapter = _adapter(tmp_path)

    with patch.object(adapter, "_retrieve_local_cache") as read_cache:
        with pytest.raises(ValueError, match="not a name the host cache is written under"):
            adapter.retrieve_content(
                "a" * 64,
                envelope=EnvelopePolicy.ENCRYPTED_ONLY,
                envelope_owner=OWNER,
                cache_name=cache_name,
            )

    read_cache.assert_not_called()


def test_retrieve_refuses_a_cache_name_it_may_not_read(tmp_path, data_key):
    adapter = _adapter(tmp_path)
    result = _cached(adapter, encrypt=True, envelope_owner=OWNER)

    with patch.object(adapter, "_retrieve_local_cache") as read_cache:
        with pytest.raises(ValueError, match="allow_local_cache=False"):
            adapter.retrieve_content(
                result.content_hash,
                envelope=EnvelopePolicy.ENCRYPTED_ONLY,
                envelope_owner=OWNER,
                allow_local_cache=False,
                cache_name=result.cache_name,
            )

    read_cache.assert_not_called()


def test_content_read_by_its_cache_name_must_hash_to_it_whatever_it_was_requested_by(
    tmp_path, data_key
):
    """A receipt's export is requested by its IPFS CID, which is no content
    hash to check the result against. Read from the cache by its recorded
    name, the result must still hash to the content hash the name begins
    with."""
    adapter = _adapter(tmp_path)
    plain = _cached(adapter, encrypt=False)
    _cached_under(adapter, plain.cache_name, b"another export's bytes")

    with pytest.raises(ValueError, match="integrity check failed"):
        adapter.retrieve_content(
            "bafy-requested-by-cid",
            ipfs_cid="bafy-requested-by-cid",
            envelope=EnvelopePolicy.OPEN,
            envelope_owner=OWNER,
            cache_name=plain.cache_name,
        )


def test_the_requested_name_is_read_when_the_recorded_name_misses(tmp_path, data_key):
    """Before #2873 a restore by CID cached what IPFS returned under the CID.
    With the recorded name's entry evicted, that copy still answers, and it
    must hash to the content hash the recorded name begins with."""
    adapter = _adapter(tmp_path)
    plain = _cached(adapter, encrypt=False)
    (adapter.cache_dir / f"{plain.cache_name}.cache").unlink()

    def retrieve():
        return adapter.retrieve_content(
            "bafy-restored-before",
            ipfs_cid="bafy-restored-before",
            envelope=EnvelopePolicy.OPEN,
            envelope_owner=OWNER,
            cache_name=plain.cache_name,
        )

    with patch.object(adapter, "_retrieve_ipfs", side_effect=ConnectionError("IPFS down")):
        with pytest.raises(ValueError, match="Content not found"):
            retrieve()
        _cached_under(adapter, "bafy-restored-before", PLAINTEXT)
        assert retrieve() == PLAINTEXT
        _cached_under(adapter, "bafy-restored-before", b"another export's bytes")
        with pytest.raises(ValueError, match="integrity check failed"):
            retrieve()


@pytest.mark.parametrize("policy", [EnvelopePolicy.OPEN, EnvelopePolicy.ENCRYPTED_ONLY])
@pytest.mark.parametrize("owner", [None, ""], ids=["none", "empty"])
def test_retrieve_opening_envelopes_requires_the_retrieving_agent(
    tmp_path, data_key, policy, owner
):
    adapter = _adapter(tmp_path)
    result = _cached(adapter, encrypt=True, envelope_owner=OWNER)

    with patch.object(adapter, "_retrieve_local_cache") as read_cache:
        with pytest.raises(ValueError, match="envelope_owner"):
            adapter.retrieve_content(
                result.content_hash, envelope=policy, envelope_owner=owner
            )

    read_cache.assert_not_called()


def test_retrieve_ignoring_envelopes_is_unchanged_for_other_callers(tmp_path, data_key):
    adapter = _adapter(tmp_path)
    legacy = _cached(adapter, encrypt=True)

    assert adapter.retrieve_content(
        legacy.content_hash, key_hash=legacy.encryption_key_hash
    ) == PLAINTEXT


def test_open_policy_decrypts_legacy_ciphertext_with_its_key_hash(tmp_path, data_key):
    adapter = _adapter(tmp_path)
    legacy = _cached(adapter, encrypt=True)

    assert adapter.retrieve_content(
        legacy.content_hash,
        key_hash=legacy.encryption_key_hash,
        envelope=EnvelopePolicy.OPEN, envelope_owner=OWNER,
    ) == PLAINTEXT


def _cached_under(adapter, name, payload):
    """Put a payload in the cache under a name of the test's choosing."""
    (adapter.cache_dir / f"{name}.cache").write_bytes(zlib.compress(payload))


def _legacy_payload(adapter, legacy):
    return zlib.decompress((adapter.cache_dir / f"{legacy.content_hash}.cache").read_bytes())


@pytest.mark.parametrize("policy", [EnvelopePolicy.OPEN, EnvelopePolicy.ENCRYPTED_ONLY])
def test_legacy_ciphertext_without_a_key_hash_decrypts_under_its_content_hash(
    tmp_path, data_key, policy
):
    """No record names the key hash, so the configured key is tried under the
    name the payload was retrieved by. An export named by its content hash
    (what ``export_sovereignty`` reports when no IPFS CID came back) opens."""
    adapter = _adapter(tmp_path)
    legacy = _cached(adapter, encrypt=True)
    assert not _legacy_payload(adapter, legacy).startswith(ENVELOPE_MAGIC)

    assert adapter.retrieve_content(legacy.content_hash, envelope=policy, envelope_owner=OWNER) == PLAINTEXT


@pytest.mark.parametrize("policy", [EnvelopePolicy.OPEN, EnvelopePolicy.ENCRYPTED_ONLY])
def test_legacy_ciphertext_under_another_content_hash_is_never_returned(
    tmp_path, data_key, policy
):
    adapter = _adapter(tmp_path)
    legacy = _cached(adapter, encrypt=True)
    _cached_under(adapter, "e" * 64, _legacy_payload(adapter, legacy))

    with pytest.raises(LegacyDecryptionUnavailableError) as caught:
        adapter.retrieve_content("e" * 64, envelope=policy, envelope_owner=OWNER)

    assert caught.value.attempted is True
    assert not isinstance(caught.value, EncryptedEnvelopeError)


def test_legacy_ciphertext_under_another_master_key_is_never_returned(
    tmp_path, data_key, monkeypatch
):
    adapter = _adapter(tmp_path)
    legacy = _cached(adapter, encrypt=True)
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())

    with pytest.raises(LegacyDecryptionUnavailableError) as caught:
        adapter.retrieve_content(legacy.content_hash, envelope=EnvelopePolicy.ENCRYPTED_ONLY, envelope_owner=OWNER)

    assert caught.value.attempted is True


def test_legacy_ciphertext_named_by_a_cid_has_no_key_to_try(tmp_path, data_key):
    """An IPFS CID is never the content hash the F187 key derives from."""
    adapter = _adapter(tmp_path)
    legacy = _cached(adapter, encrypt=True)
    cid = "bafy" + legacy.content_hash
    _cached_under(adapter, cid, _legacy_payload(adapter, legacy))

    with patch.object(adapter, "_derive_content_key") as derive:
        with pytest.raises(LegacyDecryptionUnavailableError) as caught:
            adapter.retrieve_content(cid, envelope=EnvelopePolicy.ENCRYPTED_ONLY, envelope_owner=OWNER)

    assert caught.value.attempted is False
    derive.assert_not_called()


def test_legacy_ciphertext_without_a_configured_key_is_refused(tmp_path, data_key):
    adapter = _adapter(tmp_path)
    legacy = _cached(adapter, encrypt=True)

    with patch(
        "kestrel_sovereign.security.encryption.get_master_key_bytes",
        return_value=None,
    ):
        with pytest.raises(DecryptionKeyUnavailableError):
            adapter.retrieve_content(legacy.content_hash, envelope=EnvelopePolicy.ENCRYPTED_ONLY, envelope_owner=OWNER)


def test_legacy_ciphertext_without_a_key_hash_never_reads_a_key_sidecar(tmp_path, data_key):
    """A pre-F187 export's random key is wrapped in a ``key_<hash>.key``
    sidecar in the host cache that co-hosted agents share (#3225). The
    record-free attempt derives its key; it never opens a sidecar, even one
    that would decrypt the payload."""
    from kestrel_sdk.security.aead import AEADCipher

    adapter = _adapter(tmp_path)
    content_hash = hashlib.sha256(PLAINTEXT).hexdigest()
    content_key = AEADCipher.generate_key()
    (adapter.cache_dir / f"key_{content_hash}.key").write_bytes(
        AEADCipher(adapter._get_master_key()).encrypt(content_key)
    )
    payload = AEADCipher(content_key).encrypt(PLAINTEXT)
    _cached_under(adapter, content_hash, payload)
    assert adapter._decrypt_content(payload, content_hash) == PLAINTEXT

    with pytest.raises(LegacyDecryptionUnavailableError) as caught:
        adapter.retrieve_content(content_hash, envelope=EnvelopePolicy.ENCRYPTED_ONLY, envelope_owner=OWNER)

    assert caught.value.attempted is True


def test_open_policy_returns_plaintext_as_before(tmp_path, data_key):
    adapter = _adapter(tmp_path)
    plain = _cached(adapter, encrypt=False)

    assert adapter.retrieve_content(plain.content_hash, envelope=EnvelopePolicy.OPEN, envelope_owner=OWNER) == PLAINTEXT


def test_encrypted_only_policy_refuses_plaintext(tmp_path, data_key):
    adapter = _adapter(tmp_path)
    plain = _cached(adapter, encrypt=False)

    with pytest.raises(UnencryptedContentError):
        adapter.retrieve_content(plain.content_hash, envelope=EnvelopePolicy.ENCRYPTED_ONLY, envelope_owner=OWNER)


class _BackupAgent(BackupMixin):
    agent_id = "did:pkh:eip155:1:0xbackup"

    def __init__(self, blob):
        config = SimpleNamespace(
            is_ephemeral=lambda: False,
            uses_temp_storage=lambda: False,
            requires_anonymization=lambda: False,
        )
        self.privacy_agent = SimpleNamespace(privacy_config=config)
        self.storage = SimpleNamespace(
            create_backup_blob=AsyncMock(return_value=blob),
            record_backup_artifact=AsyncMock(return_value="node"),
        )

    def recorded(self):
        return self.storage.record_backup_artifact.await_args.args[1]


@pytest.mark.asyncio
async def test_backup_command_seals_its_encrypted_backup(tmp_path, monkeypatch, data_key):
    """``!backup`` writes the same ``backup_artifact`` that
    ``import_sovereignty`` restores, so it is sealed the same way."""
    monkeypatch.chdir(tmp_path)
    agent = _BackupAgent(PLAINTEXT)

    await agent._command_backup("!backup")

    result = agent.recorded()
    assert result.encrypted is True
    master_key = FilecoinAdapter()._get_master_key()
    owner_id = envelope_owner_id(master_key, agent.agent_id)
    name = envelope_cache_name(result.content_hash, owner_id)
    stored = zlib.decompress((tmp_path / "storage_cache" / f"{name}.cache").read_bytes())
    assert stored.startswith(ENVELOPE_MAGIC)
    assert _split(stored)[0]["owner_id"] == owner_id
    assert result.cache_name == name
    assert FilecoinAdapter().retrieve_content(
        result.content_hash,
        envelope=EnvelopePolicy.ENCRYPTED_ONLY,
        envelope_owner=agent.agent_id,
        cache_name=result.cache_name,
    ) == PLAINTEXT


@pytest.mark.asyncio
async def test_backup_command_without_encryption_stays_plaintext(tmp_path, monkeypatch, data_key):
    monkeypatch.chdir(tmp_path)
    agent = _BackupAgent(PLAINTEXT)

    await agent._command_backup("!backup --no-encrypt")

    result = agent.recorded()
    assert result.encrypted is False
    stored = zlib.decompress((tmp_path / "storage_cache" / f"{result.content_hash}.cache").read_bytes())
    assert stored == PLAINTEXT
