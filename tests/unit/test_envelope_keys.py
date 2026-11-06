"""Real encryption checks for durable, rotatable repository roots."""

import secrets

import pytest

from kestrel_sdk.resources import ResourceError
from kestrel_sdk.security.aead import AEADCipher, KEY_SIZE
from kestrel_sovereign.security.envelope_keys import resolve_envelope_key


def resolve(cipher, wrapped, *, has_dependents=False):
    return resolve_envelope_key(
        cipher, wrapped, has_dependents=has_dependents,
        integrity_error=ResourceError, missing_message="missing original root",
        unavailable_message="original root unavailable", invalid_message="invalid root",
    )


def test_random_root_survives_reopening_and_master_key_rewrapping():
    old = AEADCipher(secrets.token_bytes(KEY_SIZE))
    new = AEADCipher(secrets.token_bytes(KEY_SIZE))
    root, wrapped = resolve(old, None)
    other_root, other_wrapped = resolve(old, None)
    assert root != other_root and wrapped != other_wrapped
    assert root not in wrapped and other_root not in other_wrapped
    assert resolve(old, wrapped, has_dependents=True) == (root, None)
    # Key rotation changes the wrapping key, never the indexed root.
    rewrapped = new.encrypt(old.decrypt(wrapped))
    assert resolve(new, rewrapped, has_dependents=True) == (root, None)
    with pytest.raises(ResourceError, match="original root unavailable"):
        resolve(old, rewrapped, has_dependents=True)


def test_missing_root_with_existing_records_cannot_create_a_new_namespace():
    cipher = AEADCipher(secrets.token_bytes(KEY_SIZE))
    with pytest.raises(ResourceError, match="missing original root"):
        resolve(cipher, None, has_dependents=True)


@pytest.mark.parametrize("size", [0, 1, KEY_SIZE - 1, KEY_SIZE + 1, 64])
def test_authenticated_wrong_size_roots_are_not_accepted(size):
    cipher = AEADCipher(secrets.token_bytes(KEY_SIZE))
    wrapped = cipher.encrypt(secrets.token_bytes(size))
    with pytest.raises(ResourceError, match="invalid root"):
        resolve(cipher, wrapped, has_dependents=True)


def test_corrupt_envelope_is_not_replaced_or_exposed():
    cipher = AEADCipher(secrets.token_bytes(KEY_SIZE))
    _, wrapped = resolve(cipher, None)
    corrupt = wrapped[:-1] + (b"A" if wrapped[-1:] != b"A" else b"B")
    with pytest.raises(ResourceError, match="original root unavailable") as error:
        resolve(cipher, corrupt, has_dependents=True)
    assert wrapped.decode() not in str(error.value)
