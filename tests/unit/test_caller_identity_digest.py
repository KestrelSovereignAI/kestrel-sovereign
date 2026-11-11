"""Durable reader attribution preserves the existing OAuth identity grammar."""

from dataclasses import replace
import hashlib
import json

import pytest

from kestrel_sovereign.auth import AuthMethod, CallerContext, CallerRole, caller_identity_digest


def test_digest_retains_the_original_oauth_namespace_and_has_no_raw_subject():
    caller = CallerContext.authenticated('owned-private-subject')
    original = hashlib.sha256(json.dumps([caller.role.value, caller.auth_method.value,
        caller.identity], separators=(',', ':')).encode()).hexdigest()
    assert caller_identity_digest(caller) == original
    assert 'owned-private-subject' not in caller_identity_digest(caller)
    assert len({caller_identity_digest(caller),
        caller_identity_digest(replace(caller, role=CallerRole.SOVEREIGN)),
        caller_identity_digest(replace(caller, auth_method=AuthMethod.JWT)),
        caller_identity_digest(replace(caller, identity='other'))}) == 4


@pytest.mark.parametrize('value', [None, {}, 'owned-private-subject'])
def test_arbitrary_identity_claims_do_not_become_native_callers(value):
    with pytest.raises(TypeError, match='native CallerContext'):
        caller_identity_digest(value)
