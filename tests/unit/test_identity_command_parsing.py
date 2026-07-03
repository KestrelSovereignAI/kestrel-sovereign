#!/usr/bin/env pytest
"""F187 command-form regression: the encrypted-export restore instruction must
be parseable by the real (positional) command parser.

The export tool tells the user to run ``!identity import <cid> true merge
<hash>``. ``Tool.parse_command_args`` binds arguments strictly positionally, so
this is the ONLY form that routes ``key_hash`` to the ``key_hash`` parameter.
The previously-documented ``key_hash=<hash>`` spelling bound the literal string
to ``verify_signature`` (a boolean) and was rejected — this test pins the fix.
"""
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.features.identity.feature import IdentityFeature


def _tools():
    agent = MagicMock()
    agent.agent_id = "did:test:parse"
    agent.did = agent.agent_id
    feat = IdentityFeature(agent)
    feat.disabled_skills = frozenset()
    return {t.schema.name: t for t in feat.get_tools()}


def test_import_positional_form_binds_key_hash():
    tool = _tools()["import_identity"]
    args = tool.parse_command_args("!identity import Qmabc123 true merge deadbeefhash")
    assert args["source"] == "Qmabc123"
    assert args["verify_signature"] is True
    assert args["merge_mode"] == "merge"
    assert args["key_hash"] == "deadbeefhash"


def test_verify_positional_form_binds_key_hash():
    tool = _tools()["verify_identity"]
    args = tool.parse_command_args("!identity verify Qmabc123 deadbeefhash")
    assert args["source"] == "Qmabc123"
    assert args["key_hash"] == "deadbeefhash"


def test_old_kwarg_form_misbinds_and_would_be_rejected():
    """The deprecated ``key_hash=`` form does NOT bind key_hash; it lands on
    verify_signature as a non-boolean string, which import_identity rejects."""
    tool = _tools()["import_identity"]
    args = tool.parse_command_args("!identity import Qmabc123 key_hash=deadbeefhash")
    # key_hash never gets the hash...
    assert args.get("key_hash") != "deadbeefhash"
    # ...instead the literal string lands on verify_signature (a boolean param).
    assert args["verify_signature"] == "key_hash=deadbeefhash"


@pytest.mark.asyncio
async def test_import_rejects_non_boolean_verify_signature():
    """import_identity guards verify_signature — so the old misbound form fails
    cleanly rather than silently importing."""
    agent = MagicMock()
    agent.agent_id = "did:test:parse"
    agent.did = agent.agent_id
    feat = IdentityFeature(agent)
    feat.disabled_skills = frozenset()
    result = await feat.import_identity(
        source="Qmabc123", verify_signature="key_hash=deadbeefhash"
    )
    assert result.status.name == "ERROR"
    assert "verify_signature must be a boolean" in result.error
