#!/usr/bin/env pytest
"""Encrypted-export restore instructions bind through the real SDK parser.

Both positional and declared named arguments must preserve the encryption key
and signature verification. Invalid boolean values must fail before any I/O.
"""

import pytest

from kestrel_sovereign.features.identity.feature import IdentityFeature


def _tools():
    feat = IdentityFeature(None)
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


def test_import_named_key_preserves_signature_verification():
    tool = _tools()["import_identity"]
    args = tool.parse_command_args("!identity import Qmabc123 key_hash=deadbeefhash")
    assert args["source"] == "Qmabc123"
    assert args["key_hash"] == "deadbeefhash"
    assert args["verify_signature"] is True
    assert args["merge_mode"] == "merge"


@pytest.mark.asyncio
async def test_import_rejects_non_boolean_verify_signature():
    """An untyped caller cannot silently disable signature verification."""
    feat = IdentityFeature(None)
    result = await feat.import_identity(
        source="Qmabc123", verify_signature="key_hash=deadbeefhash"
    )
    assert result.status.name == "ERROR"
    assert "verify_signature must be a boolean" in result.error
