"""F126: run_script approval gate must fail CLOSED when no queue is reachable.

When a script requires approval (risk >= threshold, not a demo server) but
``SecurityFeature`` — or its ``approval_queue`` — is absent, ``run_script``
must reject the script instead of falling through to execution.
"""

import asyncio
import os
import tempfile

import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.compute.feature import ComputeFeature
from kestrel_sovereign.features.compute.models import (
    ComputePolicy,
    ComputeScript,
    ScriptState,
)
from kestrel_sovereign.features.compute.script_analyzer import ScriptAnalyzer
from kestrel_sovereign.features.compute.script_signer import ScriptSigner
from kestrel_sovereign.features.compute.script_store import ScriptStore


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def signer_with_ecdsa_keys(temp_db):
    from cryptography.hazmat.primitives.asymmetric import ec

    signer = ScriptSigner("did:ethr:0xtest", temp_db)
    signer._private_key = ec.generate_private_key(ec.SECP256K1())
    signer._public_key = signer._private_key.public_key()

    async def _mock_load_keys():
        return True

    signer._load_keys = _mock_load_keys
    return signer


async def _make_feature(temp_db, signer):
    """Build a ComputeFeature wired to a store/signer/analyzer/executor but
    an agent with NO SecurityFeature (features={}), bypassing full init."""
    store = ScriptStore(temp_db)
    await store.initialize()

    agent = MagicMock()
    agent.features = {}  # no SecurityFeature → no reachable approval queue

    feature = ComputeFeature(agent)
    feature.script_store = store
    feature.signer = signer
    feature.analyzer = ScriptAnalyzer()
    # Default policy: auto_approve_below_risk = 0 → every script requires
    # approval, so even a benign script exercises the approval gate.
    feature.policy = ComputePolicy()

    executor = MagicMock()
    executor.supports_language.return_value = True
    executor.execute = AsyncMock(
        side_effect=AssertionError("script must NOT execute when failing closed")
    )
    feature.executors = {"uv": executor}

    feature._initialized = True
    feature._init_lock = asyncio.Lock()
    return feature, store, executor


async def _make_signed_script(store, signer):
    script = ComputeScript(
        id="failclosed-001",
        name="benign",
        language="python",
        content='print("hello")',
        purpose="approval fail-closed test",
        state=ScriptState.SIGNED,
    )
    await store.save(script)
    await signer.sign_and_update(script)
    script.state = ScriptState.SIGNED
    await store.update(script)
    return script


@pytest.mark.asyncio
async def test_missing_security_feature_fails_closed(temp_db, signer_with_ecdsa_keys):
    """No SecurityFeature → approval required → REJECTED, never executed."""
    feature, store, executor = await _make_feature(temp_db, signer_with_ecdsa_keys)
    script = await _make_signed_script(store, signer_with_ecdsa_keys)

    result = await feature.run_script(script.id, executor="uv")

    # ToolResult reports failure.
    assert result.status is ToolResultStatus.ERROR
    assert "approval" in result.error.lower()

    # The executor was never invoked.
    executor.execute.assert_not_called()

    # Persisted state is REJECTED — never reached QUEUED/execution.
    refreshed = await store.get(script.id)
    assert refreshed.state == ScriptState.REJECTED


@pytest.mark.asyncio
async def test_missing_approval_queue_fails_closed(temp_db, signer_with_ecdsa_keys):
    """SecurityFeature present but approval_queue is None → still fail closed."""
    feature, store, executor = await _make_feature(temp_db, signer_with_ecdsa_keys)

    security_feature = MagicMock()
    security_feature.approval_queue = None
    feature.agent.features = {"SecurityFeature": security_feature}

    script = await _make_signed_script(store, signer_with_ecdsa_keys)

    result = await feature.run_script(script.id, executor="uv")

    assert result.status is ToolResultStatus.ERROR
    executor.execute.assert_not_called()

    refreshed = await store.get(script.id)
    assert refreshed.state == ScriptState.REJECTED
