import pytest
import asyncio
from pathlib import Path
import os
import shutil
import json

from kestrel_sovereign.inception_service import create_kestrel_identity_async

@pytest.mark.anyio
@pytest.mark.asyncio
async def test_successful_inception(tmp_path):
    """
    Tests that the create_kestrel_identity function completes successfully
    and creates the expected artifacts.
    """
    output_dir = tmp_path / "test_agent"
    
    # The constitution must exist relative to the execution path
    constitution_path = Path("docs/principles/KESTREL_CONSTITUTION.md")
    assert constitution_path.exists(), "Test requires the Kestrel Constitution to exist at the project root."

    credentials = await create_kestrel_identity_async(str(output_dir))

    # Default inception (#2399) mints a born-hybrid did:web identity.
    assert credentials.agent_did.startswith("did:web:")
    assert Path(credentials.db_path).exists()
    assert "CRITICAL: Agent" in credentials.backup_prompt

    # Check that the files were created: DID document, DB, and the
    # encrypted hybrid key material (born-hybrid never writes plaintext).
    files = list(output_dir.iterdir())
    assert any(f.name.endswith('_did.json') for f in files)
    assert any(f.name.endswith('.db') for f in files)
    assert any(f.name.endswith('_ed25519.key.enc') for f in files)
    assert any(f.name.endswith('_mldsa65.bytes.enc') for f in files)
    assert not any(f.name.endswith('.pem') for f in files) 