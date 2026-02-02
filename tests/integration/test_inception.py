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

    assert credentials.agent_did.startswith("did:pkh:eip155:1:0x")
    assert Path(credentials.db_path).exists()
    assert "CRITICAL: Agent" in credentials.backup_prompt

    # Check that the files were created
    files = list(output_dir.iterdir())
    assert any(f.name.endswith('.json') for f in files)
    assert any(f.name.endswith('.db') for f in files)
    # Key file: .pem (plaintext fallback) or .key.enc (encrypted when KESTREL_DATA_KEY is set)
    assert any(f.name.endswith('.pem') or f.name.endswith('.key.enc') for f in files), \
        f"Expected key file (.pem or .key.enc) in {[f.name for f in files]}" 