import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.fernet import Fernet

from kestrel_sovereign.security import encryption
from kestrel_sovereign.security import key_rotation as key_rotation_module
from kestrel_sovereign.security.key_rotation import KeyRotationService


class TestKeyRotationTaskLifecycle:
    @pytest.mark.asyncio
    async def test_shutdown_cancels_tracked_rotation_tasks(self):
        service = KeyRotationService(storage=MagicMock())
        started = asyncio.Event()

        async def never_finishes():
            started.set()
            await asyncio.Event().wait()

        task = service._track_rotation_task(never_finishes(), "rotation-1")
        await started.wait()

        await service.shutdown()

        assert task.done()
        assert task.cancelled()
        assert service._rotation_tasks == set()

    @pytest.mark.asyncio
    async def test_start_rotation_tracks_background_rotation_task(self, monkeypatch):
        service = KeyRotationService(storage=MagicMock())
        service._get_in_progress_rotation = AsyncMock(return_value=None)
        service._count_encrypted_records = AsyncMock(return_value=0)
        service._save_rotation = AsyncMock()

        old_key = Fernet.generate_key().decode()
        new_key = Fernet.generate_key().decode()
        started = asyncio.Event()

        monkeypatch.setattr(
            key_rotation_module,
            "get_fernet",
            lambda: Fernet(old_key),
        )
        monkeypatch.setattr(
            encryption,
            "_get_data_key",
            lambda: old_key,
        )

        async def never_finishes(rotation, old_fernet, new_fernet):
            started.set()
            await asyncio.Event().wait()

        service._execute_rotation = never_finishes

        rotation_id = await service.start_rotation(new_key=new_key)
        await started.wait()

        assert rotation_id
        assert len(service._rotation_tasks) == 1

        task = next(iter(service._rotation_tasks))
        assert task.get_name() == f"key-rotation-{rotation_id}"

        await service.shutdown()

        assert task.done()
        assert task.cancelled()
        assert service._rotation_tasks == set()
