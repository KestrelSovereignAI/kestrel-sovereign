"""
Concurrent Access Tests for Kestrel Agent

Tests multi-client scenarios to verify thread-safety and proper
handling of simultaneous requests using async storage.

L5 from IMPLEMENTATION_PLAN.md
"""
import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import concurrent.futures

from kestrel_sovereign import storage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestAsyncStorageConcurrency:
    """Test concurrent access using async storage."""
    
    @pytest.mark.asyncio
    async def test_concurrent_file_writes_no_corruption(self, tmp_path):
        """Multiple concurrent file writes should not corrupt data."""
        from kestrel_sovereign.storage import AsyncStorage
        
        db_path = tmp_path / "kestrel.db"
        
        async with AsyncStorage(str(db_path)) as storage:
            num_concurrent = 20
            writes_per_task = 10
            results = {}
            
            async def write_files(task_id: int):
                task_results = []
                for i in range(writes_per_task):
                    content = f"Task {task_id} wrote file {i} with unique content".encode()
                    content_hash = await storage.store_file(content, f"task_{task_id}_file_{i}.txt")
                    task_results.append((content_hash, content))
                return task_results
            
            # Run concurrent writes
            all_results = await asyncio.gather(*[
                write_files(i) for i in range(num_concurrent)
            ])
            
            # Flatten results
            for task_results in all_results:
                for content_hash, content in task_results:
                    results[content_hash] = content
            
            # Verify all files exist and have correct content
            assert len(results) == num_concurrent * writes_per_task
            
            for content_hash, expected in results.items():
                retrieved = await storage.retrieve_file(content_hash)
                assert retrieved is not None, f"File {content_hash} missing"
                assert retrieved == expected, f"File {content_hash} corrupted"
    
    @pytest.mark.asyncio
    async def test_concurrent_same_content_deduplication(self, tmp_path):
        """Same content written concurrently should produce same hash."""
        from kestrel_sovereign.storage import AsyncStorage
        
        db_path = tmp_path / "kestrel.db"
        same_content = b"This is identical content for all concurrent writes"
        
        async with AsyncStorage(str(db_path)) as storage:
            num_concurrent = 10
            
            async def store_same(task_id: int):
                return await storage.store_file(same_content, f"file_{task_id}.txt")
            
            hashes = await asyncio.gather(*[
                store_same(i) for i in range(num_concurrent)
            ])
            
            # All hashes should be identical (content-addressed)
            assert len(set(hashes)) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_conversation_updates(self, tmp_path):
        """Concurrent conversation updates should not lose messages."""
        from kestrel_sovereign.storage import AsyncStorage
        
        db_path = tmp_path / "kestrel.db"
        
        async with AsyncStorage(str(db_path)) as storage:
            num_concurrent = 10
            messages_per_task = 20
            
            async def add_messages(task_id: int):
                for i in range(messages_per_task):
                    await storage.add_conversation(
                        role='user',
                        content=f"Task {task_id} message {i}",
                        metadata={'task': task_id, 'index': i}
                    )
            
            # Run concurrent message additions
            await asyncio.gather(*[
                add_messages(i) for i in range(num_concurrent)
            ])
            
            # Verify all messages exist
            history = await storage.get_conversation_history(limit=1000)
            assert len(history) == num_concurrent * messages_per_task
            
            # Verify all messages are present
            contents = {msg['content'] for msg in history}
            for task_id in range(num_concurrent):
                for i in range(messages_per_task):
                    expected = f"Task {task_id} message {i}"
                    assert expected in contents, f"Missing: {expected}"
    
    @pytest.mark.asyncio
    async def test_concurrent_graph_operations(self, tmp_path):
        """Concurrent graph node additions should not corrupt the graph."""
        from kestrel_sovereign.storage import AsyncStorage
        from kestrel_sovereign.storage.async_graph_store import GraphNode
        
        db_path = tmp_path / "kestrel.db"
        
        async with AsyncStorage(str(db_path)) as storage:
            num_concurrent = 10
            nodes_per_task = 20
            
            async def add_nodes(task_id: int):
                for i in range(nodes_per_task):
                    node = GraphNode(
                        node_id=f"node_t{task_id}_i{i}",
                        node_type="test_node",
                        label=f"Node {task_id}-{i}",
                        properties={"task": task_id, "index": i}
                    )
                    await storage.add_node(node)
            
            await asyncio.gather(*[
                add_nodes(i) for i in range(num_concurrent)
            ])
            
            # Verify all nodes exist
            for task_id in range(num_concurrent):
                for i in range(nodes_per_task):
                    node_id = f"node_t{task_id}_i{i}"
                    node = await storage.get_node(node_id)
                    assert node is not None, f"Node {node_id} not found"
                    assert node.properties['task'] == task_id
                    assert node.properties['index'] == i
    
    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, tmp_path):
        """Mixed concurrent operations (files, conversations, graph)."""
        from kestrel_sovereign.storage import AsyncStorage
        from kestrel_sovereign.storage.async_graph_store import GraphNode
        
        db_path = tmp_path / "kestrel.db"
        
        async with AsyncStorage(str(db_path)) as storage:
            num_tasks = 5
            ops_per_task = 10
            
            async def mixed_operations(task_id: int):
                for i in range(ops_per_task):
                    # File operation
                    content = f"Mixed task {task_id} file {i}".encode()
                    await storage.store_file(content, f"mixed_{task_id}_{i}.txt")
                    
                    # Conversation operation
                    await storage.add_conversation(
                        role='user',
                        content=f"Mixed task {task_id} msg {i}"
                    )
                    
                    # Graph operation
                    node = GraphNode(
                        node_id=f"mixed_t{task_id}_i{i}",
                        node_type="mixed",
                        label=f"Mixed {task_id}-{i}",
                        properties={"type": "mixed"}
                    )
                    await storage.add_node(node)
            
            await asyncio.gather(*[
                mixed_operations(i) for i in range(num_tasks)
            ])
            
            # Verify counts
            history = await storage.get_conversation_history(limit=1000)
            assert len(history) == num_tasks * ops_per_task
            
            nodes = await storage.get_nodes_by_type("mixed")
            assert len(nodes) == num_tasks * ops_per_task


class TestConcurrentAPIRequests:
    """Test concurrent API request handling."""
    
    @pytest.fixture
    def test_client(self, tmp_path, monkeypatch):
        """Create test client with isolated storage."""
        import threading
        from fastapi.testclient import TestClient
        from server import app
        from kestrel_sovereign.inception_service import create_kestrel_identity

        monkeypatch.setattr(storage, "get_default_agent_data_dir", lambda: str(tmp_path))
        monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path))

        constitution_path = Path(__file__).parent.parent.parent / "docs" / "principles" / "KESTREL_CONSTITUTION.md"
        if constitution_path.exists():
            create_kestrel_identity(str(tmp_path), str(constitution_path))

        threads_before = set(threading.enumerate())

        with TestClient(app) as client:
            yield client

        # Wait for aiosqlite threads to finish
        threads_after = set(threading.enumerate())
        for t in threads_after - threads_before:
            if t.is_alive() and not t.daemon:
                t.join(timeout=2.0)
    
    def test_concurrent_health_checks(self, test_client):
        """Multiple concurrent health checks should succeed."""
        num_concurrent = 20
        results = []
        
        def health_check():
            return test_client.get("/health").status_code
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(health_check) for _ in range(num_concurrent)]
            results = [f.result() for f in futures]
        
        assert all(status == 200 for status in results)
        assert len(results) == num_concurrent


class TestAsyncStorageTransactions:
    """Test transaction handling in concurrent scenarios."""
    
    @pytest.mark.asyncio
    async def test_transaction_isolation(self, tmp_path):
        """Transactions should be properly isolated."""
        from kestrel_sovereign.storage import AsyncStorage
        
        db_path = tmp_path / "kestrel.db"
        
        async with AsyncStorage(str(db_path)) as storage:
            # Start multiple operations that might conflict
            async def transactional_write(task_id: int):
                content = f"Transaction {task_id}".encode()
                hash1 = await storage.store_file(content, f"tx_{task_id}_1.txt")
                await asyncio.sleep(0.001)  # Simulate work
                hash2 = await storage.store_file(content, f"tx_{task_id}_2.txt")
                return hash1, hash2
            
            results = await asyncio.gather(*[
                transactional_write(i) for i in range(10)
            ])
            
            # All transactions should complete
            assert len(results) == 10
            for hash1, hash2 in results:
                # Same content = same hash
                assert hash1 == hash2


class TestAsyncStorageReconnection:
    """Test storage behavior with connection issues."""
    
    @pytest.mark.asyncio
    async def test_multiple_storage_instances(self, tmp_path):
        """Multiple storage instances to same DB should work."""
        from kestrel_sovereign.storage import AsyncStorage
        
        db_path = tmp_path / "kestrel.db"
        
        # Create first instance and write
        async with AsyncStorage(str(db_path)) as storage1:
            hash1 = await storage1.store_file(b"Content 1", "file1.txt")
        
        # Create second instance and verify + write
        async with AsyncStorage(str(db_path)) as storage2:
            content = await storage2.retrieve_file(hash1)
            assert content == b"Content 1"
            
            hash2 = await storage2.store_file(b"Content 2", "file2.txt")
        
        # Create third instance and verify both
        async with AsyncStorage(str(db_path)) as storage3:
            assert await storage3.retrieve_file(hash1) == b"Content 1"
            assert await storage3.retrieve_file(hash2) == b"Content 2"


class TestHighConcurrencyStress:
    """Stress tests for high concurrency scenarios."""
    
    @pytest.mark.asyncio
    async def test_100_concurrent_writes(self, tmp_path):
        """100 concurrent file writes should all succeed."""
        from kestrel_sovereign.storage import AsyncStorage
        
        db_path = tmp_path / "kestrel.db"
        
        async with AsyncStorage(str(db_path)) as storage:
            async def write(i: int):
                content = f"High concurrency test file {i}".encode()
                return await storage.store_file(content, f"stress_{i}.txt")
            
            hashes = await asyncio.gather(*[write(i) for i in range(100)])
            
            # All should succeed with unique hashes
            assert len(hashes) == 100
            assert len(set(hashes)) == 100  # All unique content
    
    @pytest.mark.asyncio
    async def test_rapid_read_write_cycles(self, tmp_path):
        """Rapid alternating reads and writes should not corrupt data."""
        from kestrel_sovereign.storage import AsyncStorage
        
        db_path = tmp_path / "kestrel.db"
        
        async with AsyncStorage(str(db_path)) as storage:
            stored_hashes = []
            
            # Write 50 files
            for i in range(50):
                content = f"Rapid cycle file {i}".encode()
                h = await storage.store_file(content, f"rapid_{i}.txt")
                stored_hashes.append((h, content))
            
            # Concurrent reads and writes
            async def read_and_write(i: int):
                # Read existing
                h, expected = stored_hashes[i % len(stored_hashes)]
                content = await storage.retrieve_file(h)
                assert content == expected
                
                # Write new
                new_content = f"New rapid content {i}".encode()
                return await storage.store_file(new_content, f"new_rapid_{i}.txt")
            
            new_hashes = await asyncio.gather(*[read_and_write(i) for i in range(100)])
            
            assert len(new_hashes) == 100
