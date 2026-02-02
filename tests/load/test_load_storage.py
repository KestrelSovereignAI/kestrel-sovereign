"""
Storage load tests for performance benchmarking.

Skipped by default - run with: pytest --run-load tests/load/

These tests verify storage performance under load:
- Concurrent write throughput
- Mixed read/write workloads
- RAG indexing at scale
- Memory efficiency
"""
import pytest
import pytest_asyncio
import asyncio
import time
import os
from pathlib import Path

pytestmark = [pytest.mark.load, pytest.mark.slow]


@pytest.fixture
def skip_unless_load(request):
    """Skip unless --run-load flag provided."""
    if not request.config.getoption("--run-load", default=False):
        pytest.skip("Load tests skipped. Use --run-load to enable.")


@pytest_asyncio.fixture
async def load_test_storage(tmp_path):
    """Create storage instance for load testing."""
    from kestrel_sovereign.storage import AsyncStorage

    db_path = tmp_path / "load_test.db"
    storage = AsyncStorage(str(db_path))
    await storage.connect()

    yield storage

    await storage.close()


@pytest.mark.asyncio
async def test_concurrent_write_throughput(skip_unless_load, load_test_storage):
    """
    Test concurrent write throughput.

    Measures how many messages can be written per second with
    multiple concurrent writers.
    """
    storage = load_test_storage
    num_writers = 10
    messages_per_writer = 100
    total_messages = num_writers * messages_per_writer

    async def write_messages(writer_id: int):
        """Write messages for a single writer."""
        for i in range(messages_per_writer):
            await storage.conversation.add_conversation(
                role="user",
                content=f"Writer {writer_id} message {i}: Lorem ipsum dolor sit amet",
                metadata={"writer_id": writer_id, "message_num": i}
            )

    # Run concurrent writers
    tasks = [write_messages(i) for i in range(num_writers)]

    start = time.perf_counter()
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    throughput = total_messages / elapsed

    print(f"\n📊 Concurrent Write Results:")
    print(f"   Writers: {num_writers}")
    print(f"   Total messages: {total_messages}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {throughput:.0f} msg/s")

    # Baseline assertion - should be able to write at least 100 msg/s
    assert throughput > 100, f"Write throughput too low: {throughput:.0f} msg/s"


@pytest.mark.asyncio
async def test_mixed_read_write_workload(skip_unless_load, load_test_storage):
    """
    Test mixed read/write workload.

    Simulates realistic usage with concurrent reads and writes.
    """
    storage = load_test_storage
    num_operations = 500
    read_ratio = 0.7  # 70% reads, 30% writes

    # Pre-populate with some data
    for i in range(100):
        await storage.conversation.add_conversation(
            role="user",
            content=f"Seed message {i}",
            metadata={"seed": True}
        )

    read_count = 0
    write_count = 0

    async def mixed_operation(op_id: int):
        """Execute a read or write based on ratio."""
        nonlocal read_count, write_count
        import random

        if random.random() < read_ratio:
            # Read operation
            await storage.conversation.get_conversation_history(limit=10)
            read_count += 1
        else:
            # Write operation
            await storage.conversation.add_conversation(
                role="user",
                content=f"Mixed workload message {op_id}",
                metadata={"mixed": True}
            )
            write_count += 1

    # Run mixed operations concurrently in batches
    batch_size = 50
    tasks = []

    start = time.perf_counter()

    for i in range(0, num_operations, batch_size):
        batch_tasks = [mixed_operation(i + j) for j in range(min(batch_size, num_operations - i))]
        await asyncio.gather(*batch_tasks)

    elapsed = time.perf_counter() - start

    ops_per_second = num_operations / elapsed

    print(f"\n📊 Mixed Workload Results:")
    print(f"   Total operations: {num_operations}")
    print(f"   Reads: {read_count}, Writes: {write_count}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {ops_per_second:.0f} ops/s")

    # Should handle at least 50 mixed ops/s
    assert ops_per_second > 50, f"Mixed workload too slow: {ops_per_second:.0f} ops/s"


@pytest.mark.asyncio
async def test_rag_indexing_at_scale(skip_unless_load, load_test_storage):
    """
    Test RAG indexing performance with many documents.

    Measures time to index and query a large document corpus.
    """
    storage = load_test_storage
    num_documents = 100
    chunks_per_doc = 10

    # Generate test documents
    documents = []
    for doc_id in range(num_documents):
        for chunk_id in range(chunks_per_doc):
            documents.append({
                "content": f"Document {doc_id} chunk {chunk_id}: This is test content for RAG indexing performance testing.",
                "metadata": {"doc_id": doc_id, "chunk_id": chunk_id}
            })

    # Index documents
    start = time.perf_counter()

    for doc in documents:
        await storage.rag.add_document_chunk(
            content=doc["content"],
            metadata=doc["metadata"]
        )

    index_time = time.perf_counter() - start

    # Run search queries
    queries = [
        "test content",
        "document chunk",
        "RAG indexing",
        "performance testing",
        "This is test"
    ]

    query_times = []
    for query in queries:
        query_start = time.perf_counter()
        results = await storage.rag.search_chunks(query, limit=10)
        query_times.append(time.perf_counter() - query_start)

    avg_query_time = sum(query_times) / len(query_times)

    print(f"\n📊 RAG Performance Results:")
    print(f"   Documents: {num_documents}")
    print(f"   Total chunks: {len(documents)}")
    print(f"   Index time: {index_time:.2f}s ({len(documents)/index_time:.0f} chunks/s)")
    print(f"   Avg query time: {avg_query_time*1000:.1f}ms")

    # Assertions
    assert index_time < 60, f"Indexing too slow: {index_time:.2f}s for {len(documents)} chunks"
    assert avg_query_time < 1.0, f"Query too slow: {avg_query_time*1000:.1f}ms average"


@pytest.mark.asyncio
async def test_large_conversation_history(skip_unless_load, load_test_storage):
    """
    Test performance with large conversation history.

    Measures memory and query performance with 10k+ messages.
    """
    storage = load_test_storage
    num_messages = 10000

    # Write large history
    print(f"\n📝 Writing {num_messages} messages...")
    start = time.perf_counter()

    # Batch writes for efficiency
    batch_size = 500
    for batch_start in range(0, num_messages, batch_size):
        batch_end = min(batch_start + batch_size, num_messages)
        tasks = []
        for i in range(batch_start, batch_end):
            role = "user" if i % 2 == 0 else "assistant"
            tasks.append(storage.conversation.add_conversation(
                role=role,
                content=f"Message {i}: {'User query' if role == 'user' else 'Assistant response'} with some context about topic {i % 100}.",
                metadata={"message_num": i}
            ))
        await asyncio.gather(*tasks)

        if batch_start % 2000 == 0:
            print(f"   Progress: {batch_start}/{num_messages}")

    write_time = time.perf_counter() - start

    # Query recent history
    query_start = time.perf_counter()
    recent = await storage.conversation.get_conversation_history(limit=100)
    recent_query_time = time.perf_counter() - query_start

    # Query with search
    search_start = time.perf_counter()
    # Note: This depends on FTS being available
    try:
        results = await storage.conversation.search_history("topic 50", limit=20)
        search_time = time.perf_counter() - search_start
    except Exception:
        search_time = 0
        results = []

    print(f"\n📊 Large History Results:")
    print(f"   Total messages: {num_messages}")
    print(f"   Write time: {write_time:.2f}s ({num_messages/write_time:.0f} msg/s)")
    print(f"   Recent query (100): {recent_query_time*1000:.1f}ms")
    if search_time > 0:
        print(f"   Search query: {search_time*1000:.1f}ms ({len(results)} results)")

    # Assertions
    assert write_time < 120, f"Writing too slow: {write_time:.2f}s"
    assert recent_query_time < 0.5, f"Recent query too slow: {recent_query_time*1000:.1f}ms"
    assert len(recent) == 100, f"Expected 100 recent messages, got {len(recent)}"


@pytest.mark.asyncio
async def test_concurrent_agent_sessions(skip_unless_load, tmp_path):
    """
    Test multiple concurrent agent sessions.

    Simulates multi-tenant workload with separate storage per agent.
    """
    from kestrel_sovereign.storage import AsyncStorage

    num_agents = 5
    messages_per_agent = 50

    async def agent_session(agent_id: int):
        """Run a complete agent session."""
        db_path = tmp_path / f"agent_{agent_id}.db"
        storage = AsyncStorage(str(db_path))
        await storage.connect()

        try:
            # Simulate agent conversation
            for i in range(messages_per_agent):
                await storage.conversation.add_conversation(
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Agent {agent_id} message {i}",
                    metadata={"agent_id": agent_id}
                )

            # Query history
            history = await storage.conversation.get_conversation_history(limit=20)
            return len(history)
        finally:
            await storage.close()

    # Run concurrent agent sessions
    start = time.perf_counter()
    results = await asyncio.gather(*[agent_session(i) for i in range(num_agents)])
    elapsed = time.perf_counter() - start

    total_messages = num_agents * messages_per_agent

    print(f"\n📊 Concurrent Agents Results:")
    print(f"   Agents: {num_agents}")
    print(f"   Messages per agent: {messages_per_agent}")
    print(f"   Total messages: {total_messages}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {total_messages/elapsed:.0f} msg/s")

    # All agents should have correct history
    for i, count in enumerate(results):
        assert count == 20, f"Agent {i} has incorrect history: {count}"

    # Should handle concurrent agents efficiently
    assert elapsed < 30, f"Concurrent agents too slow: {elapsed:.2f}s"
