"""Unit tests for the continuation cursor store (#808 / #806)."""

import threading

import pytest

from kestrel_sovereign.llm.continuation_store import (
    ContinuationCursor,
    InMemoryContinuationStore,
)


class TestContinuationCursor:
    def test_immutable(self):
        cursor = ContinuationCursor(
            last_response_id="resp_1",
            last_message_count=3,
            last_request_signature="abc123",
        )
        with pytest.raises((AttributeError, Exception)):
            cursor.last_response_id = "resp_2"  # frozen dataclass

    def test_signature_optional(self):
        cursor = ContinuationCursor(last_response_id="r", last_message_count=0)
        assert cursor.last_request_signature is None


class TestInMemoryContinuationStore:
    def test_get_missing_returns_none(self):
        store = InMemoryContinuationStore()
        assert store.get("openai_plan", "conv-1") is None

    def test_put_then_get(self):
        store = InMemoryContinuationStore()
        cursor = ContinuationCursor("resp_1", 3, "sig1")
        store.put("openai_plan", "conv-1", cursor)
        assert store.get("openai_plan", "conv-1") == cursor

    def test_namespacing_by_adapter_name(self):
        # Same session_id across adapters does not collide.
        store = InMemoryContinuationStore()
        a = ContinuationCursor("resp_a", 1, "sig_a")
        b = ContinuationCursor("resp_b", 5, "sig_b")
        store.put("openai_plan", "conv-1", a)
        store.put("anthropic_plan", "conv-1", b)
        assert store.get("openai_plan", "conv-1") == a
        assert store.get("anthropic_plan", "conv-1") == b

    def test_namespacing_by_session_id(self):
        store = InMemoryContinuationStore()
        a = ContinuationCursor("resp_a", 1)
        b = ContinuationCursor("resp_b", 5)
        store.put("openai_plan", "conv-1", a)
        store.put("openai_plan", "conv-2", b)
        assert store.get("openai_plan", "conv-1") == a
        assert store.get("openai_plan", "conv-2") == b

    def test_put_overwrites(self):
        store = InMemoryContinuationStore()
        store.put("openai_plan", "conv-1", ContinuationCursor("r1", 1))
        store.put("openai_plan", "conv-1", ContinuationCursor("r2", 5))
        assert store.get("openai_plan", "conv-1").last_response_id == "r2"

    def test_clear(self):
        store = InMemoryContinuationStore()
        store.put("openai_plan", "conv-1", ContinuationCursor("r", 1))
        store.clear("openai_plan", "conv-1")
        assert store.get("openai_plan", "conv-1") is None

    def test_clear_missing_is_noop(self):
        # Idempotent — agent layer can call clear on every conversation delete
        # without checking existence first.
        store = InMemoryContinuationStore()
        store.clear("openai_plan", "never-existed")  # does not raise

    def test_threadsafe_concurrent_put(self):
        store = InMemoryContinuationStore()
        N = 200

        def writer(start: int) -> None:
            for i in range(start, start + N):
                store.put("openai_plan", f"conv-{i}", ContinuationCursor(f"r{i}", i))

        threads = [
            threading.Thread(target=writer, args=(t * N,)) for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(store) == 4 * N
        # Spot-check a few cursors landed correctly (no torn writes).
        for i in (0, N, 2 * N, 3 * N - 1):
            assert store.get("openai_plan", f"conv-{i}").last_response_id == f"r{i}"
