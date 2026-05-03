"""Layer 2: Memory - Knowledge/Context checks."""
from typing import List
from hashlib import sha256

from .base import HealthChecker
from ..models import HealthCheck, ReflectionLayer, CheckStatus, Severity


class MemoryChecker(HealthChecker):
    """Layer 2: Can I access and understand what I know?"""

    async def run_all(self) -> List[HealthCheck]:
        return [
            await self.check_constitution_access(),
            await self.check_conversation_decryption(),
            await self.check_knowledge_graph(),
            await self.check_rag_quality(),
        ]

    async def check_constitution_access(self) -> HealthCheck:
        """Can we read our constitution and verify integrity?"""
        check = HealthCheck(
            id="memory.constitution",
            layer=ReflectionLayer.MEMORY,
            name="Constitution Access",
            description="Verify constitution is accessible and intact",
            status=CheckStatus.SKIP,
        )

        try:
            storage = getattr(self.agent, 'storage', None)
            if not storage or not hasattr(storage, 'graph'):
                check.status = CheckStatus.SKIP
                check.message = "Graph store not available"
                return check

            # Find constitution - may be stored as type "constitution" or "document" with label
            nodes = await storage.graph.get_nodes_by_type("constitution")
            if not nodes:
                # Fallback: search by label (inception stores as document type)
                all_docs = await storage.graph.get_nodes_by_type("document")
                nodes = [n for n in all_docs if getattr(n, 'label', '') == 'KESTREL_CONSTITUTION']
            if not nodes:
                check.status = CheckStatus.FAIL
                check.severity = Severity.CRITICAL
                check.message = "Constitution not found"
                check.suggested_fix = "Run inception_service.py to reinitialize"
                return check

            constitution = nodes[0] if isinstance(nodes, list) else nodes
            # GraphNode uses properties dict, not direct attributes
            props = getattr(constitution, 'properties', {}) if hasattr(constitution, 'properties') else constitution
            stored_hash = props.get("hash") or props.get("content_hash")

            if stored_hash:
                content = props.get("content", "")
                if content:
                    computed = sha256(content.encode()).hexdigest()
                    if computed != stored_hash:
                        check.status = CheckStatus.FAIL
                        check.severity = Severity.CRITICAL
                        check.message = "Constitution hash mismatch - tampering detected"
                        return check

            check.status = CheckStatus.PASS
            check.message = "Constitution accessible and intact"

        except Exception as e:
            check.status = CheckStatus.FAIL
            check.severity = Severity.CRITICAL
            check.message = f"Constitution access error: {e}"

        return check

    async def check_conversation_decryption(self) -> HealthCheck:
        """Can we actually read our conversation history?"""
        check = HealthCheck(
            id="memory.conversations",
            layer=ReflectionLayer.MEMORY,
            name="Conversation Decryption",
            description="Verify we can read our own memories",
            status=CheckStatus.SKIP,
        )

        try:
            storage = getattr(self.agent, 'storage', None)
            conv_store = getattr(storage, 'conversation', None) if storage else None

            if not conv_store:
                check.status = CheckStatus.SKIP
                check.message = "Conversation store not available"
                return check

            history = await conv_store.get_conversation_history(limit=5)

            if not history:
                check.status = CheckStatus.PASS
                check.message = "No conversation history yet"
                return check

            # Check if content looks encrypted (legacy Fernet 'gAAAAA' or v2 'KSAv2:').
            # Min-length guard avoids tripping on a literal short string a user
            # might paste — both real prefixes are followed by a long base64 body
            # (Fernet token >= 73 chars, v2 token >= ~46 chars). 50 is a safe lower
            # bound that still rejects short pasted strings.
            def _looks_encrypted(c: str) -> bool:
                return len(c) > 50 and (c.startswith("gAAAAA") or c.startswith("KSAv2:"))

            encrypted_count = sum(
                1 for msg in history
                if _looks_encrypted(msg.get("content", ""))
            )

            if encrypted_count > 0:
                check.status = CheckStatus.FAIL
                check.severity = Severity.CRITICAL
                check.message = f"{encrypted_count}/{len(history)} messages still encrypted - wrong KESTREL_DATA_KEY?"
                check.suggested_fix = "Set correct KESTREL_DATA_KEY to match when data was encrypted"
                check.file_path = ".env"
            else:
                check.status = CheckStatus.PASS
                check.message = f"Can read {len(history)} messages"

        except Exception as e:
            check.status = CheckStatus.FAIL
            check.severity = Severity.CRITICAL
            check.message = f"Conversation access error: {e}"

        return check

    async def check_knowledge_graph(self) -> HealthCheck:
        """Is our knowledge graph connected or fragmented?"""
        check = HealthCheck(
            id="memory.graph",
            layer=ReflectionLayer.MEMORY,
            name="Knowledge Graph",
            description="Check graph connectivity",
            status=CheckStatus.SKIP,
        )

        try:
            storage = getattr(self.agent, 'storage', None)
            if not storage or not hasattr(storage, 'graph'):
                check.status = CheckStatus.SKIP
                check.message = "Graph store not available"
                return check

            # Just verify we can query the graph - get constitution type nodes as simple test
            # Full graph analysis would be expensive
            test_nodes = await storage.graph.get_nodes_by_type("constitution")

            check.status = CheckStatus.PASS
            check.message = "Knowledge graph accessible"
            check.details = {"test_query": "success"}

        except Exception as e:
            check.status = CheckStatus.WARN
            check.severity = Severity.MEDIUM
            check.message = f"Graph check error: {e}"

        return check

    async def check_rag_quality(self) -> HealthCheck:
        """Do embeddings return relevant results?"""
        check = HealthCheck(
            id="memory.rag",
            layer=ReflectionLayer.MEMORY,
            name="RAG Quality",
            description="Test semantic search",
            status=CheckStatus.SKIP,
        )

        try:
            storage = getattr(self.agent, 'storage', None)
            rag = getattr(storage, 'rag', None) if storage else None

            if not rag:
                check.status = CheckStatus.SKIP
                check.message = "RAG not available"
                return check

            # Search for constitution (should always exist)
            results = await rag.search_chunks("Kestrel constitution governance", limit=3)

            if not results:
                check.status = CheckStatus.WARN
                check.severity = Severity.MEDIUM
                check.message = "RAG returned no results"
            else:
                check.status = CheckStatus.PASS
                check.message = f"RAG returning results ({len(results)} hits)"

        except Exception as e:
            check.status = CheckStatus.WARN
            check.severity = Severity.LOW
            check.message = f"RAG check error: {e}"

        return check
