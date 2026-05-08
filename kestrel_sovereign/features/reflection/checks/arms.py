"""Layer 1: Arms - Physical/Functional checks."""
import asyncio
import uuid
from typing import List

from .base import HealthChecker
from ..models import HealthCheck, ReflectionLayer, CheckStatus, Severity


class ArmsChecker(HealthChecker):
    """Layer 1: Do my components work mechanically?"""

    async def run_all(self) -> List[HealthCheck]:
        return [
            await self.check_llm_connectivity(),
            await self.check_storage_read_write(),
            await self.check_encryption_keys(),
            await self.check_feature_initialization(),
            await self.check_tool_execution(),
            # Functional checks - actually invoke tools
            await self.check_github_operations(),
            await self.check_model_operations(),
            await self.check_web_search(),
            await self.check_security_permissions(),
        ]

    async def check_llm_connectivity(self) -> HealthCheck:
        """Can we reach the LLM provider?"""
        check = HealthCheck(
            id="arms.llm",
            layer=ReflectionLayer.ARMS,
            name="LLM Connectivity",
            description="Verify LLM provider is reachable",
            status=CheckStatus.SKIP,
        )
        start = self._start_timer()

        try:
            llm = getattr(self.agent, 'llm_service', None)
            if not llm:
                check.status = CheckStatus.FAIL
                check.severity = Severity.CRITICAL
                check.message = "No LLM service configured"
                return check

            response = await llm.generate(
                system_prompt="Reply with exactly: OK",
                user_prompt="Health check",
            )
            response_text = response if isinstance(response, str) else getattr(response, 'content', str(response))

            check.status = CheckStatus.PASS
            check.message = f"LLM responding"
            check.details = {"provider": getattr(llm, 'active_provider', 'unknown')}

        except Exception as e:
            check.status = CheckStatus.FAIL
            check.severity = Severity.CRITICAL
            check.message = f"LLM unreachable: {e}"
            check.suggested_fix = "Check OPENAI_API_KEY or Ollama service"
            check.file_path = "kestrel.toml"

        check.duration_ms = self._elapsed_ms(start)
        return check

    async def check_storage_read_write(self) -> HealthCheck:
        """Can we read and write to storage?"""
        check = HealthCheck(
            id="arms.storage",
            layer=ReflectionLayer.ARMS,
            name="Storage Read/Write",
            description="Verify storage is accessible",
            status=CheckStatus.SKIP,
        )
        start = self._start_timer()

        try:
            storage = getattr(self.agent, 'storage', None)
            if not storage:
                check.status = CheckStatus.FAIL
                check.severity = Severity.CRITICAL
                check.message = "No storage configured"
                return check

            graph = getattr(storage, 'graph', None)
            if not graph:
                check.status = CheckStatus.WARN
                check.severity = Severity.MEDIUM
                check.message = "Graph store not available"
                return check

            # Write test - use GraphNode object
            from kestrel_sovereign.storage.async_graph_store import GraphNode
            test_id = f"_health_{uuid.uuid4().hex[:8]}"
            test_node = GraphNode(
                node_id=test_id,
                node_type="health_check",
                label="Health Check Node",
                properties={"test": True},
            )
            await graph.add_node(test_node)

            # Read test
            node = await graph.get_node(test_id)

            # Cleanup
            if hasattr(graph, 'delete_node'):
                await graph.delete_node(test_id)

            check.status = CheckStatus.PASS
            check.message = "Storage read/write working"

        except Exception as e:
            check.status = CheckStatus.FAIL
            check.severity = Severity.CRITICAL
            check.message = f"Storage error: {e}"
            check.suggested_fix = "Check database path and permissions"

        check.duration_ms = self._elapsed_ms(start)
        return check

    async def check_encryption_keys(self) -> HealthCheck:
        """Can we decrypt our own data?"""
        check = HealthCheck(
            id="arms.encryption",
            layer=ReflectionLayer.ARMS,
            name="Encryption Keys",
            description="Verify encryption/decryption works",
            status=CheckStatus.SKIP,
        )

        try:
            from kestrel_sovereign.security.encryption import get_fernet, encrypt_string_fernet, decrypt_string_fernet

            fernet = get_fernet()
            if fernet is None:
                check.status = CheckStatus.WARN
                check.severity = Severity.MEDIUM
                check.message = "Encryption disabled (KESTREL_DATA_KEY not set)"
                return check

            # Test round-trip
            test_data = "health_check_secret_123"
            encrypted, _ = encrypt_string_fernet(test_data, fernet)
            decrypted = decrypt_string_fernet(encrypted, {"enc": True}, fernet)

            if decrypted == test_data:
                check.status = CheckStatus.PASS
                check.message = "Encryption working"
            else:
                check.status = CheckStatus.FAIL
                check.severity = Severity.CRITICAL
                check.message = "Encryption round-trip failed"
                check.suggested_fix = "Check KESTREL_DATA_KEY is correct"
                check.file_path = ".env"

        except Exception as e:
            check.status = CheckStatus.FAIL
            check.severity = Severity.CRITICAL
            check.message = f"Encryption error: {e}"

        return check

    async def check_feature_initialization(self) -> HealthCheck:
        """Are all features properly initialized?"""
        check = HealthCheck(
            id="arms.features",
            layer=ReflectionLayer.ARMS,
            name="Feature Initialization",
            description="Verify features are loaded",
            status=CheckStatus.SKIP,
        )

        features = getattr(self.agent, 'features', {})
        if not features:
            check.status = CheckStatus.WARN
            check.severity = Severity.MEDIUM
            check.message = "No features registered"
            return check

        initialized = list(features.keys())
        check.status = CheckStatus.PASS
        check.message = f"{len(initialized)} features loaded"
        check.details = {"features": initialized}

        return check

    async def check_tool_execution(self) -> HealthCheck:
        """Can tools execute?"""
        check = HealthCheck(
            id="arms.tools",
            layer=ReflectionLayer.ARMS,
            name="Tool Execution",
            description="Verify tools are callable",
            status=CheckStatus.SKIP,
        )

        registry = getattr(self.agent, 'tool_registry', None)
        if not registry:
            check.status = CheckStatus.SKIP
            check.message = "No tool registry"
            return check

        try:
            tools = registry.list_tools() if hasattr(registry, 'list_tools') else []
            check.status = CheckStatus.PASS
            check.message = f"{len(tools)} tools registered"
            check.details = {"count": len(tools)}
        except Exception as e:
            check.status = CheckStatus.WARN
            check.message = f"Tool check error: {e}"

        return check

    # =========================================================================
    # FUNCTIONAL CHECKS - Actually invoke tools and verify outputs
    # =========================================================================

    async def check_github_operations(self) -> HealthCheck:
        """Test GitHub API - actually call get_self_repo_info()."""
        check = HealthCheck(
            id="arms.github",
            layer=ReflectionLayer.ARMS,
            name="GitHub Operations",
            description="Verify GitHub API connectivity and authentication",
            status=CheckStatus.SKIP,
        )
        start = self._start_timer()

        try:
            github_feature = self.agent.features.get("GitHubFeature")
            if not github_feature:
                check.status = CheckStatus.SKIP
                check.message = "GitHubFeature not loaded"
                return check

            # Actually call the tool. As of #1061 wave 14 the github
            # @tool methods return a ToolResult envelope; honor the
            # status field so an auth/API failure doesn't pass the
            # health check just because the envelope object itself is
            # truthy.
            if hasattr(github_feature, 'get_self_repo_info'):
                from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

                result = await github_feature.get_self_repo_info()

                if isinstance(result, ToolResult):
                    if result.status is ToolResultStatus.ERROR:
                        check.status = CheckStatus.FAIL
                        check.severity = Severity.HIGH
                        check.message = f"GitHub API failed: {result.error}"
                        check.suggested_fix = "Check GITHUB_PAT or GITHUB_TOKEN in .env"
                        check.file_path = ".env"
                    else:
                        body = result.confirmation or ""
                        if "Repository" in body or "kestrel" in body.lower():
                            check.status = CheckStatus.PASS
                            check.message = "GitHub API working"
                            check.details = {"response_type": "tool_result"}
                        else:
                            check.status = CheckStatus.WARN
                            check.severity = Severity.MEDIUM
                            check.message = "GitHub returned data but unexpected format"
                elif result:
                    if isinstance(result, str):
                        if "Repository" in result or "kestrel" in result.lower():
                            check.status = CheckStatus.PASS
                            check.message = "GitHub API working"
                            check.details = {"response_type": "formatted_string"}
                        else:
                            check.status = CheckStatus.WARN
                            check.severity = Severity.MEDIUM
                            check.message = "GitHub returned data but unexpected format"
                    elif isinstance(result, dict):
                        repo_name = result.get("name") or result.get("full_name")
                        if repo_name:
                            check.status = CheckStatus.PASS
                            check.message = f"GitHub API working (repo: {repo_name})"
                            check.details = {
                                "repo": repo_name,
                                "has_description": bool(result.get("description")),
                            }
                        else:
                            check.status = CheckStatus.WARN
                            check.severity = Severity.MEDIUM
                            check.message = "GitHub returned data but missing repo name"
                    else:
                        check.status = CheckStatus.PASS
                        check.message = "GitHub API responding"
                        check.details = {"response_type": type(result).__name__}
                else:
                    check.status = CheckStatus.FAIL
                    check.severity = Severity.HIGH
                    check.message = "GitHub API returned no data"
                    check.suggested_fix = "Check GITHUB_PAT or GITHUB_TOKEN in .env"
                    check.file_path = ".env"
            else:
                check.status = CheckStatus.WARN
                check.message = "GitHubFeature missing get_self_repo_info method"

        except Exception as e:
            error_str = str(e).lower()
            if "401" in error_str or "unauthorized" in error_str or "authentication" in error_str:
                check.status = CheckStatus.FAIL
                check.severity = Severity.HIGH
                check.message = f"GitHub authentication failed: {e}"
                check.suggested_fix = "Set valid GITHUB_PAT or GITHUB_TOKEN in .env"
                check.file_path = ".env"
            elif "rate limit" in error_str:
                check.status = CheckStatus.WARN
                check.severity = Severity.MEDIUM
                check.message = "GitHub rate limited"
            else:
                check.status = CheckStatus.FAIL
                check.severity = Severity.HIGH
                check.message = f"GitHub API error: {e}"

        check.duration_ms = self._elapsed_ms(start)
        return check

    async def check_model_operations(self) -> HealthCheck:
        """Test model listing - actually call list_models() and get_current_model()."""
        check = HealthCheck(
            id="arms.models",
            layer=ReflectionLayer.ARMS,
            name="Model Operations",
            description="Verify model listing and selection works",
            status=CheckStatus.SKIP,
        )
        start = self._start_timer()

        try:
            llm = getattr(self.agent, 'llm_service', None)
            if not llm:
                check.status = CheckStatus.SKIP
                check.message = "No LLM service configured"
                return check

            models = []
            current_model = None

            # Try to list models - LLMService uses discover_all_models from ModelDiscoveryMixin
            if hasattr(llm, 'discover_all_models'):
                models = await llm.discover_all_models(use_cache=True)
                if not isinstance(models, list):
                    models = list(models) if models else []
            elif hasattr(llm, 'list_models'):
                method = llm.list_models
                if asyncio.iscoroutinefunction(method):
                    models = await method()
                else:
                    models = method()
                if not isinstance(models, list):
                    models = list(models) if models else []

            # Try to get current model
            if hasattr(llm, 'get_current_model'):
                current_model = llm.get_current_model() if callable(llm.get_current_model) else llm.get_current_model
            elif hasattr(llm, 'model'):
                current_model = llm.model
            elif hasattr(llm, 'active_model'):
                current_model = llm.active_model

            if models:
                check.status = CheckStatus.PASS
                check.message = f"Model operations working ({len(models)} models available)"
                check.details = {
                    "model_count": len(models),
                    "current_model": str(current_model) if current_model else "unknown",
                    "sample_models": [str(m)[:50] for m in models[:3]],
                }
            else:
                check.status = CheckStatus.WARN
                check.severity = Severity.MEDIUM
                check.message = "No models found from list_models()"
                check.suggested_fix = "Check LLM provider configuration"
                check.file_path = "kestrel.toml"

        except Exception as e:
            check.status = CheckStatus.FAIL
            check.severity = Severity.HIGH
            check.message = f"Model operations error: {e}"
            check.suggested_fix = "Check LLM service configuration"
            check.file_path = "kestrel.toml"

        check.duration_ms = self._elapsed_ms(start)
        return check

    async def check_web_search(self) -> HealthCheck:
        """Test web search - actually perform a search query."""
        check = HealthCheck(
            id="arms.websearch",
            layer=ReflectionLayer.ARMS,
            name="Web Search",
            description="Verify web search tool works",
            status=CheckStatus.SKIP,
        )
        start = self._start_timer()

        try:
            # Look for WebSearchFeature or web search tool
            websearch_feature = self.agent.features.get("WebSearchFeature")

            if not websearch_feature:
                # Try to find it in tool registry
                registry = getattr(self.agent, 'tool_registry', None)
                if registry and hasattr(registry, 'get_tool'):
                    websearch_tool = registry.get_tool('web_search') or registry.get_tool('search')
                    if websearch_tool:
                        websearch_feature = websearch_tool

            if not websearch_feature:
                check.status = CheckStatus.SKIP
                check.message = "WebSearchFeature not loaded"
                return check

            # Actually perform a search
            search_method = None
            for method_name in ['search', 'web_search', 'query']:
                if hasattr(websearch_feature, method_name):
                    search_method = getattr(websearch_feature, method_name)
                    break

            if search_method:
                # Use a simple test query
                if asyncio.iscoroutinefunction(search_method):
                    result = await search_method("kestrel ai agent")
                else:
                    result = search_method("kestrel ai agent")

                if result:
                    result_count = len(result) if isinstance(result, list) else 1
                    check.status = CheckStatus.PASS
                    check.message = f"Web search working ({result_count} results)"
                    check.details = {"result_count": result_count}
                else:
                    check.status = CheckStatus.WARN
                    check.severity = Severity.MEDIUM
                    check.message = "Web search returned no results"
            else:
                check.status = CheckStatus.WARN
                check.message = "WebSearchFeature has no search method"

        except Exception as e:
            error_str = str(e).lower()
            if "api key" in error_str or "unauthorized" in error_str:
                check.status = CheckStatus.FAIL
                check.severity = Severity.HIGH
                check.message = f"Web search API key issue: {e}"
                check.suggested_fix = "Set TAVILY_API_KEY or equivalent in .env"
                check.file_path = ".env"
            else:
                check.status = CheckStatus.WARN
                check.severity = Severity.MEDIUM
                check.message = f"Web search error: {e}"

        check.duration_ms = self._elapsed_ms(start)
        return check

    async def check_security_permissions(self) -> HealthCheck:
        """Test security feature - check permissions and hooks."""
        check = HealthCheck(
            id="arms.security",
            layer=ReflectionLayer.ARMS,
            name="Security Permissions",
            description="Verify security controls are active",
            status=CheckStatus.SKIP,
        )
        start = self._start_timer()

        try:
            security_feature = self.agent.features.get("SecurityFeature")

            if not security_feature:
                # Check for security hooks directly on agent
                hooks = getattr(self.agent, 'security_hooks', None) or getattr(self.agent, 'hooks', None)
                if hooks:
                    check.status = CheckStatus.PASS
                    check.message = "Security hooks active"
                    check.details = {"hook_count": len(hooks) if isinstance(hooks, (list, dict)) else 1}
                    return check

                check.status = CheckStatus.SKIP
                check.message = "SecurityFeature not loaded"
                return check

            # Try to list permissions or check security status
            permissions = None
            if hasattr(security_feature, 'list_permissions'):
                method = security_feature.list_permissions
                if asyncio.iscoroutinefunction(method):
                    permissions = await method()
                else:
                    permissions = method()
            elif hasattr(security_feature, 'get_permissions'):
                method = security_feature.get_permissions
                if asyncio.iscoroutinefunction(method):
                    permissions = await method()
                else:
                    permissions = method()
            elif hasattr(security_feature, 'permissions'):
                permissions = security_feature.permissions

            # Check for approval queue
            approval_queue = getattr(security_feature, 'approval_queue', None)

            # Check for rate limiter
            rate_limiter = getattr(security_feature, 'rate_limiter', None)

            details = {}
            if permissions:
                details["permission_count"] = len(permissions) if isinstance(permissions, (list, dict)) else 1
            if approval_queue is not None:
                details["approval_queue_active"] = True
            if rate_limiter is not None:
                details["rate_limiter_active"] = True

            if permissions or approval_queue or rate_limiter:
                check.status = CheckStatus.PASS
                check.message = "Security controls active"
                check.details = details
            else:
                check.status = CheckStatus.WARN
                check.severity = Severity.MEDIUM
                check.message = "SecurityFeature loaded but no controls found"
                check.suggested_fix = "Verify security configuration"

        except Exception as e:
            check.status = CheckStatus.WARN
            check.severity = Severity.LOW
            check.message = f"Security check error: {e}"

        check.duration_ms = self._elapsed_ms(start)
        return check
