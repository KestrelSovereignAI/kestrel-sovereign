"""Retain the original published host through native MCP HTTP delivery."""


class NativeMCPHostMiddleware:
    """Wrap the official SDK without rewriting its messages, routes or credentials."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        application = scope.get("app")
        registry = getattr(getattr(application, "state", None), "native_mcp_http", None)
        if scope["type"] != "http" or registry is None:
            await self.app(scope, receive, send)
            return
        from .transport import NativeMCPHTTPRegistry

        if not isinstance(registry, NativeMCPHTTPRegistry):
            raise TypeError("Original native MCP HTTP registry required")
        binding = registry.resolve(scope)
        if binding is None:
            await self.app(scope, receive, send)
            return
        if not callable(binding.request_lifetime):
            raise TypeError("Original native MCP host request lifetime required")
        async with binding.request_lifetime() as current:
            async def guarded_send(message):
                current()
                await send(message)
                current()
            await self.app(scope, receive, guarded_send)
