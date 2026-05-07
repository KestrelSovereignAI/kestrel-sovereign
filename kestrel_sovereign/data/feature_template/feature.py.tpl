"""{{class_name}} -- TODO: describe your feature."""

from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory


class {{class_name}}(Feature):
    """TODO: describe what this feature does."""

    @property
    def tool_description(self) -> str:
        return "TODO: describe capabilities for the orchestrator"

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    @tool("hello", "Say hello to someone", category=ToolCategory.UTILITY)
    async def hello(self, name: str = "world") -> str:
        """Greet someone by name.

        Args:
            name: The name of the person to greet
        """
        return f"Hello, {name}! From {{class_name}}."
