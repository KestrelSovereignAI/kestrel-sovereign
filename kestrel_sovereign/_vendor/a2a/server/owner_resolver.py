# Modified by Kestrel: isolated SDK namespace; see VENDORED.json.
from collections.abc import Callable

from kestrel_sovereign._vendor.a2a.server.context import ServerCallContext


# Definition
OwnerResolver = Callable[[ServerCallContext], str]


# Example Default Implementation
def resolve_user_scope(context: ServerCallContext) -> str:
    """Resolves the owner scope based on the user in the context."""
    return context.user.user_name
