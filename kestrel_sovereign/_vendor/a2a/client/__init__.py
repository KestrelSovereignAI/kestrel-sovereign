# Modified by Kestrel: isolated SDK namespace; see VENDORED.json.
"""Client-side components for interacting with an A2A agent."""

from kestrel_sovereign._vendor.a2a.client.auth import (
    AuthInterceptor,
    CredentialService,
    InMemoryContextCredentialStore,
)
from kestrel_sovereign._vendor.a2a.client.base_client import BaseClient
from kestrel_sovereign._vendor.a2a.client.card_resolver import A2ACardResolver
from kestrel_sovereign._vendor.a2a.client.client import (
    Client,
    ClientCallContext,
    ClientConfig,
)
from kestrel_sovereign._vendor.a2a.client.client_factory import (
    ClientFactory,
    create_client,
    minimal_agent_card,
)
from kestrel_sovereign._vendor.a2a.client.errors import (
    A2AClientError,
    A2AClientTimeoutError,
    AgentCardResolutionError,
)
from kestrel_sovereign._vendor.a2a.client.interceptors import ClientCallInterceptor


__all__ = [
    'A2ACardResolver',
    'A2AClientError',
    'A2AClientTimeoutError',
    'AgentCardResolutionError',
    'AuthInterceptor',
    'BaseClient',
    'Client',
    'ClientCallContext',
    'ClientCallInterceptor',
    'ClientConfig',
    'ClientFactory',
    'CredentialService',
    'InMemoryContextCredentialStore',
    'create_client',
    'minimal_agent_card',
]
