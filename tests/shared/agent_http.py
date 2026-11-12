"""Mount a booted agent behind the production authentication middleware."""

from fastapi import FastAPI

from kestrel_sovereign import server
from kestrel_sovereign.api_errors import register_api_error_handlers
from kestrel_sovereign.rate_limit import limiter
from kestrel_sovereign.security.private_delivery import PrivateDeliveryMiddleware
from kestrel_sovereign.security.request_auth import (
    AuthenticatedSessionMiddleware,
    InteractiveAuthenticationLifetimeMiddleware,
    RequestAuthenticationService,
)


def authenticated_agent_app(agent, *, session_secret):
    """Use the test-owned host credential with the actual server auth rail.

    The caller owns the credential environment, agent boot/retirement and routers.
    This does not run the complete host lifespan or create a network listener.
    """
    app = FastAPI()
    app.state.agent = agent
    app.state.limiter = limiter
    app.state.request_authentication = RequestAuthenticationService(
        api_key=lambda: server.get_api_key(generate_if_missing=False))
    app.add_middleware(InteractiveAuthenticationLifetimeMiddleware)
    app.middleware("http")(server.auth_middleware)
    app.add_middleware(
        AuthenticatedSessionMiddleware, secret_key=session_secret,
        session_cookie="kestrel_session", max_age=600,
    )
    app.add_middleware(PrivateDeliveryMiddleware)
    register_api_error_handlers(app)
    return app
