"""Compose official MCP OAuth handlers with the host's native authority and resource."""

import asyncio
import hmac
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlencode

from authlib.oauth2.rfc6749 import OAuth2Error
from authlib.oauth2.rfc6749.authenticate_client import ClientAuthentication
from authlib.oauth2.rfc6749.errors import (
    InvalidClientError,
    InvalidGrantError,
    InvalidRequestError,
    UnsupportedGrantTypeError,
)
from authlib.oauth2.rfc6749.requests import BasicOAuth2Payload
from authlib.oauth2.rfc7636.challenge import (
    CODE_VERIFIER_PATTERN,
    create_s256_code_challenge,
)
from mcp.server.auth.handlers.authorize import AuthorizationHandler
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    IdentityAssertionParams,
    RefreshToken,
    RegistrationError,
    TokenError,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    is_execution_control_error,
)
from kestrel_sovereign.storage.db.interface import TransactionError

from .grants import MCPAuthorizationGrants
from .pending import PendingMCPQuotaExceeded
from .records import MCPRecordKind


@dataclass
class _OAuthClientRequest:
    """Authlib's HTTP client-authentication contract over the original form."""

    form: dict[str, str]
    headers: Mapping[str, str]
    auth_method: str | None = None

    @property
    def payload(self) -> BasicOAuth2Payload:
        return BasicOAuth2Payload(self.form)


@dataclass
class _OAuthRequest:
    request: Request
    task: asyncio.Task
    purpose: Literal["authorize", "token", "revoke"]
    resource: str
    active: bool = True
    control: BaseException | None = field(default=None, repr=False)


class NativeMCPOAuthProvider:
    """Native OAuth authorization server and official MCP token verifier."""

    def __init__(self, grants: MCPAuthorizationGrants, consent_url: str):
        origin = grants.policy.console_origin
        if not consent_url.startswith(origin + "/") or "?" in consent_url or "#" in consent_url:
            raise ValueError("MCP consent must use the configured first-party Console")
        self.grants, self.pending, self.consent_url = grants, grants.pending, consent_url
        self._require_current = grants.consent_authority.require_current_host
        self._request: ContextVar[_OAuthRequest | None] = ContextVar("mcp_oauth_request", default=None)
        self._authorization = AuthorizationHandler(self)
        self._authenticate_client = ClientAuthentication(self.pending.policy.registered_client)

    @contextmanager
    def _own_request(self, request: Request, purpose: Literal["authorize", "token", "revoke"]) -> Iterator[_OAuthRequest]:
        self._require_current()
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("A native OAuth HTTP request owner is required")
        owner = _OAuthRequest(request, task, purpose, self.pending.resource_url)
        binding = self._request.set(owner)
        try:
            yield owner
        finally:
            owner.active = False
            self._request.reset(binding)

    def _owner(self, *purposes: str) -> _OAuthRequest:
        self._require_current()
        owner = self._request.get()
        if (owner is None or not owner.active or owner.task is not asyncio.current_task()
                or owner.purpose not in purposes):
            raise RuntimeError("The original native OAuth HTTP operation is required")
        return owner

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        self._require_current()
        client = self.pending.policy.registered_client(client_id)
        return client.oauth_information() if client is not None else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        raise RegistrationError("invalid_client_metadata", "Explicit client registration is required")

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        owner = self._owner("authorize")
        if params.resource != owner.resource:
            raise AuthorizeError("invalid_target", "The exact MCP resource is required")
        try:
            handle = await self.pending.create(client.client_id, params)
        except PendingMCPQuotaExceeded as error:
            raise AuthorizeError("temporarily_unavailable", "Pending MCP consent capacity is full") from error
        except (ExecutionAuthorityError, TransactionError, BaseExceptionGroup) as error:
            # The official authorization handler catches Exception. Retain a
            # classified native receipt so the original HTTP owner can raise
            # it before publishing the handler's ordinary error response.
            if is_execution_control_error(error):
                owner.control = error
            raise
        return self.consent_url + "?" + urlencode({"request": handle})

    async def load_authorization_code(self, client: OAuthClientInformationFull, authorization_code: str) -> AuthorizationCode | None:
        return await self.grants.load(MCPRecordKind.CODE, authorization_code,
            resource=self._owner("token").resource, client_id=client.client_id)

    async def exchange_authorization_code(self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode) -> OAuthToken:
        if client.client_id != authorization_code.client_id:
            raise TokenError("invalid_grant", "The original OAuth client is required")
        return await self.grants.exchange(MCPRecordKind.CODE, authorization_code,
            resource=self._owner("token").resource)

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        owner = self._owner("token", "revoke")
        loader = self.grants.load_for_revocation if owner.purpose == "revoke" else self.grants.load
        return await loader(MCPRecordKind.REFRESH, refresh_token,
            resource=owner.resource, client_id=client.client_id)

    async def exchange_refresh_token(self, client: OAuthClientInformationFull, refresh_token: RefreshToken,
                                     scopes: list[str]) -> OAuthToken:
        if client.client_id != refresh_token.client_id:
            raise TokenError("invalid_grant", "The original OAuth client is required")
        return await self.grants.exchange(MCPRecordKind.REFRESH, refresh_token,
            resource=self._owner("token").resource, scopes=scopes)

    async def load_access_token(self, token: str) -> AccessToken | None:
        self._owner("revoke")
        return await self.grants.load_for_revocation(MCPRecordKind.ACCESS, token,
            resource=self.pending.resource_url, client_id=None)

    async def verify_token(self, token: str) -> AccessToken | None:
        self._require_current()
        return await self.grants.load(MCPRecordKind.ACCESS, token, resource=self.pending.resource_url)

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        kind = MCPRecordKind.ACCESS if isinstance(token, AccessToken) else MCPRecordKind.REFRESH
        await self.grants.revoke(kind, token.token, resource=self._owner("revoke").resource,
            client_id=token.client_id)

    async def exchange_identity_assertion(self, client: OAuthClientInformationFull,
                                          params: IdentityAssertionParams) -> OAuthToken:
        raise TokenError("unsupported_grant_type", "Identity assertion grants are not enabled")

    async def authorization_endpoint(self, request: Request) -> Response:
        with self._own_request(request, "authorize") as owner:
            response = await self._authorization.handle(request)
            if owner.control is not None:
                raise owner.control
            self._require_current()
            return response

    async def _form(self, request: Request) -> dict[str, str]:
        form = await request.form()
        if any(len(form.getlist(key)) != 1 or not isinstance(value, str) for key, value in form.items()):
            raise InvalidRequestError(description="OAuth parameters must occur once and contain text")
        return dict(form)

    def _client(self, request: Request, form: dict[str, str], endpoint: str) -> OAuthClientInformationFull:
        if len(request.headers.getlist("authorization")) > 1:
            raise InvalidRequestError(description="Only one client authentication method is permitted")
        header = request.headers.get("authorization")
        if header:
            if "client_secret" in form:
                raise InvalidRequestError(description="Only one client authentication method is permitted")
            methods = ["client_secret_basic"]
        else:
            methods = ["client_secret_post"] if "client_secret" in form else ["none"]
        native = _OAuthClientRequest(form, request.headers)
        client = self._authenticate_client(native, methods, endpoint)
        if "client_id" in form and form["client_id"] != client.client_id:
            raise InvalidClientError(description="The client identities differ", status_code=401)
        return client.oauth_information()

    @staticmethod
    def _response(value: dict, status: int = 200, headers: Mapping[str, str] | None = None) -> Response:
        return JSONResponse(value, status_code=status, headers={
            "Cache-Control": "no-store", "Pragma": "no-cache", **(headers or {})})

    @classmethod
    def _oauth_error(cls, error: OAuth2Error | TokenError) -> Response:
        if isinstance(error, OAuth2Error):
            return cls._response(dict(error.get_body()), error.status_code, dict(error.get_headers()))
        return cls._response({"error": error.error, "error_description": error.error_description}, 400)

    async def token_endpoint(self, request: Request) -> Response:
        with self._own_request(request, "token"):
            try:
                form = await self._form(request)
                if form.get("resource") != self.pending.resource_url:
                    raise TokenError("invalid_target", "The exact MCP resource is required")
                client = self._client(request, form, "token")
                kind = form.get("grant_type")
                if not kind:
                    raise InvalidRequestError(description="The grant_type is required")
                if kind == "authorization_code":
                    code, verifier = form.get("code"), form.get("code_verifier")
                    if not code or not verifier or not CODE_VERIFIER_PATTERN.fullmatch(verifier):
                        raise InvalidRequestError(description="An authorization code and valid PKCE verifier are required")
                    credential = await self.load_authorization_code(client, code)
                    if credential is None:
                        raise InvalidGrantError(description="The authorization code is unavailable")
                    supplied_redirect = form.get("redirect_uri")
                    if ((credential.redirect_uri_provided_explicitly or supplied_redirect is not None)
                            and supplied_redirect != str(credential.redirect_uri)):
                        raise InvalidGrantError(description="The original redirect URI is required")
                    if not hmac.compare_digest(create_s256_code_challenge(verifier), credential.code_challenge):
                        raise InvalidGrantError(description="The PKCE verifier does not match")
                    tokens = await self.exchange_authorization_code(client, credential)
                elif kind == "refresh_token":
                    secret = form.get("refresh_token")
                    if not secret:
                        raise InvalidRequestError(description="The refresh_token is required")
                    credential = await self.load_refresh_token(client, secret)
                    if credential is None:
                        raise InvalidGrantError(description="The refresh token is unavailable")
                    scopes = form["scope"].split() if "scope" in form else credential.scopes
                    tokens = await self.exchange_refresh_token(client, credential, scopes)
                else:
                    raise UnsupportedGrantTypeError()
                self._require_current()
                return self._response(tokens.model_dump(mode="json", exclude_none=True))
            except (OAuth2Error, TokenError) as error:
                return self._oauth_error(error)

    async def revocation_endpoint(self, request: Request) -> Response:
        with self._own_request(request, "revoke"):
            try:
                form = await self._form(request)
                client = self._client(request, form, "revocation")
                secret = form.get("token")
                if not secret:
                    raise InvalidRequestError(description="The token is required")
                kinds = [MCPRecordKind.ACCESS, MCPRecordKind.REFRESH]
                if form.get("token_type_hint") == "refresh_token":
                    kinds.reverse()
                for kind in kinds:
                    token = await self.grants.load_for_revocation(kind, secret,
                        resource=self.pending.resource_url, client_id=client.client_id)
                    if token is not None:
                        await self.revoke_token(token)
                        break
                self._require_current()
                return Response(status_code=200, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})
            except (OAuth2Error, TokenError) as error:
                return self._oauth_error(error)
