"""Standard OAuth endpoints backed by native SQL grants and Authlib authentication."""

import asyncio
import hmac
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlencode

from authlib.oauth2.rfc6749 import OAuth2Error
from authlib.oauth2.rfc6749.authenticate_client import ClientAuthentication
from authlib.oauth2.rfc6749.errors import (
    InvalidClientError,
    InvalidGrantError,
    InvalidRequestError,
    InvalidScopeError,
    UnsupportedGrantTypeError,
    UnsupportedResponseTypeError,
)
from authlib.oauth2.rfc6749.requests import BasicOAuth2Payload
from authlib.oauth2.rfc7636.challenge import (
    CODE_CHALLENGE_PATTERN,
    CODE_VERIFIER_PATTERN,
    create_s256_code_challenge,
)
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    TokenError,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.datastructures import FormData, QueryParams
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

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


class NativeMCPAuthorizationServer:
    """Native OAuth authorization server and official MCP token verifier."""

    def __init__(self, grants: MCPAuthorizationGrants, consent_url: str):
        origin = grants.policy.console_origin
        if not consent_url.startswith(origin + "/") or "?" in consent_url or "#" in consent_url:
            raise ValueError("MCP consent must use the configured first-party Console")
        self.grants, self.pending, self.consent_url = grants, grants.pending, consent_url
        self._request: ContextVar[_OAuthRequest | None] = ContextVar("mcp_oauth_request", default=None)
        self._authenticate_client = ClientAuthentication(self.pending.policy.registered_client)

    def _require_current(self) -> None:
        self.grants.consent_authority.require_current_host()

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

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        owner = self._owner("authorize")
        if params.resource != owner.resource:
            raise TokenError("invalid_target", "The exact MCP resource is required")
        try:
            handle = await self.pending.create(client.client_id, params)
        except PendingMCPQuotaExceeded as error:
            raise OAuth2Error(error="temporarily_unavailable",
                description="Pending MCP consent capacity is full") from error
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
        return await self.grants.load(MCPRecordKind.REFRESH, refresh_token,
            resource=self._owner("token").resource, client_id=client.client_id)

    async def exchange_refresh_token(self, client: OAuthClientInformationFull, refresh_token: RefreshToken,
                                     scopes: list[str]) -> OAuthToken:
        if client.client_id != refresh_token.client_id:
            raise TokenError("invalid_grant", "The original OAuth client is required")
        return await self.grants.exchange(MCPRecordKind.REFRESH, refresh_token,
            resource=self._owner("token").resource, scopes=scopes)

    async def verify_token(self, token: str) -> AccessToken | None:
        self._require_current()
        return await self.grants.load(MCPRecordKind.ACCESS, token, resource=self.pending.resource_url)

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        kind = MCPRecordKind.ACCESS if isinstance(token, AccessToken) else MCPRecordKind.REFRESH
        await self.grants.revoke(kind, token.token, resource=self._owner("revoke").resource,
            client_id=token.client_id)

    @staticmethod
    def _scopes(value: str | None, defaults: list[str]) -> list[str]:
        selected = list(defaults) if value is None else value.split(" ")
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(defaults):
            raise InvalidScopeError(description="Only the original registered scopes may be requested")
        return selected

    def authorization_response(self, redirect_uri: str, values: dict[str, str]) -> Response:
        """Append response parameters without rewriting the registered URI query."""
        encoded = urlencode({**values, "iss": self.grants.issuer})
        separator = "&" if "?" in redirect_uri else "?"
        return RedirectResponse(redirect_uri + separator + encoded, status_code=302,
            headers=self._response_headers())

    async def authorization_endpoint(self, request: Request) -> Response:
        with self._own_request(request, "authorize"):
            redirect, state = None, None
            try:
                parameters = self._parameters(request.query_params) if request.method == "GET" else await self._form(request)
                self._require_current()
                client = self.pending.policy.registered_client(parameters.get("client_id", ""))
                if client is None:
                    raise InvalidRequestError(description="Explicit OAuth client registration is required")
                supplied_redirect = parameters.get("redirect_uri")
                if supplied_redirect is None:
                    if len(client.redirect_uris) != 1:
                        raise InvalidRequestError(description="The redirect_uri is required")
                    supplied_redirect = client.redirect_uris[0]
                native_redirect = client.native_redirect_uri(supplied_redirect)
                if native_redirect is None:
                    raise InvalidRequestError(description="The exact registered redirect URI is required")
                redirect, state = str(native_redirect), parameters.get("state")
                response_type = parameters.get("response_type")
                if not response_type:
                    raise InvalidRequestError(description="The response_type is required")
                if response_type != "code":
                    raise UnsupportedResponseTypeError()
                if parameters.get("resource") != self.pending.resource_url:
                    raise TokenError("invalid_target", "The exact MCP resource is required")
                challenge = parameters.get("code_challenge", "")
                if (parameters.get("code_challenge_method") != "S256" or len(challenge) != 43
                        or not CODE_CHALLENGE_PATTERN.fullmatch(challenge)):
                    raise InvalidRequestError(description="An S256 PKCE challenge is required")
                scopes = self._scopes(parameters.get("scope"), list(client.scopes))
                params = AuthorizationParams(state=state, scopes=scopes, code_challenge=challenge,
                    redirect_uri=native_redirect, redirect_uri_provided_explicitly="redirect_uri" in parameters,
                    resource=self.pending.resource_url)
                location = await self.authorize(client.oauth_information(), params)
                self._require_current()
                return RedirectResponse(location, status_code=302, headers=self._response_headers())
            except (OAuth2Error, TokenError) as error:
                self._require_current()
                if redirect is None:
                    return self._oauth_error(error)
                values = self._error_values(error)
                if state is not None:
                    values["state"] = state
                return self.authorization_response(redirect, values)

    @staticmethod
    def _parameters(parameters: FormData | QueryParams) -> dict[str, str]:
        result = {}
        for key in parameters:
            values = parameters.getlist(key)
            if len(values) != 1 or not isinstance(values[0], str):
                raise InvalidRequestError(description="OAuth parameters must occur once and contain text")
            result[key] = values[0]
        return result

    async def _form(self, request: Request) -> dict[str, str]:
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type != "application/x-www-form-urlencoded":
            raise InvalidRequestError(description="OAuth requests require application/x-www-form-urlencoded")
        return self._parameters(await request.form())

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
    def _response_headers() -> dict[str, str]:
        return {"Cache-Control": "no-store", "Pragma": "no-cache", "Referrer-Policy": "no-referrer"}

    @classmethod
    def _response(cls, value: dict, status: int = 200, headers: Mapping[str, str] | None = None) -> Response:
        return JSONResponse(value, status_code=status, headers={**cls._response_headers(), **(headers or {})})

    @staticmethod
    def _error_values(error: OAuth2Error | TokenError) -> dict[str, str]:
        if isinstance(error, OAuth2Error):
            return dict(error.get_body())
        result = {"error": error.error}
        if error.error_description is not None:
            result["error_description"] = error.error_description
        return result

    @classmethod
    def _oauth_error(cls, error: OAuth2Error | TokenError) -> Response:
        status, headers = (error.status_code, dict(error.get_headers())) if isinstance(error, OAuth2Error) else (400, {})
        return cls._response(cls._error_values(error), status, headers)

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
                    scopes = self._scopes(form.get("scope"), credential.scopes)
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
                return Response(status_code=200, headers=self._response_headers())
            except (OAuth2Error, TokenError) as error:
                return self._oauth_error(error)
