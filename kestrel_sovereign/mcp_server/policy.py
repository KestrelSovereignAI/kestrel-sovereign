"""Explicit remote-server identity, registered clients and finite OAuth budgets."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from kestrel_sovereign.mcp_oauth import MCPOAuthHostPolicy

if TYPE_CHECKING:
    from mcp.shared.auth import OAuthClientInformationFull


MemoryScope = Literal["memory:read", "memory:write", "conversation:write"]


class MCPServerClient(BaseModel):
    """One explicitly registered OAuth client; no automatic client discovery."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)
    client_id: str = Field(min_length=1, max_length=256)
    client_name: str = Field(min_length=1, max_length=128)
    redirect_uris: tuple[str, ...] = Field(min_length=1, max_length=8)
    scopes: tuple[MemoryScope, ...] = Field(default=("memory:read",), min_length=1)
    token_endpoint_auth_method: Literal["none", "client_secret_post", "client_secret_basic"] = "client_secret_post"
    client_secret: SecretStr | None = None

    @field_validator("client_id", "client_name")
    @classmethod
    def control_free_label(cls, value: str) -> str:
        if value != value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("MCP client identity must be nonempty and control-free")
        return value

    @field_validator("redirect_uris")
    @classmethod
    def registered_https_redirects(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("MCP client redirect URIs must be unique")
        for value in values:
            if value != value.strip() or any(ord(char) < 33 or ord(char) == 127 for char in value):
                raise ValueError("MCP client redirect URI must be control-free HTTPS")
            try:
                parsed = urlsplit(value)
                port = parsed.port
            except ValueError:
                raise ValueError("MCP client redirect URI must be valid HTTPS") from None
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                    or parsed.password is not None or "#" in value or port == 0):
                raise ValueError("MCP client redirect URI must be credential-free HTTPS without a fragment")
        return values  # Retain the registered URI; do not rewrite the redirect identity.

    @model_validator(mode="after")
    def explicit_client_authority(self) -> Self:
        if len(set(self.scopes)) != len(self.scopes):
            raise ValueError("MCP client scopes must be unique")
        if self.token_endpoint_auth_method == "none":
            if self.client_secret is not None:
                raise ValueError("A public MCP client must not configure a secret")
        elif self.client_secret is None or not self.client_secret.get_secret_value().strip():
            raise ValueError("A confidential MCP client requires a secret")
        if tuple(str(uri) for uri in self.oauth_information().redirect_uris) != self.redirect_uris:
            raise ValueError("The native SDK must preserve the registered redirect identity")
        return self

    def oauth_information(self) -> "OAuthClientInformationFull":
        """Give the official authorization handler its registered client record."""
        from mcp.shared.auth import OAuthClientInformationFull

        return OAuthClientInformationFull(
            client_id=self.client_id,
            client_name=self.client_name,
            redirect_uris=list(self.redirect_uris),
            scope=" ".join(self.scopes),
            token_endpoint_auth_method=self.token_endpoint_auth_method,
            client_secret=self.client_secret.get_secret_value() if self.client_secret is not None else None,
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        )


class MCPServerPolicy(BaseModel):
    """Opt-in server configuration; it grants no client access to any agent."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True,
        allow_inf_nan=False, hide_input_in_errors=True)
    console_origin: str
    clients: tuple[MCPServerClient, ...] = Field(min_length=1, max_length=64)
    max_pending_per_agent: int = Field(default=32, gt=0, le=256)
    authorization_seconds: float = Field(default=600.0, gt=0, le=1800)
    access_token_seconds: int = Field(default=900, gt=0, le=3600)
    refresh_token_seconds: int = Field(default=604800, gt=0, le=2592000)
    max_request_bytes: int = Field(default=65536, gt=0, le=1048576)

    @field_validator("console_origin")
    @classmethod
    def public_console_origin(cls, value: str) -> str:
        return MCPOAuthHostPolicy.public_console_origin(value)

    @model_validator(mode="after")
    def distinct_registered_clients(self) -> Self:
        if len({client.client_id for client in self.clients}) != len(self.clients):
            raise ValueError("MCP server client IDs must be unique")
        if self.refresh_token_seconds < self.access_token_seconds:
            raise ValueError("MCP refresh lifetime must cover the access lifetime")
        return self

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> Self | None:
        raw = environ.get("KESTREL_MCP_SERVER")
        return None if raw is None else cls.model_validate_json(raw)
