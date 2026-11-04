"""Host-owned MCP OAuth identity and authenticated connection boundaries.

The feature owns native OAuth and MCP I/O. Core owns operator credentials,
agent membership, private delivery and the public web client identity.
"""

from __future__ import annotations

from collections.abc import Mapping
import ipaddress
import json
from typing import Self
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator, model_validator


MCP_OAUTH_CALLBACK_PATH = "/oauth/mcp/callback"
MCP_OAUTH_CLIENT_METADATA_PATH = "/oauth/mcp/client-metadata.json"


class MCPOAuthHostPolicy(BaseModel):
    """Explicit installation identity and finite host budgets; never vendor grants."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False, hide_input_in_errors=True)
    console_origin: str
    client_name: str = "Kestrel Sovereign"
    max_operations: int = Field(default=16, gt=0, le=256)
    max_operations_per_principal: int = Field(default=2, gt=0, le=16)
    authorization_seconds: float = Field(default=600.0, gt=0, le=1800)
    network_seconds: float = Field(default=20.0, gt=0, le=60)
    operation_seconds: float = Field(default=90.0, gt=0, le=300)
    response_max_bytes: int = Field(default=65_536, gt=0, le=1_048_576)
    max_token_bytes: int = Field(default=16_384, gt=0, le=1_048_576)
    max_request_bytes: int = Field(default=16_384, gt=0, le=1_048_576)
    lease_seconds: float = Field(default=900.0, gt=0, le=3600)

    @field_validator("console_origin")
    @classmethod
    def public_console_origin(cls, value: str) -> str:
        if value != value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("MCP OAuth console origin must be a control-free HTTPS origin")
        url = AnyHttpUrl(value)
        if ("@" in urlsplit(value).netloc or url.scheme != "https"
                or url.username is not None or url.password is not None
                or url.query is not None or url.fragment is not None or url.path not in (None, "/")):
            raise ValueError("MCP OAuth console origin must be an exact credential-free HTTPS origin")
        hostname = url.host.strip("[]")
        if hostname == "localhost" or hostname.endswith(".localhost"):
            raise ValueError("MCP OAuth web client identity requires a public origin")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("MCP OAuth web client identity requires a public origin")
        return str(url).rstrip("/")

    @field_validator("client_name")
    @classmethod
    def client_display_name(cls, value: str) -> str:
        if (not value.strip() or len(value) > 128
                or any(ord(char) < 32 or ord(char) == 127 for char in value)):
            raise ValueError("MCP OAuth client name must be nonempty and control-free")
        return value

    @model_validator(mode="after")
    def bounded_principal_quota(self) -> Self:
        if self.max_operations_per_principal > self.max_operations:
            raise ValueError("MCP OAuth per-principal quota exceeds the host quota")
        return self

    @property
    def redirect_uri(self) -> str:
        return self.console_origin + MCP_OAUTH_CALLBACK_PATH

    @property
    def client_id(self) -> str:
        return self.console_origin + MCP_OAUTH_CLIENT_METADATA_PATH

    def client_document(self) -> dict:
        """Render one public web CIMD identity, with no per-principal scopes or secrets."""
        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "redirect_uris": [self.redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            "application_type": "web",
        }

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> Self | None:
        """Read the explicit public origin and optional JSON policy settings.

        Raises:
            ValueError: Configured identity or budgets are invalid or ambiguous.
        """
        origin = environ.get("KESTREL_MCP_OAUTH_CONSOLE_ORIGIN")
        raw = environ.get("KESTREL_MCP_OAUTH_POLICY")
        if origin is None:
            if raw is not None:
                raise ValueError("MCP OAuth policy requires KESTREL_MCP_OAUTH_CONSOLE_ORIGIN")
            return None
        try:
            settings = {} if raw is None else json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("MCP OAuth policy must be a JSON object") from error
        if not isinstance(settings, dict) or "console_origin" in settings:
            raise ValueError("MCP OAuth policy must contain settings with the origin supplied only by its environment field")
        return cls(console_origin=origin, **settings)
