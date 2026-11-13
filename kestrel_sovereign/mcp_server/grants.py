"""Durable native OAuth credentials with atomic consumption and family revocation."""

import asyncio
import hashlib
import hmac
import json
from contextlib import asynccontextmanager

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    RefreshToken,
    TokenError,
)
from mcp.shared.auth import OAuthToken

from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied

from .consent import MCPConsentApproval, MCPConsentAuthority
from .pending import PendingMCPAuthorization, PendingMCPAuthorizations
from .records import (
    MCPAuthorizationRecords,
    MCPRecordKind,
    hash_mcp_secret,
    new_mcp_secret,
)


class MCPAuthorizationCommitRequired(RuntimeError):
    """A credential operation cannot publish from an enclosing transaction."""


class MCPAuthorizationDecisionUnavailable(ValueError):
    """The original pending request cannot produce a publishable decision."""


class MCPAuthorizationGrantQuotaExceeded(ValueError):
    """The agent already has its configured durable grant quota."""


class MCPAuthorizationGrants:
    """Borrow SQL and host authority; the original host owns both lifetimes.

    HTTP composition must retain the original agent/privacy authority reader and
    validate the token request's resource before entering these operations.
    Nothing here authenticates an HTTP client or replaces OAuth PKCE verification.
    """

    def __init__(self, pending: PendingMCPAuthorizations, issuer: str, *,
                 consent_authority: MCPConsentAuthority):
        if not issuer.startswith(pending.policy.console_origin + "/") or "#" in issuer or "?" in issuer:
            raise ValueError("The MCP authorization issuer must belong to its configured host")
        self.pending, self.issuer = pending, issuer
        self.backend, self.policy, self.agent_id = pending.backend, pending.policy, pending.agent_id
        if (consent_authority.agent_id != pending.agent_id or consent_authority.policy is not pending.policy
                or consent_authority.resource_url != pending.resource_url):
            raise ValueError("The original target agent's MCP consent authority is required")
        self._consent_authority = consent_authority
        self._require_current = consent_authority.require_current_host
        self._records = MCPAuthorizationRecords(self.agent_id, self.policy.max_request_bytes)

    @property
    def consent_authority(self) -> MCPConsentAuthority:
        return self._consent_authority

    def _require_top_level(self) -> None:
        self._require_current()
        if self.backend.owns_open_transaction is not False:
            raise MCPAuthorizationCommitRequired("MCP credentials require a certified top-level SQL commit boundary")

    async def initialize(self) -> None:
        self._require_top_level()
        await self.pending.initialize()
        async with self.backend.transaction():
            await self.backend.execute("""CREATE TABLE IF NOT EXISTS mcp_server_oauth_grants (
                agent_id TEXT NOT NULL, grant_hash TEXT NOT NULL, ciphertext TEXT NOT NULL,
                expires_at BIGINT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0 CHECK (revoked IN (0, 1)),
                PRIMARY KEY (agent_id, grant_hash)
            )""")
            await self.backend.execute("""CREATE TABLE IF NOT EXISTS mcp_server_oauth_credentials (
                agent_id TEXT NOT NULL, kind TEXT NOT NULL, secret_hash TEXT NOT NULL,
                grant_hash TEXT NOT NULL, ciphertext TEXT NOT NULL, expires_at BIGINT NOT NULL,
                consumed INTEGER NOT NULL DEFAULT 0 CHECK (consumed IN (0, 1)),
                PRIMARY KEY (agent_id, kind, secret_hash)
            )""")
            self._require_current()
        self._require_current()

    def _resource(self, resource: str) -> None:
        self._require_current()
        if resource != self.pending.resource_url:
            raise TokenError("invalid_target", "The exact target MCP resource is required")

    def _registration(self, client_id: str) -> str | None:
        client = self.policy.registered_client(client_id)
        if client is None:
            return None
        # The native SDK model retains the actual secret; SecretStr's public
        # redacted dump cannot distinguish a rotated client credential.
        raw = client.oauth_information().model_dump(mode="json")
        return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _current_grant(self, value: dict, expires_at: int) -> bool:
        self._require_current()
        registration = self._registration(value["client_id"])
        fingerprint = self.consent_authority.current_sovereign_fingerprint()
        return (expires_at > self.pending.now() and registration is not None
            and hmac.compare_digest(registration, value["registration"])
            and fingerprint is not None and hmac.compare_digest(fingerprint, value["credential_fingerprint"])
            and value["resource"] == self.pending.resource_url and value["issuer"] == self.issuer)

    async def _grant(self, grant_hash: str):
        row = await self.backend.fetch_one("""SELECT ciphertext, expires_at, revoked
            FROM mcp_server_oauth_grants WHERE agent_id = ? AND grant_hash = ?""",
            (self.agent_id, grant_hash))
        self._require_current()
        if row is None or row[2]:
            return None
        value = self._records.open(MCPRecordKind.GRANT, grant_hash, row[1], row[0])
        return (value, row[1]) if self._current_grant(value, row[1]) else None

    async def _stored_credential(self, kind: MCPRecordKind, secret: str, client_id: str | None):
        try:
            digest = hash_mcp_secret(secret)
        except ValueError:
            return None
        row = await self.backend.fetch_one("""SELECT ciphertext, expires_at, consumed, grant_hash
            FROM mcp_server_oauth_credentials WHERE agent_id = ? AND kind = ? AND secret_hash = ?""",
            (self.agent_id, kind.value, digest))
        self._require_current()
        if row is None:
            return None
        value = self._records.open(kind, digest, row[1], row[0])
        if value["grant_hash"] != row[3] or (client_id is not None and value["client_id"] != client_id):
            return None
        return digest, value, row[1], bool(row[2]), row[3]

    async def _resolve_credential(self, kind: MCPRecordKind, secret: str, client_id: str | None):
        """Within serialized SQL, settle authenticated replay before checking use."""
        row = await self._stored_credential(kind, secret, client_id)
        if row is None:
            return None
        if row[3]:
            await self._revoke(row[4])
            return None
        grant = await self._grant(row[4])
        if grant is None:
            return None
        return (*row, grant)

    async def _revoke(self, grant_hash: str) -> None:
        await self.backend.execute("UPDATE mcp_server_oauth_grants SET revoked = 1 WHERE agent_id = ? AND grant_hash = ?",
            (self.agent_id, grant_hash))
        self._require_current()

    async def read_authorization_request(self, handle: str) -> PendingMCPAuthorization | None:
        """Read Console consent only under the original sovereign request."""
        authentication = self.consent_authority.require_sovereign()
        self._require_top_level()
        pending = await self.pending.read(handle)
        self.consent_authority.require_sovereign(expected=authentication)
        return pending

    async def deny_authorization_request(self, handle: str) -> PendingMCPAuthorization | None:
        """Consume a denied request at its original COMMIT, without minting a grant."""
        authentication = self.consent_authority.require_sovereign()
        self._require_top_level()
        async with self.pending.serialized():
            self.consent_authority.require_sovereign(expected=authentication)
            pending = await self.pending.claim(handle)
            self.consent_authority.require_sovereign(expected=authentication)
        self.consent_authority.require_sovereign(expected=authentication)
        return pending if pending is not None and pending.expires_at > self.pending.now() else None

    async def issue_code(self, handle: str, approval: MCPConsentApproval) -> AuthorizationCode:
        """Consume the original pending request and mint its code in one commit."""
        self._require_top_level()
        self.consent_authority.require_approval(approval)
        async with self.pending.serialized():
            self._require_current()
            now = self.pending.now()
            await self.backend.execute("""DELETE FROM mcp_server_oauth_credentials WHERE agent_id = ?
                AND grant_hash IN (SELECT grant_hash FROM mcp_server_oauth_grants WHERE agent_id = ? AND expires_at <= ?)""",
                (self.agent_id, self.agent_id, now))
            await self.backend.execute("DELETE FROM mcp_server_oauth_grants WHERE agent_id = ? AND expires_at <= ?",
                (self.agent_id, now))
            count = await self.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_grants WHERE agent_id = ?",
                (self.agent_id,))
            full = count[0] >= self.policy.max_grants_per_agent
            pending = None if full else await self.pending.read(handle)
            if pending is not None:
                self.consent_authority.require_approval(approval, handle=handle, pending=pending)
                claimed = await self.pending.claim(handle)
                if claimed is None:
                    pending = None
            if pending is not None:
                self.consent_authority.require_approval(approval, handle=handle, pending=claimed)
                grant_hash = hash_mcp_secret(new_mcp_secret())
                expires = now + self.policy.refresh_token_seconds
                value = {"client_id": approval.client_id, "resource": approval.resource,
                    "issuer": self.issuer, "scopes": list(approval.scopes), "subject": approval.subject,
                    "credential_fingerprint": approval.credential_fingerprint,
                    "registration": self._registration(approval.client_id)}
                await self.backend.execute("""INSERT INTO mcp_server_oauth_grants
                    (agent_id, grant_hash, ciphertext, expires_at) VALUES (?, ?, ?, ?)""",
                    (self.agent_id, grant_hash, self._records.seal(MCPRecordKind.GRANT, grant_hash, expires, value), expires))
                code = AuthorizationCode(code=new_mcp_secret(), scopes=list(approval.scopes),
                    expires_at=min(pending.expires_at, expires), client_id=approval.client_id,
                    code_challenge=pending.parameters.code_challenge, redirect_uri=pending.parameters.redirect_uri,
                    redirect_uri_provided_explicitly=pending.parameters.redirect_uri_provided_explicitly,
                    resource=approval.resource, subject=approval.subject)
                await self._insert(MCPRecordKind.CODE, code.code, grant_hash, int(code.expires_at),
                    code.model_dump(mode="json", exclude={"code"}))
            self.consent_authority.require_approval(approval)
            self._require_current()
        if full:
            raise MCPAuthorizationGrantQuotaExceeded("The agent's MCP authorization grant quota is full")
        if pending is None:
            raise MCPAuthorizationDecisionUnavailable("The original pending MCP request is unavailable")
        self.consent_authority.require_approval(approval)
        if code.expires_at <= self.pending.now():
            raise MCPAuthorizationDecisionUnavailable("The MCP authorization code expired before publication")
        return code

    async def _insert(self, kind: MCPRecordKind, secret: str, grant_hash: str,
                      expires_at: int, value: dict) -> None:
        digest = hash_mcp_secret(secret)
        ciphertext = self._records.seal(kind, digest, expires_at, {**value, "grant_hash": grant_hash})
        await self.backend.execute("""INSERT INTO mcp_server_oauth_credentials
            (agent_id, kind, secret_hash, grant_hash, ciphertext, expires_at) VALUES (?, ?, ?, ?, ?, ?)""",
            (self.agent_id, kind.value, digest, grant_hash, ciphertext, expires_at))
        self._require_current()

    async def load(self, kind: MCPRecordKind, secret: str, *, resource: str, client_id: str | None = None):
        self._require_top_level()
        self._resource(resource)
        if kind not in (MCPRecordKind.CODE, MCPRecordKind.ACCESS, MCPRecordKind.REFRESH):
            raise ValueError("Only native OAuth credentials may be loaded")
        async with self.pending.serialized():
            row = await self._resolve_credential(kind, secret, client_id)
        return self._usable_credential(kind, secret, row)

    def _usable_credential(self, kind, secret, row):
        self._require_current()
        if row is None or row[2] <= self.pending.now() or not self._current_grant(*row[5]):
            return None
        return self._native_credential(kind, secret, row[1])

    @asynccontextmanager
    async def access_lifetime(self, secret: str, *, resource: str, client_id: str):
        """Retain the original SQL authorization lock through physical delivery."""
        self._require_top_level()
        self._resource(resource)
        owner = asyncio.current_task()
        async with self.pending.serialized():
            row = await self._resolve_credential(MCPRecordKind.ACCESS, secret, client_id)
            active = True

            def current_access():
                if (not active or asyncio.current_task() is not owner
                        or self.backend.owns_open_transaction is not True):
                    raise HostCapabilityAccessDenied("Original native MCP access lifetime is unavailable")
                token = self._usable_credential(MCPRecordKind.ACCESS, secret, row)
                if token is None:
                    raise HostCapabilityAccessDenied("Original native MCP access grant is unavailable")
                return token

            try:
                current_access()
                yield current_access
                current_access()
            finally:
                active = False

    def _native_credential(self, kind: MCPRecordKind, secret: str, stored: dict):
        value = {key: value for key, value in stored.items() if key != "grant_hash"}
        if kind is MCPRecordKind.CODE:
            client = self.policy.registered_client(value["client_id"])
            redirect = client.native_redirect_uri(value["redirect_uri"])
            if redirect is None:
                raise ValueError("The stored MCP redirect is no longer registered")
            return AuthorizationCode.model_validate({**value, "code": secret, "redirect_uri": redirect})
        model = AccessToken if kind is MCPRecordKind.ACCESS else RefreshToken
        return model.model_validate({**value, "token": secret})

    async def load_for_revocation(self, kind: MCPRecordKind, secret: str, *, resource: str,
                                  client_id: str | None = None) -> AccessToken | RefreshToken | None:
        """Authenticate stored ownership without granting current token usability."""
        self._require_top_level()
        self._resource(resource)
        if kind not in (MCPRecordKind.ACCESS, MCPRecordKind.REFRESH):
            raise ValueError("Only access and refresh tokens may be revoked")
        row = await self._stored_credential(kind, secret, client_id)
        return self._native_credential(kind, secret, row[1]) if row is not None else None

    async def exchange(self, kind: MCPRecordKind, credential: AuthorizationCode | RefreshToken, *,
                       resource: str, scopes: list[str] | None = None) -> OAuthToken:
        self._require_top_level()
        self._resource(resource)
        if kind not in (MCPRecordKind.CODE, MCPRecordKind.REFRESH):
            raise ValueError("Only codes and refresh tokens may be exchanged")
        secret = credential.code if kind is MCPRecordKind.CODE else credential.token
        async with self.pending.serialized():
            row = await self._resolve_credential(kind, secret, credential.client_id)
            refused = row is None
            if row is not None:
                count = await self.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_credentials WHERE agent_id = ? AND grant_hash = ?",
                    (self.agent_id, row[4]))
                refused = row[2] <= self.pending.now() or count[0] + 2 > self.policy.max_credentials_per_grant
                if refused:
                    await self._revoke(row[4])
                else:
                    selected = list(credential.scopes) if scopes is None else list(scopes)
                    expected = credential.model_dump(mode="json", exclude={"code", "token"})
                    original = {key: value for key, value in row[1].items() if key != "grant_hash"}
                    refused = expected != original or not selected or len(set(selected)) != len(selected) or not set(selected) <= set(original["scopes"])
                    if not refused:
                        await self.backend.execute("""UPDATE mcp_server_oauth_credentials SET consumed = 1
                            WHERE agent_id = ? AND kind = ? AND secret_hash = ?""",
                            (self.agent_id, kind.value, row[0]))
                        grant, deadline = row[5]
                        now = self.pending.now()
                        access_expiry = min(deadline, now + self.policy.access_token_seconds)
                        access, refresh = new_mcp_secret(), new_mcp_secret()
                        access_value = {"client_id": grant["client_id"], "scopes": selected, "expires_at": access_expiry,
                            "resource": resource, "subject": grant["subject"], "claims": {"iss": self.issuer}}
                        refresh_value = {"client_id": grant["client_id"], "scopes": selected, "expires_at": deadline,
                            "resource": resource, "subject": grant["subject"]}
                        await self._insert(MCPRecordKind.ACCESS, access, row[4], access_expiry, access_value)
                        await self._insert(MCPRecordKind.REFRESH, refresh, row[4], deadline, refresh_value)
                        token = OAuthToken(access_token=access, refresh_token=refresh,
                            token_type="Bearer", expires_in=access_expiry - now, scope=" ".join(selected))
            self._require_current()
        # A refusal follows COMMIT so replay revocation cannot roll itself back.
        if refused or row[2] <= self.pending.now() or not self._current_grant(*row[5]):
            raise TokenError("invalid_grant", "The original MCP credential is unavailable")
        if access_expiry <= self.pending.now():
            raise TokenError("invalid_grant", "The MCP access token expired before publication")
        return token.model_copy(update={"expires_in": access_expiry - self.pending.now()})

    async def revoke(self, kind: MCPRecordKind, secret: str, *, resource: str, client_id: str) -> None:
        self._require_top_level()
        self._resource(resource)
        if kind not in (MCPRecordKind.ACCESS, MCPRecordKind.REFRESH):
            raise ValueError("Only access and refresh tokens may be revoked")
        async with self.pending.serialized():
            row = await self._stored_credential(kind, secret, client_id)
            if row is not None:
                await self._revoke(row[4])
        self._require_current()
