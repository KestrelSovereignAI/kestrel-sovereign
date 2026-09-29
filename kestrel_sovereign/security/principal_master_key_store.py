"""Shared persistence for principal-scoped master credentials.

The delegated-master payer policies (``HOST_MASTER_PROVISIONED``,
``USER_MASTER_PROVISIONED`` and ``SPONSOR``) each keep a funding principal's
master credential in its own table:

========================  =============================  ==================
Facade                    Table                          Encryption identity
========================  =============================  ==================
``HostKeyStorage``        ``host_service_keys``          ``"host"``
``UserMasterKeyStorage``  ``user_master_service_keys``   the user's DID
``SponsorKeyStorage``     ``sponsor_master_service_keys`` the sponsor's DID
========================  =============================  ==================

The mechanics are identical, so they live here once: provider bootstrap,
SHA-256 fingerprinting, identity-derived encryption with the legacy-decrypt
fallback, atomic replace, active lookup, listing, and atomic delete. The
public facades stay thin and keep their own tables, identity scopes and info
records.

SQL is built only from :class:`MasterKeyTable` members, a closed set of
validated identifiers. Callers choose a member; they never supply table or
column text.

Wire formats are unchanged: ciphertext is
``base64(encrypt(identity, "service-keys", utf8(api_key)))`` and the key hash
is the first 32 hex characters of the key's SHA-256, so rows written before
this module existed read back as they always did.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Callable, Generic, List, Optional, TypeVar

from kestrel_sovereign.security.agent_encryption import encrypt
from kestrel_sovereign.security.exceptions import (
    DecryptionError,
    KeyNotConfiguredError,
)
from kestrel_sovereign.security.legacy_decrypt import (
    decrypt_with_legacy_fallback as decrypt,
)
from kestrel_sovereign.security.service_key_storage import KNOWN_PROVIDERS
from kestrel_sovereign.storage.timestamps import timestamp_column_value

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)

#: The purpose string every master credential is encrypted under. Changing it
#: changes the derived key and makes every stored row unreadable.
MASTER_KEY_PURPOSE = "service-keys"

#: How much of a principal DID a log line or error message may carry.
_PRINCIPAL_LABEL_CHARS = 30

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class MasterKeyTable(Enum):
    """The tables that hold principal master credentials.

    Each member names its table and, for multi-principal tables, the column
    holding the principal. The host table has no principal column: there is
    one host per deployment.
    """

    HOST = ("host_service_keys", None)
    USER_MASTER = ("user_master_service_keys", "master_did")
    SPONSOR = ("sponsor_master_service_keys", "master_did")

    def __init__(self, table_name: str, principal_column: Optional[str]) -> None:
        for identifier in (table_name, principal_column):
            if identifier is not None and not _IDENTIFIER.match(identifier):
                raise ValueError(f"invalid SQL identifier: {identifier!r}")
        self.table_name = table_name
        self.principal_column = principal_column


@dataclass(frozen=True)
class MasterKeyRecord:
    """One stored master credential's metadata. Carries no secret material."""

    id: str
    principal: Optional[str]
    provider_id: str
    is_active: bool
    created_at: datetime


InfoT = TypeVar("InfoT")


@dataclass(frozen=True)
class PrincipalMasterKeyScope(Generic[InfoT]):
    """Typed configuration binding one principal to one master-key table.

    Attributes:
        table: Where the credentials live.
        principal: The value stored in ``table.principal_column``; ``None``
            exactly when the table has no principal column.
        encryption_identity: The identity salt for ``encrypt``/``decrypt``.
            ``"host"`` for the operator, the principal's DID otherwise.
        kind: Safe human label (``"host"``, ``"user"``, ``"sponsor"``) used in
            log lines and error messages.
        project_info: Builds the facade's public info record from a
            :class:`MasterKeyRecord`.
    """

    table: MasterKeyTable
    principal: Optional[str]
    encryption_identity: str
    kind: str
    project_info: Callable[[MasterKeyRecord], InfoT]

    def __post_init__(self) -> None:
        if not isinstance(self.table, MasterKeyTable):
            raise TypeError("table must be a MasterKeyTable member")
        if self.table.principal_column is None:
            if self.principal is not None:
                raise ValueError(f"{self.table.table_name} has no principal column")
        elif not self.principal:
            raise ValueError(f"{self.table.table_name} requires a principal value")
        if not self.encryption_identity:
            raise ValueError("encryption_identity is required")
        if not self.kind:
            raise ValueError("kind is required")

    @property
    def principal_label(self) -> Optional[str]:
        """The truncated principal that log lines and errors may carry."""
        if self.principal is None:
            return None
        return f"{self.principal[:_PRINCIPAL_LABEL_CHARS]}..."


def master_key_hash(api_key: str) -> str:
    """Fingerprint a key for lookup without decryption."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]


class PrincipalMasterKeyStore(Generic[InfoT]):
    """Encrypted persistence of one principal's master credentials."""

    def __init__(
        self, db: "AsyncDatabase", scope: PrincipalMasterKeyScope[InfoT]
    ) -> None:
        self._db = db
        self._scope = scope
        table = scope.table
        self._table = table.table_name
        column = table.principal_column
        if column is None:
            self._scope_predicate = "provider_id = ?"
            self._metadata_columns = "id, NULL, provider_id, is_active, created_at"
            self._insert_columns = (
                "id, provider_id, encrypted_key, key_hash, is_active, created_at"
            )
            self._insert_values = "?, ?, ?, ?, 1, CURRENT_TIMESTAMP"
            self._conflict_target = "provider_id"
        else:
            self._scope_predicate = f"{column} = ? AND provider_id = ?"
            self._metadata_columns = f"id, {column}, provider_id, is_active, created_at"
            self._insert_columns = (
                f"id, {column}, provider_id, encrypted_key, key_hash, "
                "is_active, created_at"
            )
            self._insert_values = "?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP"
            self._conflict_target = f"{column}, provider_id"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(table={self._table!r}, kind={self._scope.kind!r})"
        )

    def _scope_params(self, provider_id: str) -> tuple:
        if self._scope.principal is None:
            return (provider_id,)
        return (self._scope.principal, provider_id)

    def _describe(self, provider_id: str) -> str:
        label = self._scope.principal_label
        if label is None:
            return f"provider={provider_id}"
        return f"{self._scope.kind}={label}, provider={provider_id}"

    async def _ensure_provider(self, provider_id: str) -> None:
        """Ensure provider exists in the shared service_providers table."""
        provider_info = KNOWN_PROVIDERS.get(provider_id, {"name": provider_id})
        await self._db.execute(
            """
            INSERT OR IGNORE INTO service_providers
            (id, name, supports_sub_accounts, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                provider_id,
                provider_info["name"],
                1 if provider_info.get("supports_sub_accounts") else 0,
            ),
        )

    async def store_key(self, provider_id: str, api_key: str) -> str:
        """Encrypt and store ``api_key``, atomically replacing any existing
        key for this principal and provider. Returns the new row id."""
        await self._ensure_provider(provider_id)

        encrypted_b64 = base64.b64encode(
            encrypt(
                self._scope.encryption_identity,
                MASTER_KEY_PURPOSE,
                api_key.encode("utf-8"),
            )
        ).decode("ascii")
        key_id = str(uuid.uuid4())

        # One statement on both backends: SQLite and PostgreSQL share the
        # ON CONFLICT ... DO UPDATE upsert, and the conflict target is the
        # table's real UNIQUE rather than the fresh per-call ``id``.
        params: tuple
        if self._scope.principal is None:
            params = (key_id, provider_id, encrypted_b64, master_key_hash(api_key))
        else:
            params = (
                key_id,
                self._scope.principal,
                provider_id,
                encrypted_b64,
                master_key_hash(api_key),
            )
        await self._db.execute(
            f"""
            INSERT INTO {self._table}
            ({self._insert_columns})
            VALUES ({self._insert_values})
            ON CONFLICT ({self._conflict_target}) DO UPDATE SET
                id = excluded.id,
                encrypted_key = excluded.encrypted_key,
                key_hash = excluded.key_hash,
                is_active = excluded.is_active,
                created_at = excluded.created_at
            """,
            params,
        )

        logger.info(
            "Stored %s master key for %s",
            self._scope.kind,
            self._describe(provider_id),
        )
        return key_id

    async def get_key(self, provider_id: str) -> str:
        """Return the decrypted active key for ``provider_id``.

        Raises:
            KeyNotConfiguredError: No active key exists for the provider.
            DecryptionError: The stored ciphertext is malformed or does not
                decrypt under this principal's identity.
        """
        row = await self._db.fetchone(
            f"""
            SELECT encrypted_key FROM {self._table}
            WHERE {self._scope_predicate} AND is_active = 1
            """,
            self._scope_params(provider_id),
        )
        if not row or not row[0]:
            message = (
                f"No {self._scope.kind} master key configured for provider "
                f"'{provider_id}'"
            )
            if self._scope.principal_label is not None:
                message += f" and {self._scope.kind} '{self._scope.principal_label}'"
            raise KeyNotConfiguredError(message)

        try:
            encrypted_bytes = base64.b64decode(row[0])
        except binascii.Error as exc:
            raise DecryptionError(
                f"Stored {self._scope.kind} master key for provider "
                f"'{provider_id}' is not valid base64"
            ) from exc
        plaintext = decrypt(
            self._scope.encryption_identity,
            MASTER_KEY_PURPOSE,
            encrypted_bytes,
        )
        try:
            return plaintext.decode("utf-8")
        except UnicodeDecodeError:
            # ``from None``: the codec error quotes the offending plaintext byte.
            raise DecryptionError(
                f"Stored {self._scope.kind} master key for provider "
                f"'{provider_id}' did not decrypt to UTF-8 text"
            ) from None

    async def has_key(self, provider_id: str) -> bool:
        """True iff an active key exists for ``provider_id``."""
        row = await self._db.fetchone(
            f"""
            SELECT 1 FROM {self._table}
            WHERE {self._scope_predicate} AND is_active = 1
            """,
            self._scope_params(provider_id),
        )
        return row is not None

    async def list_keys(self) -> List[InfoT]:
        """Metadata for every key of this principal, newest first."""
        if self._scope.principal is None:
            where, params = "", ()
        else:
            where = f"WHERE {self._scope.table.principal_column} = ?"
            params = (self._scope.principal,)
        rows = await self._db.fetchall(
            f"""
            SELECT {self._metadata_columns}
            FROM {self._table}
            {where}
            ORDER BY created_at DESC, provider_id ASC
            """,
            params,
        )
        return [
            self._scope.project_info(
                MasterKeyRecord(
                    id=row[0],
                    principal=row[1],
                    provider_id=row[2],
                    is_active=bool(row[3]),
                    # The schema defaults created_at, so NULL only arises from
                    # a hand-written row; it has always listed as "now".
                    created_at=(
                        timestamp_column_value(row[4])
                        if row[4] is not None
                        else datetime.utcnow()
                    ),
                )
            )
            for row in rows
        ]

    async def delete_key(self, provider_id: str) -> bool:
        """Hard-delete the active key for ``provider_id``.

        One conditional ``DELETE`` whose affected-row count is the result, so
        of two concurrent deletes exactly one reports ``True``. An inactive
        row is neither removed nor reported, matching :meth:`has_key`.
        """
        removed = await self._db.execute(
            f"""
            DELETE FROM {self._table}
            WHERE {self._scope_predicate} AND is_active = 1
            """,
            self._scope_params(provider_id),
        )
        if removed <= 0:
            return False
        logger.info(
            "Deleted %s master key for %s",
            self._scope.kind,
            self._describe(provider_id),
        )
        return True


__all__ = [
    "MASTER_KEY_PURPOSE",
    "MasterKeyRecord",
    "MasterKeyTable",
    "PrincipalMasterKeyScope",
    "PrincipalMasterKeyStore",
    "master_key_hash",
]
