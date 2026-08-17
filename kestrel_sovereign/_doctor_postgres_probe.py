"""Minimal environment-isolated PostgreSQL worker for ``kestrel doctor``.

This module intentionally imports no Kestrel runtime code.  Doctor launches it
in a short-lived process whose ``PG*`` namespace has been removed, preventing
libpq-only environment settings from changing the asyncpg connection that the
parent already translated into an explicit DSN.
"""

from __future__ import annotations

import json
import os
import shutil
import ssl
import sys
import tempfile
from pathlib import Path

_IS_WINDOWS = sys.platform == "win32"


class ProbeError(RuntimeError):
    """A driver failure safe to return over the worker protocol."""


class ProbeConnectionError(ProbeError):
    """The equivalent libpq connection itself failed."""


class ProbeQueryError(ProbeError):
    """The connection opened, but the governance query failed."""


class ProbeRuntimeConfigurationError(ProbeError):
    """The asyncpg runtime lacks a capability libpq used to connect."""


class ProbeDiagnosticCapabilityError(ProbeError):
    """The worker cannot establish cross-driver connection parity."""


def _emit_phase(*, connected: bool) -> None:
    """Flush one bounded, non-secret breadcrumb outside the JSON channel."""
    phase = "connected; querying" if connected else "connecting"
    sys.stderr.write(f"PostgreSQL diagnostic phase: {phase}\n")
    sys.stderr.flush()


def _runtime_gss_module(dsn_parameters: dict[str, str]) -> str:
    """Return the optional module asyncpg imports for effective GSS auth."""
    gsslib = dsn_parameters.get("gsslib")
    if not gsslib:
        gsslib = "sspi" if _IS_WINDOWS else "gssapi"
    return "sspilib" if gsslib == "sspi" else "gssapi"


def _check_gss_runtime_parity(
    connection,
    dsn_parameters: dict[str, str],
    runtime_gss: dict[str, object] | None,
) -> None:
    """Reject a libpq GSS success that bare asyncpg cannot reproduce.

    PostgreSQL 12 added ``pg_stat_gssapi``, which reports the authentication
    method for this backend without exposing another session's identity. The
    parent can safely validate asyncpg's optional module initialization, but a
    successful libpq exchange is not proof that asyncpg can complete the same
    server challenge sequence. Doctor therefore fails diagnostic-capability
    whenever GSS/SSPI was actually used unless the probe itself is eventually
    replaced with a bounded asyncpg connection/query.
    """
    info = getattr(connection, "info", None)
    server_version = getattr(info, "server_version", None)
    if server_version is None:
        # Lightweight driver doubles do not model libpq connection metadata.
        # Production psycopg2 always supplies it.
        return

    module = _runtime_gss_module(dsn_parameters)
    if (
        not isinstance(runtime_gss, dict)
        or runtime_gss.get("module") != module
        or type(runtime_gss.get("available")) is not bool
    ):
        raise ProbeDiagnosticCapabilityError(
            "PostgreSQL diagnostic worker did not receive a verified "
            "spawned-runtime GSSAPI/SSPI capability"
        )
    module_available = runtime_gss["available"]
    if getattr(info, "used_password", False):
        # A password-authenticated connection did not use GSS.
        return
    if server_version < 120000:
        capability = (
            "initialized its optional module"
            if module_available
            else f"lacks usable optional module {module!r}"
        )
        raise ProbeDiagnosticCapabilityError(
            "PostgreSQL diagnostic driver cannot determine whether this "
            "pre-12 connection used GSSAPI/SSPI authentication; the spawned "
            f"asyncpg runtime {capability}"
        )

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT gss_authenticated FROM pg_catalog.pg_stat_gssapi "
                "WHERE pid = pg_catalog.pg_backend_pid()"
            )
            rows = cursor.fetchall()
    except Exception as exc:
        raise ProbeDiagnosticCapabilityError(
            "PostgreSQL diagnostic driver could not verify whether the "
            "successful connection used GSSAPI/SSPI authentication"
        ) from exc

    if (
        len(rows) != 1
        or not isinstance(rows[0], (list, tuple))
        or len(rows[0]) != 1
        or type(rows[0][0]) is not bool
    ):
        raise ProbeDiagnosticCapabilityError(
            "PostgreSQL diagnostic driver returned an invalid GSSAPI/SSPI "
            "authentication verification result"
        )
    used_gss = rows[0][0]
    if used_gss and not module_available:
        raise ProbeRuntimeConfigurationError(
            "the PostgreSQL server selected GSSAPI/SSPI authentication, but "
            f"the spawned asyncpg runtime lacks usable optional module {module!r}"
        )
    if used_gss:
        raise ProbeDiagnosticCapabilityError(
            "the PostgreSQL diagnostic connection used GSSAPI/SSPI "
            "authentication; module initialization alone cannot verify the "
            "spawned asyncpg runtime's complete server exchange"
        )


def _copy_private_key(source: Path, destination: Path) -> None:
    """Copy readable runtime key bytes into a worker-owned 0600 file."""
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if os.name != "nt":
            # ``open(..., 0o600)`` is still filtered by the caller's umask.
            # Materialized key custody is an exact contract, including under
            # unusually restrictive service umasks.
            os.fchmod(descriptor, 0o600)
        with (
            source.open("rb") as input_file,
            os.fdopen(descriptor, "wb", closefd=False) as output_file,
        ):
            shutil.copyfileobj(input_file, output_file)
    finally:
        os.close(descriptor)


def _asyncpg_client_identity(
    parameters: dict[str, str],
) -> tuple[Path, Path | None] | None:
    """Validate and return the client identity asyncpg eagerly loads.

    This duplicates only asyncpg 0.30's client-chain branch.  The parent has
    already run the installed asyncpg parser in the agent's exact Python
    environment to validate the complete TLS context (root/CRL, password, and
    protocol bounds) before this worker can connect.
    """
    if parameters.get("sslmode", "prefer") == "disable":
        return None

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    password = parameters.get("sslpassword") or ""
    certificate = parameters.get("sslcert")
    key = parameters.get("sslkey")
    default_dir = Path.home() / ".postgresql"
    if not key:
        default_key = default_dir / "postgresql.key"
        key = str(default_key) if default_key.exists() else None

    try:
        if certificate:
            context.load_cert_chain(
                certificate,
                keyfile=key,
                password=lambda: password,
            )
            return Path(certificate), Path(key) if key else None

        default_certificate = default_dir / "postgresql.crt"
        try:
            context.load_cert_chain(
                default_certificate,
                keyfile=key,
                password=lambda: password,
            )
        except (FileNotFoundError, NotADirectoryError):
            # Asyncpg's optional-default branch intentionally swallows a
            # missing default certificate *or* a missing explicit/default key.
            return None
        return default_certificate, Path(key) if key else None
    except (OSError, ssl.SSLError, ValueError) as exc:
        raise ProbeRuntimeConfigurationError(
            "spawned asyncpg rejected its PostgreSQL client TLS identity"
        ) from exc


def _disable_libpq_client_identity(driver, dsn: str) -> str:
    """Suppress a default chain asyncpg tried but intentionally ignored."""
    try:
        return driver.extensions.make_dsn(dsn, sslcertmode="disable")
    except Exception as exc:
        raise ProbeDiagnosticCapabilityError(
            "installed PostgreSQL diagnostic driver cannot suppress a client "
            "TLS identity that spawned asyncpg did not load"
        ) from exc


def _materialize_libpq_tls_key(driver, dsn: str, temp_dir: str) -> str:
    """Give libpq private key material with asyncpg-equivalent semantics.

    Python's ``SSLContext.load_cert_chain`` accepts readable POSIX key files
    regardless of their mode and accepts a certificate/private-key combined
    PEM when ``keyfile`` is omitted. Libpq rejects broad key modes and searches
    for a separate default key. A worker-private copy both satisfies libpq's
    stricter custody check and lets a combined certificate act as its key.
    """
    parameters = driver.extensions.parse_dsn(dsn)
    if parameters.get("sslmode", "prefer") == "disable":
        return dsn

    identity = _asyncpg_client_identity(parameters)
    if identity is None:
        default_certificate = Path.home() / ".postgresql" / "postgresql.crt"
        if not parameters.get("sslcert") and default_certificate.exists():
            # The optional default certificate was present but asyncpg ignored
            # the chain (for example, an explicitly missing lone sslkey). Tell
            # libpq not to reinterpret the same files as a required identity.
            return _disable_libpq_client_identity(driver, dsn)
        return dsn

    certificate, key = identity
    # ``load_cert_chain(keyfile=None)`` reads the private key from the
    # certificate PEM itself. Point libpq at an equivalent private copy.
    key = key or certificate

    private_key = Path(temp_dir) / "client.key"
    _copy_private_key(key, private_key)
    return driver.extensions.make_dsn(
        dsn,
        sslcert=str(certificate),
        sslkey=str(private_key),
    )


def _query_postgres_driver(
    driver,
    dsn: str,
    sql: str,
    params,
    *,
    dsn_parameters: dict[str, str],
    runtime_gss: dict[str, object] | None,
) -> list:
    """Run one query through a psycopg2-compatible driver."""
    _emit_phase(connected=False)
    try:
        connection = driver.connect(dsn)
    except Exception as exc:
        raise ProbeConnectionError(str(exc)) from exc
    _emit_phase(connected=True)
    try:
        _check_gss_runtime_parity(connection, dsn_parameters, runtime_gss)
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchall()
        except Exception as exc:
            raise ProbeQueryError(str(exc)) from exc
    finally:
        connection.close()


def fetch_rows_in_process(
    dsn: str,
    sql: str,
    params: tuple | list,
    *,
    absent_passfile_sentinel: str,
    probe_temp_dir: str | None = None,
    runtime_gss: dict[str, object] | None = None,
    driver=None,
) -> list:
    """Connect and query inside the already-isolated worker process."""
    if driver is None:
        import psycopg2 as driver

    if probe_temp_dir is not None:
        return _fetch_rows_with_temp_dir(
            dsn,
            sql,
            params,
            absent_passfile_sentinel=absent_passfile_sentinel,
            runtime_gss=runtime_gss,
            driver=driver,
            temp_dir=probe_temp_dir,
        )
    with tempfile.TemporaryDirectory(prefix="kestrel-doctor-") as temp_dir:
        return _fetch_rows_with_temp_dir(
            dsn,
            sql,
            params,
            absent_passfile_sentinel=absent_passfile_sentinel,
            runtime_gss=runtime_gss,
            driver=driver,
            temp_dir=temp_dir,
        )


def _fetch_rows_with_temp_dir(
    dsn: str,
    sql: str,
    params: tuple | list,
    *,
    absent_passfile_sentinel: str,
    runtime_gss: dict[str, object] | None,
    driver,
    temp_dir: str,
) -> list:
    """Execute using private material whose lifecycle is owned by the caller."""
    effective_dsn = dsn
    # Windows libpq searches %APPDATA%/postgresql for default TLS files,
    # while asyncpg searches %USERPROFILE%/.postgresql.  The parent has frozen
    # any asyncpg-visible files into the DSN, so hide libpq's unrelated
    # defaults for the duration of this one child-local query.
    old_appdata = os.environ.get("APPDATA")
    had_appdata = "APPDATA" in os.environ
    if _IS_WINDOWS:
        os.environ["APPDATA"] = str(Path(temp_dir) / "absent-appdata")
    try:
        try:
            if (
                driver.extensions.parse_dsn(dsn).get("passfile")
                == absent_passfile_sentinel
            ):
                # The worker directory has private mkdtemp custody. Its
                # ``absent`` child is never created, so no process can race a
                # passfile into place between translation and libpq's open.
                passfile = Path(temp_dir) / "absent" / "pgpass"
                effective_dsn = driver.extensions.make_dsn(dsn, passfile=str(passfile))

            effective_dsn = _materialize_libpq_tls_key(driver, effective_dsn, temp_dir)
            dsn_parameters = driver.extensions.parse_dsn(effective_dsn)
            return _query_postgres_driver(
                driver,
                effective_dsn,
                sql,
                params,
                dsn_parameters=dsn_parameters,
                runtime_gss=runtime_gss,
            )
        except ProbeError as exc:
            message = str(exc).replace(temp_dir, "<probe-temp>")
            raise type(exc)(message) from exc
        except Exception as exc:
            message = str(exc).replace(temp_dir, "<probe-temp>")
            raise ProbeError(message) from exc
    finally:
        if _IS_WINDOWS:
            if had_appdata:
                os.environ["APPDATA"] = old_appdata or ""
            else:
                os.environ.pop("APPDATA", None)


def main() -> None:
    """Serve one JSON request on stdin and one JSON response on stdout."""
    try:
        request = json.loads(sys.stdin.read())
        rows = fetch_rows_in_process(
            request["dsn"],
            request["sql"],
            request.get("params", []),
            absent_passfile_sentinel=request["absent_passfile_sentinel"],
            probe_temp_dir=request.get("probe_temp_dir"),
            runtime_gss=request.get("runtime_gss"),
        )
        # Deliberately no ``default=str``: every current governance query
        # returns JSON-native scalars.  A future query that does not must fail
        # closed rather than silently changing bytes/datetime/Decimal values.
        output = json.dumps({"ok": True, "rows": rows})
    except ProbeConnectionError as exc:
        output = json.dumps({"ok": False, "kind": "connection", "error": str(exc)})
    except ProbeQueryError as exc:
        output = json.dumps({"ok": False, "kind": "query", "error": str(exc)})
    except ProbeRuntimeConfigurationError as exc:
        output = json.dumps(
            {"ok": False, "kind": "runtime_configuration", "error": str(exc)}
        )
    except ProbeDiagnosticCapabilityError as exc:
        output = json.dumps(
            {"ok": False, "kind": "diagnostic_capability", "error": str(exc)}
        )
    except Exception as exc:  # noqa: BLE001 - protocol carries safe text only
        output = json.dumps({"ok": False, "kind": "diagnostic", "error": str(exc)})
    sys.stdout.write(output)


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    main()
