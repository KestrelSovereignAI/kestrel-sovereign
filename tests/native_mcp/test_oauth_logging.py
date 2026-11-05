"""Mandatory real-SDK logging coverage in the installed MCP dependency lane."""

from tests.unit.test_mcp_oauth_public import _assert_oauth_diagnostic_redaction


def test_actual_native_sdk_oauth_logger_drops_secrets_and_exception_traceback(caplog):
    from mcp.client.auth import oauth2

    _assert_oauth_diagnostic_redaction(oauth2.logger, caplog)
