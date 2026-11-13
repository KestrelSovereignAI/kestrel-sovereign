"""Join the original worker stream close without treating peer loss as debt."""
import asyncio
from asyncio import sslproto

from kestrel_sovereign.execution_custody import execution_terminal_error


async def join_original_writer_close(writer):
    """Return an original terminal peer error only after its actual close join.

    Browser IPC and broker TLS use this same boundary. Recognition requires
    the pinned stdlib writer, protocol, Future and exception identities. The
    TLS timeout case additionally requires the original SSLProtocol to have
    finished connection_lost and released its underlying transport. Other
    errors and caller cancellation remain failures of the owned join.
    """
    protocol = writer._protocol if type(writer) is asyncio.StreamWriter else None
    close_waiter = (protocol._get_close_waiter(writer)
                    if type(protocol) is asyncio.StreamReaderProtocol else None)
    transport = writer.transport if type(writer) is asyncio.StreamWriter else None
    tls = (transport._ssl_protocol
           if type(transport) is sslproto._SSLProtocolTransport else None)
    try:
        await writer.wait_closed()
    except (ConnectionResetError, BrokenPipeError, TimeoutError) as error:
        try:
            original_closed = (
                type(close_waiter) is asyncio.Future
                and writer._protocol is protocol
                and protocol._get_close_waiter(writer) is close_waiter
                and close_waiter.done() and not close_waiter.cancelled()
                and close_waiter.exception() is error
                and writer.is_closing()
            )
            terminal_tls_timeout = (
                type(error) is TimeoutError
                and error.args == ("SSL shutdown timed out",)
                and type(transport) is sslproto._SSLProtocolTransport
                and writer.transport is transport
                and type(tls) is sslproto.SSLProtocol
                and transport._ssl_protocol is tls and transport._closed
                and tls._state is sslproto.SSLProtocolState.UNWRAPPED
                and tls._app_state is sslproto.AppProtocolState.STATE_CON_LOST
                and tls._transport is None
                and tls._app_transport is None and tls._app_protocol is None
                and tls._conn_lost > 0
            )
            terminal_peer = original_closed and (
                type(error) in (ConnectionResetError, BrokenPipeError)
                or terminal_tls_timeout
            )
        except BaseException as verification:
            raise execution_terminal_error(error, verification, retain_originals=True)
        if not terminal_peer:
            raise
        return error
    return None
