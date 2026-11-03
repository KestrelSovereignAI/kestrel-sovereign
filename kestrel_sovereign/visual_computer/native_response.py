"""Bounded response mechanics for the finite closed original HTTP owners.

The concrete owner validates provenance; this base issues no operation or grant.
"""

import threading

import httpx

from kestrel_sovereign.execution_custody import ExecutionAuthorityError

MAX_ORIGINAL_RESPONSE_BYTES = 1024 * 1024


class _BoundedOriginalResponse:
    def _request_started(self, native, original):
        self._require_worker(original)
        if self._request is not None:
            raise ExecutionAuthorityError(
                "Original acquisition permits one native request"
            )
        self._binding._require_actual_request(native)
        self._request = (native, original)

    def _socket_borrowed(self, original_socket, original):
        self._require_worker(original)
        if self._request is None or self._request[1] is not original:
            raise ExecutionAuthorityError("Original acquisition socket differs")
        self._sockets.add(original_socket)

    def _response_started(self, response, original):
        self._require_worker(original)
        if (
            self._request is None
            or self._request[1] is not original
            or self._response is not None
            or type(response) is not httpx.Response
            or response.request is not self._request[0]
        ):
            raise ExecutionAuthorityError("Original acquisition response differs")
        self._response = response
        # This event precedes HTTPX's iter_bytes/_get_content_decoder/read.
        encoding = response.headers.get_list("content-encoding")
        if encoding and (
            len(encoding) != 1 or encoding[0].strip().lower() != "identity"
        ):
            raise ExecutionAuthorityError(
                "Original acquisition refuses encoded response"
            )
        lengths = response.headers.get_list("content-length")
        if lengths and (
            len(lengths) != 1
            or not lengths[0].isascii()
            or not lengths[0].isdigit()
            or len(lengths[0]) > 7
            or int(lengths[0]) > MAX_ORIGINAL_RESPONSE_BYTES
        ):
            raise ExecutionAuthorityError(
                "Original acquisition response exceeds byte bounds"
            )

    def _raw_chunk(self, data, original):
        self._require_worker(original)
        if (
            self._request[1] is not original
            or self._response is not original.response
            or type(data) is not bytes
            or self._exhausted
            or len(data) > MAX_ORIGINAL_RESPONSE_BYTES - self._raw_bytes
        ):
            raise ExecutionAuthorityError(
                "Original acquisition response exceeds byte bounds"
            )
        self._raw_bytes += len(data)

    def _body_exhausted(self, original):
        self._require_worker(original)
        if self._request[1] is not original or self._response is not original.response:
            raise ExecutionAuthorityError("Original acquisition body differs")
        self._exhausted = True

    def _body_closed(self, original):
        self._require_worker(original)
        if self._request[1] is not original or self._response is not original.response:
            raise ExecutionAuthorityError("Original acquisition close differs")
        self._response_closed = True


class _CurrentRunpodNativeResponse(_BoundedOriginalResponse):
    """Mechanical worker guards for the finite original current HTTP owners."""

    def _enter_worker(self, client, program):
        from kestrel_sovereign.execution_sync_http import _THREAD

        thread = _THREAD.get()
        if (
            self._worker_entered
            or self._state != "prepared"
            or client is not self._client
            or program is not self._program
            or thread is not self._thread
            or thread.environment_quote is not self
        ):
            raise ExecutionAuthorityError("Original quote native worker differs")
        self._worker, self._worker_entered, self._state = (
            threading.current_thread(),
            True,
            "running",
        )


    def _require_worker(self, original):
        from kestrel_sovereign.execution_sync_http import _THREAD

        if (
            not self._worker_entered
            or self._worker_exited
            or threading.current_thread() is not self._worker
            or self._state != "running"
            or _THREAD.get() is not self._thread
            or self._thread.environment_quote is not self
            or original.thread is not self._thread
            or original.client is not self._client
        ):
            raise ExecutionAuthorityError("Original quote physical worker differs")
        self._mutation._require_registered()
        self._mutation._require_captured_native_lifetime()


    def check_parse_current(self):
        from kestrel_sovereign.execution_sync_http import _THREAD, _require_sync_work

        if (
            not self._worker_entered
            or self._worker_exited
            or threading.current_thread() is not self._worker
            or self._state != "running"
            or _THREAD.get() is not self._thread
            or not self._thread.active
        ):
            raise ExecutionAuthorityError(
                "Original quote parse requires its native worker"
            )
        self._mutation._require_registered()
        self._mutation._require_captured_native_lifetime()
        self._host._canonical.require_original()
        self._check_original()
        self._control.transport._prepared_configuration()
        _require_sync_work()


    def _require_actual_request(self, request):
        from .runpod_acquisition import _parts

        self.check_parse_current()
        if _parts(request) != self._native_parts:
            raise ExecutionAuthorityError("Original quote native request differs")


    def _invocation_current(self):
        if threading.current_thread() is self._worker:
            self.check_parse_current()
        else:
            self._require_origin()
            self._mutation.check_current()
