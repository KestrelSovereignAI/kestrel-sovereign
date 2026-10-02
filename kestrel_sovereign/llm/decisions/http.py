"""One HTTP path for ``/v1/systemone``-shaped decision requests.

Adapters own their dialect (URL, envelope fields, auth headers); this helper
owns the transport rules every route shares: an explicit timeout on every
call, typed errors, and error messages that never echo the request or the
vendor's error body (either may contain the decision ``state``).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

import httpx

from kestrel_sdk._frozen_json import thaw_json
from kestrel_sdk.llm.decisions import (
    DecisionProtocolError,
    DecisionTransportError,
    ValidatedDecisionRequest,
)


class DecisionHTTPError(DecisionTransportError):
    """The route answered with an HTTP error status."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def systemone_body(
    model: str,
    request: ValidatedDecisionRequest,
    extra: Optional[Mapping[str, Any]] = None,
) -> bytes:
    """The canonical systemone request plus ``model`` and route envelope fields."""

    body: Dict[str, Any] = {"model": model}
    body.update(thaw_json(request.wire))
    if extra:
        body.update(extra)
    return json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


async def post_systemone(
    url: str,
    content: bytes,
    *,
    timeout: float,
    headers: Optional[Mapping[str, str]] = None,
    route: str,
) -> Dict[str, Any]:
    """POST one decision request and return the decoded JSON body."""

    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            response = await http.post(url, content=content, headers=request_headers)
    except httpx.HTTPError as error:
        raise DecisionTransportError(
            f"{route}: decision request failed ({type(error).__name__})"
        ) from None
    if response.status_code >= 400:
        raise DecisionHTTPError(
            f"{route}: decision request returned HTTP {response.status_code}",
            status_code=response.status_code,
        )
    try:
        body = response.json()
    except ValueError:
        raise DecisionProtocolError(f"{route}: decision response is not JSON") from None
    if not isinstance(body, dict):
        raise DecisionProtocolError(f"{route}: decision response is not a JSON object")
    return body
