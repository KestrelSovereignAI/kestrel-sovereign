"""Lossless path-safe encoding for host-routed agent names."""

from __future__ import annotations

import base64
import binascii


def encode_agent_route_name(agent_name: str) -> str:
    """Encode any Unicode agent name as one URL-safe path segment."""

    if not isinstance(agent_name, str) or not agent_name:
        raise ValueError("agent name must be non-empty text")
    return base64.urlsafe_b64encode(agent_name.encode("utf-8")).rstrip(b"=").decode("ascii")


def decode_agent_route_name(segment: str) -> str:
    """Decode a canonical agent-name segment, rejecting aliases and damage."""

    if not isinstance(segment, str) or not segment:
        raise ValueError("encoded agent name must be non-empty text")
    padding = "=" * (-len(segment) % 4)
    try:
        raw = base64.b64decode(
            segment + padding,
            altchars=b"-_",
            validate=True,
        )
        agent_name = raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as error:
        raise ValueError("encoded agent name is invalid") from error
    if not agent_name or encode_agent_route_name(agent_name) != segment:
        raise ValueError("encoded agent name is not canonical")
    return agent_name


__all__ = ["decode_agent_route_name", "encode_agent_route_name"]
