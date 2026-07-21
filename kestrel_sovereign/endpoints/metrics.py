"""Prometheus metrics endpoint — returns metrics in Prometheus text exposition format.

If prometheus-client is not installed, returns 404 with an informative message.
"""

from fastapi import APIRouter, Response

from kestrel_sovereign.api_errors import api_error_response
from kestrel_sdk.metrics import (
    PROMETHEUS_AVAILABLE,
    generate_metrics,
    get_content_type,
)

router = APIRouter(tags=["observability"])


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Expose Prometheus-compatible metrics for scraping.

    Returns Prometheus text exposition format when prometheus-client is
    installed, or 404 with a helpful message otherwise.
    """
    if not PROMETHEUS_AVAILABLE:
        return api_error_response(
            status_code=404,
            code="metrics_unavailable",
            message=(
                "Prometheus metrics not available. "
                "Install with: pip install kestrel_sovereign[observability]"
            ),
        )

    body = generate_metrics()
    return Response(content=body, media_type=get_content_type())
