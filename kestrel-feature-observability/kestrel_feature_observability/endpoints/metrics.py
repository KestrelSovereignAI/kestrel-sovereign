"""Prometheus metrics endpoint — returns metrics in Prometheus text exposition format.

If prometheus-client is not installed, returns 404 with an informative message.
"""

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from kestrel_feature_observability.metrics import (
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
        return JSONResponse(
            status_code=404,
            content={
                "detail": "Prometheus metrics not available. "
                "Install with: pip install kestrel-feature-observability[prometheus]"
            },
        )

    body = generate_metrics()
    return Response(content=body, media_type=get_content_type())
