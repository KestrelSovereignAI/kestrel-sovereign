"""Canonical API error envelope contracts (#2651)."""

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from kestrel_sovereign.api_errors import (
    ApiHTTPException,
    register_api_error_handlers,
)


class _Payload(BaseModel):
    name: str


def _app() -> FastAPI:
    app = FastAPI()
    register_api_error_handlers(app)

    @app.get("/legacy")
    async def legacy_error():
        raise HTTPException(status_code=404, detail="Resource not found.")

    @app.get("/typed")
    async def typed_error():
        raise ApiHTTPException(
            status_code=409,
            code="resource_conflict",
            message="Resource already exists.",
            details=[{"location": ["path", "name"], "message": "Choose another name."}],
        )

    @app.post("/validated")
    async def validated(_payload: _Payload):
        return {"ok": True}

    @app.get("/unexpected")
    async def unexpected():
        raise RuntimeError("secret database path /private/data")

    return app


def test_legacy_http_exception_uses_envelope_and_preserves_detail():
    response = TestClient(_app()).get("/legacy")

    assert response.status_code == 404
    assert response.json()["detail"] == "Resource not found."
    assert response.json()["error"]["code"] == "http_404"
    assert response.json()["error"]["message"] == "Resource not found."
    assert response.headers["X-Correlation-ID"]


def test_typed_exception_exposes_stable_code_and_safe_details():
    response = TestClient(_app()).get("/typed")

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "resource_conflict"
    assert error["message"] == "Resource already exists."
    assert error["details"] == [
        {"location": ["path", "name"], "message": "Choose another name."}
    ]


def test_validation_envelope_omits_submitted_input_and_context():
    response = TestClient(_app()).post(
        "/validated",
        json={"name": {"password": "do-not-echo"}},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["details"] == payload["detail"]
    assert "do-not-echo" not in response.text
    assert all("input" not in detail and "ctx" not in detail for detail in payload["detail"])


def test_unhandled_exception_is_sanitized_in_canonical_envelope():
    response = TestClient(_app(), raise_server_exceptions=False).get("/unexpected")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert response.json()["detail"] == "An internal error occurred."
    assert "/private/data" not in response.text

