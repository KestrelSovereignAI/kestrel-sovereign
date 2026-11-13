"""Read a bounded JSON body before entering private capability operations."""


class InvalidJSONRequest(ValueError):
    """Invalid bounded JSON input, distinct from host or native operation errors."""


async def read_json_model(request, model, *, max_bytes: int):
    """Validate the original media type and streamed bytes, then the schema."""
    if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
        raise InvalidJSONRequest("JSON request required")
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > max_bytes:
            raise InvalidJSONRequest("JSON request exceeds byte limit")
        content.extend(chunk)
    return model.model_validate_json(bytes(content))
