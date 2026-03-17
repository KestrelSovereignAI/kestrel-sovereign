Direct contract proof was added for weak endpoint groups.

Added `tests/unit/test_endpoint_contract_suite.py` covering:

- database explorer response shape
- file HEAD contract via existence checks
- observability event serialization
- structured saved-item creation contract
- OpenAI-compatible `/v1/models` and `/v1/chat/completions`

Verification:

- `uv run pytest tests/unit/test_endpoint_contract_suite.py -v`
- result: passed
