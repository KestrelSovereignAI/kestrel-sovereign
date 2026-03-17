Issue complete.

Deliverables now in the repo:

- `tests/unit/test_endpoint_contract_suite.py`
- `docs/audit/API_ENDPOINT_MATRIX.md`
- `docs/audit/API_SURFACE_RECONCILIATION.md`

Verification:

- `uv run pytest tests/unit/test_endpoint_contract_suite.py -v`
- passing

Result:

- weak endpoint groups now have direct contract proof
- the live API surface is inventoried in audit docs
