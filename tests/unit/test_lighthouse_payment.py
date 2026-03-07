"""
Unit tests for Lighthouse payment methods.

Tests pay_for_storage() and get_balance() functionality with mocked REST client.
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock

from kestrel_sovereign.storage.providers.lighthouse_provider import (
    LighthouseProvider,
    LIGHTHOUSE_PERPETUAL_COST_PER_GB,
)


def _make_provider(tmp_path):
    """Create a provider with mocked REST client."""
    provider = LighthouseProvider(
        api_key="test-api-key",
        cache_dir=str(tmp_path),
    )
    # Mock the REST client's _get_client to return a mock httpx client
    mock_http = AsyncMock()
    provider._client._get_client = AsyncMock(return_value=mock_http)
    provider._available = True
    return provider, mock_http


class TestLighthousePayment:
    """Tests for Lighthouse payment functionality."""

    @pytest.fixture
    def provider(self, tmp_path):
        """Create provider with temp cache."""
        provider, self.mock_http = _make_provider(tmp_path)
        yield provider

    @pytest.mark.asyncio
    async def test_pay_for_storage_success(self, provider):
        """Payment should succeed with valid parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "paymentId": "payment_123",
            "dealId": "deal_456",
            "expiresAt": "2027-02-15T00:00:00Z"
        }
        self.mock_http.post = AsyncMock(return_value=mock_response)

        result = await provider.pay_for_storage(
            amount_usd=Decimal("20.00"),
            currency="USDC",
            wallet_address="t1test123"
        )

        assert result["status"] == "success"
        assert result["payment_id"] == "payment_123"
        assert result["deal_id"] == "deal_456"
        assert result["currency"] == "USDC"
        assert result["amount_usd"] == "20.00"
        assert result["wallet_address"] == "t1test123"

    @pytest.mark.asyncio
    async def test_pay_for_storage_with_fil(self, provider):
        """Payment with FIL should convert USD to FIL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "paymentId": "payment_fil",
            "dealId": "deal_fil",
        }
        self.mock_http.post = AsyncMock(return_value=mock_response)

        result = await provider.pay_for_storage(
            amount_usd=Decimal("55.00"),
            currency="FIL",
            wallet_address="t1test456"
        )

        assert result["status"] == "success"
        assert result["currency"] == "FIL"

    @pytest.mark.asyncio
    async def test_pay_for_storage_api_failure(self, provider):
        """Payment should handle API errors gracefully."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "Insufficient balance"
        }
        self.mock_http.post = AsyncMock(return_value=mock_response)

        result = await provider.pay_for_storage(
            amount_usd=Decimal("100.00"),
            currency="USDC",
            wallet_address="t1poor"
        )

        assert result["status"] == "failed"
        assert "Insufficient balance" in result["error"]

    @pytest.mark.asyncio
    async def test_pay_for_storage_unsupported_currency(self, provider):
        """Payment should reject unsupported currencies."""
        with pytest.raises(ValueError, match="Unsupported currency"):
            await provider.pay_for_storage(
                amount_usd=Decimal("10.00"),
                currency="BTC",
                wallet_address="t1test"
            )

    @pytest.mark.asyncio
    async def test_pay_for_storage_network_error(self, provider):
        """Payment should handle network errors."""
        import httpx
        self.mock_http.post = AsyncMock(side_effect=httpx.HTTPError("Network timeout"))

        with pytest.raises(ConnectionError, match="Failed to connect"):
            await provider.pay_for_storage(
                amount_usd=Decimal("10.00"),
                currency="USDC",
                wallet_address="t1test"
            )

    @pytest.mark.asyncio
    async def test_pay_for_storage_provider_unavailable(self, tmp_path):
        """Payment should fail if provider is unavailable."""
        provider = LighthouseProvider(api_key=None, cache_dir=str(tmp_path))
        provider._available = False

        with pytest.raises(ConnectionError, match="not available"):
            await provider.pay_for_storage(
                amount_usd=Decimal("10.00"),
                currency="USDC",
                wallet_address="t1test"
            )


class TestLighthouseBalance:
    """Tests for Lighthouse balance checking."""

    @pytest.fixture
    def provider(self, tmp_path):
        """Create provider with temp cache."""
        provider, self.mock_http = _make_provider(tmp_path)
        # Mock the get_balance REST call
        self._mock_balance_response = None
        yield provider

    def _set_balance_response(self, data_used: str, data_limit: str):
        """Set the mocked balance response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "dataUsed": data_used,
                "dataLimit": data_limit,
            }
        }
        mock_response.raise_for_status = Mock()
        self.mock_http.get = AsyncMock(return_value=mock_response)

    @pytest.mark.asyncio
    async def test_get_balance_success(self, provider):
        """Balance check should return available storage quota."""
        self._set_balance_response("1073741824", "5368709120")  # 1GB used, 5GB limit

        balance = await provider.get_balance(
            wallet_address="t1test",
            currency="USDC"
        )

        # 4 GB available * $4/GB = $16 in USDC
        assert balance == Decimal("16")

    @pytest.mark.asyncio
    async def test_get_balance_with_fil(self, provider):
        """Balance check should convert to FIL."""
        self._set_balance_response("0", "5368709120")  # 0 used, 5GB limit

        balance = await provider.get_balance(
            wallet_address="t1test",
            currency="FIL"
        )

        # 5 GB * $4/GB = $20 USD / $5.50 per FIL ≈ 3.636 FIL
        expected = Decimal("20") / Decimal("5.50")
        assert abs(balance - expected) < Decimal("0.01")

    @pytest.mark.asyncio
    async def test_get_balance_full_quota_used(self, provider):
        """Balance should be 0 when quota is full."""
        self._set_balance_response("5368709120", "5368709120")

        balance = await provider.get_balance(
            wallet_address="t1test",
            currency="USDC"
        )

        assert balance == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_balance_over_quota(self, provider):
        """Balance should be 0 when over quota."""
        self._set_balance_response("6442450944", "5368709120")

        balance = await provider.get_balance(
            wallet_address="t1test",
            currency="USDC"
        )

        assert balance == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_balance_api_error(self, provider):
        """Balance check should return 0 on API error."""
        self.mock_http.get = AsyncMock(side_effect=Exception("API error"))

        balance = await provider.get_balance(
            wallet_address="t1test",
            currency="USDC"
        )

        assert balance == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_balance_unsupported_currency(self, provider):
        """Balance check should reject unsupported currencies."""
        with pytest.raises(ValueError, match="Unsupported currency"):
            await provider.get_balance(
                wallet_address="t1test",
                currency="ETH"
            )

    @pytest.mark.asyncio
    async def test_get_balance_provider_unavailable(self, tmp_path):
        """Balance check should fail if provider unavailable."""
        provider = LighthouseProvider(api_key=None, cache_dir=str(tmp_path))
        provider._available = False

        with pytest.raises(ConnectionError, match="not available"):
            await provider.get_balance(
                wallet_address="t1test",
                currency="USDC"
            )


class TestLighthousePaymentIntegration:
    """Integration tests for payment workflow."""

    @pytest.fixture
    def provider(self, tmp_path):
        """Create provider with temp cache."""
        provider, self.mock_http = _make_provider(tmp_path)
        yield provider

    @pytest.mark.asyncio
    async def test_payment_workflow(self, provider):
        """Test complete payment workflow: check balance → pay → verify."""
        # Step 1: Check initial balance
        balance_response = Mock()
        balance_response.status_code = 200
        balance_response.json.return_value = {
            "data": {
                "dataUsed": "1073741824",
                "dataLimit": "5368709120"
            }
        }
        balance_response.raise_for_status = Mock()
        self.mock_http.get = AsyncMock(return_value=balance_response)

        initial_balance = await provider.get_balance("t1test", "USDC")
        assert initial_balance > Decimal("0")

        # Step 2: Make payment
        payment_response = Mock()
        payment_response.status_code = 200
        payment_response.json.return_value = {
            "paymentId": "pay_123",
            "dealId": "deal_456"
        }
        self.mock_http.post = AsyncMock(return_value=payment_response)

        payment_result = await provider.pay_for_storage(
            amount_usd=Decimal("10.00"),
            currency="USDC",
            wallet_address="t1test"
        )

        assert payment_result["status"] == "success"

        # Step 3: Check updated balance
        balance_response2 = Mock()
        balance_response2.status_code = 200
        balance_response2.json.return_value = {
            "data": {
                "dataUsed": "3221225472",
                "dataLimit": "5368709120"
            }
        }
        balance_response2.raise_for_status = Mock()
        self.mock_http.get = AsyncMock(return_value=balance_response2)

        new_balance = await provider.get_balance("t1test", "USDC")
        assert new_balance >= Decimal("0")
