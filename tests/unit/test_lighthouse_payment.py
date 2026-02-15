"""
Unit tests for Lighthouse payment methods.

Tests pay_for_storage() and get_balance() functionality with mocked SDK.
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from kestrel_sovereign.storage.providers.lighthouse_provider import (
    LighthouseProvider,
    LIGHTHOUSE_PERPETUAL_COST_PER_GB,
)


class TestLighthousePayment:
    """Tests for Lighthouse payment functionality."""

    @pytest.fixture
    def provider(self, tmp_path):
        """Create provider with temp cache."""
        with patch("kestrel_sovereign.storage.providers.lighthouse_provider.LIGHTHOUSE_AVAILABLE", True):
            with patch("kestrel_sovereign.storage.providers.lighthouse_provider.Lighthouse") as mock_lighthouse:
                # Mock the Lighthouse client
                mock_client = Mock()
                mock_lighthouse.return_value = mock_client

                provider = LighthouseProvider(
                    api_key="test-api-key",
                    cache_dir=str(tmp_path)
                )
                provider._client = mock_client
                provider._available = True

                yield provider

    @pytest.mark.asyncio
    async def test_pay_for_storage_success(self, provider):
        """Payment should succeed with valid parameters."""
        with patch("requests.post") as mock_post:
            # Mock successful payment response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "paymentId": "payment_123",
                "dealId": "deal_456",
                "expiresAt": "2027-02-15T00:00:00Z"
            }
            mock_post.return_value = mock_response

            result = await provider.pay_for_storage(
                amount_usd=Decimal("20.00"),
                currency="USDC",
                wallet_address="t1test123"
            )

            # Verify result
            assert result["status"] == "success"
            assert result["payment_id"] == "payment_123"
            assert result["deal_id"] == "deal_456"
            assert result["currency"] == "USDC"
            assert result["amount_usd"] == "20.00"
            assert result["wallet_address"] == "t1test123"

            # Verify API was called correctly
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            assert "json" in call_kwargs
            assert call_kwargs["json"]["currency"] == "USDC"
            assert call_kwargs["json"]["wallet_address"] == "t1test123"

    @pytest.mark.asyncio
    async def test_pay_for_storage_with_fil(self, provider):
        """Payment with FIL should convert USD to FIL."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "paymentId": "payment_fil",
                "dealId": "deal_fil",
            }
            mock_post.return_value = mock_response

            result = await provider.pay_for_storage(
                amount_usd=Decimal("55.00"),  # Should be ~10 FIL at $5.50/FIL
                currency="FIL",
                wallet_address="t1test456"
            )

            assert result["status"] == "success"
            assert result["currency"] == "FIL"

            # Check the amount was converted (should be 10 FIL)
            call_kwargs = mock_post.call_args.kwargs
            amount_sent = Decimal(call_kwargs["json"]["amount"])
            assert amount_sent == Decimal("10")  # 55 / 5.50 = 10

    @pytest.mark.asyncio
    async def test_pay_for_storage_api_failure(self, provider):
        """Payment should handle API errors gracefully."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {
                "error": "Insufficient balance"
            }
            mock_post.return_value = mock_response

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
                currency="BTC",  # Not supported
                wallet_address="t1test"
            )

    @pytest.mark.asyncio
    async def test_pay_for_storage_network_error(self, provider):
        """Payment should handle network errors."""
        with patch("requests.post") as mock_post:
            import requests
            mock_post.side_effect = requests.RequestException("Network timeout")

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
        with patch("kestrel_sovereign.storage.providers.lighthouse_provider.LIGHTHOUSE_AVAILABLE", True):
            with patch("kestrel_sovereign.storage.providers.lighthouse_provider.Lighthouse") as mock_lighthouse:
                mock_client = Mock()
                mock_lighthouse.return_value = mock_client

                provider = LighthouseProvider(
                    api_key="test-api-key",
                    cache_dir=str(tmp_path)
                )
                provider._client = mock_client
                provider._available = True

                yield provider

    @pytest.mark.asyncio
    async def test_get_balance_success(self, provider):
        """Balance check should return available storage quota."""
        # Mock SDK response
        provider._client.getBalance.return_value = {
            "data": {
                "dataUsed": "1073741824",  # 1 GB used
                "dataLimit": "5368709120"  # 5 GB limit
            }
        }

        balance = await provider.get_balance(
            wallet_address="t1test",
            currency="USDC"
        )

        # 4 GB available * $4/GB = $16 in USDC
        assert balance == Decimal("16")

    @pytest.mark.asyncio
    async def test_get_balance_with_fil(self, provider):
        """Balance check should convert to FIL."""
        provider._client.getBalance.return_value = {
            "data": {
                "dataUsed": "0",
                "dataLimit": "5368709120"  # 5 GB
            }
        }

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
        provider._client.getBalance.return_value = {
            "data": {
                "dataUsed": "5368709120",  # 5 GB used
                "dataLimit": "5368709120"  # 5 GB limit
            }
        }

        balance = await provider.get_balance(
            wallet_address="t1test",
            currency="USDC"
        )

        assert balance == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_balance_over_quota(self, provider):
        """Balance should be 0 when over quota."""
        provider._client.getBalance.return_value = {
            "data": {
                "dataUsed": "6442450944",  # 6 GB used
                "dataLimit": "5368709120"  # 5 GB limit
            }
        }

        balance = await provider.get_balance(
            wallet_address="t1test",
            currency="USDC"
        )

        assert balance == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_balance_api_error(self, provider):
        """Balance check should return 0 on API error."""
        provider._client.getBalance.side_effect = Exception("API error")

        balance = await provider.get_balance(
            wallet_address="t1test",
            currency="USDC"
        )

        # Graceful fallback
        assert balance == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_balance_unexpected_format(self, provider):
        """Balance check should handle unexpected response format."""
        provider._client.getBalance.return_value = {
            "unexpected": "format"
        }

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
        with patch("kestrel_sovereign.storage.providers.lighthouse_provider.LIGHTHOUSE_AVAILABLE", True):
            with patch("kestrel_sovereign.storage.providers.lighthouse_provider.Lighthouse") as mock_lighthouse:
                mock_client = Mock()
                mock_lighthouse.return_value = mock_client

                provider = LighthouseProvider(
                    api_key="test-api-key",
                    cache_dir=str(tmp_path)
                )
                provider._client = mock_client
                provider._available = True

                yield provider

    @pytest.mark.asyncio
    async def test_payment_workflow(self, provider):
        """Test complete payment workflow: check balance → pay → verify."""
        # Step 1: Check initial balance
        provider._client.getBalance.return_value = {
            "data": {
                "dataUsed": "1073741824",
                "dataLimit": "5368709120"
            }
        }

        initial_balance = await provider.get_balance("t1test", "USDC")
        assert initial_balance > Decimal("0")

        # Step 2: Make payment
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "paymentId": "pay_123",
                "dealId": "deal_456"
            }
            mock_post.return_value = mock_response

            payment_result = await provider.pay_for_storage(
                amount_usd=Decimal("10.00"),
                currency="USDC",
                wallet_address="t1test"
            )

            assert payment_result["status"] == "success"

        # Step 3: Check updated balance (would be lower in real scenario)
        provider._client.getBalance.return_value = {
            "data": {
                "dataUsed": "3221225472",  # More used now
                "dataLimit": "5368709120"
            }
        }

        new_balance = await provider.get_balance("t1test", "USDC")
        # In real scenario, new_balance < initial_balance
        assert new_balance >= Decimal("0")
