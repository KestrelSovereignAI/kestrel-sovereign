"""
Unit tests for Filecoin miner selection logic.

Tests the improved _find_suitable_miner() implementation with filtering
and scoring based on miner characteristics.
"""

import pytest
from unittest.mock import Mock, patch
from decimal import Decimal

from kestrel_sovereign.filecoin_adapter import FilecoinAdapter


class TestMinerSelection:
    """Tests for miner selection algorithm."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked Lotus client."""
        with patch("kestrel_sovereign.filecoin_adapter.LotusClient"):
            with patch("kestrel_sovereign.filecoin_adapter.HttpJsonRpcConnector"):
                adapter = FilecoinAdapter(
                    lotus_rpc_url="http://test:1234/rpc/v0",
                    lotus_token="test-token"
                )

                # Mock successful initialization
                mock_client = Mock()
                adapter.lotus_client = mock_client
                adapter._lotus_available = True

                yield adapter

    def test_find_suitable_miner_no_lotus(self):
        """Should return None when Lotus is unavailable."""
        adapter = FilecoinAdapter()
        adapter._lotus_available = False

        result = adapter._find_suitable_miner(size_bytes=1024)

        assert result is None

    def test_find_suitable_miner_no_miners_found(self, adapter):
        """Should return None when no miners on network."""
        adapter.lotus_client.StateListMiners.return_value = []

        result = adapter._find_suitable_miner(size_bytes=1024)

        assert result is None

    def test_find_suitable_miner_filters_inactive(self, adapter):
        """Should filter out miners with no power."""
        # Mock 3 miners, only one active
        adapter.lotus_client.StateListMiners.return_value = [
            "t01000",  # Active
            "t01001",  # Inactive (no power)
            "t01002",  # Inactive (no power)
        ]

        # Mock miner info responses
        def mock_miner_info(miner):
            return {
                "SectorCount": 100,
            }

        def mock_miner_power(miner):
            if miner == "t01000":
                return {
                    "MinerPower": {
                        "RawBytePower": "1099511627776"  # 1 TiB
                    }
                }
            else:
                return {
                    "MinerPower": {
                        "RawBytePower": "0"  # No power
                    }
                }

        def mock_available_balance(miner):
            return "1000000000000000000"  # 1 FIL

        adapter.lotus_client.StateMinerInfo.side_effect = mock_miner_info
        adapter.lotus_client.StateMinerPower.side_effect = mock_miner_power
        adapter.lotus_client.StateMinerAvailableBalance.side_effect = mock_available_balance

        result = adapter._find_suitable_miner(size_bytes=1024)

        # Should select the only active miner
        assert result == "t01000"

    def test_find_suitable_miner_filters_no_balance(self, adapter):
        """Should filter out miners with zero balance."""
        adapter.lotus_client.StateListMiners.return_value = [
            "t01000",  # Has balance
            "t01001",  # No balance
        ]

        def mock_miner_info(miner):
            return {"SectorCount": 100}

        def mock_miner_power(miner):
            return {
                "MinerPower": {
                    "RawBytePower": "1099511627776"  # 1 TiB
                }
            }

        def mock_available_balance(miner):
            if miner == "t01000":
                return "1000000000000000000"  # 1 FIL
            else:
                return "0"  # No balance

        adapter.lotus_client.StateMinerInfo.side_effect = mock_miner_info
        adapter.lotus_client.StateMinerPower.side_effect = mock_miner_power
        adapter.lotus_client.StateMinerAvailableBalance.side_effect = mock_available_balance

        result = adapter._find_suitable_miner(size_bytes=1024)

        assert result == "t01000"

    def test_find_suitable_miner_selects_highest_score(self, adapter):
        """Should select miner with highest score."""
        adapter.lotus_client.StateListMiners.return_value = [
            "t01000",  # Low score (small power)
            "t01001",  # High score (large power, many sectors)
            "t01002",  # Medium score
        ]

        def mock_miner_info(miner):
            if miner == "t01000":
                return {"SectorCount": 10}
            elif miner == "t01001":
                return {"SectorCount": 1000}  # Lots of sectors
            else:
                return {"SectorCount": 100}

        def mock_miner_power(miner):
            if miner == "t01000":
                power = "1099511627776"  # 1 TiB
            elif miner == "t01001":
                power = "1099511627776000"  # 1000 TiB (large)
            else:
                power = "10995116277760"  # 10 TiB

            return {
                "MinerPower": {
                    "RawBytePower": power
                }
            }

        def mock_available_balance(miner):
            if miner == "t01001":
                return "10000000000000000000"  # 10 FIL
            else:
                return "1000000000000000000"  # 1 FIL

        adapter.lotus_client.StateMinerInfo.side_effect = mock_miner_info
        adapter.lotus_client.StateMinerPower.side_effect = mock_miner_power
        adapter.lotus_client.StateMinerAvailableBalance.side_effect = mock_available_balance

        result = adapter._find_suitable_miner(size_bytes=1024)

        # Should select t01001 (highest power, sectors, balance)
        assert result == "t01001"

    def test_find_suitable_miner_handles_api_errors(self, adapter):
        """Should handle API errors gracefully."""
        adapter.lotus_client.StateListMiners.return_value = [
            "t01000",
            "t01001",
        ]

        # Mock API errors for one miner
        def mock_miner_info(miner):
            if miner == "t01000":
                raise Exception("API error")
            return {"SectorCount": 100}

        def mock_miner_power(miner):
            return {
                "MinerPower": {
                    "RawBytePower": "1099511627776"
                }
            }

        def mock_available_balance(miner):
            return "1000000000000000000"

        adapter.lotus_client.StateMinerInfo.side_effect = mock_miner_info
        adapter.lotus_client.StateMinerPower.side_effect = mock_miner_power
        adapter.lotus_client.StateMinerAvailableBalance.side_effect = mock_available_balance

        result = adapter._find_suitable_miner(size_bytes=1024)

        # Should skip t01000 and select t01001
        assert result == "t01001"

    def test_find_suitable_miner_returns_none_when_all_fail_scoring(self, adapter):
        """#1676: when no miner can be vetted, return None (fail closed) rather
        than an unvetted fallback — the caller raises 'no suitable miner'."""
        adapter.lotus_client.StateListMiners.return_value = ["t01000", "t01001"]

        # Mock all miners fail scoring
        adapter.lotus_client.StateMinerInfo.side_effect = Exception("API error")
        adapter.lotus_client.StateMinerPower.side_effect = Exception("API error")

        result = adapter._find_suitable_miner(size_bytes=1024)

        # No vetted miner -> None, not an unvetted fallback.
        assert result is None

    def test_find_suitable_miner_excludes_miner_on_balance_check_failure(self, adapter):
        """#1679: a miner whose balance check fails is treated as unfit and
        excluded (fail closed), not allowed to fall through as suitable."""
        adapter.lotus_client.StateListMiners.return_value = ["t01000"]

        def mock_miner_info(miner):
            return {"SectorCount": 100}

        def mock_miner_power(miner):
            return {
                "MinerPower": {
                    "RawBytePower": "1099511627776"
                }
            }

        adapter.lotus_client.StateMinerInfo.side_effect = mock_miner_info
        adapter.lotus_client.StateMinerPower.side_effect = mock_miner_power
        # Balance check fails -> the only candidate is excluded -> None.
        adapter.lotus_client.StateMinerAvailableBalance.side_effect = Exception("Balance API error")

        result = adapter._find_suitable_miner(size_bytes=1024)

        # The unverifiable miner is excluded; no suitable miner remains.
        assert result is None

    def test_find_suitable_miner_score_calculation(self, adapter):
        """Test score calculation gives reasonable values."""
        adapter.lotus_client.StateListMiners.return_value = ["t01000"]

        adapter.lotus_client.StateMinerInfo.return_value = {
            "SectorCount": 500  # Should contribute to score
        }

        adapter.lotus_client.StateMinerPower.return_value = {
            "MinerPower": {
                "RawBytePower": "1099511627776"  # 1 TiB
            }
        }

        adapter.lotus_client.StateMinerAvailableBalance.return_value = "5000000000000000000"  # 5 FIL

        result = adapter._find_suitable_miner(size_bytes=1024)

        # Should succeed
        assert result == "t01000"


class TestMinerSelectionIntegration:
    """Integration tests for miner selection in storage workflow."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked dependencies."""
        with patch("kestrel_sovereign.filecoin_adapter.LotusClient"):
            with patch("kestrel_sovereign.filecoin_adapter.HttpJsonRpcConnector"):
                adapter = FilecoinAdapter()
                mock_client = Mock()
                adapter.lotus_client = mock_client
                adapter._lotus_available = True
                adapter._ipfs_available = True

                yield adapter

    def test_create_deal_uses_miner_selection(self, adapter):
        """Creating a deal should use improved miner selection."""
        # Mock miner selection
        adapter.lotus_client.StateListMiners.return_value = ["t01000", "t01001"]

        def mock_miner_info(miner):
            if miner == "t01000":
                return {"SectorCount": 10}  # Few sectors
            else:
                return {"SectorCount": 1000}  # Many sectors (better)

        def mock_miner_power(miner):
            if miner == "t01000":
                power = "1099511627776"  # 1 TiB
            else:
                power = "1099511627776000"  # 1000 TiB (much better)

            return {
                "MinerPower": {
                    "RawBytePower": power
                }
            }

        def mock_available_balance(miner):
            if miner == "t01000":
                return "1000000000000000000"  # 1 FIL
            else:
                return "10000000000000000000"  # 10 FIL (better)

        adapter.lotus_client.StateMinerInfo.side_effect = mock_miner_info
        adapter.lotus_client.StateMinerPower.side_effect = mock_miner_power
        adapter.lotus_client.StateMinerAvailableBalance.side_effect = mock_available_balance

        # Mock other required methods
        adapter.lotus_client.WalletDefaultAddress.return_value = "t1wallet"
        adapter.lotus_client.ClientStartDeal.return_value = {"/": "bafy123deal"}

        # Create deal with size metadata
        metadata = {"size_bytes": 1073741824}  # 1 GB
        deal_id = adapter._create_filecoin_deal("QmTest123", metadata)

        # Should use the better miner (t01001)
        # Verify by checking ClientStartDeal was called with t01001
        call_args = adapter.lotus_client.ClientStartDeal.call_args
        proposal = call_args[0][0]
        assert proposal["Miner"] == "t01001"

        assert deal_id == "bafy123deal"

    def test_create_deal_without_size_metadata(self, adapter):
        """Creating a deal without size should still work."""
        adapter.lotus_client.StateListMiners.return_value = ["t01000"]

        adapter.lotus_client.StateMinerInfo.return_value = {"SectorCount": 100}
        adapter.lotus_client.StateMinerPower.return_value = {
            "MinerPower": {
                "RawBytePower": "1099511627776"
            }
        }
        adapter.lotus_client.StateMinerAvailableBalance.return_value = "1000000000000000000"
        adapter.lotus_client.WalletDefaultAddress.return_value = "t1wallet"
        adapter.lotus_client.ClientStartDeal.return_value = {"/": "bafy123"}

        # No size in metadata
        deal_id = adapter._create_filecoin_deal("QmTest123", metadata=None)

        assert deal_id == "bafy123"


class TestMinerSelectionEdgeCases:
    """Test edge cases in miner selection."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked Lotus client."""
        with patch("kestrel_sovereign.filecoin_adapter.LotusClient"):
            with patch("kestrel_sovereign.filecoin_adapter.HttpJsonRpcConnector"):
                adapter = FilecoinAdapter()
                mock_client = Mock()
                adapter.lotus_client = mock_client
                adapter._lotus_available = True

                yield adapter

    def test_all_miners_filtered_out(self, adapter):
        """#1676: when every miner is unsuitable, return None (fail closed),
        not an unvetted fallback."""
        adapter.lotus_client.StateListMiners.return_value = [
            "t01000",
            "t01001",
        ]

        # All miners have no power
        def mock_miner_power(miner):
            return {
                "MinerPower": {
                    "RawBytePower": "0"
                }
            }

        adapter.lotus_client.StateMinerInfo.return_value = {"SectorCount": 0}
        adapter.lotus_client.StateMinerPower.side_effect = mock_miner_power

        result = adapter._find_suitable_miner(size_bytes=1024)

        # No suitable miner -> None, not a fallback to an unvetted one.
        assert result is None

    def test_single_miner_network(self, adapter):
        """Should work with single miner networks."""
        adapter.lotus_client.StateListMiners.return_value = ["t01000"]

        adapter.lotus_client.StateMinerInfo.return_value = {"SectorCount": 100}
        adapter.lotus_client.StateMinerPower.return_value = {
            "MinerPower": {
                "RawBytePower": "1099511627776"
            }
        }
        adapter.lotus_client.StateMinerAvailableBalance.return_value = "1000000000000000000"

        result = adapter._find_suitable_miner(size_bytes=1024)

        assert result == "t01000"

    def test_large_file_size(self, adapter):
        """Should handle large file sizes correctly."""
        adapter.lotus_client.StateListMiners.return_value = ["t01000"]

        adapter.lotus_client.StateMinerInfo.return_value = {"SectorCount": 1000}
        adapter.lotus_client.StateMinerPower.return_value = {
            "MinerPower": {
                "RawBytePower": "1099511627776000"  # 1 PiB
            }
        }
        adapter.lotus_client.StateMinerAvailableBalance.return_value = "1000000000000000000"

        # 100 GB file
        result = adapter._find_suitable_miner(size_bytes=107374182400)

        assert result == "t01000"
