"""
Filecoin Calibration Testnet Adapter

Provides real blockchain integration for wallet operations on the Filecoin
Calibration testnet. This is for testing ONLY - not for mainnet.

Environment Variables:
    FILECOIN_NETWORK: 'calibration' (default) or 'mainnet'
    FILECOIN_RPC_URL: RPC endpoint URL
    FILECOIN_WALLET_ADDRESS: Wallet address for balance checks
    FILECOIN_PRIVATE_KEY: Private key for signing (encrypted)
"""

import asyncio
import hashlib
import logging
import os
from decimal import Decimal
from typing import Optional, Dict, Any
from enum import Enum

import httpx

# HTTP timeout - use framework constant if available, otherwise default to 30s
try:
    from kestrel_sdk.config.constants import HTTP_TIMEOUT_DEFAULT
except ImportError:
    HTTP_TIMEOUT_DEFAULT = 30

logger = logging.getLogger(__name__)


class FilecoinNetwork(Enum):
    """Supported Filecoin networks."""
    CALIBRATION = "calibration"
    MAINNET = "mainnet"


# Network configurations
NETWORK_CONFIG = {
    FilecoinNetwork.CALIBRATION: {
        "rpc_url": "https://api.calibration.node.glif.io/rpc/v1",
        "explorer_url": "https://calibration.filfox.info/en",
        "faucet_url": "https://faucet.calibration.fildev.network/",
        "chain_id": 314159,
    },
    FilecoinNetwork.MAINNET: {
        "rpc_url": "https://api.node.glif.io/rpc/v1",
        "explorer_url": "https://filfox.info/en",
        "faucet_url": None,  # No faucet for mainnet
        "chain_id": 314,
    },
}


class FilecoinTestnetAdapter:
    """
    Filecoin Calibration testnet adapter for real blockchain transactions.

    This adapter connects to the Filecoin Calibration testnet for testing
    wallet functionality with real (test) FIL tokens.

    Usage:
        adapter = FilecoinTestnetAdapter()
        balance = await adapter.get_balance("f1...")
        tx_hash = await adapter.send_fil("f1...", Decimal("1.5"))
    """

    def __init__(
        self,
        network: Optional[str] = None,
        rpc_url: Optional[str] = None,
        wallet_address: Optional[str] = None,
    ):
        """
        Initialize the Filecoin testnet adapter.

        Args:
            network: Network name ('calibration' or 'mainnet'), defaults to env var
            rpc_url: RPC endpoint URL, defaults to env var or network default
            wallet_address: Wallet address for operations, defaults to env var
        """
        # Determine network
        network_str = network or os.environ.get("FILECOIN_NETWORK", "calibration")
        try:
            self.network = FilecoinNetwork(network_str.lower())
        except ValueError:
            logger.warning(f"Unknown network '{network_str}', defaulting to calibration")
            self.network = FilecoinNetwork.CALIBRATION

        # Block mainnet usage until explicitly enabled
        if self.network == FilecoinNetwork.MAINNET:
            if os.environ.get("FILECOIN_MAINNET_ENABLED") != "true":
                raise ValueError(
                    "Mainnet is disabled. Set FILECOIN_MAINNET_ENABLED=true to enable. "
                    "WARNING: This uses REAL money!"
                )

        # Get network config
        config = NETWORK_CONFIG[self.network]

        # Set RPC URL
        self.rpc_url = rpc_url or os.environ.get("FILECOIN_RPC_URL", config["rpc_url"])
        self.explorer_url = config["explorer_url"]
        self.faucet_url = config["faucet_url"]
        self.chain_id = config["chain_id"]

        # Wallet address (optional, for convenience)
        self.wallet_address = wallet_address or os.environ.get("FILECOIN_WALLET_ADDRESS")

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(
            f"FilecoinTestnetAdapter initialized for {self.network.value} network "
            f"(RPC: {self.rpc_url})"
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_DEFAULT)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _rpc_call(self, method: str, params: list = None) -> Any:
        """
        Make a JSON-RPC call to the Filecoin node.

        Args:
            method: RPC method name (e.g., 'Filecoin.WalletBalance')
            params: Method parameters

        Returns:
            RPC result

        Raises:
            Exception: On RPC error
        """
        client = await self._get_client()

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or [],
            "id": 1,
        }

        try:
            response = await client.post(self.rpc_url, json=payload)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                error = data["error"]
                raise Exception(f"RPC error: {error.get('message', error)}")

            return data.get("result")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling {method}: {e}")
            raise

    # =========================================================================
    # Balance Operations
    # =========================================================================

    async def get_balance(self, address: Optional[str] = None) -> Decimal:
        """
        Get the FIL balance for an address.

        Args:
            address: Filecoin address (f1... or f3...), defaults to configured wallet

        Returns:
            Balance in FIL (not attoFIL)

        Raises:
            ValueError: If no address provided and no default configured
        """
        address = address or self.wallet_address
        if not address:
            raise ValueError("No address provided and no default wallet configured")

        try:
            # Filecoin.WalletBalance returns balance in attoFIL (10^-18 FIL)
            result = await self._rpc_call("Filecoin.WalletBalance", [address])

            # Convert attoFIL to FIL
            atto_fil = Decimal(result)
            fil = atto_fil / Decimal("1000000000000000000")

            logger.debug(f"Balance for {address}: {fil} FIL")
            return fil

        except httpx.HTTPError as e:
            logger.error(f"Failed to get balance for {address} (HTTP error): {e}", exc_info=True)
            raise
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Failed to get balance for {address} (data parsing error): {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to get balance for {address}: {e}", exc_info=True)
            raise

    async def get_balance_atto(self, address: Optional[str] = None) -> int:
        """
        Get the balance in attoFIL (smallest unit).

        Args:
            address: Filecoin address

        Returns:
            Balance in attoFIL
        """
        address = address or self.wallet_address
        if not address:
            raise ValueError("No address provided")

        result = await self._rpc_call("Filecoin.WalletBalance", [address])
        return int(result)

    # =========================================================================
    # Transaction Operations (Requires Private Key)
    # =========================================================================

    async def send_fil(
        self,
        to_address: str,
        amount: Decimal,
        from_address: Optional[str] = None,
    ) -> str:
        """
        Send FIL to another address.

        NOTE: This requires a private key to be configured, which is not
        implemented in this initial version for security reasons.

        Args:
            to_address: Recipient address
            amount: Amount in FIL to send
            from_address: Sender address (defaults to configured wallet)

        Returns:
            Transaction CID (hash)

        Raises:
            NotImplementedError: Private key signing not yet implemented
        """
        # For safety, we don't implement actual sending in the initial version.
        # This would require:
        # 1. Private key management (secure storage, encryption)
        # 2. Transaction construction (nonce, gas estimation)
        # 3. Signature generation (secp256k1 or BLS)
        # 4. Transaction broadcast and confirmation

        raise NotImplementedError(
            "Transaction signing not yet implemented. "
            "Use the Filecoin CLI or a wallet app to send FIL, "
            "then record the deposit via !wallet-deposit."
        )

    async def estimate_gas(
        self,
        to_address: str,
        amount: Decimal,
        from_address: Optional[str] = None,
    ) -> Dict[str, Decimal]:
        """
        Estimate gas for a transfer.

        Args:
            to_address: Recipient address
            amount: Amount in FIL
            from_address: Sender address

        Returns:
            Dict with gas estimates (gas_limit, gas_fee_cap, gas_premium)
        """
        from_address = from_address or self.wallet_address
        if not from_address:
            raise ValueError("No from address provided")

        # Convert FIL to attoFIL
        amount_atto = str(int(amount * Decimal("1000000000000000000")))

        # Construct unsigned message
        message = {
            "To": to_address,
            "From": from_address,
            "Value": amount_atto,
            "Method": 0,  # Transfer method
            "Params": "",
        }

        try:
            # Estimate gas
            result = await self._rpc_call("Filecoin.GasEstimateMessageGas", [
                message,
                None,  # Max fee (null = default)
                [],    # Tipset key (empty = latest)
            ])

            return {
                "gas_limit": Decimal(result.get("GasLimit", 0)),
                "gas_fee_cap": Decimal(result.get("GasFeeCap", "0")) / Decimal("1e18"),
                "gas_premium": Decimal(result.get("GasPremium", "0")) / Decimal("1e18"),
            }

        except httpx.HTTPError as e:
            logger.error(f"Failed to estimate gas (HTTP error): {e}", exc_info=True)
            raise
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Failed to estimate gas (data parsing error): {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to estimate gas: {e}", exc_info=True)
            raise

    # =========================================================================
    # Transaction Status
    # =========================================================================

    async def get_transaction(self, tx_cid: str) -> Optional[Dict]:
        """
        Get transaction details by CID.

        Args:
            tx_cid: Transaction CID

        Returns:
            Transaction details or None if not found
        """
        try:
            result = await self._rpc_call("Filecoin.ChainGetMessage", [
                {"/": tx_cid}
            ])
            return result
        except httpx.HTTPError as e:
            logger.error(f"Failed to get transaction {tx_cid} (HTTP error): {e}", exc_info=True)
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Failed to get transaction {tx_cid} (data parsing error): {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Failed to get transaction {tx_cid}: {e}", exc_info=True)
            return None

    async def wait_for_confirmation(
        self,
        tx_cid: str,
        timeout: int = 120,
        confidence: int = 5,
    ) -> bool:
        """
        Wait for a transaction to be confirmed.

        Args:
            tx_cid: Transaction CID to wait for
            timeout: Maximum seconds to wait
            confidence: Number of confirmations required

        Returns:
            True if confirmed, False if timeout
        """
        try:
            # StateWaitMsg waits for message inclusion with specified confidence
            result = await asyncio.wait_for(
                self._rpc_call("Filecoin.StateWaitMsg", [
                    {"/": tx_cid},
                    confidence,
                ]),
                timeout=timeout
            )

            if result and result.get("Receipt"):
                exit_code = result["Receipt"].get("ExitCode", -1)
                return exit_code == 0

            return False

        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for transaction {tx_cid}")
            return False
        except httpx.HTTPError as e:
            logger.error(f"Error waiting for transaction {tx_cid} (HTTP error): {e}", exc_info=True)
            return False
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Error waiting for transaction {tx_cid} (data parsing error): {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Error waiting for transaction {tx_cid}: {e}", exc_info=True)
            return False

    # =========================================================================
    # Network Info
    # =========================================================================

    async def get_chain_head(self) -> Dict:
        """Get the current chain head."""
        return await self._rpc_call("Filecoin.ChainHead", [])

    async def get_network_version(self) -> int:
        """Get the current network version."""
        head = await self.get_chain_head()
        tipset_key = head.get("Cids", [])
        result = await self._rpc_call("Filecoin.StateNetworkVersion", [tipset_key])
        return result

    async def is_connected(self) -> bool:
        """Check if we can connect to the RPC endpoint."""
        try:
            await self.get_chain_head()
            return True
        except (httpx.HTTPError, ConnectionError, TimeoutError):
            return False
        except Exception:
            return False

    def get_explorer_url(self, tx_cid: str) -> str:
        """Get block explorer URL for a transaction."""
        return f"{self.explorer_url}/message/{tx_cid}"

    def get_address_url(self, address: str) -> str:
        """Get block explorer URL for an address."""
        return f"{self.explorer_url}/address/{address}"

    # =========================================================================
    # Faucet (Testnet Only)
    # =========================================================================

    def get_faucet_url(self) -> Optional[str]:
        """
        Get the faucet URL for requesting test FIL.

        Returns:
            Faucet URL or None if not available (mainnet)
        """
        return self.faucet_url

    async def request_test_fil(self, address: Optional[str] = None) -> str:
        """
        Get instructions for requesting test FIL from the faucet.

        Args:
            address: Address to fund (defaults to configured wallet)

        Returns:
            Instructions message
        """
        if self.network != FilecoinNetwork.CALIBRATION:
            return "❌ Faucet is only available on Calibration testnet"

        address = address or self.wallet_address
        if not address:
            return "❌ No address provided. Specify an address or set FILECOIN_WALLET_ADDRESS"

        return f"""🚰 **Request Test FIL**

1. Visit the faucet: {self.faucet_url}
2. Enter your address: `{address}`
3. Complete the captcha and submit
4. Wait ~2 minutes for the transaction to confirm
5. Use `!wallet-deposit` to record the received FIL

Faucet dispenses small amounts of test FIL for development.
View your balance: {self.get_address_url(address)}"""
