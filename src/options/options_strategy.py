"""
Options Strategy — toggleable via config: options.enabled

When enabled, converts directional stock signals into options trades:
  - Bullish signal → Buy call (~30 delta, 7-30 DTE)
  - Bearish signal → Buy put (~30 delta, 7-30 DTE)

Stop-loss: close at 50% premium loss
No short options (unlimited risk not appropriate for $500 account).
"""

import logging
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.config import get_config, get_alpaca_credentials, is_paper_trading
from src.strategies.base_strategy import TradeSignal

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE = "https://api.alpaca.markets"


class OptionsStrategy:
    """
    Handles options contract selection and order sizing.
    Only active when options.enabled = true in settings.yaml.
    """

    def __init__(self):
        cfg = get_config()
        self._cfg = cfg["options"]
        self._enabled = self._cfg["enabled"]
        self._api_key, self._secret_key = get_alpaca_credentials()
        self._base_url = PAPER_BASE if is_paper_trading() else LIVE_BASE
        self._headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
        }

    def is_enabled(self) -> bool:
        return self._cfg.get("enabled", False)

    def get_option_contract(self, signal: TradeSignal) -> dict | None:
        """
        Find the best option contract for a given directional signal.

        Returns contract dict or None if no suitable contract found.
        """
        if not self.is_enabled():
            return None

        option_type = "call" if signal.is_long() else "put"
        today = datetime.now(ET).date()
        min_expiry = today + timedelta(days=self._cfg["min_dte"])
        max_expiry = today + timedelta(days=self._cfg["max_dte"])

        try:
            contracts = self._fetch_contracts(
                symbol=signal.symbol,
                option_type=option_type,
                min_expiry=min_expiry.isoformat(),
                max_expiry=max_expiry.isoformat(),
            )
        except Exception as exc:
            logger.error("Failed to fetch option contracts for %s: %s", signal.symbol, exc)
            return None

        if not contracts:
            logger.info("No option contracts found for %s %s", signal.symbol, option_type)
            return None

        # Select contract nearest to target delta (0.30)
        target_delta = self._cfg["preferred_delta"]
        best = min(
            contracts,
            key=lambda c: abs(float(c.get("delta", 0)) - target_delta),
            default=None,
        )

        if best is None:
            return None

        # IV check
        iv_pct = float(best.get("implied_volatility", 0))
        if iv_pct > self._cfg["max_iv_percentile"] / 100:
            logger.info(
                "Skipping %s option: IV %.1f%% exceeds max %d%%",
                signal.symbol, iv_pct * 100, self._cfg["max_iv_percentile"]
            )
            return None

        logger.info(
            "Selected %s %s %s delta=%.2f ask=%.2f",
            signal.symbol, best.get("expiration_date"), option_type,
            float(best.get("delta", 0)), float(best.get("ask_price", 0)),
        )
        return best

    def calculate_contracts(self, premium_per_contract: float, capital: float) -> int:
        """
        Calculate number of contracts to buy.
        Each contract = 100 shares. Max 20% of capital per position.
        """
        max_cost = capital * 0.20
        cost_per_contract = premium_per_contract * 100
        if cost_per_contract <= 0:
            return 0
        contracts = int(max_cost / cost_per_contract)
        return max(1, min(contracts, self._cfg["max_contracts_per_trade"]))

    def get_stop_price(self, entry_premium: float) -> float:
        """Close option if premium drops by 50%."""
        return entry_premium * (1 - self._cfg["stop_loss_pct"])

    def _fetch_contracts(
        self,
        symbol: str,
        option_type: str,
        min_expiry: str,
        max_expiry: str,
    ) -> list[dict]:
        """Fetch option contracts from Alpaca Options API."""
        url = f"{self._base_url}/v2/options/contracts"
        params = {
            "underlying_symbol": symbol,
            "type": option_type,
            "expiration_date_gte": min_expiry,
            "expiration_date_lte": max_expiry,
            "status": "active",
            "limit": 50,
        }
        resp = requests.get(url, headers=self._headers, params=params, timeout=10)
        if resp.status_code != 200:
            logger.warning("Options API returned %d: %s", resp.status_code, resp.text[:200])
            return []
        data = resp.json()
        return data.get("option_contracts", [])
