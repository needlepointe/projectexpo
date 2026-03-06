"""Base class for all trading bots."""

import logging
from abc import ABC, abstractmethod

from src.risk.risk_manager import RiskManager
from src.execution.alpaca_client import AlpacaClient
from src.data.market_data import MarketDataClient
from src.database import init_db, log_event

logger = logging.getLogger(__name__)


class BaseBot(ABC):
    def __init__(self, bot_type: str, starting_capital: float):
        self.bot_type = bot_type
        self.starting_capital = starting_capital
        self._running = False

        init_db()
        self._risk = RiskManager(bot_type, starting_capital)
        self._client = AlpacaClient()
        self._data = MarketDataClient()

        # Sync capital from Alpaca account on startup
        try:
            equity = self._client.get_equity()
            self._risk.set_capital(equity)
            logger.info("[%s] Initialized. Equity: $%.2f", bot_type.upper(), equity)
        except Exception as exc:
            logger.warning("[%s] Could not fetch equity: %s. Using starting capital.", bot_type.upper(), exc)

    @abstractmethod
    def run_cycle(self):
        """Execute one trading cycle (called by scheduler)."""
        ...

    @abstractmethod
    def on_market_open(self):
        """Called once when market opens."""
        ...

    @abstractmethod
    def on_market_close(self):
        """Called before market closes. Must close all positions for day trader."""
        ...

    def start(self):
        self._running = True
        log_event("start", f"{self.bot_type} bot started", self.bot_type)
        logger.info("[%s] Bot started.", self.bot_type.upper())

    def stop(self):
        self._running = False
        log_event("stop", f"{self.bot_type} bot stopped", self.bot_type)
        logger.info("[%s] Bot stopped.", self.bot_type.upper())

    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict:
        return {
            "bot_type": self.bot_type,
            "running": self._running,
            **self._risk.status(),
        }
