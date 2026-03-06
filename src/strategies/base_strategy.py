"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd


@dataclass
class TradeSignal:
    """A fully-specified trade recommendation from a strategy."""
    symbol: str
    direction: str          # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float       # 0–1 combined TA+ML confidence
    reason: str             # Human-readable entry rationale
    ml_signal: int          # 1 / -1 / 0
    ml_confidence: float
    ta_signal: int          # 1 / -1 / 0
    timeframe: str          # "5min", "15min", "1day"

    def is_long(self) -> bool:
        return self.direction == "long"

    def risk_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    def reward_distance(self) -> float:
        return abs(self.take_profit - self.entry_price)

    def rr_ratio(self) -> float:
        risk = self.risk_distance()
        return self.reward_distance() / risk if risk > 0 else 0


class BaseStrategy(ABC):
    """
    All strategies inherit from this.
    Implement generate_signals() to return a list of TradeSignal objects.
    """

    @abstractmethod
    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[TradeSignal]:
        """
        Analyze bars and return trade signals.

        Args:
            bars: {symbol: DataFrame with OHLCV + feature columns}

        Returns:
            List of TradeSignal objects (may be empty)
        """
        ...

    @abstractmethod
    def should_exit(self, symbol: str, position: dict, current_bars: pd.DataFrame) -> tuple[bool, str]:
        """
        Check if an open position should be closed early (before SL/TP).

        Returns: (should_exit: bool, reason: str)
        """
        ...
