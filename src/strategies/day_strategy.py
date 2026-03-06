"""
Day Trading Strategy — Momentum/Breakout
Timeframe: 5-minute and 15-minute bars
Hours: 9:45 AM – 3:45 PM ET

Entry criteria (all must be met):
  TA:
    - Price within 0.2% of VWAP (for VWAP plays) OR breaking resistance
    - RSI > 50 (uptrend momentum) or < 50 (downtrend) depending on direction
    - Relative volume ≥ 1.5x average
    - Price in allowed range ($5–$500)
  ML:
    - Random Forest signal agrees with TA direction
    - ML confidence ≥ 60%
"""

import logging
import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy, TradeSignal
from src.ml.feature_engineering import compute_features
from src.ml.signal_predictor import SignalPredictor
from src.config import get_config

logger = logging.getLogger(__name__)


class DayStrategy(BaseStrategy):
    def __init__(self):
        cfg = get_config()
        self._cfg = cfg["day_trader"]
        self._predictor = SignalPredictor("day")

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[TradeSignal]:
        """
        Scan all symbols and return momentum/breakout signals.
        bars: {symbol: DataFrame of 5-min or 15-min OHLCV bars}
        """
        signals: list[TradeSignal] = []

        for symbol, df in bars.items():
            if df.empty or len(df) < 30:
                continue
            try:
                signal = self._analyze(symbol, df)
                if signal is not None:
                    signals.append(signal)
            except Exception as exc:
                logger.warning("Signal error for %s: %s", symbol, exc)

        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals

    def _analyze(self, symbol: str, df: pd.DataFrame) -> TradeSignal | None:
        df = compute_features(df)
        if df.empty:
            return None

        latest = df.iloc[-1]
        price = float(latest["close"])

        # -- Price filter --
        if not (self._cfg["min_price"] <= price <= self._cfg["max_price"]):
            return None

        # -- Relative volume filter --
        rel_vol = float(latest.get("volume_ratio_20", 0))
        if rel_vol < self._cfg["min_relative_volume"]:
            return None

        # -- VWAP setup --
        vwap_dev = float(latest.get("vwap_deviation", 0))
        rsi = float(latest.get("rsi_14", 50))
        macd_hist = float(latest.get("macd_hist", 0))

        ta_signal, entry_reason = self._ta_signal(latest, price, rsi, vwap_dev, macd_hist)
        if ta_signal == 0:
            return None

        # -- ML signal --
        ml_signal, ml_conf = self._predictor.predict(df)

        # Both TA and ML must agree (or ML neutral is OK if TA is strong)
        if ml_signal != 0 and ml_signal != ta_signal:
            return None

        combined_conf = (0.6 * ml_conf + 0.4 * min(rel_vol / 3.0, 1.0)) if ml_conf > 0 else 0.5

        if ml_conf > 0 and ml_conf < self._cfg["ml_confidence_threshold"]:
            return None

        # -- Stop and target calculation --
        atr = float(latest.get("atr_pct", 0.01)) * price
        atr = max(atr, price * 0.005)  # Minimum 0.5% ATR

        if ta_signal == 1:  # Long
            stop_loss = price - (1.5 * atr)
            take_profit = price + (3.0 * atr)  # 2:1 R:R minimum
        else:  # Short
            stop_loss = price + (1.5 * atr)
            take_profit = price - (3.0 * atr)

        return TradeSignal(
            symbol=symbol,
            direction="long" if ta_signal == 1 else "short",
            entry_price=price,
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            confidence=round(combined_conf, 3),
            reason=entry_reason,
            ml_signal=ml_signal,
            ml_confidence=ml_conf,
            ta_signal=ta_signal,
            timeframe="5min",
        )

    def _ta_signal(
        self,
        row: pd.Series,
        price: float,
        rsi: float,
        vwap_dev: float,
        macd_hist: float,
    ) -> tuple[int, str]:
        """
        Determine TA signal direction using VWAP, RSI, and MACD.
        Returns (signal: int, reason: str)
        """
        vwap_threshold = self._cfg.get("vwap_deviation_pct", 0.002)
        rsi_min = self._cfg.get("rsi_momentum_min", 50)

        # LONG: price near or above VWAP, RSI > 50, MACD turning positive
        if (
            vwap_dev >= -vwap_threshold  # At or above VWAP
            and rsi >= rsi_min
            and macd_hist > 0
        ):
            return 1, f"VWAP long: RSI={rsi:.0f}, VWAP_dev={vwap_dev:.3f}, MACD_hist={macd_hist:.4f}"

        # LONG breakout: price just broke above recent high, strong RSI
        if rsi >= 60 and macd_hist > 0 and vwap_dev > vwap_threshold:
            return 1, f"Breakout long: RSI={rsi:.0f} above VWAP"

        # SHORT: price below VWAP, RSI < 50, MACD negative
        if (
            vwap_dev <= vwap_threshold
            and rsi < (100 - rsi_min)
            and macd_hist < 0
        ):
            return -1, f"VWAP short: RSI={rsi:.0f}, VWAP_dev={vwap_dev:.3f}"

        return 0, "No signal"

    def should_exit(self, symbol: str, position: dict, current_bars: pd.DataFrame) -> tuple[bool, str]:
        """Check for momentum reversal exits."""
        if current_bars.empty:
            return False, ""

        df = compute_features(current_bars)
        if df.empty:
            return False, ""

        latest = df.iloc[-1]
        rsi = float(latest.get("rsi_14", 50))
        macd_hist = float(latest.get("macd_hist", 0))
        vwap_dev = float(latest.get("vwap_deviation", 0))
        side = position.get("side", "long")

        # Long position: exit if RSI overbought AND MACD diverging
        if side in ("long", "buy") and rsi > self._cfg.get("rsi_overbought", 70) and macd_hist < 0:
            return True, f"Long exit: RSI overbought ({rsi:.0f}) + MACD divergence"

        # Short position: exit if RSI oversold AND MACD turning up
        if side in ("short", "sell") and rsi < self._cfg.get("rsi_oversold", 30) and macd_hist > 0:
            return True, f"Short exit: RSI oversold ({rsi:.0f}) + MACD convergence"

        return False, ""

    def train_ml(self, historical_bars: pd.DataFrame) -> dict:
        """Train (or retrain) the ML model on historical data."""
        logger.info("Training day trader ML model...")
        return self._predictor.train(historical_bars)

    def ml_needs_retraining(self) -> bool:
        return self._predictor.needs_retraining()
