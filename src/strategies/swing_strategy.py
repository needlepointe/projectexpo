"""
Swing Trading Strategy — Trend Following + Momentum
Timeframe: Daily bars
Hold time: 2–10 days

Entry criteria:
  TA:
    - Uptrend: 20 MA > 50 MA, price above both (for longs)
    - Pullback entry: RSI ≤ 45 while in uptrend (buy the dip)
    - OR Breakout entry: price closes above 20-day high on 1.3x volume
  ML:
    - Signal agrees with TA
    - Confidence ≥ 55%
"""

import logging
import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy, TradeSignal
from src.ml.feature_engineering import compute_features
from src.ml.signal_predictor import SignalPredictor
from src.config import get_config

logger = logging.getLogger(__name__)


class SwingStrategy(BaseStrategy):
    def __init__(self):
        cfg = get_config()
        self._cfg = cfg["swing_trader"]
        self._predictor = SignalPredictor("swing")

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[TradeSignal]:
        signals: list[TradeSignal] = []

        for symbol, df in bars.items():
            if df.empty or len(df) < 55:
                continue
            try:
                signal = self._analyze(symbol, df)
                if signal is not None:
                    signals.append(signal)
            except Exception as exc:
                logger.warning("Swing signal error for %s: %s", symbol, exc)

        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals

    def _analyze(self, symbol: str, df: pd.DataFrame) -> TradeSignal | None:
        df = compute_features(df)
        if df.empty or len(df) < 55:
            return None

        latest = df.iloc[-1]
        price = float(latest["close"])
        ma20 = float(latest.get("ma_20", 0))
        ma50 = float(latest.get("ma_50", 0))
        rsi = float(latest.get("rsi_14", 50))
        rel_vol = float(latest.get("volume_ratio_20", 1.0))

        ta_signal, entry_type, reason = self._ta_signal(
            df, latest, price, ma20, ma50, rsi, rel_vol
        )
        if ta_signal == 0:
            return None

        # ML confirmation
        ml_signal, ml_conf = self._predictor.predict(df)
        if ml_signal != 0 and ml_signal != ta_signal:
            return None
        if ml_conf > 0 and ml_conf < self._cfg["ml_confidence_threshold"]:
            return None

        # Stop-loss: below recent swing low (or below 20 MA for pullbacks)
        atr = float(latest.get("atr_pct", 0.015)) * price
        atr = max(atr, price * 0.01)

        if ta_signal == 1:  # Long
            if entry_type == "pullback":
                stop_loss = min(ma20 - atr, price - 2 * atr)
            else:  # breakout
                # Stop below the breakout level
                breakout_level = float(df["high"].iloc[-20:].max())
                stop_loss = breakout_level - atr
            stop_loss = min(stop_loss, price - atr)  # At least 1 ATR below entry
            take_profit = price + abs(price - stop_loss) * 2.5  # 2.5:1 R:R for swings
        else:  # Short
            stop_loss = max(ma20 + atr, price + 2 * atr)
            take_profit = price - abs(stop_loss - price) * 2.5

        combined_conf = (0.5 * ml_conf + 0.3 * min(rel_vol, 2) / 2 + 0.2) if ml_conf > 0 else 0.5

        return TradeSignal(
            symbol=symbol,
            direction="long" if ta_signal == 1 else "short",
            entry_price=price,
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            confidence=round(combined_conf, 3),
            reason=reason,
            ml_signal=ml_signal,
            ml_confidence=ml_conf,
            ta_signal=ta_signal,
            timeframe="1day",
        )

    def _ta_signal(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        price: float,
        ma20: float,
        ma50: float,
        rsi: float,
        rel_vol: float,
    ) -> tuple[int, str, str]:
        """Returns (signal, entry_type, reason)."""
        cfg = self._cfg

        uptrend = ma20 > ma50 and price > ma20
        downtrend = ma20 < ma50 and price < ma20

        # -- Pullback to MA in uptrend (strongest setup) --
        if uptrend and rsi <= cfg.get("rsi_pullback_max", 45) and rsi >= 25:
            return 1, "pullback", (
                f"MA pullback long: price={price:.2f}, MA20={ma20:.2f}, "
                f"RSI={rsi:.0f} (oversold in uptrend)"
            )

        # -- Breakout above 20-day high --
        high_20d = float(df["high"].iloc[-21:-1].max()) if len(df) > 21 else price
        if (
            price > high_20d
            and rel_vol >= cfg.get("breakout_volume_ratio", 1.3)
            and rsi > 50
        ):
            return 1, "breakout", (
                f"Breakout long: price={price:.2f} > 20d high={high_20d:.2f}, "
                f"rel_vol={rel_vol:.1f}x, RSI={rsi:.0f}"
            )

        # -- Short: breakdown below 20-day low in downtrend --
        low_20d = float(df["low"].iloc[-21:-1].min()) if len(df) > 21 else price
        if (
            downtrend
            and price < low_20d
            and rel_vol >= cfg.get("breakout_volume_ratio", 1.3)
            and rsi < 50
        ):
            return -1, "breakdown", (
                f"Breakdown short: price={price:.2f} < 20d low={low_20d:.2f}, "
                f"RSI={rsi:.0f}"
            )

        return 0, "", "No signal"

    def should_exit(self, symbol: str, position: dict, current_bars: pd.DataFrame) -> tuple[bool, str]:
        """
        Swing exit triggers:
          - MA cross against position
          - RSI extreme (overbought longs, oversold shorts)
          - Price closes back below breakout level
        """
        if current_bars.empty or len(current_bars) < 20:
            return False, ""

        df = compute_features(current_bars)
        latest = df.iloc[-1]

        rsi = float(latest.get("rsi_14", 50))
        ma20 = float(latest.get("ma_20", 0))
        price = float(latest["close"])
        side = position.get("side", "long")
        entry_price = float(position.get("avg_entry_price", price))

        if side in ("long", "buy"):
            # Exit if RSI overbought (taking profits into strength)
            if rsi > 75:
                return True, f"Swing exit: RSI overbought ({rsi:.0f})"
            # Exit if price closes back below MA20 (trend reversal)
            if price < ma20 and (entry_price - price) / entry_price < -0.02:
                return True, "Swing exit: Price closed below MA20"

        elif side in ("short", "sell"):
            if rsi < 25:
                return True, f"Swing exit: RSI oversold ({rsi:.0f})"
            if price > ma20:
                return True, "Swing exit: Price closed back above MA20"

        return False, ""

    def train_ml(self, historical_bars: pd.DataFrame) -> dict:
        logger.info("Training swing trader ML model...")
        return self._predictor.train(historical_bars)

    def ml_needs_retraining(self) -> bool:
        return self._predictor.needs_retraining()
