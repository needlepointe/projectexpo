"""
AB Trades Strategy
Replicates the swing trading style of @ABTradess (OWLS Discord).

Core methodology (from public information):
  - Market structure analysis: higher highs / higher lows, break of structure
  - Fibonacci retracements: 38.2%, 50%, 61.8% pullback zones
  - ABC corrections: 3-wave pullback in uptrend before continuation
  - Breakouts: price breaks above 20-day swing high with volume confirmation
  - EMA trend filter: EMA20 > EMA50 > EMA200

Entry types:
  1. FIBONACCI PULLBACK — price retraces to 38.2–61.8% fib level in uptrend
                          after an ABC correction completes at support
  2. BREAKOUT            — price closes above 20-day high with vol ≥ 1.5× avg
"""

import logging

import pandas as pd
import numpy as np

from src.strategies.base_strategy import BaseStrategy, TradeSignal
from src.ml.feature_engineering import compute_features
from src.ml.signal_predictor import SignalPredictor
from src.config import get_config

logger = logging.getLogger(__name__)


class ABStrategy(BaseStrategy):
    """
    AB Trades-style swing strategy using market structure + Fibonacci + ABC patterns.
    """

    def __init__(self):
        cfg = get_config()
        self._cfg = cfg.get("ab_trader", {})
        self._predictor = SignalPredictor("ab")

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[TradeSignal]:
        """Scan a dict of {symbol: DataFrame} and return ranked trade signals."""
        signals = []
        for symbol, df in bars.items():
            if df.empty or len(df) < 60:
                continue
            try:
                signal = self._analyze(symbol, df)
                if signal:
                    signals.append(signal)
            except Exception as exc:
                logger.warning("[AB] Signal error %s: %s", symbol, exc)

        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals

    def should_exit(self, symbol: str, position: dict, current_bars: pd.DataFrame) -> tuple[bool, str]:
        """Exit early if trend breaks (EMA20 cross) or RSI extreme reached."""
        if current_bars.empty or len(current_bars) < 22:
            return False, ""
        try:
            df = compute_features(current_bars)
            latest = df.iloc[-1]
            price = float(latest["close"])
            ema20 = float(latest.get("ema_20", 0))
            rsi = float(latest.get("rsi_14", 50))
            side = position.get("side", "long")

            # Long exit: price drops below EMA20 (trend broken) or RSI overbought
            if side == "long":
                if ema20 > 0 and price < ema20 * 0.99:
                    return True, "price_below_ema20"
                if rsi > 78:
                    return True, "rsi_overbought"

            # Short exit: price rises above EMA20 or RSI oversold
            if side == "short":
                if ema20 > 0 and price > ema20 * 1.01:
                    return True, "price_above_ema20"
                if rsi < 22:
                    return True, "rsi_oversold"
        except Exception as exc:
            logger.debug("[AB] should_exit error %s: %s", symbol, exc)
        return False, ""

    def train_ml(self, historical_bars: pd.DataFrame) -> dict:
        return self._predictor.train(historical_bars)

    def ml_needs_retraining(self) -> bool:
        return self._predictor.needs_retraining()

    # ------------------------------------------------------------------
    # Core signal analysis
    # ------------------------------------------------------------------

    def _analyze(self, symbol: str, df: pd.DataFrame) -> TradeSignal | None:
        df = compute_features(df)
        df = self._add_ema(df)
        latest = df.iloc[-1]

        price = float(latest["close"])
        atr = float(latest.get("atr_pct", 0.02)) * price
        if atr <= 0:
            atr = price * 0.02

        ema20 = float(latest.get("ema_20", 0))
        ema50 = float(latest.get("ema_50", 0))
        ema200 = float(latest.get("ema_200", 0))
        rsi = float(latest.get("rsi_14", 50))
        macd_hist = float(latest.get("macd_hist", 0))
        volume_ratio = float(latest.get("volume_ratio_20", 1.0))

        # ── Trend structure ─────────────────────────────────────────────
        strong_uptrend = (ema20 > 0 and ema50 > 0 and ema200 > 0
                          and ema20 > ema50 and price > ema200)
        strong_downtrend = (ema20 > 0 and ema50 > 0 and ema200 > 0
                            and ema20 < ema50 and price < ema200)

        # ── Market structure ────────────────────────────────────────────
        ms = self._compute_market_structure(df)

        # ── Fibonacci levels ─────────────────────────────────────────────
        fib = self._compute_fibonacci(df)

        # ── ABC pattern ──────────────────────────────────────────────────
        abc_complete = self._detect_abc_pattern(df)

        # ── Breakout ──────────────────────────────────────────────────────
        breakout_vol_ratio = self._cfg.get("breakout_volume_ratio", 1.5)
        is_breakout = (
            ms.get("bos_bull", False)
            and volume_ratio >= breakout_vol_ratio
            and rsi >= 50
        )

        # ── LONG signal ──────────────────────────────────────────────────
        fib_entry_long = (
            fib.get("at_fib_zone", False)
            and abc_complete
            and 32 <= rsi <= 52
            and macd_hist > -0.5 * abs(macd_hist + 0.001)  # MACD not deeply negative
            and ms.get("higher_low", False)
        )

        if strong_uptrend and (fib_entry_long or is_breakout):
            stop = price - (1.0 * atr)
            target = price + (2.0 * atr)
            if stop <= 0 or target <= price:
                return None

            reason_parts = []
            if fib_entry_long:
                reason_parts.append(f"fib_pullback({fib.get('nearest_fib_pct', 0):.0%})")
            if abc_complete:
                reason_parts.append("abc_complete")
            if is_breakout:
                reason_parts.append("breakout")
            if ms.get("higher_low"):
                reason_parts.append("HL_structure")
            reason = " | ".join(reason_parts) or "ab_long"

            # ML confirmation (veto only — don't require agreement)
            ml_signal, ml_conf = self._predictor.predict(df)
            if ml_signal == -1 and ml_conf > 0.65:
                return None  # ML strongly disagrees

            combined_conf = 0.55 + (0.10 if fib_entry_long else 0) + (0.10 if is_breakout else 0)
            combined_conf = min(combined_conf, 0.95)

            return TradeSignal(
                symbol=symbol,
                direction="long",
                entry_price=price,
                stop_loss=stop,
                take_profit=target,
                confidence=combined_conf,
                reason=reason,
                ml_signal=ml_signal,
                ml_confidence=ml_conf,
                ta_signal=1,
                timeframe="1day",
            )

        # ── SHORT signal ─────────────────────────────────────────────────
        fib_entry_short = (
            fib.get("at_fib_resistance", False)
            and 52 <= rsi <= 68
            and macd_hist < 0
            and ms.get("lower_high", False)
        )

        if strong_downtrend and fib_entry_short:
            stop = price + (1.0 * atr)
            target = price - (2.0 * atr)
            if stop >= price * 2 or target <= 0:
                return None

            ml_signal, ml_conf = self._predictor.predict(df)
            if ml_signal == 1 and ml_conf > 0.65:
                return None

            return TradeSignal(
                symbol=symbol,
                direction="short",
                entry_price=price,
                stop_loss=stop,
                take_profit=target,
                confidence=0.55,
                reason="fib_resistance | LH_structure | ab_short",
                ml_signal=ml_signal,
                ml_confidence=ml_conf,
                ta_signal=-1,
                timeframe="1day",
            )

        return None

    # ------------------------------------------------------------------
    # Market structure helpers
    # ------------------------------------------------------------------

    def _compute_market_structure(self, df: pd.DataFrame) -> dict:
        """
        Identify swing highs/lows and higher highs/lows pattern.
        Uses rolling 10-bar windows to find local extremes.
        """
        result = {
            "higher_high": False,
            "higher_low": False,
            "lower_high": False,
            "lower_low": False,
            "bos_bull": False,   # Bullish break of structure
            "bos_bear": False,
        }

        if len(df) < 25:
            return result

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(highs)

        # Find swing highs and lows (local extremes over 5-bar windows)
        swing_highs = []
        swing_lows = []
        window = 5

        for i in range(window, n - window):
            local_high = highs[i - window:i + window + 1]
            if highs[i] == max(local_high):
                swing_highs.append((i, highs[i]))

            local_low = lows[i - window:i + window + 1]
            if lows[i] == min(local_low):
                swing_lows.append((i, lows[i]))

        if len(swing_highs) >= 2:
            latest_sh = swing_highs[-1][1]
            prior_sh = swing_highs[-2][1]
            result["higher_high"] = latest_sh > prior_sh
            result["lower_high"] = latest_sh < prior_sh
            # Break of structure: current close above latest swing high
            result["bos_bull"] = float(closes[-1]) > latest_sh

        if len(swing_lows) >= 2:
            latest_sl = swing_lows[-1][1]
            prior_sl = swing_lows[-2][1]
            result["higher_low"] = latest_sl > prior_sl
            result["lower_low"] = latest_sl < prior_sl
            result["bos_bear"] = float(closes[-1]) < latest_sl

        return result

    # ------------------------------------------------------------------
    # Fibonacci helpers
    # ------------------------------------------------------------------

    def _compute_fibonacci(self, df: pd.DataFrame) -> dict:
        """
        Find the most recent swing high and swing low (last 30 bars),
        compute standard Fibonacci retracement levels, and check if
        current price is within the 38.2–61.8% zone.
        """
        result = {
            "at_fib_zone": False,
            "at_fib_resistance": False,
            "nearest_fib_pct": 0.0,
            "fib_38": 0.0,
            "fib_50": 0.0,
            "fib_62": 0.0,
        }

        if len(df) < 20:
            return result

        lookback = df.iloc[-30:]
        swing_high = float(lookback["high"].max())
        swing_low = float(lookback["low"].min())
        price = float(df["close"].iloc[-1])

        if swing_high <= swing_low or swing_high == 0:
            return result

        fib_range = swing_high - swing_low
        fib_38 = swing_high - fib_range * 0.382
        fib_50 = swing_high - fib_range * 0.500
        fib_62 = swing_high - fib_range * 0.618

        result["fib_38"] = fib_38
        result["fib_50"] = fib_50
        result["fib_62"] = fib_62

        tol = self._cfg.get("fib_zone_tolerance", 0.01)

        # Support zone: price near 38.2–61.8% retracement (pullback into support)
        in_fib_support = fib_62 * (1 - tol) <= price <= fib_38 * (1 + tol)
        result["at_fib_zone"] = in_fib_support

        # Resistance zone: price bounced and approaching prior levels from below
        in_fib_resistance = fib_62 * (1 - tol) <= price <= fib_38 * (1 + tol)
        result["at_fib_resistance"] = in_fib_resistance  # Used for shorts

        # Which fib level is nearest?
        dists = {0.382: abs(price - fib_38), 0.500: abs(price - fib_50), 0.618: abs(price - fib_62)}
        result["nearest_fib_pct"] = min(dists, key=dists.get)

        return result

    # ------------------------------------------------------------------
    # ABC pattern detection
    # ------------------------------------------------------------------

    def _detect_abc_pattern(self, df: pd.DataFrame) -> bool:
        """
        Detect a completed ABC (3-wave) correction in an uptrend.

        A wave: initial pullback from recent high (decline > 3%)
        B wave: partial recovery (38–78% of A wave)
        C wave: second leg down ending near A low or fib level (within 2%)

        Returns True when C wave appears complete (price stabilizing at C low).
        """
        if len(df) < 20:
            return False

        closes = df["close"].values[-20:]
        b_min = self._cfg.get("abc_b_wave_min_retrace", 0.38)
        b_max = self._cfg.get("abc_b_wave_max_retrace", 0.78)

        # Scan for ABC pattern in the last 20 bars
        # Look for: local high → lower → bounce → lower again
        try:
            peak_idx = int(np.argmax(closes[:-3]))  # Peak (start of A)
            peak = closes[peak_idx]
            if peak_idx >= len(closes) - 3:
                return False

            post_peak = closes[peak_idx:]
            if len(post_peak) < 4:
                return False

            # A wave: trough after peak
            a_trough_idx = int(np.argmin(post_peak[:len(post_peak) // 2 + 1]))
            a_trough = post_peak[a_trough_idx]
            a_size = peak - a_trough
            if a_size / peak < 0.03:  # A wave must be at least 3% decline
                return False

            # B wave: bounce from A trough
            post_a = post_peak[a_trough_idx:]
            if len(post_a) < 3:
                return False

            b_peak_idx = int(np.argmax(post_a[:len(post_a) // 2 + 1]))
            b_peak = post_a[b_peak_idx]
            b_retrace = (b_peak - a_trough) / a_size if a_size > 0 else 0
            if not (b_min <= b_retrace <= b_max):
                return False

            # C wave: decline from B peak, ending near or below A trough
            post_b = post_a[b_peak_idx:]
            if len(post_b) < 2:
                return False

            c_trough = float(min(post_b))
            c_size = b_peak - c_trough

            # C wave should be roughly equal to A wave (0.6–1.4× A)
            c_vs_a = c_size / a_size if a_size > 0 else 0
            if not (0.5 <= c_vs_a <= 1.6):
                return False

            # C trough should be near or slightly below A trough
            c_near_a = abs(c_trough - a_trough) / peak < 0.04

            return c_near_a

        except Exception:
            return False

    # ------------------------------------------------------------------
    # EMA computation (exponential MAs — more responsive than SMA)
    # ------------------------------------------------------------------

    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA20, EMA50, EMA200 columns if not already present."""
        df = df.copy()
        close = df["close"]

        if "ema_20" not in df.columns:
            df["ema_20"] = close.ewm(span=20, adjust=False).mean()
        if "ema_50" not in df.columns:
            df["ema_50"] = close.ewm(span=50, adjust=False).mean()
        if "ema_200" not in df.columns:
            df["ema_200"] = close.ewm(span=200, adjust=False).mean()

        return df
