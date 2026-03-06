"""
Feature engineering — converts raw OHLCV bars into ML-ready features.
All technical indicators computed here feed both the ML model and TA strategies.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicator features to a bar DataFrame.

    Input columns expected: open, high, low, close, volume (+ optional vwap)
    Returns the same DataFrame with additional feature columns.
    """
    if df.empty or len(df) < 20:
        return df

    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # -- RSI --
    df["rsi_14"] = _rsi(close, 14)
    # RSI change: positive = RSI rising (momentum building), negative = falling (exhausting)
    df["rsi_change"] = df["rsi_14"].diff()

    # -- MACD --
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # -- Bollinger Bands --
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = bb_upper - bb_lower
    df["bb_upper_pct"] = (bb_upper - close) / bb_range.replace(0, np.nan)
    df["bb_lower_pct"] = (close - bb_lower) / bb_range.replace(0, np.nan)
    df["bb_position"] = (close - bb_lower) / bb_range.replace(0, np.nan)

    # -- Volume ratio --
    vol_ma20 = volume.rolling(20).mean()
    df["volume_ratio_20"] = volume / vol_ma20.replace(0, np.nan)

    # -- VWAP deviation (resets each trading day) --
    if "vwap" in df.columns:
        df["vwap_deviation"] = (close - df["vwap"]) / df["vwap"].replace(0, np.nan)
    else:
        # Compute intraday VWAP with daily reset — group by calendar date so each
        # trading day starts fresh.  A rolling cumulative over the whole dataset
        # produces a multi-month average that is useless as an intraday indicator.
        typical = (high + low + close) / 3
        tp_vol = typical * volume
        dates = pd.Series(df.index.date, index=df.index, dtype=object)
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        cum_vol_day = volume.groupby(dates).cumsum()
        vwap_calc = cum_tp_vol / cum_vol_day.replace(0, np.nan)
        df["vwap_deviation"] = (close - vwap_calc) / vwap_calc.replace(0, np.nan)

    # Previous-bar VWAP deviation — enables detecting VWAP crosses (price transitioning
    # from one side of the daily average to the other).  Event-based signals like
    # "price just crossed VWAP" are far more predictive than level-based signals like
    # "RSI is currently in range X" because they capture a genuine regime change.
    df["vwap_deviation_prev"] = df["vwap_deviation"].shift(1)

    # -- Price momentum --
    df["price_momentum_5"] = close.pct_change(5)
    df["price_momentum_10"] = close.pct_change(10)
    df["price_momentum_1"] = close.pct_change(1)

    # -- ATR (normalized) --
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    df["atr_pct"] = atr14 / close.replace(0, np.nan)

    # -- Moving averages --
    df["ma_20"] = close.rolling(20).mean()
    df["ma_50"] = close.rolling(50).mean()
    df["ma_200"] = close.rolling(200).mean()
    df["price_vs_ma20"] = (close - df["ma_20"]) / df["ma_20"].replace(0, np.nan)
    df["price_vs_ma50"] = (close - df["ma_50"]) / df["ma_50"].replace(0, np.nan)
    df["ma20_vs_ma50"] = (df["ma_20"] - df["ma_50"]) / df["ma_50"].replace(0, np.nan)

    # -- Stochastic %K --
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    return df


def make_labels(
    df: pd.DataFrame,
    forward_bars: int = 6,
    bullish_thr: float = 0.008,
    bearish_thr: float = -0.008,
    use_percentile: bool = True,
    percentile: float = 0.33,
) -> pd.Series:
    """
    Generate classification labels from forward returns.
    1 = bullish, -1 = bearish, 0 = neutral.

    use_percentile=True (default): label top/bottom N% of forward returns as
    bullish/bearish, guaranteeing balanced classes regardless of symbol volatility.
    This is the recommended mode — fixed thresholds cause 95%+ neutral labels on
    low-volatility symbols like SPY on short timeframes.

    use_percentile=False: use fixed bullish_thr / bearish_thr values instead.
    """
    fwd_return = df["close"].pct_change(forward_bars).shift(-forward_bars)
    labels = pd.Series(0, index=df.index, dtype=int)

    if use_percentile:
        valid = fwd_return.dropna()
        bull_cut = float(valid.quantile(1.0 - percentile))
        bear_cut = float(valid.quantile(percentile))
        labels[fwd_return >= bull_cut] = 1
        labels[fwd_return <= bear_cut] = -1
    else:
        labels[fwd_return >= bullish_thr] = 1
        labels[fwd_return <= bearish_thr] = -1

    return labels


def get_feature_columns(cfg_features: list[str] | None = None) -> list[str]:
    """Return the list of feature column names used for ML training."""
    default = [
        "rsi_14", "rsi_change", "macd", "macd_signal", "macd_hist",
        "bb_upper_pct", "bb_lower_pct", "bb_position",
        "volume_ratio_20", "vwap_deviation", "vwap_deviation_prev",
        "price_momentum_5", "price_momentum_10", "price_momentum_1",
        "atr_pct", "price_vs_ma20", "price_vs_ma50", "ma20_vs_ma50",
        "stoch_k", "stoch_d",
    ]
    if cfg_features:
        return [f for f in cfg_features if f in default] or default
    return default


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
