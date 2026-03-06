"""
Market data client wrapping Alpaca's historical and live data APIs.
Provides OHLCV bars, quotes, and universe scanning.
"""

import logging
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.config import get_alpaca_credentials, get_config

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

# S&P 500 large-cap sample (full list requires a separate data source)
SP500_SAMPLE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "JPM", "UNH",
    "V", "XOM", "JNJ", "MA", "PG", "HD", "AVGO", "MRK", "CVX", "ABBV",
    "LLY", "COST", "PEP", "KO", "BAC", "TMO", "WMT", "MCD", "ABT", "CSCO",
    "CRM", "DHR", "ACN", "NKE", "LIN", "NEE", "TXN", "PM", "VZ", "ADBE",
    "CMCSA", "WFC", "INTC", "RTX", "HON", "BMY", "AMGN", "QCOM", "LOW", "UPS",
]

RUSSELL2000_SAMPLE = [
    "AMC", "GME", "BBBY", "CLOV", "WKHS", "SPCE", "RIDE", "NKLA", "HYLN", "RKT",
    "HOOD", "COUR", "UWMC", "OPEN", "MTHW", "GREE", "ATER", "VVOS", "GOEV", "PAYO",
    "MAPS", "MYPS", "BARK", "BIRD", "BRDS", "CERT", "PHAT", "SHOT", "RCKT", "ORGN",
]


class MarketDataClient:
    def __init__(self):
        api_key, secret_key = get_alpaca_credentials()
        self._client = StockHistoricalDataClient(api_key, secret_key)
        self._cfg = get_config()

    # ------------------------------------------------------------------
    # Core bar fetching
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbols: list[str],
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV bars for one or more symbols.
        Returns {symbol: DataFrame with columns open/high/low/close/volume/vwap}.
        """
        if end is None:
            end = datetime.now(ET)

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
            feed="iex",  # Use IEX feed (included in free plan)
        )

        try:
            barset = self._client.get_stock_bars(req)
        except Exception as exc:
            logger.error("get_stock_bars API call failed: %s", exc)
            return {}

        result: dict[str, pd.DataFrame] = {}

        # Primary: use barset.df — a MultiIndex DataFrame (symbol, timestamp)
        try:
            full_df = barset.df
            if not full_df.empty:
                idx_names = list(full_df.index.names)
                if "symbol" in idx_names:
                    for sym in symbols:
                        try:
                            df = full_df.loc[sym].copy()
                            df = self._normalize_df(df)
                            if not df.empty:
                                result[sym] = df
                        except KeyError:
                            pass
                else:
                    # Single-symbol request: index is just timestamp
                    sym = symbols[0]
                    df = full_df.copy()
                    df = self._normalize_df(df)
                    if not df.empty:
                        result[sym] = df
                logger.debug("Fetched bars via .df: %s", {s: len(d) for s, d in result.items()})
                return result
        except Exception as exc:
            logger.debug("barset.df failed (%s), falling back to iteration", exc)

        # Fallback: iterate barset as dict {symbol: [Bar, ...]}
        try:
            for sym, bars_list in barset.items():
                if not bars_list:
                    continue
                try:
                    records = [
                        {
                            "timestamp": b.timestamp,
                            "open": float(b.open),
                            "high": float(b.high),
                            "low": float(b.low),
                            "close": float(b.close),
                            "volume": float(b.volume),
                            "vwap": float(b.vwap) if getattr(b, "vwap", None) else None,
                        }
                        for b in bars_list
                    ]
                    df = pd.DataFrame(records).set_index("timestamp")
                    df = self._normalize_df(df)
                    if not df.empty:
                        result[sym] = df
                except Exception as exc2:
                    logger.warning("Could not parse bars for %s: %s", sym, exc2)
        except Exception as exc:
            logger.error("barset iteration failed: %s", exc)

        logger.debug("Fetched bars via iteration: %s", {s: len(d) for s, d in result.items()})
        return result

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent index timezone and lowercase column names."""
        if df.empty:
            return df
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        else:
            df.index = df.index.tz_convert(ET)
        return df

    def get_daily_bars(
        self,
        symbols: list[str],
        lookback_days: int = 60,
    ) -> dict[str, pd.DataFrame]:
        start = datetime.now(ET) - timedelta(days=lookback_days)
        return self.get_bars(symbols, TimeFrame.Day, start)

    def get_intraday_bars(
        self,
        symbols: list[str],
        minutes: int = 5,
        lookback_days: int = 1,
    ) -> dict[str, pd.DataFrame]:
        tf = TimeFrame(minutes, TimeFrameUnit.Minute)
        start = datetime.now(ET) - timedelta(days=lookback_days + 1)
        return self.get_bars(symbols, tf, start)

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Single symbol historical data for backtesting."""
        data = self.get_bars([symbol], timeframe, start, end)
        return data.get(symbol, pd.DataFrame())

    # ------------------------------------------------------------------
    # Latest quotes
    # ------------------------------------------------------------------

    def get_latest_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Get current bid/ask for symbols."""
        req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        quotes = self._client.get_stock_latest_quote(req)
        result = {}
        for sym, q in quotes.items():
            result[sym] = {
                "bid": float(q.bid_price or 0),
                "ask": float(q.ask_price or 0),
                "mid": (float(q.bid_price or 0) + float(q.ask_price or 0)) / 2,
                "bid_size": int(q.bid_size or 0),
                "ask_size": int(q.ask_size or 0),
            }
        return result

    def get_latest_price(self, symbol: str) -> float:
        quotes = self.get_latest_quotes([symbol])
        return quotes.get(symbol, {}).get("mid", 0.0)

    # ------------------------------------------------------------------
    # Universe scanning
    # ------------------------------------------------------------------

    def get_universe(self, universe: str = "sp500") -> list[str]:
        """Return ticker list for the requested universe."""
        if universe == "sp500":
            return SP500_SAMPLE
        elif universe == "russell2000":
            return RUSSELL2000_SAMPLE
        elif universe == "both":
            return list(set(SP500_SAMPLE + RUSSELL2000_SAMPLE))
        return SP500_SAMPLE

    def scan_day_trading_universe(self, max_symbols: int = 20) -> list[str]:
        """
        Filter stocks meeting day-trader entry criteria:
          - Price $5–$500
          - Avg daily volume ≥ 1M
          - Relative volume ≥ 1.5x
        Returns top candidates by relative volume.
        """
        cfg = self._cfg["day_trader"]
        base = SP500_SAMPLE  # Start with liquid large caps

        # Fetch recent bars to compute volume metrics
        try:
            bars = self.get_daily_bars(base, lookback_days=25)
        except Exception as exc:
            logger.error("Universe scan failed: %s", exc)
            return base[:10]

        candidates = []
        for sym, df in bars.items():
            if df.empty or len(df) < 20:
                continue
            latest_price = float(df["close"].iloc[-1])
            if not (cfg["min_price"] <= latest_price <= cfg["max_price"]):
                continue
            avg_vol = float(df["volume"].iloc[-20:].mean())
            if avg_vol < cfg["min_avg_daily_volume"]:
                continue
            today_vol = float(df["volume"].iloc[-1])
            rel_vol = today_vol / avg_vol if avg_vol > 0 else 0
            if rel_vol >= cfg["min_relative_volume"]:
                candidates.append((sym, rel_vol))

        candidates.sort(key=lambda x: x[1], reverse=True)
        result = [s for s, _ in candidates[:max_symbols]]
        logger.info("Day trading universe: %d symbols (rel-vol filtered)", len(result))
        return result if result else base[:10]

    def get_snapshots(self, symbols: list[str]) -> dict:
        """Get full market snapshot (quote + bar + trade) for symbols."""
        req = StockSnapshotRequest(symbol_or_symbols=symbols)
        try:
            return dict(self._client.get_stock_snapshot(req))
        except Exception as exc:
            logger.error("Snapshot fetch failed: %s", exc)
            return {}
