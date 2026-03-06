"""
Full US Market Universe Scanner
Fetches all tradeable US equities from Alpaca, filters by price/volume,
and pre-ranks by recent momentum so the AB bot scans the most relevant stocks.
"""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.config import get_alpaca_credentials, get_config

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

# Hard-coded fallback universe (liquid mid/large caps) if Alpaca asset list fails
FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "MA",
    "UNH", "XOM", "JNJ", "PG", "HD", "AVGO", "LLY", "COST", "MRK", "ABBV",
    "BAC", "KO", "PEP", "TMO", "WMT", "MCD", "CSCO", "CRM", "ACN", "NKE",
    "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "KLAC", "LRCX", "ASML", "TSM",
    "PYPL", "SQ", "SHOP", "SNAP", "UBER", "LYFT", "ABNB", "COIN", "HOOD", "RBLX",
    "NET", "CRWD", "ZS", "OKTA", "DDOG", "MDB", "SNOW", "PLTR", "GTLB", "PATH",
    "SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "XLF", "XLE", "XLK", "XBI",
    "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI", "RIVN", "LCID", "FSR",
    "DIS", "NFLX", "WBD", "PARA", "CMCSA", "T", "VZ", "TMUS", "AMC", "GME",
    "GS", "MS", "BLK", "C", "WFC", "USB", "PNC", "TFC", "COF", "AXP",
    "CVX", "COP", "SLB", "HAL", "EOG", "PXD", "DVN", "MRO", "APA", "OXY",
    "PFE", "MRNA", "BNTX", "REGN", "BIIB", "GILD", "VRTX", "AMGN", "BMY", "AZN",
    "BA", "LMT", "RTX", "NOC", "GD", "HII", "L3H", "TDG", "HEI", "KTOS",
]


class UniverseScanner:
    """
    Scans the full US equity market for the AB trader bot.
    Uses Alpaca's asset list + momentum pre-filtering to narrow to top candidates.
    """

    def __init__(self, api_key: str = None, secret_key: str = None):
        creds_api, creds_secret = get_alpaca_credentials()
        self._api_key = api_key or creds_api
        self._secret_key = secret_key or creds_secret
        self._cfg = get_config()
        self._ab_cfg = self._cfg.get("ab_trader", {})
        self._cached_universe: list[str] | None = None
        self._cache_date: str | None = None

    def _get_trading_client(self) -> TradingClient:
        paper = self._cfg["account"]["paper_trading"]
        return TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=paper,
        )

    def _get_data_client(self) -> StockHistoricalDataClient:
        return StockHistoricalDataClient(self._api_key, self._secret_key)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def get_scannable_universe(
        self,
        min_price: float = None,
        max_price: float = None,
        min_volume: int = None,
        top_n: int = None,
    ) -> list[str]:
        """
        Return a filtered, momentum-ranked list of US stocks ready for AB scanning.
        Results are cached for the trading day to avoid repeated API calls.
        """
        today = datetime.now(ET).date().isoformat()
        if self._cached_universe and self._cache_date == today:
            logger.debug("[UNIVERSE] Using cached universe (%d symbols)", len(self._cached_universe))
            return self._cached_universe

        cfg = self._ab_cfg
        min_price = min_price or cfg.get("min_price", 5.0)
        max_price = max_price or cfg.get("max_price", 2000.0)
        min_volume = min_volume or cfg.get("min_avg_volume", 200_000)
        top_n = top_n or cfg.get("max_symbols_per_scan", 200)

        # Step 1: Get base universe
        raw = self._get_all_us_equities()
        logger.info("[UNIVERSE] Raw universe: %d symbols", len(raw))

        # Step 2: Momentum filter — fetch 20-day bars, rank by 5-day return × volume
        ranked = self._rank_by_momentum(raw, min_price, max_price, min_volume, top_n)
        logger.info("[UNIVERSE] Post-filter momentum universe: %d symbols", len(ranked))

        self._cached_universe = ranked
        self._cache_date = today
        return ranked

    # ------------------------------------------------------------------
    # Asset listing
    # ------------------------------------------------------------------

    def _get_all_us_equities(self) -> list[str]:
        """Fetch all active, tradeable US equity symbols from Alpaca."""
        try:
            client = self._get_trading_client()
            req = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            assets = client.get_all_assets(req)
            symbols = [
                a.symbol for a in assets
                if a.tradable
                and a.fractionable is not False
                and "." not in a.symbol   # Skip BRK.B style / foreign
                and len(a.symbol) <= 5     # Skip options-like tickers
            ]
            logger.info("[UNIVERSE] Alpaca returned %d tradeable US equities", len(symbols))
            return symbols
        except Exception as exc:
            logger.warning("[UNIVERSE] Could not fetch Alpaca asset list: %s — using fallback", exc)
            return FALLBACK_UNIVERSE

    # ------------------------------------------------------------------
    # Momentum ranking
    # ------------------------------------------------------------------

    def _rank_by_momentum(
        self,
        symbols: list[str],
        min_price: float,
        max_price: float,
        min_volume: int,
        top_n: int,
    ) -> list[str]:
        """
        Fetch 20-day daily bars in batches, filter by price/volume,
        and rank by 5-day price momentum × volume ratio.
        AB targets stocks that are MOVING — this pre-filter focuses the scan.
        """
        data_client = self._get_data_client()
        start = datetime.now(ET) - timedelta(days=30)
        batch_size = 100
        scored: list[tuple[str, float]] = []

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=start,
                    feed="iex",
                )
                barset = self._get_batch_bars(data_client, req, batch)
                for sym, df in barset.items():
                    score = self._score_symbol(df, min_price, max_price, min_volume)
                    if score is not None:
                        scored.append((sym, score))
            except Exception as exc:
                logger.debug("[UNIVERSE] Batch %d–%d failed: %s", i, i + batch_size, exc)

        scored.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in scored[:top_n]]

    def _get_batch_bars(
        self,
        client: StockHistoricalDataClient,
        req: StockBarsRequest,
        symbols: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Fetch bars for a batch, return {symbol: DataFrame}."""
        result = {}
        try:
            barset = client.get_stock_bars(req)
            try:
                full_df = barset.df
                if not full_df.empty and "symbol" in full_df.index.names:
                    for sym in symbols:
                        try:
                            df = full_df.loc[sym].copy()
                            df.columns = [c.lower() for c in df.columns]
                            if not df.empty:
                                result[sym] = df
                        except KeyError:
                            pass
                    return result
            except Exception:
                pass
            # Fallback: iterate
            for sym, bars_list in barset.items():
                if bars_list:
                    try:
                        records = [
                            {"close": float(b.close), "volume": float(b.volume)}
                            for b in bars_list
                        ]
                        result[sym] = pd.DataFrame(records)
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("[UNIVERSE] Batch bars error: %s", exc)
        return result

    def _score_symbol(
        self,
        df: pd.DataFrame,
        min_price: float,
        max_price: float,
        min_volume: int,
    ) -> float | None:
        """
        Score a symbol for momentum. Returns None if it fails filters.
        Score = 5-day return % × (recent volume / 20-day avg volume)
        Higher score = stronger momentum with volume confirmation (AB's preference).
        """
        if df.empty or len(df) < 6:
            return None

        close_col = "close" if "close" in df.columns else df.columns[0]
        vol_col = "volume" if "volume" in df.columns else None

        try:
            price = float(df[close_col].iloc[-1])
            price_5d_ago = float(df[close_col].iloc[-6])
        except (IndexError, ValueError):
            return None

        # Price filter
        if not (min_price <= price <= max_price):
            return None

        # Volume filter
        if vol_col and len(df) >= 20:
            avg_vol = float(df[vol_col].iloc[-20:].mean())
            if avg_vol < min_volume:
                return None
            recent_vol = float(df[vol_col].iloc[-5:].mean())
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        # 5-day momentum
        momentum_5d = (price - price_5d_ago) / price_5d_ago if price_5d_ago > 0 else 0.0

        # Combined score: momentum × volume confirmation
        # AB targets stocks making moves with participation
        score = momentum_5d * vol_ratio
        return score
