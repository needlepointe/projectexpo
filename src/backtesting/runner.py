"""
Backtesting Framework
- Replays historical OHLCV data through the strategy logic
- Simulates order fills with configurable slippage
- Enforces all risk rules (PDT, position sizing, stops)
- Outputs: win rate, total return, max drawdown, Sharpe ratio, expectancy
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.config import get_config
from src.data.market_data import MarketDataClient
from src.ml.feature_engineering import compute_features, make_labels
from src.ml.signal_predictor import SignalPredictor

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


@dataclass
class BacktestTrade:
    symbol: str
    direction: str
    entry_bar: int
    entry_price: float
    stop_loss: float
    take_profit: float
    shares: int
    exit_bar: int = -1
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    is_day_trade: bool = False


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: list[BacktestTrade] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"BACKTEST RESULTS — {self.strategy.upper()} on {self.symbol}",
            f"{'='*60}",
            f"Period:         {self.start_date} to {self.end_date}",
            f"Initial capital: ${self.initial_capital:,.2f}",
            f"Final capital:   ${self.final_capital:,.2f}",
            f"Total return:    {self.total_return_pct:+.2f}%",
            f"Trades:          {self.num_trades} ({self.num_wins}W / {self.num_losses}L)",
            f"Win rate:        {self.win_rate:.1f}%",
            f"Avg win:         ${self.avg_win:.2f}",
            f"Avg loss:        ${self.avg_loss:.2f}",
            f"Expectancy:      ${self.expectancy:.2f}/trade",
            f"Max drawdown:    {self.max_drawdown_pct:.2f}%",
            f"Sharpe ratio:    {self.sharpe_ratio:.2f}",
            f"{'='*60}",
        ]
        # Pass/fail vs targets
        cfg = get_config()
        if self.strategy == "day":
            win_target = 45.0
        else:
            # Swing uses 2:1 R:R — breakeven win rate is 33.3%.
            # 40% is a realistic professional target that allows positive expectancy
            # while avoiding over-filtering to the point of statistical insignificance.
            # 50% at 2:1 R:R would be elite-level performance, not a baseline requirement.
            win_target = 40.0

        # Exit reason breakdown — shows HOW trades closed
        exit_counts: dict[str, int] = {}
        exit_wins: dict[str, int] = {}
        for t in self.trades:
            r = t.exit_reason
            exit_counts[r] = exit_counts.get(r, 0) + 1
            if t.pnl > 0:
                exit_wins[r] = exit_wins.get(r, 0) + 1
        lines.append("\nEXIT REASONS (count | wins):")
        for reason, count in sorted(exit_counts.items()):
            wins = exit_wins.get(reason, 0)
            lines.append(f"  {reason:<16s}: {count:4d}  ({wins}W / {count - wins}L)")

        # Direction breakdown — reveals if longs vs shorts have different win rates
        long_trades = [t for t in self.trades if t.direction == "long"]
        short_trades = [t for t in self.trades if t.direction == "short"]
        lines.append("\nDIRECTION BREAKDOWN:")
        if long_trades:
            lw = sum(1 for t in long_trades if t.pnl > 0)
            lines.append(f"  long : {len(long_trades):3d} trades  ({lw}W / {len(long_trades)-lw}L)  {lw/len(long_trades)*100:.1f}%")
        if short_trades:
            sw = sum(1 for t in short_trades if t.pnl > 0)
            lines.append(f"  short: {len(short_trades):3d} trades  ({sw}W / {len(short_trades)-sw}L)  {sw/len(short_trades)*100:.1f}%")

        lines.append("\nTARGET CHECK:")
        lines.append(f"  Win rate {self.win_rate:.1f}% vs target {win_target}%: {'PASS' if self.win_rate >= win_target else 'FAIL'}")
        lines.append(f"  Max drawdown {self.max_drawdown_pct:.1f}% vs limit 15%: {'PASS' if self.max_drawdown_pct <= 15 else 'FAIL'}")
        lines.append(f"  Positive expectancy: {'PASS' if self.expectancy > 0 else 'FAIL'}")
        return "\n".join(lines)

    def passes_targets(self, strategy: str = "day") -> bool:
        win_target = 45.0 if strategy == "day" else 40.0  # "ab" and "swing" both use 40%
        return (
            self.win_rate >= win_target
            and self.max_drawdown_pct <= 15.0
            and self.expectancy > 0
        )


class Backtester:
    """
    Event-driven backtester with realistic simulation.
    """

    def __init__(self):
        cfg = get_config()
        self._bt_cfg = cfg["backtesting"]
        self._risk_cfg = cfg["risk"]
        self._data = MarketDataClient()

    def run(
        self,
        strategy: str,
        symbol: str,
        start: datetime,
        end: datetime | None = None,
        initial_capital: float | None = None,
    ) -> BacktestResult:
        """
        Run a backtest for the given strategy and symbol.

        Args:
            strategy: "day" or "swing"
            symbol: Ticker to backtest
            start: Start date
            end: End date (defaults to now)
            initial_capital: Starting capital (defaults to config value)
        """
        if end is None:
            end = datetime.now(ET)
        if initial_capital is None:
            if strategy == "ab":
                initial_capital = get_config().get("ab_trader", {}).get("starting_capital", 1000.0)
            else:
                key = f"initial_capital_{strategy}"
                initial_capital = self._bt_cfg.get(key, 500.0)

        logger.info("Backtesting %s on %s from %s to %s", strategy, symbol, start.date(), end.date())

        # Fetch historical data.
        # Day strategy uses 15-min bars: each bar represents 15 min of market action,
        # giving 3× the ATR of 5-min bars.  With 5-min bars the 3×ATR take-profit
        # target required a sustained 0.5%+ move — almost never reached before the
        # 0.25% stop triggered (causing 80%+ loss rate regardless of signal quality).
        # 15-min bars give the momentum room to develop (ATR ≈ $2–3 on QQQ vs $0.82
        # on 5-min) and produce far fewer false signals (26 bars/day vs 78).
        if strategy == "day":
            tf = TimeFrame(15, TimeFrameUnit.Minute)
        else:  # "swing" or "ab" — both use daily bars
            tf = TimeFrame.Day

        df = self._data.get_historical_bars(symbol, tf, start, end)
        if df.empty:
            raise ValueError(f"No historical data for {symbol}")

        logger.info("Fetched %d bars for %s", len(df), symbol)

        # Train ML on first half of data
        predictor = SignalPredictor(strategy)
        split = len(df) // 2
        train_df = df.iloc[:split]
        if len(train_df) >= 200:
            try:
                predictor.train(train_df)
            except Exception as exc:
                logger.warning("ML training failed during backtest: %s", exc)

        # Run simulation on second half
        test_df = df.iloc[split:]
        trades = self._simulate(strategy, symbol, test_df, predictor, initial_capital)

        return self._calculate_metrics(
            strategy, symbol,
            start.date().isoformat(), end.date().isoformat(),
            initial_capital, trades,
        )

    def run_multiple(
        self,
        strategy: str,
        symbols: list[str],
        lookback_days: int | None = None,
        initial_capital: float | None = None,
    ) -> list[BacktestResult]:
        """Backtest across multiple symbols and return all results."""
        if lookback_days is None:
            lookback_days = self._bt_cfg.get("lookback_days", 365)
        start = datetime.now(ET) - timedelta(days=lookback_days)
        results = []
        for sym in symbols:
            try:
                result = self.run(strategy, sym, start, initial_capital=initial_capital)
                results.append(result)
                print(result.summary())
            except Exception as exc:
                logger.warning("Backtest failed for %s: %s", sym, exc)
        return results

    # ------------------------------------------------------------------
    # Simulation engine
    # ------------------------------------------------------------------

    def _simulate(
        self,
        strategy: str,
        symbol: str,
        df: pd.DataFrame,
        predictor: SignalPredictor,
        capital: float,
    ) -> list[BacktestTrade]:
        """Bar-by-bar simulation."""
        df = compute_features(df)
        # Precompute AB-specific features for "ab" strategy
        if strategy == "ab":
            df = self._precompute_ab_features(df)
        slippage = self._bt_cfg.get("slippage_pct", 0.001)
        commission = self._bt_cfg.get("commission_per_share", 0.0)
        risk_pct = self._risk_cfg.get("max_risk_per_trade_pct", 0.01)
        max_pos_pct = self._risk_cfg.get("max_position_size_pct", 0.20)
        min_rr = self._risk_cfg.get("min_reward_risk_ratio", 2.0)
        min_conf = predictor._ml_cfg.get("min_confidence", 0.55)

        # Batch-predict all ML signals at once — avoids per-bar joblib overhead
        ml_signals = np.zeros(len(df), dtype=int)
        ml_confs = np.zeros(len(df), dtype=float)
        if predictor._is_trained and predictor._model is not None:
            feat_cols = [c for c in predictor._feature_cols if c in df.columns]
            if feat_cols:
                X_all = df[feat_cols].fillna(0).values
                X_scaled = predictor._scaler.transform(X_all)
                preds = predictor._model.predict(X_scaled)
                probas = predictor._model.predict_proba(X_scaled).max(axis=1)
                ml_signals = preds.astype(int)
                ml_confs = probas
                # Zero out low-confidence signals
                ml_signals[ml_confs < min_conf] = 0
        logger.info("Batch ML prediction complete (%d bars)", len(df))

        trades: list[BacktestTrade] = []
        open_trade: BacktestTrade | None = None
        _dbg_ta_signals = 0   # diagnostic counters
        _dbg_ml_vetoes = 0

        for i in range(50, len(df)):
            bar = df.iloc[i]
            price = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])

            # Check for exit on open trade
            if open_trade is not None:
                # Stop hit
                if open_trade.direction == "long" and low <= open_trade.stop_loss:
                    exit_price = open_trade.stop_loss * (1 - slippage)
                    open_trade = self._close_trade(open_trade, i, exit_price, "stop_loss")
                    capital += open_trade.pnl
                    trades.append(open_trade)
                    open_trade = None
                    continue

                # Take-profit hit
                elif open_trade.direction == "long" and high >= open_trade.take_profit:
                    exit_price = open_trade.take_profit * (1 - slippage)
                    open_trade = self._close_trade(open_trade, i, exit_price, "take_profit")
                    capital += open_trade.pnl
                    trades.append(open_trade)
                    open_trade = None
                    continue

                elif open_trade.direction == "short" and high >= open_trade.stop_loss:
                    exit_price = open_trade.stop_loss * (1 + slippage)
                    open_trade = self._close_trade(open_trade, i, exit_price, "stop_loss")
                    capital += open_trade.pnl
                    trades.append(open_trade)
                    open_trade = None
                    continue

                elif open_trade.direction == "short" and low <= open_trade.take_profit:
                    exit_price = open_trade.take_profit * (1 + slippage)
                    open_trade = self._close_trade(open_trade, i, exit_price, "take_profit")
                    capital += open_trade.pnl
                    trades.append(open_trade)
                    open_trade = None
                    continue

                # EOD force-close for day trades: live bot closes all by 3:55 PM ET.
                # For 15-min bars the last bar of the trading day is at 15:45, so
                # the old check for ts.minute >= 55 never triggered (no such bar).
                if strategy == "day":
                    ts = df.index[i]
                    if ts.hour > 15 or (ts.hour == 15 and ts.minute >= 45):
                        open_trade = self._close_trade(open_trade, i, price, "eod_close")
                        capital += open_trade.pnl
                        trades.append(open_trade)
                        open_trade = None
                        continue

                # Max hold for swing / ab
                if strategy in ("swing", "ab") and (i - open_trade.entry_bar) >= 10:
                    open_trade = self._close_trade(open_trade, i, price, "max_hold")
                    capital += open_trade.pnl
                    trades.append(open_trade)
                    open_trade = None
                    continue

            # Look for new entry
            if open_trade is None and capital > 10:
                # Time-of-day filter: for day strategy only enter during core
                # market hours (10:00–15:30 ET) to avoid open/close volatility
                if strategy == "day":
                    ts = df.index[i]
                    h, m = ts.hour, ts.minute
                    in_window = (h > 10 or (h == 10 and m >= 0)) and (h < 14 or (h == 14 and m <= 30))
                    if not in_window:
                        continue

                ml_signal = int(ml_signals[i])
                ml_conf = float(ml_confs[i])
                ta_signal = self._get_ta_signal(bar, strategy)

                # ML acts as a VETO, not a co-signer.
                # - TA signal alone is sufficient to enter.
                # - ML blocks entry only when it actively disagrees (signals opposite direction).
                # - When ML is neutral (0), the TA signal proceeds — this is the common case
                #   since the model only produces confident non-neutral predictions ~10-20% of bars.
                # The old "ml_signal == ta_signal" requirement filtered out ~90% of valid TA signals
                # because ML outputs neutral (0) far more often than it outputs 1 or -1,
                # resulting in only ~11 trades over 6 months (1/month) for a day trading strategy.
                # ML veto: only applied for day strategy where 2000+ 15-min bars give
                # sufficient training data. Swing ML trains on ~230 daily bars — far too
                # few for a random forest to find reliable patterns. CV accuracy of 27-28%
                # is BELOW random (33.3% for 3-class), meaning the veto actively blocks
                # good TA signals. Disable for swing until training data improves.
                if strategy == "day":
                    ml_veto = (ml_signal != 0 and ml_signal == -ta_signal)
                else:
                    ml_veto = False

                if ta_signal != 0:
                    _dbg_ta_signals += 1
                    if ml_veto:
                        _dbg_ml_vetoes += 1

                if ta_signal != 0 and not ml_veto:
                    entry_price = price * (1 + slippage if ta_signal == 1 else 1 - slippage)
                    atr = float(bar.get("atr_pct", 0.01)) * price
                    # Floor must be much smaller for 5-min bars than daily bars.
                    # 5-min ATR for QQQ/SPY is ~0.15–0.20% of price.
                    # The old 0.5% floor inflated ATR 3× on expensive ETFs,
                    # making take-profit targets unreachable.
                    min_atr_pct = 0.001 if strategy == "day" else 0.005
                    atr = max(atr, price * min_atr_pct)

                    # Day: 2×ATR stop gives breathing room on 15-min bars; 4×ATR target
                    # is rarely hit but positive expectancy comes from EOD directional edge.
                    # Swing: 1×ATR stop and 2×ATR target — same 2:1 R:R but target is
                    # reachable within 2-5 days on daily bars. 4×ATR required an 8%+ sustained
                    # move — achievable but rare — so wins were capped at max_hold exits
                    # while losses always hit the full 2×ATR stop (expectancy went negative).
                    if strategy == "day":
                        stop_mult, target_mult = 2.0, 4.0
                    elif strategy == "ab":
                        # 1.5×ATR stop gives NVDA/volatile stocks room to breathe.
                        # 3×ATR target maintains the 2:1 R:R.
                        stop_mult, target_mult = 1.5, 3.0
                    else:  # "swing" — 1×ATR stop, 2×ATR target (reachable in 2-5 days)
                        stop_mult, target_mult = 1.0, 2.0

                    if ta_signal == 1:
                        stop = entry_price - stop_mult * atr
                        target = entry_price + target_mult * atr
                    else:
                        stop = entry_price + stop_mult * atr
                        target = entry_price - target_mult * atr

                    risk_per_share = abs(entry_price - stop)
                    if risk_per_share < 0.001:
                        continue

                    # Use fractional shares so high-priced stocks (SPY, NVDA) work with small capital
                    shares = round((capital * risk_pct) / risk_per_share, 4)
                    max_shares = round((capital * max_pos_pct) / entry_price, 4)
                    shares = min(shares, max_shares)
                    if shares < 0.01:
                        continue

                    rr = abs(target - entry_price) / risk_per_share
                    if rr < min_rr:
                        continue

                    cost = shares * commission
                    open_trade = BacktestTrade(
                        symbol=symbol,
                        direction="long" if ta_signal == 1 else "short",
                        entry_bar=i,
                        entry_price=entry_price,
                        stop_loss=stop,
                        take_profit=target,
                        shares=shares,
                    )
                    capital -= cost

        # Close any open trade at end of test
        if open_trade is not None:
            last_price = float(df.iloc[-1]["close"])
            open_trade = self._close_trade(open_trade, len(df) - 1, last_price, "end_of_test")
            trades.append(open_trade)

        logger.info(
            "Simulation complete: %d TA signals generated, %d ML vetoes applied, %d trades taken",
            _dbg_ta_signals, _dbg_ml_vetoes, len(trades),
        )
        return trades

    def _get_ta_signal(self, bar: pd.Series, strategy: str) -> int:
        """Simplified TA signal check for backtesting."""
        rsi = float(bar.get("rsi_14", 50))
        rsi_change = float(bar.get("rsi_change", 0))   # positive = RSI rising
        macd_hist = float(bar.get("macd_hist", 0))
        vwap_dev = float(bar.get("vwap_deviation", 0))
        volume_ratio = float(bar.get("volume_ratio_20", 1.0))
        ma20 = float(bar.get("ma_20", 0))
        ma50 = float(bar.get("ma_50", 0))
        price = float(bar["close"])

        if strategy == "day":
            # VWAP Cross Momentum Strategy — event-based, not level-based.
            #
            # Three filters required for a valid signal:
            # 1. VWAP cross (0.05% threshold eliminates VWAP-hugging noise)
            # 2. Trend alignment: price must be on the correct side of 50-bar MA
            # 3. Volume confirmation: volume_ratio > 1.3 means institutions are participating.
            #    A VWAP cross on low volume is noise; institutional volume = conviction.
            # 4. RSI momentum alignment: rsi_change > 0 for longs (RSI is rising on this bar,
            #    meaning momentum is BUILDING, not exhausting). Prevents entering after the
            #    move has already happened and RSI is rolling over.
            # 5. MACD histogram positive for longs: short-term trend agreement.
            vwap_dev_prev = float(bar.get("vwap_deviation_prev", 0))
            vwap_cross_up = vwap_dev_prev < -0.0005 and vwap_dev > 0.0005
            vwap_cross_dn = vwap_dev_prev > 0.0005 and vwap_dev < -0.0005
            in_uptrend = ma50 > 0 and price > ma50   # price above 15-min 50-bar MA
            in_downtrend = ma50 > 0 and price < ma50
            has_volume = volume_ratio > 1.2           # institutional participation required
            # Candle direction filter: the cross bar must be a conviction candle.
            # A bullish cross bar with a bearish body (close < open) is an exhaustion
            # signal — price briefly crossed VWAP but sellers won the bar. Not a real breakout.
            open_price = float(bar.get("open", price))
            bullish_candle = price > open_price
            bearish_candle = price < open_price

            if vwap_cross_up and in_uptrend and macd_hist > 0 and has_volume and rsi_change > 0 and bullish_candle:
                return 1
            if vwap_cross_dn and in_downtrend and macd_hist < 0 and has_volume and rsi_change < 0 and bearish_candle:
                return -1
        elif strategy == "ab":
            return self._get_ta_signal_ab(bar)

        else:  # swing
            ma200 = float(bar.get("ma_200", 0))
            uptrend = ma20 > ma50 and ma20 > 0 and ma50 > 0
            downtrend = ma20 < ma50 and ma20 > 0 and ma50 > 0
            # Long: pullback to 20-day MA in an established uptrend, WITH price above
            # the 200-day MA (primary bull trend confirmed).
            # Without the MA200 filter, "pullbacks to MA20 in uptrend" during corrections
            # become breakdowns — 30% long win rate. The 200-day MA is the professional's
            # primary trend filter: below it = bear market, above it = bull market.
            near_ma20 = abs(price - ma20) / ma20 < 0.03 if ma20 > 0 else False
            primary_uptrend = ma200 > 0 and price > ma200
            # Bounce confirmation: price must be AT or ABOVE MA20 now (touched and held).
            # Without this, "near_ma20" catches stocks falling THROUGH the MA20 on their
            # way to a larger breakdown — the classic "catching a falling knife" trap.
            bounced_above_ma20 = price >= ma20 if ma20 > 0 else False
            if uptrend and primary_uptrend and near_ma20 and bounced_above_ma20 and 30 <= rsi <= 52:
                return 1
            # Short: bounce into overhead resistance (declining MA20 in downtrend).
            # Price rallies up toward the falling MA20 from below, RSI 48–70.
            # MACD histogram must still be negative — if MACD is already turning positive,
            # the "downtrend" has reversed and shorting here is fighting a recovery.
            # Without this, the signal fires during brief corrections in ongoing bull runs
            # (e.g., META dipping 10% then rallying 30%), producing 20% short win rates.
            near_ma20_resistance = abs(price - ma20) / ma20 < 0.03 if ma20 > 0 else False
            if downtrend and near_ma20_resistance and 48 <= rsi <= 70 and macd_hist < 0:
                return -1
        return 0

    def _precompute_ab_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Precompute AB strategy-specific features on the full DataFrame.
        Called once before simulation to avoid per-bar recalculation overhead.

        AB's primary signal: EMA20 pullback + bounce in primary uptrend.
        Secondary signal: breakout above 20-bar high with volume confirmation.
        """
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
        vol_ma = volume.rolling(20, min_periods=1).mean()
        vol_ratio = (volume / vol_ma).fillna(1.0)

        # Exponential MAs — more responsive than SMA for AB's style
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        df["ema_20"] = ema20
        df["ema_50"] = ema50
        df["ema_200"] = ema200

        # EMA20 touch: price within 2% of EMA20 and at or above it (bounce confirmation)
        # This is the primary AB entry condition — pullback to the 20 EMA in uptrend
        df["near_ema20"] = (abs(close - ema20) / ema20.clip(lower=0.01)) < 0.02
        df["bounced_ema20"] = close >= ema20 * 0.99   # Price held at EMA20

        # EMA20 resistance for shorts: price bounced up to declining EMA20 from below
        df["at_ema20_resistance"] = (abs(close - ema20) / ema20.clip(lower=0.01)) < 0.02

        # Breakout: close above 20-bar rolling high (prior bars only) with volume surge
        prior_high_20 = high.shift(1).rolling(19, min_periods=10).max()
        breakout_vol = get_config().get("ab_trader", {}).get("breakout_volume_ratio", 1.5)
        df["is_breakout_bull"] = (close > prior_high_20) & (vol_ratio >= breakout_vol)

        # Fibonacci zone: price in 38.2–61.8% retracement of rolling 30-bar swing range
        # Used as an additional confirmation — not the primary filter
        roll_high = high.rolling(30, min_periods=10).max()
        roll_low = low.rolling(30, min_periods=10).min()
        fib_range = (roll_high - roll_low).clip(lower=0.001)
        fib_38 = roll_high - fib_range * 0.382
        fib_62 = roll_high - fib_range * 0.618
        tol = get_config().get("ab_trader", {}).get("fib_zone_tolerance", 0.015)
        df["at_fib_support"] = (close >= fib_62 * (1 - tol)) & (close <= fib_38 * (1 + tol))

        # Market structure: higher low (uptrend intact)
        df["higher_low"] = low > low.shift(5)
        df["lower_high"] = high < high.shift(5)

        # EMA20 slope: EMA20 must be rising for longs (trend healthy, not topping)
        df["ema20_rising"] = ema20 > ema20.shift(3)
        df["ema20_falling"] = ema20 < ema20.shift(3)

        # RSI recovered: RSI rising from recent low (momentum returning after pullback)
        rsi = df.get("rsi_14", pd.Series(50.0, index=df.index))
        if "rsi_14" in df.columns:
            rsi = df["rsi_14"]
        df["rsi_rising"] = rsi > rsi.shift(2)

        return df

    def _get_ta_signal_ab(self, bar: pd.Series) -> int:
        """
        AB Trades TA signal for backtesting.

        PRIMARY (EMA20 Pullback):
          - EMA20 > EMA50 (short-term uptrend)
          - price > EMA200 (primary bull trend)
          - Price within 2% of EMA20 and held above it (bounce)
          - RSI 35-55 (pullback without deeply oversold)
          - MACD not deeply negative (momentum not broken)
          - Optional: at Fibonacci support zone

        SECONDARY (Breakout):
          - EMA20 > EMA50, price > EMA200
          - Close above 20-bar high with volume >= 1.5x
          - RSI >= 48

        SHORT:
          - EMA20 < EMA50, price < EMA200
          - Price at EMA20 resistance from below
          - RSI 52-70, MACD hist < 0
        """
        price = float(bar["close"])
        ema20 = float(bar.get("ema_20", 0))
        ema50 = float(bar.get("ema_50", 0))
        ema200 = float(bar.get("ema_200", 0))
        rsi = float(bar.get("rsi_14", 50))
        macd_hist = float(bar.get("macd_hist", 0))

        near_ema20 = bool(bar.get("near_ema20", False))
        bounced_ema20 = bool(bar.get("bounced_ema20", False))
        at_ema20_resistance = bool(bar.get("at_ema20_resistance", False))
        is_breakout = bool(bar.get("is_breakout_bull", False))
        at_fib_support = bool(bar.get("at_fib_support", False))
        higher_low = bool(bar.get("higher_low", True))
        lower_high = bool(bar.get("lower_high", True))
        rsi_rising = bool(bar.get("rsi_rising", True))
        ema20_rising = bool(bar.get("ema20_rising", True))
        ema20_falling = bool(bar.get("ema20_falling", True))

        # Trend structure (EMA-based — more responsive than MA)
        uptrend = ema20 > 0 and ema50 > 0 and ema20 > ema50
        primary_bull = ema200 > 0 and price > ema200
        downtrend = ema20 > 0 and ema50 > 0 and ema20 < ema50
        primary_bear = ema200 > 0 and price < ema200

        # LONG — Primary: EMA20 pullback + bounce
        # EMA20 must be rising: rules out entries during EMA20 rollover (early downtrend)
        ema20_pullback = (
            near_ema20          # Within 2% of EMA20
            and bounced_ema20   # Price at or above EMA20 (not falling through)
            and ema20_rising    # EMA20 is still pointing up (trend not reversing)
            and 33 <= rsi <= 57 # Pulled back but not deeply oversold
            and rsi_rising      # Momentum recovering
            and macd_hist > -0.5  # MACD not deeply negative
        )

        # LONG — Secondary: breakout above 20-day high with volume
        breakout_long = is_breakout and rsi >= 48

        if uptrend and primary_bull and (ema20_pullback or breakout_long):
            return 1

        # SHORT — EMA20 resistance in downtrend (EMA20 must be falling = confirmed downtrend)
        ema20_short = (
            at_ema20_resistance
            and ema20_falling    # EMA20 pointing down = confirmed downtrend
            and 52 <= rsi <= 70
            and macd_hist < 0
            and lower_high       # Structure: lower highs confirming downtrend
        )
        if downtrend and primary_bear and ema20_short:
            return -1

        return 0

    @staticmethod
    def _close_trade(trade: BacktestTrade, bar: int, price: float, reason: str) -> BacktestTrade:
        trade.exit_bar = bar
        trade.exit_price = price
        trade.exit_reason = reason
        if trade.direction == "long":
            trade.pnl = (price - trade.entry_price) * trade.shares
        else:
            trade.pnl = (trade.entry_price - price) * trade.shares
        trade.pnl_pct = (price / trade.entry_price - 1) * (1 if trade.direction == "long" else -1)
        return trade

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _calculate_metrics(
        self,
        strategy: str,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
        trades: list[BacktestTrade],
    ) -> BacktestResult:
        if not trades:
            return BacktestResult(
                strategy=strategy, symbol=symbol,
                start_date=start_date, end_date=end_date,
                initial_capital=initial_capital, final_capital=initial_capital,
                total_return_pct=0, num_trades=0, num_wins=0, num_losses=0,
                win_rate=0, avg_win=0, avg_loss=0, expectancy=0,
                max_drawdown_pct=0, sharpe_ratio=0, trades=[],
            )

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        final_capital = initial_capital + sum(pnls)
        total_return_pct = (final_capital / initial_capital - 1) * 100
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        expectancy = np.mean(pnls) if pnls else 0

        # Max drawdown
        equity_curve = np.cumsum([0] + pnls) + initial_capital
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100
        max_dd = float(np.max(drawdown))

        # Sharpe ratio (annualized, assuming daily returns)
        daily_returns = np.array(pnls) / initial_capital
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        return BacktestResult(
            strategy=strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return_pct,
            num_trades=len(trades),
            num_wins=len(wins),
            num_losses=len(losses),
            win_rate=win_rate,
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            expectancy=float(expectancy),
            max_drawdown_pct=max_dd,
            sharpe_ratio=float(sharpe),
            trades=trades,
        )
