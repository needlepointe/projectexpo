"""
Day Trader Bot
- Runs every 5 minutes during 9:45 AM – 3:45 PM ET
- Closes ALL positions by 3:55 PM ET
- Enforces PDT: max 3 day trades per 5-day rolling window
- When PDT limit reached: queues signals but does NOT execute new day trades
"""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.bots.base_bot import BaseBot
from src.strategies.day_strategy import DayStrategy
from src.options.options_strategy import OptionsStrategy
from src.config import get_config, is_options_enabled
from src.database import get_connection, log_event

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class DayTraderBot(BaseBot):
    def __init__(self):
        cfg = get_config()
        super().__init__("day", cfg["account"]["starting_capital_day"])
        self._cfg = cfg["day_trader"]
        self._strategy = DayStrategy()
        self._options = OptionsStrategy()
        self._queued_signals: list = []  # When PDT limit hit
        self._open_positions: dict = {}   # {symbol: position_info}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_market_open(self):
        """Called at 9:45 AM ET."""
        equity = self._client.get_equity()
        self._risk.reset_daily(equity)
        self._open_positions = {}
        log_event("market_open", f"Day trader capital: ${equity:.2f}", "day")
        logger.info("[DAY] Market open. Capital: $%.2f", equity)

        # Retrain ML weekly
        if self._strategy.ml_needs_retraining():
            self._retrain_ml()

    def on_market_close(self):
        """Called at 3:55 PM ET — close EVERY position, no exceptions."""
        logger.warning("[DAY] Market close: force-closing all positions.")
        positions = self._client.get_positions()

        for pos in positions:
            sym = pos["symbol"]
            result = self._client.close_position(sym)
            pnl = pos["unrealized_pl"]
            self._risk.update_pnl(pnl)
            self._record_trade_close(sym, pos, pnl, "time_exit")
            logger.info("[DAY] Force closed %s P&L: $%.2f", sym, pnl)

        self._open_positions = {}
        log_event("market_close", "All day positions closed", "day")

    # ------------------------------------------------------------------
    # Main cycle (every 5 minutes)
    # ------------------------------------------------------------------

    def run_cycle(self):
        if not self._running:
            return

        now = datetime.now(ET)
        if not self._is_trading_hours(now):
            return

        ok, reason = self._risk.can_trade()
        if not ok:
            logger.warning("[DAY] Trading blocked: %s", reason)
            return

        # Step 1: Check existing positions for exits
        self._manage_open_positions()

        # Step 2: Scan for new entries
        ok, pdt_reason = self._risk.can_day_trade()
        if not ok:
            logger.info("[DAY] PDT limit reached — queueing signals only. %s", pdt_reason)
            self._scan_and_queue()
            return

        self._scan_and_trade()

    # ------------------------------------------------------------------
    # Signal scanning
    # ------------------------------------------------------------------

    def _scan_and_trade(self):
        """Scan universe and execute best signal if found."""
        universe = self._data.scan_day_trading_universe(max_symbols=15)
        if not universe:
            return

        tf = TimeFrame(5, TimeFrameUnit.Minute)
        start = datetime.now(ET) - timedelta(days=2)
        bars_5m = self._data.get_bars(universe, tf, start)

        if not bars_5m:
            return

        signals = self._strategy.generate_signals(bars_5m)

        for signal in signals[:3]:  # Try top 3 signals
            if self._open_positions.get(signal.symbol):
                continue  # Already have a position in this stock

            ok, reason = self._risk.can_day_trade()
            if not ok:
                logger.info("[DAY] PDT reached during scan: %s", reason)
                break

            self._execute_signal(signal, bars_5m.get(signal.symbol))

    def _scan_and_queue(self):
        """When PDT-limited, just queue signals for monitoring."""
        universe = self._data.scan_day_trading_universe(max_symbols=10)
        if not universe:
            return
        tf = TimeFrame(5, TimeFrameUnit.Minute)
        start = datetime.now(ET) - timedelta(days=2)
        bars_5m = self._data.get_bars(universe, tf, start)
        if bars_5m:
            self._queued_signals = self._strategy.generate_signals(bars_5m)
            if self._queued_signals:
                logger.info("[DAY] PDT queue: %d signals ready (will execute when PDT resets)",
                            len(self._queued_signals))

    def _execute_signal(self, signal, bars: pd.DataFrame | None):
        """Place order for a signal after full risk validation."""
        try:
            pos = self._risk.calculate_position(
                entry_price=signal.entry_price,
                stop_price=signal.stop_loss,
            )
        except Exception as exc:
            logger.warning("[DAY] Position calc failed for %s: %s", signal.symbol, exc)
            return

        approved, reason = self._risk.validate_trade(
            symbol=signal.symbol,
            entry_price=signal.entry_price,
            stop_price=pos["stop_loss"],
            take_profit=pos["take_profit"],
            shares=pos["shares"],
            is_day_trade=True,
        )
        if not approved:
            logger.info("[DAY] Trade rejected %s: %s", signal.symbol, reason)
            return

        # Try stock first
        try:
            if signal.is_long():
                order = self._client.buy_market(
                    signal.symbol, pos["shares"],
                    pos["stop_loss"], pos["take_profit"],
                )
            else:
                order = self._client.sell_market(
                    signal.symbol, pos["shares"],
                    pos["stop_loss"], pos["take_profit"],
                )

            self._risk.record_day_trade(signal.symbol)
            self._record_trade_open(signal, pos, order)
            logger.info(
                "[DAY] ENTERED %s %s: %d shares @ $%.2f SL=$%.2f TP=$%.2f",
                signal.direction.upper(), signal.symbol,
                pos["shares"], signal.entry_price,
                pos["stop_loss"], pos["take_profit"],
            )

            # Options overlay (if enabled)
            if is_options_enabled() and self._options.is_enabled():
                self._try_options_trade(signal)

        except Exception as exc:
            logger.error("[DAY] Order failed for %s: %s", signal.symbol, exc)
            log_event("error", f"Order failed {signal.symbol}: {exc}", "day")

    def _try_options_trade(self, signal):
        """Optionally layer an options trade alongside the stock trade."""
        contract = self._options.get_option_contract(signal)
        if not contract:
            return
        ask = float(contract.get("ask_price", 0))
        if ask <= 0:
            return
        num_contracts = self._options.calculate_contracts(ask, self._risk.current_capital)
        if num_contracts <= 0:
            return
        try:
            self._client.buy_option(contract["symbol"], num_contracts, limit_price=ask)
            logger.info("[DAY] Options: bought %d %s contracts @ $%.2f",
                        num_contracts, contract["symbol"], ask)
        except Exception as exc:
            logger.warning("[DAY] Options order failed: %s", exc)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _manage_open_positions(self):
        """Check open positions for manual exit signals."""
        positions = self._client.get_positions()
        for pos in positions:
            sym = pos["symbol"]
            tf = TimeFrame(5, TimeFrameUnit.Minute)
            start = datetime.now(ET) - timedelta(hours=4)
            bars = self._data.get_bars([sym], tf, start).get(sym)
            if bars is None or bars.empty:
                continue
            should_exit, reason = self._strategy.should_exit(sym, pos, bars)
            if should_exit:
                logger.info("[DAY] Early exit %s: %s", sym, reason)
                result = self._client.close_position(sym)
                pnl = pos["unrealized_pl"]
                self._risk.update_pnl(pnl)
                self._record_trade_close(sym, pos, pnl, reason)

    # ------------------------------------------------------------------
    # Database recording
    # ------------------------------------------------------------------

    def _record_trade_open(self, signal, pos: dict, order: dict):
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO open_positions
                (bot_type, symbol, side, entry_time, entry_price, quantity,
                 stop_loss, take_profit, entry_reason, ml_confidence, alpaca_order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "day", signal.symbol,
                "buy" if signal.is_long() else "sell",
                datetime.now(ET).isoformat(),
                signal.entry_price, pos["shares"],
                pos["stop_loss"], pos["take_profit"],
                signal.reason, signal.ml_confidence,
                order.get("id", ""),
            ))

    def _record_trade_close(self, symbol: str, position: dict, pnl: float, reason: str):
        entry_price = float(position.get("avg_entry_price", 0))
        exit_price = float(position.get("current_price", entry_price))
        qty = float(position.get("qty", 0))
        pnl_pct = pnl / (entry_price * qty) if entry_price * qty > 0 else 0

        with get_connection() as conn:
            conn.execute("""
                INSERT INTO trades
                (bot_type, symbol, side, entry_price, exit_price, quantity,
                 pnl, pnl_pct, exit_reason, is_day_trade)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                "day", symbol, position.get("side", ""),
                entry_price, exit_price, qty,
                pnl, pnl_pct, reason,
            ))
            conn.execute("DELETE FROM open_positions WHERE bot_type='day' AND symbol=?", (symbol,))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_trading_hours(self, now: datetime) -> bool:
        start_h, start_m = [int(x) for x in self._cfg["trading_start"].split(":")]
        end_h, end_m = [int(x) for x in self._cfg["trading_end"].split(":")]
        t = now.hour * 60 + now.minute
        return (start_h * 60 + start_m) <= t <= (end_h * 60 + end_m)

    def _is_force_close_time(self, now: datetime) -> bool:
        close_h, close_m = [int(x) for x in self._cfg["force_close_time"].split(":")]
        return now.hour * 60 + now.minute >= close_h * 60 + close_m

    def _retrain_ml(self):
        try:
            start = datetime.now(ET) - timedelta(days=365)
            tf = TimeFrame(5, TimeFrameUnit.Minute)
            # Train on SPY as a representative liquid stock
            bars = self._data.get_historical_bars("SPY", tf, start)
            if not bars.empty:
                metrics = self._strategy.train_ml(bars)
                logger.info("[DAY] ML retrained: %s", metrics)
        except Exception as exc:
            logger.error("[DAY] ML retraining failed: %s", exc)
