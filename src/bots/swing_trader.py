"""
Swing Trader Bot
- Runs once daily (after market close)
- Scans S&P 500 + Russell 2000 for multi-day setups
- Holds positions 2–10 days
- Does NOT count as day trades (different-day entry/exit)
"""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.timeframe import TimeFrame

from src.bots.base_bot import BaseBot
from src.strategies.swing_strategy import SwingStrategy
from src.options.options_strategy import OptionsStrategy
from src.config import get_config, is_options_enabled
from src.database import get_connection, log_event

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class SwingTraderBot(BaseBot):
    def __init__(self):
        cfg = get_config()
        super().__init__("swing", cfg["account"]["starting_capital_swing"])
        self._cfg = cfg["swing_trader"]
        self._strategy = SwingStrategy()
        self._options = OptionsStrategy()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_market_open(self):
        """Check for stop-loss and take-profit hits on open positions."""
        equity = self._client.get_equity()
        self._risk.reset_daily(equity)
        log_event("market_open", f"Swing trader capital: ${equity:.2f}", "swing")
        logger.info("[SWING] Market open. Capital: $%.2f", equity)

        if self._strategy.ml_needs_retraining():
            self._retrain_ml()

    def on_market_close(self):
        """
        Main swing scan happens at close. Find setups for next-day entry.
        Swing positions are NOT force-closed at end of day.
        """
        logger.info("[SWING] Running end-of-day scan...")
        self.run_cycle()

    def run_cycle(self):
        """Scan for swing setups and manage existing positions."""
        if not self._running:
            return

        ok, reason = self._risk.can_trade()
        if not ok:
            logger.warning("[SWING] Trading blocked: %s", reason)
            return

        # Step 1: Manage existing swing positions
        self._manage_open_positions()

        # Step 2: Check max hold time
        self._enforce_max_hold()

        # Step 3: Scan for new entries
        self._scan_and_trade()

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan_and_trade(self):
        # Use curated watchlist if configured, otherwise fall back to universe scan.
        # Watchlist only contains symbols validated by backtesting — avoids trading
        # names with no backtested edge.
        watchlist = self._cfg.get("symbols", [])
        if watchlist:
            universe = watchlist
            logger.info("[SWING] Using curated watchlist: %s", universe)
        else:
            universe = self._data.get_universe(self._cfg.get("universe", "sp500"))
            # Limit scan to 50 symbols per cycle to avoid rate limits
            universe = universe[:50]

        bars = self._data.get_daily_bars(universe, lookback_days=60)
        if not bars:
            logger.warning("[SWING] No daily bars returned for universe scan")
            return

        signals = self._strategy.generate_signals(bars)
        logger.info("[SWING] Found %d swing signals", len(signals))

        # Only take top 2 signals per scan (preserve capital)
        positions = self._client.get_positions()
        held_symbols = {p["symbol"] for p in positions}
        max_positions = max(1, int(self._risk.current_capital / 100))  # Rough cap

        for signal in signals[:5]:
            if signal.symbol in held_symbols:
                continue
            if len(held_symbols) >= max_positions:
                logger.info("[SWING] Max positions reached (%d)", max_positions)
                break

            self._execute_signal(signal, bars.get(signal.symbol))
            held_symbols.add(signal.symbol)

    def _execute_signal(self, signal, bars=None):
        try:
            pos = self._risk.calculate_position(
                entry_price=signal.entry_price,
                stop_price=signal.stop_loss,
            )
        except Exception as exc:
            logger.warning("[SWING] Position calc failed for %s: %s", signal.symbol, exc)
            return

        approved, reason = self._risk.validate_trade(
            symbol=signal.symbol,
            entry_price=signal.entry_price,
            stop_price=pos["stop_loss"],
            take_profit=pos["take_profit"],
            shares=pos["shares"],
            is_day_trade=False,
        )
        if not approved:
            logger.info("[SWING] Trade rejected %s: %s", signal.symbol, reason)
            return

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

            self._record_trade_open(signal, pos, order)
            logger.info(
                "[SWING] ENTERED %s %s: %d shares @ $%.2f SL=$%.2f TP=$%.2f | %s",
                signal.direction.upper(), signal.symbol,
                pos["shares"], signal.entry_price,
                pos["stop_loss"], pos["take_profit"],
                signal.reason,
            )

            if is_options_enabled() and self._options.is_enabled():
                self._try_options_trade(signal)

        except Exception as exc:
            logger.error("[SWING] Order failed for %s: %s", signal.symbol, exc)
            log_event("error", f"Swing order failed {signal.symbol}: {exc}", "swing")

    def _try_options_trade(self, signal):
        contract = self._options.get_option_contract(signal)
        if not contract:
            return
        ask = float(contract.get("ask_price", 0))
        if ask <= 0:
            return
        num_contracts = self._options.calculate_contracts(ask, self._risk.current_capital)
        if num_contracts > 0:
            try:
                self._client.buy_option(contract["symbol"], num_contracts, limit_price=ask)
                logger.info("[SWING] Options: bought %d %s @ $%.2f",
                            num_contracts, contract["symbol"], ask)
            except Exception as exc:
                logger.warning("[SWING] Options order failed: %s", exc)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _manage_open_positions(self):
        """Check positions for technical exit signals."""
        positions = self._client.get_positions()
        for pos in positions:
            sym = pos["symbol"]
            bars = self._data.get_daily_bars([sym], lookback_days=30).get(sym)
            if bars is None or bars.empty:
                continue
            should_exit, reason = self._strategy.should_exit(sym, pos, bars)
            if should_exit:
                logger.info("[SWING] Early exit %s: %s", sym, reason)
                result = self._client.close_position(sym)
                pnl = float(pos["unrealized_pl"])
                self._risk.update_pnl(pnl)
                self._record_trade_close(sym, pos, pnl, reason)

    def _enforce_max_hold(self):
        """Force-close positions held longer than max_hold_days."""
        max_days = self._cfg.get("max_hold_days", 10)
        with get_connection() as conn:
            old_positions = conn.execute("""
                SELECT symbol, entry_time FROM open_positions
                WHERE bot_type = 'swing'
                AND entry_time < datetime('now', ? )
            """, (f"-{max_days} days",)).fetchall()

        for row in old_positions:
            sym = row["symbol"]
            pos = self._client.get_position(sym)
            if pos:
                logger.info("[SWING] Max hold exceeded for %s — closing", sym)
                result = self._client.close_position(sym)
                pnl = float(pos.get("unrealized_pl", 0))
                self._risk.update_pnl(pnl)
                self._record_trade_close(sym, pos, pnl, "max_hold_time")

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
                "swing", signal.symbol,
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                "swing", symbol, position.get("side", ""),
                entry_price, exit_price, qty,
                pnl, pnl_pct, reason,
            ))
            conn.execute("DELETE FROM open_positions WHERE bot_type='swing' AND symbol=?", (symbol,))

    def _retrain_ml(self):
        try:
            start = datetime.now(ET) - timedelta(days=365)
            bars = self._data.get_historical_bars("SPY", TimeFrame.Day, start)
            if not bars.empty:
                metrics = self._strategy.train_ml(bars)
                logger.info("[SWING] ML retrained: %s", metrics)
        except Exception as exc:
            logger.error("[SWING] ML retraining failed: %s", exc)
