"""
AB Trader Bot
Replicates the swing trading style of @ABTradess (OWLS Discord).

Key features:
  - Scans full US equity market (all liquid stocks)
  - EMA20 pullback + breakout strategy (AB's public methodology)
  - Self-improving ML: learns from AB's actual picks + own trade outcomes
  - Weekly target tracking: reports progress toward 10% weekly goal
  - PDT-safe: all trades held 2+ days (is_day_trade=False)
  - Separate Alpaca account (ALPACA_AB_API_KEY / ALPACA_AB_SECRET_KEY)
  - Options overlay on highest-confidence signals
"""

import logging
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.timeframe import TimeFrame

from src.bots.base_bot import BaseBot
from src.strategies.ab_strategy import ABStrategy
from src.ml.ab_predictor import ABPredictor
from src.monitoring.ab_tracker import ABTracker
from src.options.options_strategy import OptionsStrategy
from src.data.universe_scanner import UniverseScanner
from src.config import get_config, is_options_enabled
from src.database import get_connection, log_event

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class WeeklyTargetTracker:
    """
    Tracks progress toward the 10% weekly return target.

    Does NOT override risk rules — instead adjusts signal confidence
    threshold slightly based on where we are vs. target for the week.
    """

    TARGET_PCT = 0.10  # 10% per week

    def __init__(self, starting_capital: float):
        self._starting_capital = starting_capital

    def get_weekly_pnl(self, client) -> dict:
        """Return realized + unrealized P&L for current week (Mon–Fri)."""
        # Week start = most recent Monday
        today = datetime.now(ET)
        days_since_monday = today.weekday()  # Mon=0, Sun=6
        week_start = (today - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()

        realized_pnl = 0.0
        with get_connection() as conn:
            row = conn.execute("""
                SELECT COALESCE(SUM(pnl), 0) as total_pnl
                FROM trades
                WHERE bot_type = 'ab'
                AND exit_time >= ?
            """, (week_start,)).fetchone()
            if row:
                realized_pnl = float(row["total_pnl"])

        # Unrealized P&L from open positions
        unrealized_pnl = 0.0
        try:
            positions = client.get_positions()
            for pos in positions:
                unrealized_pnl += float(pos.get("unrealized_pl", 0))
        except Exception as exc:
            logger.debug("[AB_WEEKLY] Could not get positions: %s", exc)

        total_pnl = realized_pnl + unrealized_pnl
        capital = self._starting_capital
        total_pct = total_pnl / capital if capital > 0 else 0.0

        return {
            "realized": realized_pnl,
            "unrealized": unrealized_pnl,
            "total": total_pnl,
            "total_pct": total_pct,
            "target_pct": self.TARGET_PCT,
            "on_pace": total_pct >= self.TARGET_PCT,
            "progress_pct": (total_pct / self.TARGET_PCT * 100) if self.TARGET_PCT > 0 else 0,
        }

    def signal_confidence_multiplier(self, client) -> float:
        """
        Adjust confidence threshold based on weekly progress.
        Target met   → raise bar (protect gains, be more selective)
        Behind pace  → keep normal threshold (don't chase risk)
        Note: NEVER VIOLATES risk rules — only affects signal selection.
        """
        try:
            weekly = self.get_weekly_pnl(client)
            if weekly["total_pct"] >= self.TARGET_PCT:
                # Target met — raise the bar, protect gains
                return 1.20
            # Behind or on pace — normal operation
            return 1.00
        except Exception:
            return 1.00

    def log_weekly_summary(self, client) -> None:
        """Log weekly P&L progress to console and database."""
        try:
            weekly = self.get_weekly_pnl(client)
            pct = weekly["total_pct"] * 100
            target = self.TARGET_PCT * 100
            status = "TARGET MET!" if weekly["on_pace"] else f"{weekly['progress_pct']:.1f}% of target"
            logger.info(
                "[AB_WEEKLY] Week P&L: $%.2f realized + $%.2f unrealized = $%.2f (%.2f%% vs %.0f%% target) — %s",
                weekly["realized"], weekly["unrealized"], weekly["total"],
                pct, target, status,
            )
        except Exception as exc:
            logger.debug("[AB_WEEKLY] Summary error: %s", exc)


class ABTraderBot(BaseBot):
    """
    AB Trades mimic bot. Swing-style, scans full US market, self-improving.
    """

    def __init__(self):
        cfg = get_config()
        ab_cfg = cfg.get("ab_trader", {})
        starting_capital = ab_cfg.get("starting_capital", 1000.0)

        # Use separate Alpaca API keys for the AB account
        api_key = os.getenv("ALPACA_AB_API_KEY", "")
        secret_key = os.getenv("ALPACA_AB_SECRET_KEY", "")

        super().__init__("ab", starting_capital)

        # Override the client with AB-specific credentials if provided
        if api_key and secret_key:
            from src.execution.alpaca_client import AlpacaClient
            self._client = AlpacaClient(api_key=api_key, secret_key=secret_key)
            logger.info("[AB] Using separate Alpaca account (AB keys)")
        else:
            logger.warning(
                "[AB] ALPACA_AB_API_KEY not set — using default account. "
                "Add ALPACA_AB_API_KEY and ALPACA_AB_SECRET_KEY to .env for separate account."
            )

        self._cfg = ab_cfg
        self._strategy = ABStrategy()
        self._predictor = ABPredictor()
        self._tracker = ABTracker()
        self._options = OptionsStrategy()
        self._scanner = UniverseScanner(
            api_key=api_key or None,
            secret_key=secret_key or None,
        )
        self._weekly = WeeklyTargetTracker(starting_capital)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_market_open(self):
        """Scan for AB's overnight tweets, log weekly status."""
        equity = self._client.get_equity()
        self._risk.reset_daily(equity)
        log_event("market_open", f"AB trader capital: ${equity:.2f}", "ab")
        logger.info("[AB] Market open. Capital: $%.2f", equity)

        # Check for AB picks tweeted after hours
        if self._cfg.get("twitter_monitor", True):
            new_picks = self._tracker.scan_recent_tweets(lookback_hours=18)
            if new_picks:
                logger.info("[AB] AB tweeted about: %s", new_picks)

        self._weekly.log_weekly_summary(self._client)

        # Retrain ML if overdue
        if self._predictor.needs_retraining():
            self._retrain_ml()

    def on_market_close(self):
        """
        Main EOD scan: find next-day entries, log predictions, scan AB's tweets.
        Swing positions are NOT force-closed at EOD.
        """
        logger.info("[AB] Running EOD scan...")

        # Check AB's tweets from today
        if self._cfg.get("twitter_monitor", True):
            self._tracker.scan_recent_tweets(lookback_hours=8)

        self.run_cycle()

        # Log weekly target progress
        self._weekly.log_weekly_summary(self._client)

    def run_cycle(self):
        """Scan universe, manage positions, execute signals."""
        if not self._running:
            return

        ok, reason = self._risk.can_trade()
        if not ok:
            logger.warning("[AB] Trading blocked: %s", reason)
            return

        # Step 1: Manage existing positions (technical exits)
        self._manage_open_positions()

        # Step 2: Check max hold time
        self._enforce_max_hold()

        # Step 3: Scan for new entries
        self._scan_and_trade()

    # ------------------------------------------------------------------
    # Scanning & execution
    # ------------------------------------------------------------------

    def _scan_and_trade(self):
        """Scan full US universe, score signals, execute top picks."""
        # Get momentum-filtered universe (full US market, top 200 by momentum)
        max_n = self._cfg.get("max_symbols_per_scan", 200)
        universe = self._scanner.get_scannable_universe(top_n=max_n)
        logger.info("[AB] Scanning %d symbols", len(universe))

        bars = self._data.get_daily_bars(universe, lookback_days=250)
        if not bars:
            logger.warning("[AB] No bars returned from universe scan")
            return

        # Generate TA signals
        signals = self._strategy.generate_signals(bars)
        logger.info("[AB] Found %d raw signals", len(signals))

        # Score signals: TA confidence + AB-style ML probability
        conf_multiplier = self._weekly.signal_confidence_multiplier(self._client)
        min_conf = self._cfg.get("ml_confidence_threshold", 0.55) * conf_multiplier

        scored_signals = []
        for sig in signals:
            df = bars.get(sig.symbol)
            if df is None:
                continue
            ab_prob = self._predictor.predict_ab_probability(df)
            # Combined score = 60% TA confidence + 40% AB-probability
            combined = sig.confidence * 0.6 + ab_prob * 0.4
            if combined >= min_conf:
                scored_signals.append((combined, sig, df))

        scored_signals.sort(key=lambda x: x[0], reverse=True)

        # Log ALL top-10 signals to ab_predictions (for accuracy tracking)
        top_predictions = [sig for _, sig, _ in scored_signals[:10]]
        self._tracker.log_bot_predictions(top_predictions)

        # Execute top 3 signals
        positions = self._client.get_positions()
        held_symbols = {p["symbol"] for p in positions}
        max_positions = max(1, int(self._risk.current_capital / 200))

        executed = 0
        for combined_conf, signal, df in scored_signals[:5]:
            if signal.symbol in held_symbols:
                continue
            if len(held_symbols) >= max_positions:
                logger.info("[AB] Max positions reached (%d)", max_positions)
                break
            if executed >= 3:  # Max 3 new entries per scan
                break

            self._execute_signal(signal)
            held_symbols.add(signal.symbol)
            executed += 1

    def _execute_signal(self, signal):
        """Execute a trade signal with position sizing and validation."""
        try:
            pos = self._risk.calculate_position(
                entry_price=signal.entry_price,
                stop_price=signal.stop_loss,
            )
        except Exception as exc:
            logger.warning("[AB] Position calc failed for %s: %s", signal.symbol, exc)
            return

        approved, reason = self._risk.validate_trade(
            symbol=signal.symbol,
            entry_price=signal.entry_price,
            stop_price=pos["stop_loss"],
            take_profit=pos["take_profit"],
            shares=pos["shares"],
            is_day_trade=False,  # Always swing — never day trade
        )
        if not approved:
            logger.info("[AB] Trade rejected %s: %s", signal.symbol, reason)
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
                "[AB] ENTERED %s %s: %.4f shares @ $%.2f SL=$%.2f TP=$%.2f | %s",
                signal.direction.upper(), signal.symbol,
                pos["shares"], signal.entry_price,
                pos["stop_loss"], pos["take_profit"],
                signal.reason,
            )

            # Options overlay on highest-confidence signals
            if is_options_enabled() and self._options.is_enabled():
                self._try_options(signal)

        except Exception as exc:
            logger.error("[AB] Order failed for %s: %s", signal.symbol, exc)
            log_event("error", f"AB order failed {signal.symbol}: {exc}", "ab")

    def _try_options(self, signal):
        """Attempt an options trade alongside the equity position."""
        contract = self._options.get_option_contract(signal)
        if not contract:
            return
        ask = float(contract.get("ask_price", 0))
        if ask <= 0:
            return
        n = self._options.calculate_contracts(ask, self._risk.current_capital)
        if n > 0:
            try:
                self._client.buy_option(contract["symbol"], n, limit_price=ask)
                logger.info("[AB] Options: bought %d %s @ $%.2f", n, contract["symbol"], ask)
            except Exception as exc:
                logger.warning("[AB] Options order failed: %s", exc)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _manage_open_positions(self):
        """Check open positions for technical exit signals (EMA20 cross, RSI extreme)."""
        positions = self._client.get_positions()
        for pos in positions:
            sym = pos["symbol"]
            bars = self._data.get_daily_bars([sym], lookback_days=60).get(sym)
            if bars is None or bars.empty:
                continue
            should_exit, reason = self._strategy.should_exit(sym, pos, bars)
            if should_exit:
                logger.info("[AB] Early exit %s: %s", sym, reason)
                self._client.close_position(sym)
                pnl = float(pos["unrealized_pl"])
                self._risk.update_pnl(pnl)
                self._record_trade_close(sym, pos, pnl, reason)

    def _enforce_max_hold(self):
        """Force-close positions held longer than max_hold_days (default 10)."""
        max_days = self._cfg.get("max_hold_days", 10)
        with get_connection() as conn:
            old_positions = conn.execute("""
                SELECT symbol, entry_time FROM open_positions
                WHERE bot_type = 'ab'
                AND entry_time < datetime('now', ?)
            """, (f"-{max_days} days",)).fetchall()

        for row in old_positions:
            sym = row["symbol"]
            pos = self._client.get_position(sym)
            if pos:
                logger.info("[AB] Max hold exceeded for %s — closing", sym)
                self._client.close_position(sym)
                pnl = float(pos.get("unrealized_pl", 0))
                self._risk.update_pnl(pnl)
                self._record_trade_close(sym, pos, pnl, "max_hold_time")

    # ------------------------------------------------------------------
    # ML retraining
    # ------------------------------------------------------------------

    def _retrain_ml(self):
        """Self-improvement: retrain ML on AB's picks + own outcomes."""
        try:
            # Step 1: Standard direction model retraining on SPY
            start = datetime.now(ET) - timedelta(days=365)
            bars = self._data.get_historical_bars("SPY", TimeFrame.Day, start)
            if not bars.empty:
                metrics = self._strategy.train_ml(bars)
                logger.info("[AB] Standard ML retrained: %s", metrics)

            # Step 2: AB-style model retraining (AB picks + outcomes)
            with get_connection() as conn:
                metrics2 = self._predictor.train_on_outcomes(conn)
                logger.info("[AB] Outcome model: %s", metrics2)

                metrics3 = self._predictor.train_on_ab_picks(conn)
                logger.info("[AB] AB-pick style model: %s", metrics3)

        except Exception as exc:
            logger.error("[AB] ML retraining failed: %s", exc)

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
                "ab", signal.symbol,
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
                "ab", symbol, position.get("side", ""),
                entry_price, exit_price, qty,
                pnl, pnl_pct, reason,
            ))
            conn.execute(
                "DELETE FROM open_positions WHERE bot_type='ab' AND symbol=?",
                (symbol,)
            )
