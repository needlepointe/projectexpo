"""
Risk Manager — enforces every risk rule non-negotiably.
No trade executes without passing validate_trade().
"""

import logging
from datetime import date, timedelta
from typing import Optional

from src.config import get_config
from src.database import get_connection, log_event

logger = logging.getLogger(__name__)


class RiskError(Exception):
    """Raised when a trade fails risk validation."""


class RiskManager:
    """
    Central risk enforcement layer.

    Rules (all non-negotiable):
      1. Max 1-2% risk per trade
      2. Stop-loss on EVERY trade
      3. Minimum 2:1 reward:risk ratio
      4. Max 20% of capital in a single position
      5. Daily loss limit 5% → trading pauses
      6. Hard stop 10% single-day loss → bot halts
      7. PDT: max 3 day trades per rolling 5-day window
    """

    def __init__(self, bot_type: str, starting_capital: float):
        cfg = get_config()
        self.bot_type = bot_type
        self.starting_capital = starting_capital
        self.current_capital = starting_capital

        self._risk_cfg = cfg["risk"]
        self._pdt_cfg = cfg["pdt"]

        self._today_pnl: float = 0.0
        self._today_starting_capital: float = starting_capital
        self._trading_halted: bool = False
        self._halt_reason: str = ""

    # ------------------------------------------------------------------
    # Daily lifecycle
    # ------------------------------------------------------------------

    def reset_daily(self, current_capital: float):
        """Call at market open each day."""
        self.current_capital = current_capital
        self._today_starting_capital = current_capital
        self._today_pnl = 0.0
        self._trading_halted = False
        self._halt_reason = ""
        logger.info("[%s] Daily reset. Capital: $%.2f", self.bot_type.upper(), current_capital)

    def update_pnl(self, realized_pnl: float):
        """Call after each closed trade."""
        self._today_pnl += realized_pnl
        self.current_capital += realized_pnl
        self._check_loss_limits()

    def set_capital(self, capital: float):
        self.current_capital = capital
        self._today_pnl = capital - self._today_starting_capital
        self._check_loss_limits()

    # ------------------------------------------------------------------
    # Loss limit enforcement
    # ------------------------------------------------------------------

    def _check_loss_limits(self):
        if self._today_starting_capital <= 0:
            return
        daily_loss_pct = self._today_pnl / self._today_starting_capital

        if daily_loss_pct <= -self._risk_cfg["hard_stop_loss_pct"]:
            self._halt(f"Hard stop: {daily_loss_pct:.1%} daily loss (limit {-self._risk_cfg['hard_stop_loss_pct']:.0%})")
        elif daily_loss_pct <= -self._risk_cfg["daily_loss_limit_pct"]:
            self._halt(f"Daily loss limit: {daily_loss_pct:.1%} (limit {-self._risk_cfg['daily_loss_limit_pct']:.0%})")

    def _halt(self, reason: str):
        if not self._trading_halted:
            self._trading_halted = True
            self._halt_reason = reason
            log_event("halt", reason, self.bot_type)
            logger.warning("[%s] TRADING HALTED: %s", self.bot_type.upper(), reason)

    def resume_trading(self):
        """Manually resume after investigating a halt."""
        self._trading_halted = False
        self._halt_reason = ""
        log_event("resume", "Trading manually resumed", self.bot_type)

    # ------------------------------------------------------------------
    # Trade gating
    # ------------------------------------------------------------------

    def can_trade(self) -> tuple[bool, str]:
        if self._trading_halted:
            return False, f"Halted: {self._halt_reason}"
        if self.current_capital < 10:
            return False, "Insufficient capital"
        return True, "OK"

    # ------------------------------------------------------------------
    # PDT tracking
    # ------------------------------------------------------------------

    def get_day_trades_used(self) -> int:
        """Count day trades in the rolling 5-day window."""
        cutoff = (date.today() - timedelta(days=self._pdt_cfg["rolling_period_days"])).isoformat()
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM pdt_log WHERE bot_type = ? AND trade_date >= ?",
                (self.bot_type, cutoff),
            ).fetchone()
        return row["cnt"] if row else 0

    def can_day_trade(self) -> tuple[bool, str]:
        used = self.get_day_trades_used()
        max_dt = self._pdt_cfg["max_day_trades_per_5_days"]
        if used >= max_dt:
            return False, f"PDT limit: {used}/{max_dt} day trades used in 5-day window"
        return True, f"{max_dt - used} day trade(s) remaining"

    def record_day_trade(self, symbol: str):
        today = date.today().isoformat()
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO pdt_log (bot_type, symbol, trade_date) VALUES (?, ?, ?)",
                (self.bot_type, symbol, today),
            )
        used = self.get_day_trades_used()
        logger.info("[%s] PDT day trade recorded: %s (%d/3)", self.bot_type.upper(), symbol, used)

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position(
        self,
        entry_price: float,
        stop_price: float,
        risk_pct: Optional[float] = None,
    ) -> dict:
        """
        Calculate shares, stop-loss, and take-profit for a trade.

        Returns dict with keys: shares, stop_loss, take_profit, risk_amount, rr_ratio
        """
        if risk_pct is None:
            risk_pct = self._risk_cfg["max_risk_per_trade_pct"]
        risk_pct = min(risk_pct, 0.02)  # Hard cap at 2%

        risk_amount = self.current_capital * risk_pct
        risk_per_share = abs(entry_price - stop_price)

        if risk_per_share < 0.01:
            raise RiskError(f"Stop too close to entry: ${risk_per_share:.4f}/share")

        shares = int(risk_amount / risk_per_share)

        # Cap by max position size
        max_shares = int((self.current_capital * self._risk_cfg["max_position_size_pct"]) / entry_price)
        shares = min(shares, max_shares)

        if shares <= 0:
            raise RiskError("Position size rounds to zero — increase capital or widen stop")

        # Calculate minimum take-profit for 2:1 R:R
        risk_dist = abs(entry_price - stop_price)
        min_reward = risk_dist * self._risk_cfg["min_reward_risk_ratio"]

        is_long = stop_price < entry_price
        take_profit = entry_price + min_reward if is_long else entry_price - min_reward
        rr_ratio = min_reward / risk_dist

        return {
            "shares": shares,
            "stop_loss": round(stop_price, 4),
            "take_profit": round(take_profit, 4),
            "risk_amount": round(risk_amount, 2),
            "position_value": round(shares * entry_price, 2),
            "rr_ratio": round(rr_ratio, 2),
        }

    # ------------------------------------------------------------------
    # Full pre-trade validation
    # ------------------------------------------------------------------

    def validate_trade(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        take_profit: float,
        shares: int,
        is_day_trade: bool = False,
    ) -> tuple[bool, str]:
        """
        Final gate before any order is placed.
        Returns (approved: bool, reason: str).
        """
        ok, reason = self.can_trade()
        if not ok:
            return False, reason

        if is_day_trade:
            ok, reason = self.can_day_trade()
            if not ok:
                return False, reason

        if stop_price <= 0:
            return False, "Stop-loss required on every trade"

        if shares <= 0:
            return False, "Invalid share count"

        risk = abs(entry_price - stop_price)
        if risk < 0.001:
            return False, "Risk distance is effectively zero"

        reward = abs(take_profit - entry_price)
        rr = reward / risk
        min_rr = self._risk_cfg["min_reward_risk_ratio"]
        if rr < min_rr:
            return False, f"R:R {rr:.2f} below minimum {min_rr}"

        position_value = shares * entry_price
        max_pos = self.current_capital * self._risk_cfg["max_position_size_pct"]
        if position_value > max_pos:
            return False, f"Position ${position_value:.0f} exceeds 20% cap (${max_pos:.0f})"

        if position_value > self.current_capital:
            return False, f"Insufficient capital: need ${position_value:.0f}, have ${self.current_capital:.0f}"

        logger.info(
            "[%s] APPROVED %s: %d shares @ $%.2f | SL $%.2f | TP $%.2f | R:R %.1f",
            self.bot_type.upper(), symbol, shares, entry_price, stop_price, take_profit, rr,
        )
        return True, f"Approved — R:R {rr:.1f}, position ${position_value:.0f}"

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        day_trades_used = self.get_day_trades_used()
        max_dt = self._pdt_cfg["max_day_trades_per_5_days"]
        daily_pnl_pct = (
            self._today_pnl / self._today_starting_capital if self._today_starting_capital > 0 else 0
        )
        return {
            "bot_type": self.bot_type,
            "capital": self.current_capital,
            "today_pnl": self._today_pnl,
            "today_pnl_pct": daily_pnl_pct,
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "day_trades_used": day_trades_used,
            "day_trades_remaining": max(0, max_dt - day_trades_used),
        }
