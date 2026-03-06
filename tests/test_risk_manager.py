"""
Unit tests for RiskManager — the most critical module.
All risk rules must hold under every condition.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.risk.risk_manager import RiskManager, RiskError


@pytest.fixture
def risk(tmp_path, monkeypatch):
    """Risk manager with mocked DB and config."""
    cfg = {
        "risk": {
            "max_risk_per_trade_pct": 0.01,
            "max_position_size_pct": 0.20,
            "daily_loss_limit_pct": 0.05,
            "hard_stop_loss_pct": 0.10,
            "min_reward_risk_ratio": 2.0,
        },
        "pdt": {
            "max_day_trades_per_5_days": 3,
            "rolling_period_days": 5,
        },
        "monitoring": {"db_path": str(tmp_path / "test.db")},
    }
    with patch("src.risk.risk_manager.get_config", return_value=cfg):
        with patch("src.risk.risk_manager.get_connection") as mock_conn:
            mock_conn.return_value.__enter__ = MagicMock()
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            rm = RiskManager("day", 500.0)
            rm._today_starting_capital = 500.0
            yield rm


class TestPositionSizing:
    def test_basic_long_position(self, risk):
        pos = risk.calculate_position(entry_price=100.0, stop_price=99.0)
        assert pos["shares"] > 0
        assert pos["stop_loss"] == pytest.approx(99.0)
        assert pos["take_profit"] == pytest.approx(102.0)  # 2:1 R:R
        assert pos["rr_ratio"] >= 2.0

    def test_position_respects_max_size(self, risk):
        # Entry at $500 with $500 capital → max 20% = $100 worth
        pos = risk.calculate_position(entry_price=500.0, stop_price=490.0)
        position_value = pos["shares"] * 500.0
        assert position_value <= 500.0 * 0.20 + 0.01  # Allow tiny float error

    def test_zero_risk_raises(self, risk):
        with pytest.raises(RiskError):
            risk.calculate_position(entry_price=100.0, stop_price=100.0)

    def test_tight_stop_raises(self, risk):
        with pytest.raises(RiskError):
            risk.calculate_position(entry_price=100.0, stop_price=100.0001)

    def test_take_profit_meets_min_rr(self, risk):
        pos = risk.calculate_position(entry_price=50.0, stop_price=49.0)
        risk_dist = abs(50.0 - pos["stop_loss"])
        reward_dist = abs(pos["take_profit"] - 50.0)
        assert reward_dist / risk_dist >= 2.0


class TestLossLimits:
    def test_daily_loss_halts_trading(self, risk):
        risk.current_capital = 500.0
        risk._today_starting_capital = 500.0
        # Simulate 5% loss
        risk._today_pnl = -25.0
        risk._check_loss_limits()
        ok, reason = risk.can_trade()
        assert not ok
        assert "limit" in reason.lower()

    def test_hard_stop_halts_trading(self, risk):
        risk.current_capital = 500.0
        risk._today_starting_capital = 500.0
        # Simulate 10% loss
        risk._today_pnl = -50.0
        risk._check_loss_limits()
        ok, reason = risk.can_trade()
        assert not ok
        assert "stop" in reason.lower() or "halted" in reason.lower()

    def test_normal_loss_allows_trading(self, risk):
        risk.current_capital = 490.0
        risk._today_starting_capital = 500.0
        risk._today_pnl = -10.0  # Only 2% loss
        risk._check_loss_limits()
        ok, _ = risk.can_trade()
        assert ok

    def test_resume_after_halt(self, risk):
        risk._halt("test halt")
        ok, _ = risk.can_trade()
        assert not ok
        risk.resume_trading()
        ok, _ = risk.can_trade()
        assert ok


class TestTradeValidation:
    def test_valid_long_trade_approved(self, risk):
        ok, reason = risk.validate_trade(
            symbol="AAPL",
            entry_price=150.0,
            stop_price=148.5,
            take_profit=153.0,
            shares=5,
            is_day_trade=False,
        )
        assert ok, f"Expected approval, got: {reason}"

    def test_no_stop_loss_rejected(self, risk):
        ok, reason = risk.validate_trade(
            symbol="AAPL",
            entry_price=150.0,
            stop_price=0,
            take_profit=153.0,
            shares=5,
        )
        assert not ok
        assert "stop" in reason.lower()

    def test_bad_rr_ratio_rejected(self, risk):
        # R:R of 0.5 (below 2.0 minimum)
        ok, reason = risk.validate_trade(
            symbol="AAPL",
            entry_price=150.0,
            stop_price=148.0,    # $2 risk
            take_profit=151.0,   # $1 reward → 0.5:1
            shares=5,
        )
        assert not ok
        assert "r:r" in reason.lower() or "ratio" in reason.lower()

    def test_oversized_position_rejected(self, risk):
        # 1000 shares of $100 stock = $100,000 with only $500 capital
        ok, reason = risk.validate_trade(
            symbol="AAPL",
            entry_price=100.0,
            stop_price=99.0,
            take_profit=102.0,
            shares=1000,
        )
        assert not ok

    def test_halted_bot_rejects_all_trades(self, risk):
        risk._halt("test")
        ok, reason = risk.validate_trade(
            symbol="AAPL",
            entry_price=150.0,
            stop_price=148.5,
            take_profit=153.0,
            shares=5,
        )
        assert not ok
        assert "halt" in reason.lower()


class TestPDT:
    def test_pdt_limit_blocks_day_trades(self, risk):
        with patch.object(risk, "get_day_trades_used", return_value=3):
            ok, reason = risk.can_day_trade()
        assert not ok
        assert "pdt" in reason.lower() or "limit" in reason.lower()

    def test_pdt_allows_trades_under_limit(self, risk):
        with patch.object(risk, "get_day_trades_used", return_value=2):
            ok, reason = risk.can_day_trade()
        assert ok

    def test_daily_trade_validation_blocked_by_pdt(self, risk):
        with patch.object(risk, "get_day_trades_used", return_value=3):
            ok, reason = risk.validate_trade(
                symbol="AAPL",
                entry_price=150.0,
                stop_price=148.5,
                take_profit=153.0,
                shares=5,
                is_day_trade=True,
            )
        assert not ok
