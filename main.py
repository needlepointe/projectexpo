"""
Trading System Entry Point

Usage:
  python main.py --mode paper --bot both        # Run both bots (paper)
  python main.py --mode paper --bot day         # Day trader only
  python main.py --mode paper --bot swing       # Swing trader only
  python main.py --backtest --strategy day --symbol SPY
  python main.py --backtest --strategy swing --symbol AAPL
  python main.py --train --strategy day         # Train ML models
  python main.py --report                       # Generate daily report
  python main.py --kill                         # Emergency kill switch
  python main.py --status                       # Show account status
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.config import get_config, is_paper_trading
from src.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/trading.log"),
    ],
)
logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def run_bots(bot_arg: str):
    """Initialize and schedule trading bots."""
    from src.bots.day_trader import DayTraderBot
    from src.bots.swing_trader import SwingTraderBot
    from src.monitoring.reporter import Reporter

    cfg = get_config()
    mode = "PAPER" if is_paper_trading() else "LIVE ⚠️"
    logger.info("=" * 60)
    logger.info("TRADING SYSTEM STARTING [%s]", mode)
    logger.info("Day capital:   $%.2f", cfg["account"]["starting_capital_day"])
    logger.info("Swing capital: $%.2f", cfg["account"]["starting_capital_swing"])
    logger.info("PDT limit:     %d/5 days", cfg["pdt"]["max_day_trades_per_5_days"])
    logger.info("=" * 60)

    init_db()
    scheduler = BlockingScheduler(timezone=ET)
    reporter = Reporter()

    bots = []

    # -- Day Trader --
    if bot_arg in ("day", "both") and cfg["day_trader"]["enabled"]:
        day_bot = DayTraderBot()
        day_bot.start()
        bots.append(day_bot)

        # Scan every 5 minutes during market hours
        scheduler.add_job(
            day_bot.run_cycle,
            IntervalTrigger(minutes=5),
            id="day_cycle",
            name="Day Trader Scan",
        )

        # Market open at 9:45 AM ET
        scheduler.add_job(
            day_bot.on_market_open,
            CronTrigger(day_of_week="mon-fri", hour=9, minute=45, timezone=ET),
            id="day_open",
            name="Day Trader Open",
        )

        # Force close at 3:55 PM ET
        scheduler.add_job(
            day_bot.on_market_close,
            CronTrigger(day_of_week="mon-fri", hour=15, minute=55, timezone=ET),
            id="day_close",
            name="Day Trader Force Close",
        )

        logger.info("[DAY] Scheduled: scan every 5min, open @9:45, close @3:55 ET")

    # -- Swing Trader --
    if bot_arg in ("swing", "both") and cfg["swing_trader"]["enabled"]:
        swing_bot = SwingTraderBot()
        swing_bot.start()
        bots.append(swing_bot)

        # Market open check
        scheduler.add_job(
            swing_bot.on_market_open,
            CronTrigger(day_of_week="mon-fri", hour=9, minute=45, timezone=ET),
            id="swing_open",
            name="Swing Trader Open",
        )

        # EOD scan at 4:05 PM ET
        scheduler.add_job(
            swing_bot.on_market_close,
            CronTrigger(day_of_week="mon-fri", hour=16, minute=5, timezone=ET),
            id="swing_scan",
            name="Swing Trader EOD Scan",
        )

        # Position check during market hours (every 30 min)
        scheduler.add_job(
            swing_bot.run_cycle,
            CronTrigger(day_of_week="mon-fri", hour="10-15", minute="0,30", timezone=ET),
            id="swing_check",
            name="Swing Position Check",
        )

        logger.info("[SWING] Scheduled: position check every 30min, EOD scan @4:05 ET")

    # -- Daily report --
    scheduler.add_job(
        reporter.generate_daily_report,
        CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone=ET),
        id="daily_report",
        name="Daily Report",
    )

    logger.info("Scheduler starting. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        for bot in bots:
            bot.stop()


def run_backtest(strategy: str, symbol: str, days: int):
    """Run backtesting on historical data."""
    from src.backtesting.runner import Backtester

    logger.info("Starting backtest: %s on %s (%d days)", strategy, symbol, days)
    init_db()

    backtester = Backtester()
    start = datetime.now(ET) - timedelta(days=days)

    result = backtester.run(strategy=strategy, symbol=symbol, start=start)
    print(result.summary())

    if result.passes_targets(strategy):
        print(f"\n[READY] READY FOR PAPER TRADING -- All targets met")
    else:
        print(f"\n[NOT READY] Review strategy parameters")

    return result


def run_training(strategy: str):
    """Train ML models on historical data."""
    from src.data.market_data import MarketDataClient
    from src.ml.signal_predictor import SignalPredictor
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    logger.info("Training ML model for %s strategy...", strategy)
    data = MarketDataClient()
    predictor = SignalPredictor(strategy)

    cfg = get_config()

    if strategy == "day":
        # Use 90 days of 15-min bars (~2,340 bars) — matches the backtester timeframe.
        # 5-min bars were swapped to 15-min because: (a) 15-min ATR is 3× larger so
        # the 3×ATR take-profit target is actually reachable within a trading day,
        # and (b) signals are far less noisy (26 bars/day vs 78).
        days = 90
        tf = TimeFrame(15, TimeFrameUnit.Minute)
        symbol = "QQQ"
    else:
        days = cfg["ml"]["training_lookback_days"]
        tf = TimeFrame.Day
        symbol = "SPY"

    start = datetime.now(ET) - timedelta(days=days)
    logger.info("Fetching %d days of %s bars for %s (this may take 30-60s)...", days, strategy, symbol)

    bars = data.get_historical_bars(symbol, tf, start)

    if bars.empty:
        logger.error(
            "No training data returned from Alpaca. "
            "Check that your API key is valid and the account is active."
        )
        return

    logger.info("Fetched %d bars. Training...", len(bars))
    metrics = predictor.train(bars)
    print("\n=== ML TRAINING COMPLETE ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def show_status():
    """Display current account and position status."""
    from src.execution.alpaca_client import AlpacaClient
    from src.monitoring.reporter import Reporter

    client = AlpacaClient()
    reporter = Reporter()

    account = client.get_account()
    pnl = reporter.get_realtime_pnl()

    print("\n=== ACCOUNT STATUS ===")
    print(f"  Equity:       ${account['equity']:,.2f}")
    print(f"  Cash:         ${account['cash']:,.2f}")
    print(f"  Buying power: ${account['buying_power']:,.2f}")
    print(f"  Day trades:   {account['day_trade_count']}")
    print(f"  PDT flag:     {account['pattern_day_trader']}")

    print(f"\n=== OPEN POSITIONS ({pnl.get('open_positions', 0)}) ===")
    for pos in pnl.get("positions", []):
        print(f"  {pos['symbol']:8s} qty={pos['qty']:6.0f} P&L={pos['unrealized_pl']:+8.2f} ({pos['unrealized_plpc']:+.1f}%)")

    print(f"\n  Total unrealized P&L: ${pnl.get('unrealized_pnl', 0):+.2f}")


def emergency_kill():
    """Invoke the kill switch."""
    from src.execution.kill_switch import kill

    print("\n⚠️  EMERGENCY KILL SWITCH")
    print("This will immediately close ALL positions and cancel ALL orders.")
    confirm = input("Type 'KILL' to confirm: ").strip()

    if confirm == "KILL":
        result = kill(reason="CLI kill switch", confirm=True)
        print("\n=== KILL SWITCH EXECUTED ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        print("Cancelled. No action taken.")


def main():
    parser = argparse.ArgumentParser(description="AI Trading System")
    parser.add_argument("--bot", choices=["day", "swing", "both"], default="both")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting")
    parser.add_argument("--strategy", choices=["day", "swing"], default="day")
    parser.add_argument("--symbol", default="SPY", help="Symbol for backtest/train")
    parser.add_argument("--days", type=int, default=365, help="Lookback days for backtest")
    parser.add_argument("--train", action="store_true", help="Train ML models")
    parser.add_argument("--report", action="store_true", help="Generate daily report")
    parser.add_argument("--status", action="store_true", help="Show account status")
    parser.add_argument("--kill", action="store_true", help="Emergency kill switch")

    args = parser.parse_args()

    # Ensure directories exist
    import os
    for d in ["logs", "data", "models"]:
        os.makedirs(d, exist_ok=True)

    if args.kill:
        emergency_kill()
    elif args.status:
        show_status()
    elif args.report:
        from src.monitoring.reporter import Reporter
        init_db()
        Reporter().generate_daily_report()
    elif args.backtest:
        run_backtest(args.strategy, args.symbol, args.days)
    elif args.train:
        run_training(args.strategy)
    else:
        # Default: run the live/paper trading bots
        run_bots(args.bot)


if __name__ == "__main__":
    main()
