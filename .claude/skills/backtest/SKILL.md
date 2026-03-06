---
name: backtest
description: "Run backtest for a strategy. Args: --strategy [day|swing] --symbol TICKER --days 365"
---

Run a backtest with these steps:
1. Parse the arguments: strategy (day or swing), symbol, lookback days (default 365)
2. Run: python main.py --backtest --strategy $STRATEGY --symbol $SYMBOL --days $DAYS
3. Parse the results output for: total return, win rate, max drawdown, Sharpe ratio, number of trades
4. Flag if results don't meet minimum thresholds:
   - Day trader: win rate > 45%, max drawdown < 15%, positive expectancy
   - Swing trader: win rate > 50%, max drawdown < 15%, positive expectancy
5. Output verdict: READY_FOR_PAPER, NEEDS_ADJUSTMENT, or FAIL with specific suggestions
