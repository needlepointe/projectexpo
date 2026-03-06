# PROJECT APEX — SESSION LOGS
> Source of truth for cross-session context. Read the last 3 entries to restore state.

---

## LOG FORMAT
```
[TIMESTAMP] - [TASK_NAME]
- Action: (What was done)
- Technical Detail: (Raw data / error codes / values — DO NOT SUMMARIZE)
- Logic State: (Current variables / parameters for AI models)
- Git Hash: (Reference the latest commit)
```

---

## [2026-03-06 | SESSION 1] - PROTOCOL INITIALIZATION

- **Action:** Initialized state management protocol. Explored project structure. Git repo managed by Claude Code (lock active — do not run concurrent git ops from Cowork).
- **Technical Detail:**
  - Entry point: `main.py`
  - Bots: `src/bots/day_trader.py`, `src/bots/swing_trader.py`
  - ML: `src/ml/signal_predictor.py`
  - Execution: `src/execution/alpaca_client.py`, `src/execution/kill_switch.py`
  - Backtesting: `src/backtesting/runner.py`
  - Monitoring: `src/monitoring/reporter.py`
  - Config: `src/config.py`, `config/` directory
  - Data: `src/data/market_data.py`
  - DB: `src/database.py`
  - Day trader: 15-min bars on QQQ, 90-day training window, scans every 5 min (9:45–3:55 ET)
  - Swing trader: Daily bars on SPY, EOD scan @4:05 ET, position check every 30 min
  - Scheduler: APScheduler (BlockingScheduler, ET timezone)
  - Paper/live mode: controlled via `src/config.py → is_paper_trading()`
- **Logic State:** Unknown — Claude Code is actively working; current task not confirmed.
- **Git Hash:** Managed by Claude Code (Cowork read-only on git during active CC session)

---
<!-- New entries go ABOVE this line, most recent first -->
