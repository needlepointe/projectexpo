---
name: kill-switch
description: Emergency kill switch - immediately closes ALL open positions and halts trading
disable-model-invocation: true
---

EMERGENCY: Close all positions and stop all trading bots immediately.

Steps:
1. Run: python -m src.execution.kill_switch --confirm --reason "Manual kill switch invoked"
2. Verify output shows positions_closed count
3. Report what positions were closed and the log file location
4. Remind user to restart the system only after investigating the cause
