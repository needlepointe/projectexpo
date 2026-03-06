---
name: daily-report
description: Generate daily P&L and performance report
disable-model-invocation: true
---

Generate today's trading performance report:
1. Run: python main.py --report
2. Read the output report from logs/report_YYYY-MM-DD.txt
3. Highlight any concerning metrics:
   - Win rate below target (45% day, 50% swing)
   - PDT trades running low (< 1 remaining)
   - Daily loss approaching limit (>3%)
   - Any halted trading alerts
4. Print the formatted report
