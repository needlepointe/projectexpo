"""
Daily P&L reporter and real-time monitoring.
- Queries SQLite for today's trades
- Calculates win rate, P&L, PDT usage
- Writes daily report to logs/
- Optional email alerts
"""

import logging
import smtplib
import json
from datetime import date, datetime
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config import get_config
from src.database import get_connection

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class Reporter:
    def __init__(self):
        self._cfg = get_config()["monitoring"]
        self._log_dir = Path(self._cfg["log_dir"])
        self._log_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Daily report
    # ------------------------------------------------------------------

    def generate_daily_report(self, bot_type: str = "all") -> dict:
        """
        Generate today's performance report for one or both bots.
        Returns report dict and writes to logs/.
        """
        today = date.today().isoformat()
        report = {
            "date": today,
            "generated_at": datetime.now(ET).isoformat(),
        }

        bots = ["day", "swing"] if bot_type == "all" else [bot_type]

        for bt in bots:
            report[bt] = self._bot_daily_stats(bt, today)

        # Write to file
        report_path = self._log_dir / f"report_{today}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        text_report = self._format_text_report(report)
        text_path = self._log_dir / f"report_{today}.txt"
        with open(text_path, "w") as f:
            f.write(text_report)

        logger.info("Daily report written to %s", text_path)
        print(text_report)

        if self._cfg.get("email_alerts_enabled"):
            self._send_email(f"Trading Report {today}", text_report)

        return report

    def _bot_daily_stats(self, bot_type: str, today: str) -> dict:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT pnl, pnl_pct, exit_reason, is_day_trade
                FROM trades
                WHERE bot_type = ? AND date(entry_time) = ?
                ORDER BY created_at
            """, (bot_type, today)).fetchall()

            open_positions = conn.execute("""
                SELECT COUNT(*) AS cnt FROM open_positions WHERE bot_type = ?
            """, (bot_type,)).fetchone()

            pdt_count = conn.execute("""
                SELECT COUNT(*) AS cnt FROM pdt_log
                WHERE bot_type = ? AND trade_date >= date('now', '-5 days')
            """, (bot_type,)).fetchone()

        trades = [dict(r) for r in rows]
        pnls = [t["pnl"] for t in trades if t["pnl"] is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        return {
            "num_trades": len(trades),
            "num_wins": len(wins),
            "num_losses": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
            "open_positions": int(open_positions["cnt"]) if open_positions else 0,
            "day_trades_used_5d": int(pdt_count["cnt"]) if pdt_count else 0,
            "day_trades_remaining": max(0, 3 - (int(pdt_count["cnt"]) if pdt_count else 0)),
        }

    def _format_text_report(self, report: dict) -> str:
        lines = [
            f"{'='*55}",
            f"DAILY TRADING REPORT — {report['date']}",
            f"Generated: {report['generated_at']}",
            f"{'='*55}",
        ]

        for bt in ["day", "swing"]:
            if bt not in report:
                continue
            s = report[bt]
            lines += [
                f"\n[{bt.upper()} TRADER]",
                f"  Trades:       {s['num_trades']} ({s['num_wins']}W / {s['num_losses']}L)",
                f"  Win rate:     {s['win_rate']}%",
                f"  Total P&L:    ${s['total_pnl']:+.2f}",
                f"  Avg win:      ${s['avg_win']:+.2f}",
                f"  Avg loss:     ${s['avg_loss']:+.2f}",
                f"  Open positions: {s['open_positions']}",
                f"  PDT used (5d): {s['day_trades_used_5d']}/3 ({s['day_trades_remaining']} remaining)",
            ]

        lines.append(f"\n{'='*55}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Real-time P&L
    # ------------------------------------------------------------------

    def get_realtime_pnl(self) -> dict:
        """Get current open position P&L from Alpaca account."""
        from src.execution.alpaca_client import AlpacaClient
        try:
            client = AlpacaClient()
            positions = client.get_positions()
            account = client.get_account()

            total_unrealized = sum(p["unrealized_pl"] for p in positions)
            return {
                "equity": account["equity"],
                "cash": account["cash"],
                "open_positions": len(positions),
                "unrealized_pnl": round(total_unrealized, 2),
                "positions": [
                    {
                        "symbol": p["symbol"],
                        "qty": p["qty"],
                        "unrealized_pl": round(p["unrealized_pl"], 2),
                        "unrealized_plpc": round(p["unrealized_plpc"] * 100, 2),
                    }
                    for p in positions
                ],
            }
        except Exception as exc:
            logger.error("Real-time P&L fetch failed: %s", exc)
            return {"error": str(exc)}

    def log_alert(self, level: str, message: str):
        """Log an alert and optionally send email."""
        timestamp = datetime.now(ET).isoformat()
        alert_path = self._log_dir / "alerts.log"
        with open(alert_path, "a") as f:
            f.write(f"[{timestamp}] {level.upper()}: {message}\n")
        logger.warning("ALERT [%s]: %s", level.upper(), message)

        if self._cfg.get("email_alerts_enabled") and level in ("critical", "error"):
            self._send_email(f"Trading Alert: {level.upper()}", message)

    # ------------------------------------------------------------------
    # Email alerts
    # ------------------------------------------------------------------

    def _send_email(self, subject: str, body: str):
        cfg = self._cfg
        if not cfg.get("email_to") or not cfg.get("email_from"):
            return
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = cfg["email_from"]
            msg["To"] = cfg["email_to"]
            with smtplib.SMTP(cfg["smtp_server"], cfg.get("smtp_port", 587)) as server:
                server.starttls()
                server.sendmail(cfg["email_from"], cfg["email_to"], msg.as_string())
            logger.info("Alert email sent: %s", subject)
        except Exception as exc:
            logger.error("Failed to send email: %s", exc)
