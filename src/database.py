"""SQLite database layer. All trade history, PDT tracking, and system events."""

import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from src.config import get_config

logger = logging.getLogger(__name__)


def get_db_path() -> str:
    return get_config()["monitoring"]["db_path"]


def init_db():
    """Create all tables if they don't exist. Safe to call multiple times."""
    db_path = get_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                asset_class TEXT DEFAULT 'stock',
                entry_time TEXT,
                exit_time TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                stop_loss REAL,
                take_profit REAL,
                pnl REAL,
                pnl_pct REAL,
                exit_reason TEXT,
                is_day_trade INTEGER DEFAULT 0,
                entry_reason TEXT,
                ml_confidence REAL,
                alpaca_order_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS open_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                asset_class TEXT DEFAULT 'stock',
                entry_time TEXT,
                entry_price REAL,
                quantity REAL,
                stop_loss REAL,
                take_profit REAL,
                entry_reason TEXT,
                ml_confidence REAL,
                alpaca_order_id TEXT,
                stop_order_id TEXT,
                tp_order_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS pdt_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                bot_type TEXT NOT NULL,
                starting_capital REAL,
                ending_capital REAL,
                total_pnl REAL,
                pnl_pct REAL,
                num_trades INTEGER DEFAULT 0,
                num_wins INTEGER DEFAULT 0,
                num_losses INTEGER DEFAULT 0,
                win_rate REAL,
                day_trades_used INTEGER DEFAULT 0,
                max_drawdown_pct REAL,
                trading_halted INTEGER DEFAULT 0,
                halt_reason TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, bot_type)
            );

            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                bot_type TEXT,
                message TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_trades_bot_date ON trades(bot_type, entry_time);
            CREATE INDEX IF NOT EXISTS idx_pdt_log_date ON pdt_log(bot_type, trade_date);
        """)
    logger.info("Database initialized at %s", db_path)


@contextmanager
def get_connection():
    db_path = get_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def log_event(event_type: str, message: str, bot_type: str = ""):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO system_events (event_type, bot_type, message) VALUES (?, ?, ?)",
            (event_type, bot_type, message),
        )
